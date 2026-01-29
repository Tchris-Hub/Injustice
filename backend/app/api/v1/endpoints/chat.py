"""
Chat Endpoints
---------------
Main chat interface for the AI Legal Advisor.
Handles conversations, messages, and escalations.
"""

import logging
import uuid
import tempfile
import os
from datetime import datetime, timezone
from typing import List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
import openai

from app.db.session import get_db
from app.db.models import (
    User,
    Conversation,
    Message,
    EscalationCase,
    EscalationState,
)
from app.schemas.chat import (
    MessageCreate,
    MessageResponse,
    AIResponse,
    ConversationSummary,
    ConversationDetail,
    ConversationList,
    EscalationRequest,
    EscalationResponse,
    SourceCitation,
    PublicChatRequest,
    PublicChatResponse,
    DocumentAnalysisRequest,
    DocumentAnalysisResponse,
    AnalysisResult,
    DocumentGenerationRequest,
    DocumentGenerationResponse
)
from app.api.deps import get_current_user
from app.services.rag_service import get_rag_service
from app.core.crypto import encrypt_text, decrypt_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


ESCALATION_RESPONSE_TIMES = {
    "critical": "Within 4 hours",
    "high": "Within 24 hours",
    "medium": "Within 48 hours",
    "low": "Within 5 business days",
}


# Legal disclaimer (always included)
LEGAL_DISCLAIMER = (
    "⚠️ **IMPORTANT NOTICE**: This information is for educational and informational "
    "purposes only. It does not constitute legal advice and should not be relied upon "
    "as such. Laws vary by jurisdiction and change over time. For advice specific to "
    "your situation, please consult a licensed attorney in Nigeria. Use of this service "
    "does not create an attorney-client relationship."
)


def _upsert_escalation_case(
    conversation: Conversation,
    *,
    db: AsyncSession,
    target_state: EscalationState,
    note: str,
    trigger_source: str,
    reason: Optional[str] = None,
    urgency: Optional[str] = None,
    contact_preference: Optional[str] = None,
    actor_user_id: Optional[uuid.UUID] = None,
) -> EscalationCase:
    """Create or update an escalation case with deterministic state changes."""
    case = conversation.escalation_case
    if case is None:
        case = EscalationCase(
            conversation_id=conversation.id,
            trigger_source=trigger_source,
            reason=encrypt_text(reason),
            urgency=urgency or "medium",
            contact_preference=contact_preference or "either",
            created_by_id=actor_user_id,
        )
        db.add(case)
        conversation.escalation_case = case
    else:
        case.trigger_source = trigger_source or case.trigger_source
        if reason:
            case.reason = encrypt_text(reason)
        if urgency:
            case.urgency = urgency
        if contact_preference:
            case.contact_preference = contact_preference
        if actor_user_id and case.created_by_id is None:
            case.created_by_id = actor_user_id

    case.append_state(target_state, note)
    conversation.is_escalated = True
    if conversation.escalated_at is None:
        conversation.escalated_at = datetime.now(timezone.utc)
    if reason:
        conversation.escalation_reason = encrypt_text(reason)

    return case


# ---------------------------------------------
# Send Message (Main Chat)
# ---------------------------------------------
@router.post(
    "/message",
    response_model=AIResponse,
    summary="Send a message and get AI response"
)
async def send_message(
    data: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Send a message to the AI Legal Advisor and receive a response.
    
    - **content**: Your question or message
    - **conversation_id**: (Optional) ID of existing conversation, or null for new
    
    The AI will:
    1. Acknowledge your situation with empathy
    2. Retrieve relevant sections from Nigerian law
    3. Provide clear, cited information
    4. Suggest practical next steps
    """
    rag_service = get_rag_service()
    
    # Get or create conversation (skip DB if suppressed)
    conversation = None
    if data.conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == data.conversation_id,
                Conversation.user_id == current_user.id
            )
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
    elif not data.suppress_storage:
        # Create new conversation only if storage is NOT suppressed
        conversation = Conversation(
            user_id=current_user.id,
            title=data.content[:50] + "..." if len(data.content) > 50 else data.content
        )
        db.add(conversation)
        await db.flush()
    
    # Save user message only if NOT suppressed
    if not data.suppress_storage and conversation:
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=encrypt_text(data.content)
        )
        db.add(user_message)
        await db.flush()
    
    # Assess risk level
    risk_assessment = rag_service.assess_risk(data.content)
    if conversation:
        conversation.risk_level = risk_assessment.get("risk_level", "medium")
        conversation.legal_topic = risk_assessment.get("legal_topic", "general_info")
    
    # Get conversation history
    history = []
    if conversation:
        history_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.desc())
            .limit(10)
        )
        history = [
            {"role": msg.role, "content": decrypt_text(msg.content)}
            for msg in reversed(history_result.scalars().all())
        ]
    
    # Generate AI response
    ai_result = rag_service.generate_response(
        user_message=data.content,
        conversation_history=history[:-1] if history else []
    )
    
    # Save AI response only if NOT suppressed
    ai_message_id = str(uuid.uuid4())
    ai_created_at = datetime.now(timezone.utc)
    
    if not data.suppress_storage and conversation:
        ai_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=encrypt_text(ai_result["content"]),
            sources=ai_result.get("sources"),
            model_version=ai_result.get("model_version"),
            confidence_score=ai_result.get("confidence_score"),
            contains_disclaimer=True
        )
        db.add(ai_message)
        await db.flush()
        ai_message_id = str(ai_message.id)
        ai_created_at = ai_message.created_at or ai_created_at
        conversation.updated_at = datetime.now(timezone.utc)
    
    # Check for escalation
    escalation_needed = bool(risk_assessment.get("escalation_needed", False))
    if escalation_needed:
        auto_reason = (
            risk_assessment.get("escalation_reason")
            or "Conversation flagged as high risk by automated assessment."
        )
        auto_urgency = risk_assessment.get("urgency") or conversation.risk_level or "medium"
        case = _upsert_escalation_case(
            conversation,
            db=db,
            target_state=EscalationState.pending_review,
            note=auto_reason,
            trigger_source="risk_engine",
            reason=auto_reason,
            urgency=auto_urgency,
            contact_preference="either",
            actor_user_id=None,
        )
        logger.warning(
            "High-risk conversation escalated: conversation=%s case=%s risk=%s",
            conversation.id,
            case.id,
            conversation.risk_level,
        )
    
    # Build response
    sources = [
        SourceCitation(**s) for s in ai_result.get("sources", [])
    ]
    
    message_response = MessageResponse(
        id=ai_message_id,
        conversation_id=str(conversation.id) if conversation else "incognito",
        role="assistant",
        content=ai_result["content"],
        sources=sources,
        confidence_score=ai_result.get("confidence_score"),
        created_at=ai_created_at
    )
    
    return AIResponse(
        message=message_response,
        conversation_id=str(conversation.id) if conversation else "incognito",
        conversation_title=conversation.title if conversation else "Incognito Chat",
        risk_level=conversation.risk_level if conversation else risk_assessment.get("risk_level"),
        escalation_recommended=escalation_needed,
        disclaimer=LEGAL_DISCLAIMER
    )


# ---------------------------------------------
# List Conversations
# ---------------------------------------------
@router.get(
    "/conversations",
    response_model=ConversationList,
    summary="List user's conversations"
)
async def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a paginated list of the user's conversations.
    
    - **page**: Page number (starting from 1)
    - **page_size**: Number of items per page (max 100)
    """
    offset = (page - 1) * page_size
    
    # Get total count
    count_result = await db.execute(
        select(func.count(Conversation.id)).where(
            Conversation.user_id == current_user.id
        )
    )
    total = count_result.scalar()
    
    # Get conversations with message count
    result = await db.execute(
        select(
            Conversation,
            func.count(Message.id).label("message_count")
        )
        .outerjoin(Message)
        .where(Conversation.user_id == current_user.id)
        .group_by(Conversation.id)
        .order_by(desc(Conversation.updated_at))
        .offset(offset)
        .limit(page_size)
    )
    
    conversations = []
    for conv, msg_count in result.all():
        conversations.append(ConversationSummary(
            id=str(conv.id),
            title=conv.title,
            legal_topic=conv.legal_topic,
            risk_level=conv.risk_level,
            is_escalated=conv.is_escalated,
            message_count=msg_count,
            created_at=conv.created_at,
            updated_at=conv.updated_at
        ))
    
    return ConversationList(
        conversations=conversations,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + page_size) < total
    )


# ---------------------------------------------
# Get Single Conversation
# ---------------------------------------------
@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get conversation details"
)
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a conversation with all its messages.
    """
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid conversation ID format"
        )
    
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conv_uuid,
            Conversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Get messages
    messages_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
    )
    
    messages = []
    for msg in messages_result.scalars().all():
        sources = None
        if msg.sources:
            sources = [SourceCitation(**s) for s in msg.sources]
        
        messages.append(MessageResponse(
            id=str(msg.id),
            conversation_id=str(msg.conversation_id),
            role=msg.role,
            content=decrypt_text(msg.content),
            sources=sources,
            confidence_score=msg.confidence_score,
            created_at=msg.created_at
        ))
    
    return ConversationDetail(
        id=str(conversation.id),
        title=conversation.title,
        legal_topic=conversation.legal_topic,
        risk_level=conversation.risk_level,
        is_escalated=conversation.is_escalated,
        escalation_reason=decrypt_text(conversation.escalation_reason),
        messages=messages,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at
    )


# ---------------------------------------------
# Delete Conversation
# ---------------------------------------------
@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation"
)
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a conversation and all its messages.
    This action cannot be undone.
    """
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid conversation ID format"
        )
    
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conv_uuid,
            Conversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    await db.delete(conversation)
    logger.info(f"Conversation deleted: {conversation_id} by user {current_user.id}")


# ---------------------------------------------
# Escalate to Human Lawyer
# ---------------------------------------------
@router.post(
    "/escalate",
    response_model=EscalationResponse,
    summary="Request human legal assistance"
)
async def escalate_conversation(
    data: EscalationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Escalate a conversation to a human legal aid partner.
    
    Use this when:
    - You have an urgent court deadline
    - Your situation involves criminal charges
    - You need personalized legal advice
    - The AI cannot fully address your needs
    
    - **conversation_id**: The conversation to escalate
    - **reason**: Why you need human assistance
    - **contact_preference**: How you prefer to be contacted
    - **urgency**: How urgent is your situation
    """
    try:
        conv_uuid = uuid.UUID(data.conversation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid conversation ID format"
        )
    
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conv_uuid,
            Conversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Mark as escalated
    reference = f"ESC-{datetime.now().strftime('%Y%m%d')}-{str(conversation.id)[:8].upper()}"

    case = _upsert_escalation_case(
        conversation,
        db=db,
        target_state=EscalationState.AWAITING_CONTACT,
        note=f"User requested escalation ({data.urgency}).",
        trigger_source="user_request",
        reason=data.reason,
        urgency=data.urgency,
        contact_preference=data.contact_preference,
        actor_user_id=current_user.id,
    )

    logger.warning(
        "Conversation escalation requested: conversation=%s case=%s user=%s urgency=%s",
        conversation.id,
        case.id,
        current_user.email,
        data.urgency,
    )

    # Add system message to conversation
    escalation_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=(
            f"🆘 **Escalation Request Submitted**\n\n"
            f"Your reference number is: **{reference}**\n\n"
            f"A legal aid partner will review your case and contact you "
            f"{ESCALATION_RESPONSE_TIMES[data.urgency].lower()}.\n\n"
            f"In the meantime, please:\n"
            f"- Gather any relevant documents\n"
            f"- Note down key dates and deadlines\n"
            f"- Keep your phone or email accessible\n\n"
            f"If this is an emergency involving immediate danger, "
            f"please contact the police or emergency services directly."
        ),
        contains_disclaimer=True
    )
    db.add(escalation_message)
    
    return EscalationResponse(
        success=True,
        message=(
            "Your case has been escalated to a legal aid partner. "
            "They will review your conversation and contact you soon."
        ),
        reference_number=reference,
        estimated_response_time=ESCALATION_RESPONSE_TIMES[data.urgency]
    )


# ---------------------------------------------
# Health Check (for the RAG service)
# ---------------------------------------------
@router.get(
    "/health",
    summary="Check chat service health"
)
async def chat_health():
    """Check if the RAG service is operational."""
    try:
        rag_service = get_rag_service()
        stats = rag_service.get_collection_stats()
        return {
            "status": "healthy",
            "rag_service": "operational",
            "vector_store": stats
        }
    except Exception as e:
        logger.error(f"Chat health check failed: {e}")
        return {
            "status": "degraded",
            "rag_service": "error",
            "error": str(e)
        }


# ==============================================
# PUBLIC ENDPOINTS (No Authentication Required)
# ==============================================
# These endpoints are for demo/public use without login

@router.post(
    "/public/message",
    response_model=PublicChatResponse,
    summary="Send a message (public/demo)",
    description="Public endpoint for chat - no authentication required."
)
async def send_message_public(
    data: PublicChatRequest
):
    """
    Send a message to the AI Legal Advisor.
    This is a simplified public endpoint for demo purposes.
    """
    try:
        # Zero-Trace Scrubbing: Explicitly avoid logging any user-identifying info
        rag_service = get_rag_service()
        
        # Generate AI response with empty history and no metadata tracking
        ai_result = rag_service.generate_response(
            user_message=data.message,
            conversation_history=[]
        )
        
        # Extract source titles as simple strings
        sources = []
        for source in ai_result.get("sources", []):
            if isinstance(source, dict):
                title = source.get("title", "")
                section = source.get("section", "")
                if section:
                    sources.append(f"{title} - {section}")
                else:
                    sources.append(title)
            else:
                sources.append(str(source))
        
        return PublicChatResponse(
            content=ai_result.get("content", "I apologize, but I couldn't generate a response."),
            sources=sources,
            confidence_score=str(ai_result.get("confidence_score", "low"))
        )
    except Exception as e:
        import traceback
        error_type = type(e).__name__
        error_msg = str(e)
        full_traceback = traceback.format_exc()
        
        logger.error(f"Public chat error [{error_type}]: {error_msg}")
        logger.error(f"Full traceback:\n{full_traceback}")
        
        # Return detailed error in development mode
        from app.core.config import settings
        if settings.debug:
            error_detail = f"[DEBUG ERROR]\nType: {error_type}\nMessage: {error_msg}\n\nCheck server logs for full traceback."
        else:
            error_detail = "I apologize, but I encountered an error processing your request. Please try again."
        
        return PublicChatResponse(
            content=error_detail,
            sources=[],
            confidence_score="0.0"
        )


@router.post(
    "/analyze-document",
    response_model=DocumentAnalysisResponse,
    summary="Analyze a document for risky clauses",
    description="Public endpoint to review contracts and find dangerous clauses."
)
async def analyze_document_public(
    data: DocumentAnalysisRequest
):
    """Analyze a legal document and return a schema-compliant response."""

    def _normalise_risk_score(raw_score: Any) -> int:
        """Clamp arbitrary model output to the 0-10 transport contract."""
        try:
            value = float(raw_score)
        except (TypeError, ValueError):
            mapping = {"low": 2, "medium": 5, "high": 8}
            value = mapping.get(str(raw_score).strip().lower(), 5)

        return max(0, min(10, int(round(value))))

    def _parse_confidence(raw_confidence: Any) -> float:
        try:
            return float(raw_confidence)
        except (TypeError, ValueError):
            return 0.0

    def _coerce_analysis_results(raw_results: Any) -> List[AnalysisResult]:
        if not isinstance(raw_results, list):
            return []

        coerced: List[AnalysisResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue

            coerced.append(AnalysisResult(
                clause_title=item.get("clause_title") or item.get("title") or "Unnamed Clause",
                clause_text=item.get("clause_text") or item.get("clause") or "Details unavailable.",
                risk_level=str(item.get("risk_level") or item.get("category") or "Medium").title(),
                explanation_ei=item.get("explanation_ei") or item.get("explanation") or item.get("why_bad") or "No explanation provided.",
                legal_principle=item.get("legal_principle") or item.get("legal_basis") or "Legal principle not identified.",
                long_term_risk=item.get("long_term_risk") or item.get("long_term_implications") or "Long-term impact unclear.",
                action_step=item.get("action_step") or item.get("recommendation") or "Consult a qualified lawyer."
            ))

        return coerced

    try:
        rag_service = get_rag_service()
        analysis = rag_service.analyze_document(data.document_text)

        document_type = str(analysis.get("document_type") or "General Legal Document")
        summary = analysis.get("summary") or "Document analyzed."
        overall_verdict = analysis.get("overall_verdict") or analysis.get("verdict") or "Caution"
        risk_score = _normalise_risk_score(analysis.get("risk_score"))
        confidence_score = _parse_confidence(analysis.get("confidence_score"))

        raw_results = analysis.get("analysis_results") or analysis.get("dangerous_clauses") or []
        analysis_results = _coerce_analysis_results(raw_results)

        response_kwargs = {
            "document_type": document_type,
            "confidence_score": confidence_score,
            "summary": summary,
            "risk_score": risk_score,
            "analysis_results": analysis_results,
            "overall_verdict": overall_verdict,
        }

        if analysis.get("disclaimer"):
            response_kwargs["disclaimer"] = analysis["disclaimer"]

        return DocumentAnalysisResponse(**response_kwargs)
    except Exception as e:
        logger.error(f"Document analysis error: {e}", exc_info=True)

        # Always return a contract-compliant fallback so clients can rely on the schema.
        return DocumentAnalysisResponse(
            summary="Unable to fully analyze the document. Please try again or consult a lawyer.",
            risk_score=5,
            analysis_results=[],
            overall_verdict="Caution",
            confidence_score=0.0
        )


@router.post(
    "/generate-document",
    response_model=DocumentGenerationResponse,
    summary="Generate a legal document template",
    description="Public endpoint to create legal document templates."
)
async def generate_document_public(
    data: DocumentGenerationRequest
):
    """
    Generate a legal document template based on the type and user details.
    Returns a template that should be reviewed by a lawyer before use.
    """
    try:
        # Zero-Trace Scrubbing: Stateless generation
        rag_service = get_rag_service()
        
        # Use RAG service to generate document without persistent indexing
        document_content = rag_service.generate_document(
            doc_type=data.doc_type,
            user_details=data.user_details
        )
        
        return DocumentGenerationResponse(
            content=document_content,
            doc_type=data.doc_type
        )
    except Exception as e:
        logger.error(f"Document generation error: {e}", exc_info=True)
        return DocumentGenerationResponse(
            content=f"Unable to generate {data.doc_type}. Please try again with more details.",
            doc_type=data.doc_type,
            warning="⚠️ Error occurred during generation. Please try again."
        )



# ---------------------------------------------
# OCR (Image to Text)
# ---------------------------------------------
@router.post("/public/extract-text")
async def extract_text_from_image(
    image: UploadFile = File(...)
):
    """
    Extract text from an uploaded image using AI Vision (OCR).
    Useful for scanning documents.
    """
    # Validate file type
    allowed_types = [
        "image/jpeg", "image/png", "image/webp", "image/heic", 
        "application/pdf", 
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain"
    ]
    content_type = image.content_type or ""
    
    # Also check by extension
    filename = image.filename or ""
    allowed_extensions = [".jpg", ".jpeg", ".png", ".webp", ".heic", ".pdf", ".docx", ".txt"]
    has_valid_ext = any(filename.lower().endswith(ext) for ext in allowed_extensions)
    
    if content_type not in allowed_types and not has_valid_ext:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid format. Supported: jpg, png, pdf, docx, txt"
        )
        
    try:
        content = await image.read()
        rag_service = get_rag_service()
        
        # Branch based on file type
        fn_lower = filename.lower()
        if content_type == "application/pdf" or fn_lower.endswith(".pdf"):
            text = rag_service.extract_text_from_pdf(content)
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or fn_lower.endswith(".docx"):
            text = rag_service.extract_text_from_docx(content)
        elif content_type == "text/plain" or fn_lower.endswith(".txt"):
            text = rag_service.extract_text_from_txt(content)
        else:
            text = rag_service.extract_text_from_image(content)
            
        return {"success": True, "text": text}
    except Exception as e:
        logger.error(f"OCR error: {e}", exc_info=True)
        # Use the actual error message from the exception (which now contains helpful info like rate limits)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e) if "OCR service is currently busy" in str(e) else "Failed to extract text. Please try again with a clearer image or a text-based document."
        )


# ---------------------------------------------
# Speech-to-Text Transcription (Whisper)
# ---------------------------------------------
@router.post("/public/transcribe")
async def transcribe_audio_public(
    audio: UploadFile = File(...)
):
    """
    Transcribe audio to text using OpenAI Whisper.
    Supports: mp3, mp4, mpeg, mpga, m4a, wav, webm
    
    Returns the transcribed text for use in chat input.
    """
    # Validate file type
    allowed_types = ["audio/mpeg", "audio/mp4", "audio/wav", "audio/webm", "audio/m4a", "audio/x-m4a"]
    content_type = audio.content_type or ""
    
    # Also check by extension for flexibility
    filename = audio.filename or ""
    allowed_extensions = [".mp3", ".mp4", ".m4a", ".wav", ".webm", ".mpeg", ".mpga"]
    has_valid_ext = any(filename.lower().endswith(ext) for ext in allowed_extensions)
    
    if content_type not in allowed_types and not has_valid_ext:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audio format. Supported: mp3, mp4, m4a, wav, webm"
        )
    
    try:
        # Save to temporary file (Whisper API requires a file)
        suffix = os.path.splitext(filename)[1] if filename else ".m4a"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Use OpenAI Whisper API
            client = openai.OpenAI()
            with open(tmp_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"  # Can be made dynamic
                )
            
            return {
                "success": True,
                "text": transcript.text,
                "language": "en"
            }
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except openai.APIError as e:
        logger.error(f"OpenAI Whisper API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech recognition service temporarily unavailable."
        )
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to transcribe audio. Please try again."
        )
