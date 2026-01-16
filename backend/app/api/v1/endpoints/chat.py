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
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
import openai

from app.db.session import get_db
from app.db.models import User, Conversation, Message
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
    DangerousClause,
    DocumentGenerationRequest,
    DocumentGenerationResponse
)
from app.api.deps import get_current_user
from app.services.rag_service import get_rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


# Legal disclaimer (always included)
LEGAL_DISCLAIMER = (
    "⚠️ **IMPORTANT NOTICE**: This information is for educational and informational "
    "purposes only. It does not constitute legal advice and should not be relied upon "
    "as such. Laws vary by jurisdiction and change over time. For advice specific to "
    "your situation, please consult a licensed attorney in Nigeria. Use of this service "
    "does not create an attorney-client relationship."
)


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
    
    # Get or create conversation
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
    else:
        # Create new conversation
        conversation = Conversation(
            user_id=current_user.id,
            title=data.content[:50] + "..." if len(data.content) > 50 else data.content
        )
        db.add(conversation)
        await db.flush()
    
    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=data.content
    )
    db.add(user_message)
    await db.flush()
    
    # Assess risk level
    risk_assessment = rag_service.assess_risk(data.content)
    conversation.risk_level = risk_assessment.get("risk_level", "medium")
    conversation.legal_topic = risk_assessment.get("legal_topic", "general_info")
    
    # Get conversation history
    history_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.desc())
        .limit(10)
    )
    history = [
        {"role": msg.role, "content": msg.content}
        for msg in reversed(history_result.scalars().all())
    ]
    
    # Generate AI response
    ai_result = rag_service.generate_response(
        user_message=data.content,
        conversation_history=history[:-1]  # Exclude current message
    )
    
    # Save AI response
    ai_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=ai_result["content"],
        sources=ai_result.get("sources"),
        model_version=ai_result.get("model_version"),
        confidence_score=ai_result.get("confidence_score"),
        contains_disclaimer=True
    )
    db.add(ai_message)
    
    # Update conversation
    conversation.updated_at = datetime.now(timezone.utc)
    
    # Check for escalation
    escalation_needed = risk_assessment.get("escalation_needed", False)
    if escalation_needed and not conversation.is_escalated:
        logger.warning(
            f"High-risk conversation detected: {conversation.id} - "
            f"Reason: {risk_assessment.get('escalation_reason')}"
        )
    
    # Build response
    sources = [
        SourceCitation(**s) for s in ai_result.get("sources", [])
    ]
    
    message_response = MessageResponse(
        id=str(ai_message.id),
        conversation_id=str(conversation.id),
        role="assistant",
        content=ai_result["content"],
        sources=sources,
        confidence_score=ai_result.get("confidence_score"),
        created_at=ai_message.created_at
    )
    
    return AIResponse(
        message=message_response,
        conversation_id=str(conversation.id),
        conversation_title=conversation.title,
        risk_level=conversation.risk_level,
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
            content=msg.content,
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
        escalation_reason=conversation.escalation_reason,
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
    conversation.is_escalated = True
    conversation.escalated_at = datetime.now(timezone.utc)
    conversation.escalation_reason = data.reason
    
    # Generate reference number
    reference = f"ESC-{datetime.now().strftime('%Y%m%d')}-{str(conversation.id)[:8].upper()}"
    
    # Determine response time based on urgency
    response_times = {
        "critical": "Within 4 hours",
        "high": "Within 24 hours",
        "medium": "Within 48 hours",
        "low": "Within 5 business days"
    }
    
    logger.warning(
        f"Conversation escalated: {conversation.id} - "
        f"User: {current_user.email} - Urgency: {data.urgency} - "
        f"Reason: {data.reason[:100]}"
    )
    
    # Add system message to conversation
    escalation_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=(
            f"🆘 **Escalation Request Submitted**\n\n"
            f"Your reference number is: **{reference}**\n\n"
            f"A legal aid partner will review your case and contact you "
            f"{response_times[data.urgency].lower()}.\n\n"
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
        estimated_response_time=response_times[data.urgency]
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
            confidence_score=0.0
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
    """
    Analyze a legal document (contract, agreement) for risky clauses.
    Returns risk assessment with explanations and recommendations.
    """
    try:
        # Zero-Trace Scrubbing: Stateless analysis
        rag_service = get_rag_service()
        
        # Use RAG service to analyze document without tracking source/user
        analysis = rag_service.analyze_document(data.document_text)
        
        # Parse the analysis result
        dangerous_clauses = []
        raw_clauses = analysis.get("dangerous_clauses", [])
        
        for clause in raw_clauses:
            if isinstance(clause, dict):
                # Map keys from LLM response (which might use clause_text/category) to schema keys
                dangerous_clauses.append(DangerousClause(
                    clause=clause.get("clause_text") or clause.get("clause") or "Unknown clause",
                    risk_level=clause.get("category") or clause.get("risk_level") or "Medium",
                    explanation=clause.get("explanation") or clause.get("why_bad") or "Could be unfavorable",
                    simplified_explanation=clause.get("simplified_explanation") or "No simplified breakdown provided.",
                    long_term_implications=clause.get("long_term_implications") or "No long-term implications identified.",
                    pros=clause.get("pros") if isinstance(clause.get("pros"), list) else [],
                    cons=clause.get("cons") if isinstance(clause.get("cons"), list) else [],
                    recommendation=clause.get("recommendation") or "Consult a lawyer"
                ))
        
        # Calculate risk score (1-10 scale to 0-100 scale)
        raw_score = analysis.get("risk_score", 5)
        try:
            risk_score = min(100, max(0, int(float(raw_score) * 10)))
        except (ValueError, TypeError):
            # If AI returned a string like "Low", map to a default number
            risk_mapping = {"low": 20, "medium": 50, "high": 80}
            risk_score = risk_mapping.get(str(raw_score).lower(), 50)
        
        # Determine verdict
        if risk_score <= 30:
            verdict = "Safe"
        elif risk_score <= 60:
            verdict = "Caution"
        else:
            verdict = "Do Not Sign"
        
        return DocumentAnalysisResponse(
            risk_score=risk_score,
            verdict=verdict,
            dangerous_clauses=dangerous_clauses,
            summary=analysis.get("summary", "Document analyzed.")
        )
    except Exception as e:
        logger.error(f"Document analysis error: {e}", exc_info=True)
        return DocumentAnalysisResponse(
            risk_score=50,
            verdict="Caution",
            dangerous_clauses=[],
            summary="Unable to fully analyze the document. Please try again or consult a lawyer."
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
