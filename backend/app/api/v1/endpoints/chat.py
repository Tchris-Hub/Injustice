"""
Chat Endpoints
---------------
Main chat interface for the AI Legal Advisor.
Handles conversations, messages, and escalations.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

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
    SourceCitation
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
