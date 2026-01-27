"""
Chat Schemas
-------------
Pydantic models for chat requests and responses.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------
# Source Citation
# ---------------------------------------------
class SourceCitation(BaseModel):
    """A citation to a legal source."""
    title: str
    section: Optional[str] = None
    excerpt: str
    document_type: str  # "constitution", "statute", "case_law"
    relevance_score: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Constitution of Nigeria 1999",
                "section": "Section 35 - Right to Personal Liberty",
                "excerpt": "Every person shall be entitled to his personal liberty...",
                "document_type": "constitution",
                "relevance_score": 0.92
            }
        }


# ---------------------------------------------
# Message Schemas
# ---------------------------------------------
class MessageCreate(BaseModel):
    """Request schema for sending a message."""
    content: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None  # If None, creates new conversation
    suppress_storage: bool = False  # If True, transcript won't be saved to DB
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "I was arrested without being told why. What are my rights?",
                "conversation_id": None,
                "suppress_storage": False
            }
        }


class MessageResponse(BaseModel):
    """Response schema for a message."""
    id: str
    conversation_id: str
    role: str  # "user" or "assistant"
    content: str
    sources: Optional[List[SourceCitation]] = None
    confidence_score: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class AIResponse(BaseModel):
    """
    The AI advisor's response with full context.
    This is the main response users will receive.
    """
    message: MessageResponse
    conversation_id: str
    conversation_title: Optional[str] = None
    risk_level: Optional[str] = None  # "low", "medium", "high"
    escalation_recommended: bool = False
    disclaimer: str = (
        "⚠️ IMPORTANT: This information is for educational purposes only "
        "and does not constitute legal advice. Laws vary and change. "
        "For advice specific to your situation, please consult a licensed attorney."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": {
                    "id": "msg_123",
                    "conversation_id": "conv_456",
                    "role": "assistant",
                    "content": "I understand this must be very stressful...",
                    "sources": [],
                    "confidence_score": "high",
                    "created_at": "2024-01-15T12:00:00Z"
                },
                "conversation_id": "conv_456",
                "conversation_title": "Arrest Rights Question",
                "risk_level": "medium",
                "escalation_recommended": False,
                "disclaimer": "⚠️ IMPORTANT: This information..."
            }
        }


# ---------------------------------------------
# Conversation Schemas
# ---------------------------------------------
class ConversationCreate(BaseModel):
    """Request schema for creating a conversation."""
    title: Optional[str] = Field(None, max_length=255)
    initial_message: str = Field(..., min_length=1, max_length=10000)


class ConversationSummary(BaseModel):
    """Summary of a conversation for listing."""
    id: str
    title: Optional[str] = None
    legal_topic: Optional[str] = None
    risk_level: Optional[str] = None
    is_escalated: bool = False
    message_count: int = 0
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ConversationDetail(BaseModel):
    """Full conversation with messages."""
    id: str
    title: Optional[str] = None
    legal_topic: Optional[str] = None
    risk_level: Optional[str] = None
    is_escalated: bool = False
    escalation_reason: Optional[str] = None
    messages: List[MessageResponse] = []
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ConversationList(BaseModel):
    """Paginated list of conversations."""
    conversations: List[ConversationSummary]
    total: int
    page: int
    page_size: int
    has_more: bool


# ---------------------------------------------
# Escalation Schemas
# ---------------------------------------------
class EscalationRequest(BaseModel):
    """Request to escalate a conversation to human review."""
    conversation_id: str
    reason: str = Field(..., min_length=10, max_length=1000)
    contact_preference: str = Field(..., pattern="^(email|phone|either)$")
    urgency: str = Field(..., pattern="^(low|medium|high|critical)$")
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_456",
                "reason": "I have a court date in 3 days and need urgent legal help",
                "contact_preference": "phone",
                "urgency": "critical"
            }
        }


class EscalationResponse(BaseModel):
    """Response after escalation request."""
    success: bool
    message: str
    reference_number: str
    estimated_response_time: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Your case has been escalated to a legal aid partner.",
                "reference_number": "ESC-2024-001234",
                "estimated_response_time": "24-48 hours"
            }
        }


# ---------------------------------------------
# Public API Schemas (No Auth Required)
# ---------------------------------------------
class PublicChatRequest(BaseModel):
    """Simple chat request for public/demo endpoint."""
    message: str = Field(..., min_length=1, max_length=10000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What are my rights as a tenant in Nigeria?"
            }
        }


class PublicChatResponse(BaseModel):
    """Simple chat response for public/demo endpoint."""
    content: str
    sources: List[str] = []
    confidence_score: Optional[str] = None
    legal_disclaimer: str = (
        "⚠️ This information is for educational purposes only and does not constitute legal advice. "
        "Please consult a licensed attorney for advice specific to your situation."
    )


class DocumentAnalysisRequest(BaseModel):
    """Request to analyze a legal document for risky clauses."""
    document_text: str = Field(..., min_length=10, max_length=50000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_text": "TENANCY AGREEMENT: The tenant agrees to pay a penalty of 50% of rent for any late payment..."
            }
        }


class AnalysisResult(BaseModel):
    """A detailed analysis of a specific clause."""
    clause_title: str
    clause_text: str
    risk_level: str  # "High", "Medium", "Low"
    explanation_ei: str
    legal_principle: str
    long_term_risk: str
    action_step: str


class DocumentAnalysisResponse(BaseModel):
    """Response from analysis with EI and legal principles."""
    document_type: str = "General Legal Document"
    confidence_score: float = 0.0
    summary: str
    risk_score: int = Field(..., ge=0, le=10)
    analysis_results: List[AnalysisResult] = []
    overall_verdict: str
    disclaimer: str = (
        "⚠️ This analysis is for informational purposes only. "
        "Have a lawyer review any document before signing."
    )


class DocumentGenerationRequest(BaseModel):
    """Request to generate a legal document template."""
    doc_type: str = Field(..., min_length=1, max_length=100)
    user_details: str = Field(..., min_length=10, max_length=5000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_type": "Demand Letter",
                "user_details": "Landlord: John Doe. Tenant: Jane Smith. Amount: 500,000 Naira."
            }
        }


class DocumentGenerationResponse(BaseModel):
    """Response with generated document template."""
    content: str
    doc_type: str
    warning: str = (
        "⚠️ TEMPLATE ONLY: This is a general template and may not be suitable for your specific situation. "
        "Have a qualified lawyer review and customize this document before using it."
    )
