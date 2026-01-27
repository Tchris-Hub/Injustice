"""
Database Models
----------------
SQLAlchemy models for User, Conversation, Message, and LegalDocument.
Designed for a legal advisory chatbot with audit trail capabilities.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    JSON,
    Integer,
    Float,
    Enum as SQLEnum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


# ---------------------------------------------
# Utility Functions
# ---------------------------------------------
def utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def generate_uuid() -> uuid.UUID:
    """Generate a new UUID."""
    return uuid.uuid4()


# ---------------------------------------------
# User Model
# ---------------------------------------------
class User(Base):
    """
    User account for the legal advisor platform.
    Stores authentication credentials and profile info.
    """
    __tablename__ = "users"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    full_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True
    )
    phone_number: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True
    )
    
    # Account status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    
    # Consent tracking (legal requirement)
    has_accepted_terms: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    terms_accepted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False
    )
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"


# ---------------------------------------------
# Refresh Token Model (for token blacklisting)
# ---------------------------------------------
class RefreshToken(Base):
    """
    Stores refresh tokens for token rotation and revocation.
    Enables logout and security monitoring.
    """
    __tablename__ = "refresh_tokens"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    token_hash: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    is_revoked: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Index for cleanup queries
    __table_args__ = (
        Index("ix_refresh_tokens_expires_at", "expires_at"),
    )


# ---------------------------------------------
# Conversation Model
# ---------------------------------------------
class Conversation(Base):
    """
    A conversation thread between a user and the AI advisor.
    Groups related messages together.
    """
    __tablename__ = "conversations"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    title: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True
    )
    
    # Context tracking
    legal_topic: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True
    )  # e.g., "arrest_rights", "tenant_rights", "employment"
    risk_level: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True
    )  # "low", "medium", "high"
    
    # Escalation tracking
    is_escalated: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    escalated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    escalation_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="conversations"
    )
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )
    escalation_case: Mapped[Optional["EscalationCase"]] = relationship(
        "EscalationCase",
        back_populates="conversation",
        cascade="all, delete-orphan",
        uselist=False
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title={self.title})>"


# ---------------------------------------------
# Message Model
# ---------------------------------------------
class Message(Base):
    """
    A single message in a conversation.
    Stores both user messages and AI responses with full provenance.
    """
    __tablename__ = "messages"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Message content
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False
    )  # "user", "assistant", "system"
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    
    # AI response metadata (for provenance & auditing)
    sources: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True
    )  # Citations: [{"title": "...", "section": "...", "excerpt": "..."}]
    model_version: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True
    )  # e.g., "gpt-4o-2024-01-25"
    confidence_score: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True
    )  # "high", "medium", "low"
    
    # Safety metadata
    contains_disclaimer: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    flagged_for_review: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True
    )
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages"
    )
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role={self.role})>"


# ---------------------------------------------
# Escalation Workflow
# ---------------------------------------------
class EscalationState(str, Enum):
    """Fixed escalation workflow states (must match PG Enum exactly)."""

    pending_review = "pending_review"
    awaiting_contact = "awaiting_contact"
    follow_up = "follow_up"
    resolved = "resolved"


class EscalationCase(Base):
    """Deterministic escalation record tied to a conversation."""

    __tablename__ = "escalation_cases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )
    created_by_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )

    # Deterministic trigger details
    trigger_source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="manual"
    )  # e.g. "risk_engine", "user_request"
    reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    urgency: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="medium"
    )
    contact_preference: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="either"
    )

    current_state: Mapped[EscalationState] = mapped_column(
        SQLEnum(EscalationState, name="escalation_state", native_enum=True),
        nullable=False,
        default=EscalationState.pending_review
    )
    state_history: Mapped[List[Dict[str, str]]] = mapped_column(
        JSON,
        default=list,
        nullable=False
    )
    last_state_changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False
    )

    # Relationships
    conversation: Mapped[Conversation] = relationship(
        "Conversation",
        back_populates="escalation_case"
    )
    created_by: Mapped[Optional[User]] = relationship("User")

    def append_state(self, state: EscalationState, note: str) -> None:
        """Append deterministic state transition to history."""

        history = list(self.state_history or [])
        history.append({
            "state": state.value,
            "note": note,
            "timestamp": utc_now().isoformat()
        })
        self.state_history = history
        self.current_state = state
        self.last_state_changed_at = utc_now()

    def __repr__(self) -> str:
        return (
            f"<EscalationCase(id={self.id}, state={self.current_state})>"
        )


# ---------------------------------------------
# User Profile & Preferences
# ---------------------------------------------
class UserProfile(Base):
    """Extended user profile information."""
    __tablename__ = "user_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False
    )
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    profession: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    bio: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship("User", backref="profile")


class UserPreference(Base):
    """User-specific app settings and privacy preferences."""
    __tablename__ = "user_preferences"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False
    )
    theme: Mapped[str] = mapped_column(String(20), default="system")
    notifications_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    data_retention_days: Mapped[int] = mapped_column(Integer, default=365)
    marketing_opt_in: Mapped[bool] = mapped_column(Boolean, default=False)

    user: Mapped["User"] = relationship("User", backref="preferences")


# ---------------------------------------------
# Chat Archives
# ---------------------------------------------
class ChatArchive(Base):
    """Long-term storage for encrypted conversation transcripts."""
    __tablename__ = "chat_archives"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False
    )
    transcript_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    archived_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )
    retention_policy: Mapped[str] = mapped_column(String(50), default="standard")


# ---------------------------------------------
# Document Analysis & Generation Tracking
# ---------------------------------------------
class DocumentAnalysisRequest(Base):
    """Audit log of document review requests."""
    __tablename__ = "document_analysis_requests"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    source_type: Mapped[str] = mapped_column(String(20))  # "upload", "text", "scan"
    document_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now
    )


class DocumentAnalysisResult(Base):
    """Stored results of document analyses."""
    __tablename__ = "document_analysis_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_analysis_requests.id", ondelete="CASCADE"),
        nullable=False
    )
    results_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    risk_score: Mapped[int] = mapped_column(Integer)
    overall_verdict: Mapped[str] = mapped_column(String(100))


class DocumentGenerationRequest(Base):
    """Audit log of document generation requests."""
    __tablename__ = "document_generation_requests"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    template_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    intake_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now
    )


class GeneratedDocument(Base):
    """Persisted record of generated documents."""
    __tablename__ = "generated_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_generation_requests.id", ondelete="CASCADE"),
        nullable=False
    )
    content_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    warning_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    download_count: Mapped[int] = mapped_column(Integer, default=0)


# ---------------------------------------------
# Legal Document Model (for RAG)
# ---------------------------------------------
class LegalDocument(Base):
    """
    Source documents for RAG retrieval.
    Stores metadata about ingested legal texts.
    """
    __tablename__ = "legal_documents"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    
    # Document identification
    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False
    )
    document_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )  # "constitution", "statute", "case_law", "guidance"
    jurisdiction: Mapped[str] = mapped_column(
        String(50),
        default="Nigeria",
        nullable=False
    )
    
    # Content
    source_url: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True
    )
    file_path: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True
    )
    
    # Indexing status
    is_indexed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    indexed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    chunk_count: Mapped[Optional[int]] = mapped_column(
        nullable=True
    )
    
    # Metadata
    document_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True
    )  # Additional info like publication date, version, etc.
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False
    )
    
    def __repr__(self) -> str:
        return f"<LegalDocument(id={self.id}, title={self.title})>"


# ---------------------------------------------
# Legal App Content & Directories
# ---------------------------------------------
class LegalTemplate(Base):
    """Catalog of document templates."""
    __tablename__ = "legal_templates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(100))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    jurisdiction: Mapped[str] = mapped_column(String(50), default="Nigeria")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class LawyerDirectory(Base):
    """Lawyer and organization listings."""
    __tablename__ = "lawyer_directory"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    org_type: Mapped[str] = mapped_column(String(50))  # "law_firm", "ngo", "legal_aid"
    specialties: Mapped[List[str]] = mapped_column(JSON, default=list)
    location_state: Mapped[str] = mapped_column(String(100))
    contact_info: Mapped[dict] = mapped_column(JSON, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)


class ConstitutionSection(Base):
    """Structured constitution data for explorer and RAG."""
    __tablename__ = "constitution_sections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    section_number: Mapped[str] = mapped_column(String(20), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    part_header: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    chapter_header: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)


class LegalAidLocation(Base):
    """Geospatial data for legal aid centers."""
    __tablename__ = "legal_aid_locations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    address: Mapped[str] = mapped_column(Text, nullable=False)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    services_offered: Mapped[List[str]] = mapped_column(JSON, default=list)


# ---------------------------------------------
# Audit Log Model
# ---------------------------------------------
class AuditLog(Base):
    """
    Audit trail for security and compliance.
    Tracks important actions in the system.
    """
    __tablename__ = "audit_logs"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    
    # Action details
    action: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )  # e.g., "user_login", "conversation_escalated", "document_indexed"
    entity_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True
    )  # "user", "conversation", "message"
    entity_id: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True
    )
    
    # Actor
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True
    )
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True
    )
    user_agent: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True
    )
    
    # Details
    details: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True
    )
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True
    )
    
    # Index for querying by action and time
    __table_args__ = (
        Index("ix_audit_logs_action_created", "action", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action={self.action})>"


# ---------------------------------------------
# Notifications & Compliance
# ---------------------------------------------
class Notification(Base):
    """User notifications across channels."""
    __tablename__ = "notifications"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    title: Mapped[str] = mapped_column(String(255))
    message: Mapped[str] = mapped_column(Text)
    channel: Mapped[str] = mapped_column(String(20))  # "in_app", "push", "email"
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now
    )


class DataRetentionEvent(Base):
    """Audit log for privacy-related data actions."""
    __tablename__ = "data_retention_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    action: Mapped[str] = mapped_column(String(50))  # "export", "delete", "archive"
    initiated_by: Mapped[str] = mapped_column(String(20))  # "user", "system", "admin"
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now
    )
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
