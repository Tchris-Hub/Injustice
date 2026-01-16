"""
Database Models
----------------
SQLAlchemy models for User, Conversation, Message, and LegalDocument.
Designed for a legal advisory chatbot with audit trail capabilities.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    JSON,
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
