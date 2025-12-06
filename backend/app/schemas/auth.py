"""
Authentication Schemas
-----------------------
Pydantic models for auth requests and responses.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator
import re


class UserRegister(BaseModel):
    """Request schema for user registration."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(None, max_length=255)
    phone_number: Optional[str] = Field(None, max_length=20)
    accept_terms: bool = Field(..., description="User must accept terms of service")
    
    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Ensure password meets security requirements."""
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        return v
    
    @field_validator("accept_terms")
    @classmethod
    def must_accept_terms(cls, v: bool) -> bool:
        """User must accept terms to register."""
        if not v:
            raise ValueError("You must accept the terms of service to continue")
        return v


class UserLogin(BaseModel):
    """Request schema for user login."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Response schema for authentication tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_at: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_at": "2024-01-15T12:00:00Z"
            }
        }


class TokenRefresh(BaseModel):
    """Request schema for token refresh."""
    refresh_token: str


class UserResponse(BaseModel):
    """Response schema for user data."""
    id: str
    email: str
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    is_active: bool
    is_verified: bool
    has_accepted_terms: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """Request schema for updating user profile."""
    full_name: Optional[str] = Field(None, max_length=255)
    phone_number: Optional[str] = Field(None, max_length=20)


class PasswordChange(BaseModel):
    """Request schema for password change."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)
    
    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Ensure new password meets security requirements."""
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        return v
