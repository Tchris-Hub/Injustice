"""
Security Module
----------------
Handles password hashing (Argon2) and JWT token management.
Implements industry-standard security practices.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from app.core.config import settings


# ---------------------------------------------
# Password Hashing (Argon2 - Most Secure)
# ---------------------------------------------
pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
    argon2__memory_cost=65536,  # 64 MB
    argon2__time_cost=3,
    argon2__parallelism=4
)


def hash_password(password: str) -> str:
    """
    Hash a plain-text password using Argon2.
    
    Args:
        password: Plain-text password
        
    Returns:
        Hashed password string
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain-text password against a hash.
    
    Args:
        plain_password: Plain-text password to verify
        hashed_password: Previously hashed password
        
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


# ---------------------------------------------
# JWT Token Models
# ---------------------------------------------
class TokenPayload(BaseModel):
    """JWT Token payload structure."""
    sub: str  # Subject (user ID)
    exp: datetime  # Expiration time
    type: str  # Token type: "access" or "refresh"
    iat: datetime  # Issued at


class TokenData(BaseModel):
    """Decoded token data."""
    user_id: str
    token_type: str


# ---------------------------------------------
# JWT Token Creation
# ---------------------------------------------
def create_access_token(user_id: str) -> Tuple[str, datetime]:
    """
    Create a JWT access token.
    
    Args:
        user_id: User's unique identifier
        
    Returns:
        Tuple of (token_string, expiration_datetime)
    """
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    
    payload = {
        "sub": str(user_id),
        "exp": expires_at,
        "type": "access",
        "iat": datetime.now(timezone.utc)
    }
    
    token = jwt.encode(
        payload,
        settings.secret_key,
        algorithm=settings.algorithm
    )
    
    return token, expires_at


def create_refresh_token(user_id: str) -> Tuple[str, datetime]:
    """
    Create a JWT refresh token (longer-lived).
    
    Args:
        user_id: User's unique identifier
        
    Returns:
        Tuple of (token_string, expiration_datetime)
    """
    expires_at = datetime.now(timezone.utc) + timedelta(
        days=settings.refresh_token_expire_days
    )
    
    payload = {
        "sub": str(user_id),
        "exp": expires_at,
        "type": "refresh",
        "iat": datetime.now(timezone.utc)
    }
    
    token = jwt.encode(
        payload,
        settings.secret_key,
        algorithm=settings.algorithm
    )
    
    return token, expires_at


def create_tokens(user_id: str) -> dict:
    """
    Create both access and refresh tokens.
    
    Args:
        user_id: User's unique identifier
        
    Returns:
        Dictionary with tokens and expiration info
    """
    access_token, access_expires = create_access_token(user_id)
    refresh_token, refresh_expires = create_refresh_token(user_id)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_at": access_expires.isoformat()
    }


# ---------------------------------------------
# JWT Token Verification
# ---------------------------------------------
def decode_token(token: str) -> Optional[TokenData]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        TokenData if valid, None if invalid/expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )
        
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if user_id is None:
            return None
            
        return TokenData(user_id=user_id, token_type=token_type)
        
    except JWTError:
        return None


def verify_access_token(token: str) -> Optional[str]:
    """
    Verify an access token and return the user ID.
    
    Args:
        token: JWT access token
        
    Returns:
        User ID if valid access token, None otherwise
    """
    token_data = decode_token(token)
    
    if token_data is None:
        return None
        
    if token_data.token_type != "access":
        return None
        
    return token_data.user_id


def verify_refresh_token(token: str) -> Optional[str]:
    """
    Verify a refresh token and return the user ID.
    
    Args:
        token: JWT refresh token
        
    Returns:
        User ID if valid refresh token, None otherwise
    """
    token_data = decode_token(token)
    
    if token_data is None:
        return None
        
    if token_data.token_type != "refresh":
        return None
        
    return token_data.user_id
