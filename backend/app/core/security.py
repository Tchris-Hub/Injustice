"""
Security Module
----------------
Handles password hashing (Argon2) and JWT token management.
Implements industry-standard security practices.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import jwt
from jwt import PyJWTError, PyJWKClient
from passlib.context import CryptContext
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)

# Supabase JWKS endpoint for ES256 token verification
# This is the public key endpoint that allows us to verify Supabase-issued JWTs
SUPABASE_PROJECT_REF = "fdjltnfkmskeqtiaqlou"
SUPABASE_JWKS_URL = f"https://{SUPABASE_PROJECT_REF}.supabase.co/auth/v1/.well-known/jwks.json"
_jwks_client: Optional[PyJWKClient] = None

def get_jwks_client() -> PyJWKClient:
    """Get or create a cached JWKS client for Supabase token verification."""
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(SUPABASE_JWKS_URL, cache_keys=True)
    return _jwks_client


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
        settings.backend_jwt_secret,
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
        settings.backend_jwt_secret,
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
        "expires_at": access_expires.isoformat(),
        "refresh_expires_at": refresh_expires.isoformat()
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
        # 1. Try decoding with our BACKEND_JWT_SECRET
        payload = jwt.decode(
            token,
            settings.backend_jwt_secret,
            algorithms=[settings.algorithm]
        )
        
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if user_id is None:
            return None
            
        return TokenData(user_id=user_id, token_type=token_type)
        
    except PyJWTError as backend_err:
        # 2. Bridge: Try decoding with Supabase JWKS (ES256)
        # Supabase uses ES256 asymmetric signing — we verify using the public key
        # fetched from their JWKS endpoint.
        logger.debug(f"Backend JWT decode failed: {type(backend_err).__name__}: {backend_err}")

        try:
            jwks_client = get_jwks_client()
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256"],
                audience="authenticated"
            )
            
            user_id: str = payload.get("sub")
            if user_id is None:
                logger.warning("Supabase token decoded but has no 'sub' claim")
                return None
                
            logger.info(f"✓ Supabase JWT bridge: accepted token for user {user_id[:8]}...")
            # Treat Supabase session tokens as valid access tokens
            return TokenData(user_id=user_id, token_type="access")
            
        except Exception as supa_err:
            logger.warning(f"Supabase JWT bridge FAILED: {type(supa_err).__name__}: {supa_err}")
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
