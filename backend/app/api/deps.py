"""
Authentication Dependencies
----------------------------
FastAPI dependencies for authentication and authorization.
"""
import logging
import uuid as uuid_mod
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.security import verify_access_token, decode_token
from app.db.session import get_db
from app.db.models import User

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user.
    
    Supports both backend-issued tokens and Supabase-bridged tokens.
    For Supabase users (e.g. Google Sign-In), auto-creates a backend
    user record on first access.
    """
    token = credentials.credentials
    
    # Decode the token (supports both backend and Supabase tokens via bridge)
    token_data = decode_token(token)
    if not token_data or token_data.token_type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    user_id = token_data.user_id
    
    # Try to convert user_id to UUID for DB lookup
    try:
        user_uuid = uuid_mod.UUID(user_id)
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user identifier in token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Fetch user from database
    result = await db.execute(
        select(User).where(User.id == user_uuid)
    )
    user = result.scalar_one_or_none()
    
    # Auto-provision for Supabase-bridged users (e.g. Google Sign-In)
    if not user:
        logger.info(f"Auto-provisioning backend user for Supabase ID: {user_id[:8]}...")
        
        # Extract email from the Supabase token if available
        import jwt as pyjwt
        try:
            # Decode without verification just to read claims (already verified above)
            unverified = pyjwt.decode(token, options={"verify_signature": False})
            email = unverified.get("email", f"{user_id}@supabase.bridge")
            full_name = unverified.get("user_metadata", {}).get("full_name", None)
        except Exception:
            email = f"{user_id}@supabase.bridge"
            full_name = None
        
        user = User(
            id=user_uuid,
            email=email,
            hashed_password="SUPABASE_OAUTH_USER",  # No password - OAuth only
            full_name=full_name,
            is_active=True,
            is_verified=True,
            has_accepted_terms=False,
        )
        db.add(user)
        await db.flush()  # Flush to get the user in the session without full commit
        logger.info(f"✓ Auto-provisioned user {email}")
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )
    
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Dependency to optionally get the current user.
    Returns None if no valid token is provided.
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None
