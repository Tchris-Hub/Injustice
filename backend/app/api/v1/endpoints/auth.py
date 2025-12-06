"""
Authentication Endpoints
-------------------------
Register, login, logout, token refresh, and password management.
"""

import hashlib
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.security import (
    hash_password,
    verify_password,
    create_tokens,
    verify_refresh_token
)
from app.db.session import get_db
from app.db.models import User, RefreshToken, AuditLog
from app.schemas.auth import (
    UserRegister,
    UserLogin,
    TokenResponse,
    TokenRefresh,
    UserResponse,
    UserUpdate,
    PasswordChange
)
from app.api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ---------------------------------------------
# Helper Functions
# ---------------------------------------------
async def log_audit(
    db: AsyncSession,
    action: str,
    user_id: str = None,
    entity_type: str = None,
    entity_id: str = None,
    details: dict = None,
    request: Request = None
):
    """Log an action to the audit trail."""
    audit = AuditLog(
        action=action,
        user_id=user_id,
        entity_type=entity_type,
        entity_id=entity_id,
        details=details,
        ip_address=request.client.host if request else None,
        user_agent=request.headers.get("user-agent") if request else None
    )
    db.add(audit)


def hash_token(token: str) -> str:
    """Hash a token for storage."""
    return hashlib.sha256(token.encode()).hexdigest()


# ---------------------------------------------
# Registration
# ---------------------------------------------
@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user"
)
async def register(
    data: UserRegister,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user account.
    
    - **email**: Valid email address (must be unique)
    - **password**: At least 8 characters with uppercase, lowercase, and number
    - **full_name**: Optional display name
    - **accept_terms**: Must be true to register
    """
    # Check if email already exists
    result = await db.execute(
        select(User).where(User.email == data.email.lower())
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists"
        )
    
    # Create user
    user = User(
        email=data.email.lower(),
        hashed_password=hash_password(data.password),
        full_name=data.full_name,
        phone_number=data.phone_number,
        has_accepted_terms=True,
        terms_accepted_at=datetime.now(timezone.utc)
    )
    db.add(user)
    await db.flush()  # Get the user ID
    
    # Create tokens
    tokens = create_tokens(str(user.id))
    
    # Store refresh token hash
    refresh_hash = hash_token(tokens["refresh_token"])
    refresh_token = RefreshToken(
        user_id=user.id,
        token_hash=refresh_hash,
        expires_at=datetime.fromisoformat(tokens["expires_at"].replace("Z", "+00:00"))
    )
    db.add(refresh_token)
    
    # Audit log
    await log_audit(
        db,
        action="user_registered",
        user_id=str(user.id),
        entity_type="user",
        entity_id=str(user.id),
        details={"email": user.email},
        request=request
    )
    
    logger.info(f"New user registered: {user.email}")
    
    return TokenResponse(**tokens)


# ---------------------------------------------
# Login
# ---------------------------------------------
@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with email and password"
)
async def login(
    data: UserLogin,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate and receive access tokens.
    
    - **email**: Your registered email
    - **password**: Your password
    """
    # Find user
    result = await db.execute(
        select(User).where(User.email == data.email.lower())
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(data.password, user.hashed_password):
        # Log failed attempt
        await log_audit(
            db,
            action="login_failed",
            details={"email": data.email.lower()},
            request=request
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been deactivated. Please contact support."
        )
    
    # Update last login
    user.last_login_at = datetime.now(timezone.utc)
    
    # Create tokens
    tokens = create_tokens(str(user.id))
    
    # Store refresh token hash
    refresh_hash = hash_token(tokens["refresh_token"])
    refresh_token = RefreshToken(
        user_id=user.id,
        token_hash=refresh_hash,
        expires_at=datetime.fromisoformat(tokens["expires_at"].replace("Z", "+00:00"))
    )
    db.add(refresh_token)
    
    # Audit log
    await log_audit(
        db,
        action="user_login",
        user_id=str(user.id),
        entity_type="user",
        entity_id=str(user.id),
        request=request
    )
    
    logger.info(f"User logged in: {user.email}")
    
    return TokenResponse(**tokens)


# ---------------------------------------------
# Token Refresh
# ---------------------------------------------
@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token"
)
async def refresh_token(
    data: TokenRefresh,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a new access token using a refresh token.
    
    - **refresh_token**: Your current refresh token
    """
    # Verify refresh token
    user_id = verify_refresh_token(data.refresh_token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    # Check if token is in database and not revoked
    token_hash = hash_token(data.refresh_token)
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.is_revoked == False
        )
    )
    stored_token = result.scalar_one_or_none()
    
    if not stored_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked"
        )
    
    # Revoke old token
    stored_token.is_revoked = True
    stored_token.revoked_at = datetime.now(timezone.utc)
    
    # Create new tokens
    tokens = create_tokens(user_id)
    
    # Store new refresh token
    new_refresh_hash = hash_token(tokens["refresh_token"])
    new_refresh_token = RefreshToken(
        user_id=user_id,
        token_hash=new_refresh_hash,
        expires_at=datetime.fromisoformat(tokens["expires_at"].replace("Z", "+00:00"))
    )
    db.add(new_refresh_token)
    
    logger.debug(f"Token refreshed for user: {user_id}")
    
    return TokenResponse(**tokens)


# ---------------------------------------------
# Logout
# ---------------------------------------------
@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Logout and revoke tokens"
)
async def logout(
    data: TokenRefresh,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Logout and revoke the refresh token.
    
    - **refresh_token**: The refresh token to revoke
    """
    token_hash = hash_token(data.refresh_token)
    
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    stored_token = result.scalar_one_or_none()
    
    if stored_token:
        stored_token.is_revoked = True
        stored_token.revoked_at = datetime.now(timezone.utc)
    
    await log_audit(
        db,
        action="user_logout",
        user_id=str(current_user.id),
        entity_type="user",
        entity_id=str(current_user.id),
        request=request
    )
    
    logger.info(f"User logged out: {current_user.email}")


# ---------------------------------------------
# Current User Profile
# ---------------------------------------------
@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile"
)
async def get_me(
    current_user: User = Depends(get_current_user)
):
    """Get the current authenticated user's profile."""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        phone_number=current_user.phone_number,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        has_accepted_terms=current_user.has_accepted_terms,
        created_at=current_user.created_at
    )


@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update current user profile"
)
async def update_me(
    data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update the current user's profile information."""
    if data.full_name is not None:
        current_user.full_name = data.full_name
    if data.phone_number is not None:
        current_user.phone_number = data.phone_number
    
    current_user.updated_at = datetime.now(timezone.utc)
    
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        phone_number=current_user.phone_number,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        has_accepted_terms=current_user.has_accepted_terms,
        created_at=current_user.created_at
    )


# ---------------------------------------------
# Password Change
# ---------------------------------------------
@router.post(
    "/change-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Change password"
)
async def change_password(
    data: PasswordChange,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Change the current user's password.
    
    - **current_password**: Your current password
    - **new_password**: Your new password (must meet security requirements)
    """
    if not verify_password(data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    current_user.hashed_password = hash_password(data.new_password)
    current_user.updated_at = datetime.now(timezone.utc)
    
    # Revoke all refresh tokens (force re-login on all devices)
    await db.execute(
        RefreshToken.__table__.update()
        .where(RefreshToken.user_id == current_user.id)
        .where(RefreshToken.is_revoked == False)
        .values(is_revoked=True, revoked_at=datetime.now(timezone.utc))
    )
    
    await log_audit(
        db,
        action="password_changed",
        user_id=str(current_user.id),
        entity_type="user",
        entity_id=str(current_user.id),
        request=request
    )
    
    logger.info(f"Password changed for user: {current_user.email}")
