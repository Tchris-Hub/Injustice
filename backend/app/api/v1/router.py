"""
API Router
-----------
Combine all API versioned routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, chat

# Create versioned router
api_v1_router = APIRouter(prefix="/api/v1")

# Include endpoint routers
api_v1_router.include_router(auth.router)
api_v1_router.include_router(chat.router)
