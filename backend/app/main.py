"""
AI Legal Advisor - Main Application
-------------------------------------
FastAPI application with security, rate limiting, and CORS.
Built to help people understand their constitutional rights in Nigeria.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.db.session import init_db, close_db
from app.api.v1.router import api_v1_router
from app.services.rag_service import get_rag_service


# ---------------------------------------------
# Logging Configuration
# ---------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------
# Rate Limiter
# ---------------------------------------------
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------
# Application Lifespan
# ---------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Jurisdiction: {settings.jurisdiction}")
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Continue anyway - DB might be pre-initialized
    
    # Initialize RAG Service (Lazy Load - disabled to reduce startup memory)
    # The RAG service will be initialized on first request instead
    # try:
    #     logger.info("Warming up RAG Service (loading LLM & Embeddings)...")
    #     get_rag_service()
    #     logger.info("RAG Service warmup complete")
    # except Exception as e:
    #     logger.error(f"RAG Service warmup failed: {e}")
    #     # Non-critical for startup, but requests will be slow/fail later
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down application...")
    await close_db()
    logger.info("Application shutdown complete")


# ---------------------------------------------
# Create FastAPI Application
# ---------------------------------------------
app = FastAPI(
    title=settings.app_name,
    description="""
## AI Legal Advisor for Nigeria 🇳🇬

An AI-powered system that helps people understand their constitutional rights 
and legal options. This service provides **legal information**, not legal advice.

### Features
- 💬 **Empathetic Chat**: Ask questions about your legal situation
- 📚 **Constitutional Knowledge**: Grounded in Nigerian law
- 🔍 **Cited Sources**: Every response includes legal references
- 🆘 **Human Escalation**: Connect with legal aid partners when needed
- 🔒 **Secure**: Encrypted, audited, and privacy-focused

### Important Disclaimer
This service provides general legal information for educational purposes only.
It does not constitute legal advice and does not create an attorney-client 
relationship. For advice specific to your situation, please consult a licensed 
attorney in Nigeria.

---
*Built to protect those who cannot afford traditional legal services.*
    """,
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)


# ---------------------------------------------
# Middleware
# ---------------------------------------------

# Rate Limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
)


# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response


# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    logger.debug(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.debug(f"Response: {response.status_code}")
    return response


# ---------------------------------------------
# Include Routers
# ---------------------------------------------
app.include_router(api_v1_router)


# ---------------------------------------------
# Root Endpoints
# ---------------------------------------------
@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint - basic service info.
    """
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "jurisdiction": settings.jurisdiction,
        "status": "operational",
        "disclaimer": (
            "This service provides legal information for educational purposes only. "
            "It does not constitute legal advice."
        )
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": settings.app_version
    }


@app.get("/debug/env", tags=["Health"])
async def check_env():
    """
    Debug endpoint to check if environment variables are set.
    Only shows if variables exist, not their values.
    """
    import os
    return {
        "openrouter_api_key_set": bool(os.getenv("OPENROUTER_API_KEY")),
        "model_name": os.getenv("MODEL_NAME", "NOT_SET"),
        "openrouter_base_url": os.getenv("OPENROUTER_BASE_URL", "NOT_SET"),
        "database_url_set": bool(os.getenv("DATABASE_URL")),
        "secret_key_set": bool(os.getenv("SECRET_KEY")),
    }


# ---------------------------------------------
# Global Exception Handler
# ---------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler.
    Logs errors and returns a safe response.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal error occurred. Please try again later.",
            "support": "If this persists, please contact support."
        }
    )


# ---------------------------------------------
# Run with Uvicorn (for development)
# ---------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    )
