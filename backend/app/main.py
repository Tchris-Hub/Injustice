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
from app.core.rate_limit import limiter
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
    
    # Initialize RAG Service (Pre-load heavy models to prevent first-request 502/timeouts)
    try:
        logger.info("Warming up RAG Service (loading LLM & Embeddings)...")
        get_rag_service()
        logger.info("RAG Service warmup complete")
    except Exception as e:
        logger.error(f"RAG Service warmup failed: {e}")
        # Non-critical for startup, but requests will be slow/fail later

    
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


# Global Error Handler for Production Debugging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler to log tracebacks for 500 errors.
    """
    import traceback
    error_trace = traceback.format_exc()
    logger.error(f"GLOBAL EXCEPTION DETECTED: {str(exc)}")
    logger.error(f"Traceback:\n{error_trace}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal Server Error. Our team has been notified.",
            "error_type": type(exc).__name__,
            "message": str(exc) if settings.debug else "Sensitive error info hidden in production."
        }
    )


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


# ---------------------------------------------
# Android App Link Verification (Digital Asset Links)
# ---------------------------------------------
@app.get("/.well-known/assetlinks.json", tags=["Security"])
async def get_assetlinks():
    """
    Android App Link verification.
    This allows the app to handle URLs on this domain automatically.
    """
    return [{
        "relation": ["delegate_permission/common.handle_all_urls"],
        "target": {
            "namespace": "android_app",
            "package_name": "com.myrights.app",
            "sha256_cert_fingerprints": [
                # NOTE: Replace with your actual production SHA256 fingerprint from Google Play Console
                "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00"
            ]
        }
    }]


@app.get("/debug/env", tags=["Health"])
async def check_env():
    """
    Enhanced diagnostic endpoint to check environment health.
    """
    # SECURITY: Disable in production or when not in debug mode
    if settings.is_production or not settings.debug:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": "Access restricted in production."}
        )
    
    import os
    
    # Check variables
    status_map = {
        "OPENROUTER_API_KEY": "SET ✅" if os.getenv("OPENROUTER_API_KEY") else "MISSING ❌",
        "MODEL_NAME": os.getenv("MODEL_NAME", "NOT SET (Using Gemini Flash Default)"),
        "DATABASE_URL": "SET ✅" if os.getenv("DATABASE_URL") else "MISSING ❌",
        "BACKEND_JWT_SECRET": "SET ✅" if os.getenv("BACKEND_JWT_SECRET") else "MISSING ❌ (SECURITY RISK)",
        "SUPABASE_JWT_SECRET": "SET ✅" if os.getenv("SUPABASE_JWT_SECRET") else "MISSING ❌ (Google Sign-In will fail!)",
        "ENVIRONMENT": os.getenv("ENVIRONMENT", "NOT SET (defaults to development)"),
    }
    
    health_score = sum(1 for v in status_map.values() if "SET ✅" in str(v))
    total_needed = 5 # Base keys
    
    return {
        "summary": "CRITICAL CONFIGURATION MISSING!" if status_map["BACKEND_JWT_SECRET"] == "MISSING ❌ (SECURITY RISK)" else "Configuration looks good",
        "variable_status": status_map,
        "health_score": f"{health_score}/{total_needed}",
        "advice": [
            "Add BACKEND_JWT_SECRET to Railway variables if it shows MISSING." if "MISSING" in status_map["BACKEND_JWT_SECRET"] else None,
            "Ensure MODEL_NAME is a valid OpenRouter slug (e.g. google/gemini-2.0-flash-exp:free)." if "NOT SET" in status_map["MODEL_NAME"] else None
        ],
        "is_production": settings.is_production
    }


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
