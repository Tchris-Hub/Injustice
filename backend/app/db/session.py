"""
Database Session Management
----------------------------
Async SQLAlchemy session with proper connection pooling.
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)
from sqlalchemy.orm import declarative_base

from app.core.config import settings


from sqlalchemy.pool import NullPool

# Create async engine with connection pooling
# SQLite doesn't support pool_size/max_overflow, so we conditionally set them
_is_sqlite = settings.database_url.startswith("sqlite")

engine_kwargs = {
    "echo": settings.debug,  # Log SQL queries in debug mode
}

if not _is_sqlite:
    db_url = settings.database_url
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")

    # Aggressively disable prepared statements for PgBouncer compatibility (Railway/Supabase)
    if "postgresql" in db_url:
        engine_kwargs.update({
            "connect_args": {
                "statement_cache_size": 0,
                "prepared_statement_cache_size": 0
            }
        })

    # Supabase/Railway Transaction Pooler specific settings
    if "pooler" in db_url or "pgbouncer" in db_url.lower():
        engine_kwargs.update({
            "poolclass": NullPool,
            # Some versions of asyncpg/sqlalchemy need this directly in execution_options
            "execution_options": {"compiled_cache": None}
        })
    else:
        # Standard pool for direct connections, but keeping it small for memory
        engine_kwargs.update({
            "pool_pre_ping": True,
            "pool_size": 2,
            "max_overflow": 5
        })


engine = create_async_engine(db_url if not _is_sqlite else settings.database_url, **engine_kwargs)

# Session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# Base class for all models
Base = declarative_base()


async def get_db() -> AsyncSession:
    """
    Dependency that provides a database session.
    Use with FastAPI's Depends().
    
    Yields:
        AsyncSession: Database session that auto-closes after request
    """
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """
    Initialize database tables.
    Call this on application startup.
    """
    async with engine.begin() as conn:
        # Import all models here to ensure they're registered
        from app.db import models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """
    Close database connections.
    Call this on application shutdown.
    """
    await engine.dispose()
