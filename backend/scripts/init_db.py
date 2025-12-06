"""
Database Initialization Script
-------------------------------
Creates database tables and initial data.
Usage: python -m scripts.init_db
"""

import asyncio
import logging

from app.db.session import init_db, engine
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Initialize the database."""
    logger.info(f"Initializing database for {settings.app_name}...")
    logger.info(f"Database URL: {settings.database_url}")
    
    try:
        await init_db()
        logger.info("✅ Database tables created successfully.")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
