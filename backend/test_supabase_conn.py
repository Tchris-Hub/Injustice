
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings

async def test_connection():
    print(f"Testing connection to: {settings.database_url}")
    # We need to use 'postgresql+asyncpg' for async connection
    db_url = settings.database_url
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
    
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    engine = create_async_engine(
        db_url,
        connect_args={
            "prepared_statement_cache_size": 0,
            "statement_cache_size": 0
        } if "pooler" in db_url else {}
    )
    try:
        async with engine.connect() as conn:
            print("Successfully connected to the database!")
            from sqlalchemy import text
            result = await conn.execute(text("SELECT version();"))
            row = result.fetchone()
            print(f"PostgreSQL version: {row[0]}")
    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_connection())
