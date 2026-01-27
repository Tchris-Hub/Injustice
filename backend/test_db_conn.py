
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import time

async def test_conn():
    url = "postgresql+asyncpg://postgres.fdjltnfkmskeqtiaqlou:SupabaseDB2026@aws-1-eu-west-1.pooler.supabase.com:6543/postgres"
    print(f"Connecting to {url}...")
    start = time.time()
    try:
        engine = create_async_engine(url)
        async with engine.connect() as conn:
            res = await conn.execute(text("SELECT 1"))
            print(f"Result: {res.scalar()}")
        print(f"Success! Time taken: {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Failed! Time taken: {time.time() - start:.2f}s")
        print(f"Error: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_conn())
