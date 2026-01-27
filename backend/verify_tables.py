
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from app.core.config import settings

async def list_tables():
    db_url = settings.database_url
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
    
    engine = create_async_engine(
        db_url,
        connect_args={
            "prepared_statement_cache_size": 0,
            "statement_cache_size": 0
        } if "pooler" in db_url else {}
    )
    
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """))
            tables = [row[0] for row in result.fetchall()]
            print(f"Found {len(tables)} tables in 'public' schema:")
            for t in tables:
                print(f" - {t}")
                
            # Verify specific production-ready tables
            target_tables = ['user_profiles', 'chat_archives', 'escalation_cases', 'lawyer_directory']
            missing = [t for t in target_tables if t not in tables]
            if not missing:
                print("\n✅ All production-ready target tables are present!")
            else:
                print(f"\n❌ Missing tables: {', '.join(missing)}")
                
    except Exception as e:
        print(f"Verification failed: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(list_tables())
