import asyncio
import uuid
import sys
import os
from datetime import datetime, timezone

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Mock environment
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/injustice.db")
os.environ.setdefault("SECRET_KEY", "test-secret-key-12345")

from backend.app.services.rag_service import get_rag_service
from backend.app.schemas.chat import MessageCreate
from backend.app.db.models import User
from backend.app.api.v1.endpoints.chat import send_message
from backend.app.db.session import async_session
from fastapi import BackgroundTasks, Request

# Dummy Request object
class MockRequest:
    def __init__(self):
        self.client = type('obj', (object,), {'host': '127.0.0.1'})
        self.headers = {"user-agent": "test-suite"}

async def test_chat_flow():
    print("🚀 Testing Authenticated Chat Flow...")
    
    # Setup
    data = MessageCreate(content="Hi there, I have a question about my rights.")
    user = User(id=uuid.uuid4(), email="test@example.com")
    bg_tasks = BackgroundTasks()
    request = MockRequest()
    
    async with async_session() as db:
        try:
            print("📡 Calling send_message endpoint logic...")
            # We call the endpoint logic directly
            # Note: We pass None for DB because the endpoint handles its own sessions 
            # OR depends on the yield. For this test, let's just see if it crashes.
            
            response = await send_message(
                request=request,
                data=data,
                background_tasks=bg_tasks,
                current_user=user,
                db=db
            )
            
            print(f"✅ Response received!")
            print(f"📝 Content excerpt: {response.message.content[:100]}...")
            
            if "Failure ID" in response.message.content:
                print("❌ ERROR: Still seeing Failure ID in response!")
            else:
                print("🎉 SUCCESS: Chat flow completed without internal error.")
                
        except Exception as e:
            print(f"💥 Crashed with exception: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_chat_flow())
