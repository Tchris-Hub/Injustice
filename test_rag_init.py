import logging
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Mock environment variables if needed
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")

from backend.app.services.rag_service import get_rag_service, reset_rag_service

logging.basicConfig(level=logging.INFO)

def test_init():
    print("🚀 Testing RAG Service Initialization...")
    try:
        service = get_rag_service()
        print("✅ RAG Service initialized successfully!")
        
        # Test a simple retrieval
        print("📡 Testing snippet retrieval...")
        chunks = service.retrieve_relevant_chunks("Hello", k=1)
        print(f"✅ Retrieved {len(chunks)} chunks.")
        
    except Exception as e:
        print(f"❌ RAG Service initialization FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_init()
