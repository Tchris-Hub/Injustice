import asyncio
import logging
from app.services.rag_service import get_rag_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    query = "What does the constitution say about unlawful arrest?"
    
    print(f"\n❓ User Question: {query}\n")
    print("⏳ Generating response... (this uses Gemini + Local Embeddings)\n")
    
    rag_service = get_rag_service()
    response = rag_service.generate_response(query)
    
    print("-" * 50)
    print("🤖 AI Response:")
    print("-" * 50)
    print(response["content"])
    print("-" * 50)
    
    print("\n📚 Sources Used:")
    for source in response["sources"]:
        print(f"- {source['title']} (Section: {source['section']}) - Relevance: {source['relevance_score']}")

if __name__ == "__main__":
    asyncio.run(main())
