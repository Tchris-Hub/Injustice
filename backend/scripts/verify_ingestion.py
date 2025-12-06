"""
Verify Ingestion
-----------------
Test the RAG system by querying the ingested constitution.
Usage: python -m scripts.verify_ingestion
"""

import asyncio
import logging

from app.services.rag_service import get_rag_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Test the RAG retrieval."""
    logger.info("Verifying RAG ingestion...")
    
    rag_service = get_rag_service()
    
    # Check stats
    stats = rag_service.get_collection_stats()
    logger.info(f"Collection Stats: {stats}")
    
    if stats.get("total_documents", 0) == 0:
        logger.error("❌ No documents found in vector store!")
        return
    
    # Test queries
    test_queries = [
        "What are my rights if I am arrested?",
        "Can the police search my house without a warrant?",
        "What does the constitution say about fair hearing?"
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: '{query}'")
        results = rag_service.retrieve_relevant_chunks(query, k=3)
        
        if not results:
            logger.warning("  ⚠️ No results found.")
            continue
            
        for i, (doc, score) in enumerate(results):
            logger.info(f"  Result {i+1} (Score: {score:.4f}):")
            logger.info(f"    Source: {doc.metadata.get('title')}")
            logger.info(f"    Excerpt: {doc.page_content[:100]}...")
    
    logger.info("\n✅ Verification complete!")


if __name__ == "__main__":
    asyncio.run(main())
