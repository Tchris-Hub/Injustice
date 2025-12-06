"""
Data Ingestion Script
----------------------
Parses and ingests legal documents into the RAG vector store.
Usage: python -m scripts.ingest_data
"""

import asyncio
import logging
import os
from pathlib import Path

from app.services.rag_service import get_rag_service
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


async def ingest_constitution():
    """Ingest the Nigerian Constitution sample."""
    file_path = DATA_DIR / "nigerian_constitution_full.txt"
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
    
    logger.info(f"Reading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    rag_service = get_rag_service()
    
    logger.info("Ingesting document...")
    num_chunks = rag_service.ingest_document(
        content=content,
        title="Constitution of the Federal Republic of Nigeria 1999 (As Amended)",
        document_type="constitution",
        metadata={
            "source": "official_pdf",
            "year": "1999",
            "jurisdiction": "Nigeria"
        }
    )
    
    logger.info(f"Successfully ingested {num_chunks} chunks into Vector DB.")


async def main():
    """Main ingestion entry point."""
    logger.info("Starting data ingestion...")
    
    # Ensure data directory exists
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found at {DATA_DIR}")
        return

    # Ingest Constitution
    await ingest_constitution()
    
    logger.info("Ingestion complete!")


if __name__ == "__main__":
    asyncio.run(main())
