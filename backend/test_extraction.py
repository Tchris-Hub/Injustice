
import sys
import os
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.services.rag_service import get_rag_service
import asyncio

async def main():
    try:
        service = get_rag_service()
        pdf_path = Path("data/nigerian_constitution_1999.pdf")
        if not pdf_path.exists():
            print(f"File {pdf_path} not found")
            return
            
        with open(pdf_path, "rb") as f:
            content = f.read()
            
        print(f"Attempting to extract text from {pdf_path}...")
        text = service.extract_text_from_pdf(content)
        print(f"Extraction successful! First 500 chars:\n{text[:500]}...")
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
