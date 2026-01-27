
import asyncio
import os
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.services.rag_service import get_rag_service

async def test_rag_robustness():
    print("🧪 Testing RAG Service Robustness...")
    
    try:
        service = get_rag_service()
        print("✅ Service initialized successfully")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return

    # 1. Test PDF Extraction (using the constitution pdf we found earlier)
    pdf_path = Path("data/nigerian_constitution_1999.pdf")
    if pdf_path.exists():
        print(f"\n📄 Testing PDF Extraction from {pdf_path}...")
        try:
            with open(pdf_path, "rb") as f:
                content = f.read()
            text = service.extract_text_from_pdf(content)
            if len(text) > 100:
                print(f"✅ PDF Extraction successful ({len(text)} chars)")
                print(f"   Snippet: {text[:100].replace(chr(10), ' ')}...")
            else:
                print("⚠️ PDF Extraction yielded little text.")
        except Exception as e:
            print(f"❌ PDF Extraction failed: {e}")
    else:
        print("\n⚠️ Skipping PDF test (file not found)")

    # 2. Test Image OCR fallback logic
    # We will try to extract text from a dummy image
    # Note: This calls the real API, so we expect it to try the models in order.
    print("\n🖼️ Testing OCR Fallback Logic...")
    small_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKwftQAAAABJRU5ErkJggg=="
    import base64
    image_bytes = base64.b64decode(small_image_b64)
    
    try:
        text = service.extract_text_from_image(image_bytes)
        print(f"✅ OCR successful. Result: {text}")
    except Exception as e:
        print(f"❌ OCR failed (expected if API keys invalid or rate limited): {e}")

if __name__ == "__main__":
    asyncio.run(test_rag_robustness())
