import asyncio
import logging
from app.services.rag_service import get_rag_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    doc_type = "Demand Letter for Unpaid Rent"
    user_details = "Landlord: John Doe. Tenant: Jane Smith. Property: 123 Lagos Street. Amount Owed: 500,000 Naira. Due Date: Last month."
    
    print(f"\n📄 Generating Document: {doc_type}\n")
    print(f"📝 Details: {user_details}\n")
    print("⏳ Generating template...\n")
    
    rag_service = get_rag_service()
    document = rag_service.generate_document(doc_type, user_details)
    
    print("-" * 50)
    print("🤖 Generated Document:")
    print("-" * 50)
    print(document)
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
