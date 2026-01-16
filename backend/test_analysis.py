import os
import asyncio
from app.services.rag_service import get_rag_service
from dotenv import load_dotenv

async def test_analysis():
    load_dotenv()
    rag_service = get_rag_service()
    
    # A clearly bad contract clause
    bad_contract = """
    TENANCY AGREEMENT
    1. The Tenant shall pay the rent in advance.
    2. The Landlord reserves the right to enter the premises at any time without notice.
    3. The Tenant agrees to pay a 100% penalty for any rent delay exceeding 2 days.
    4. The Landlord is not liable for any injury or death occurring on the premises, even if caused by Landlord's negligence.
    """
    
    print("Analyzing bad contract...")
    result = rag_service.analyze_document(bad_contract)
    
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(test_analysis())
