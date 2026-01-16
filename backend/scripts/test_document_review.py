import asyncio
import logging
from app.services.rag_service import get_rag_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BAD_CONTRACT = """
EMPLOYMENT AGREEMENT

1. SALARY: The Employee shall be paid 50,000 Naira per month.
2. TERMINATION: The Employer may terminate this agreement at any time without notice and without paying any outstanding salary.
3. OVERTIME: The Employee agrees to work up to 14 hours a day, 7 days a week, with no additional pay.
4. LIABILITY: If the Employee gets injured on the job, the Employer is not responsible for any medical bills.
5. NON-COMPETE: Upon leaving, the Employee cannot work in any similar industry in Nigeria for 10 years.
"""

async def main():
    print(f"\n📄 Analyzing Contract:\n{BAD_CONTRACT}\n")
    print("⏳ Scanning for dangerous clauses...\n")
    
    rag_service = get_rag_service()
    analysis = rag_service.analyze_document(BAD_CONTRACT)
    
    print("-" * 50)
    print("🤖 AI Analysis:")
    print("-" * 50)
    print(f"Risk Score: {analysis.get('risk_score')}/10")
    print(f"Verdict: {analysis.get('overall_verdict')}")
    print(f"Summary: {analysis.get('summary')}")
    print("\n⚠️ Dangerous Clauses Found:")
    
    for clause in analysis.get("dangerous_clauses", []):
        print(f"\n🔴 Clause: \"{clause['clause_text']}\"")
        print(f"   Category: {clause.get('category', 'Unknown')}")
        print(f"   Why it's bad: {clause['explanation']}")
        print(f"   Recommendation: {clause['recommendation']}")

if __name__ == "__main__":
    asyncio.run(main())
