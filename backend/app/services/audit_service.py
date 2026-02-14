"""
Audit Service
--------------
Background auditing of user queries using a specialized "Guardian" AI.
Logs compliance and safety metrics to the audit_logs table.
"""

import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
import openai

from app.core.config import settings
from app.db.models import AuditLog
from app.db.session import async_session

logger = logging.getLogger(__name__)

GUARDIAN_PROMPT = """You are the 'Shadow Guardian' for the Injustice (My Rights) AI Legal Advisor.
Your job is to audit user queries for compliance, safety, and jurisdictional relevance.

## Your Goals:
1. **Safety Check**: Is the user asking for something illegal, harmful, or trying to bypass AI safety filters?
2. **Jurisdiction Check**: Is the query relevant to Nigerian law?
3. **Complexity Check**: Is this query so complex it MUST be escalated to a human lawyer?

## Evaluation Criteria (JSON Response):
- **is_safe**: boolean
- **is_jurisdictionally_relevant**: boolean (Nigeria)
- **needs_escalation**: boolean
- **risk_category**: one of ["none", "safety_violation", "out_of_jurisdiction", "high_complexity", "ambiguous"]
- **audit_note**: A brief (1 sentence) explanation of your verdict.

Query: "{query}"

Respond ONLY with a JSON object.
"""

class AuditService:
    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        self.base_url = settings.OPENROUTER_BASE_URL
        self.model = settings.MODEL_CONFIG.get("analysis", settings.MODEL_NAME)

    async def audit_query(
        self, 
        query: str, 
        user_id: Optional[uuid.UUID] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Asynchronously audit a user query and log it to the database.
        Runs as a shadow process to avoid blocking the main chat response.
        """
        try:
            # 1. Call Guardian AI
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": GUARDIAN_PROMPT.replace("{query}", query)}
                ],
                response_format={"type": "json_object"}
            )
            
            audit_result = json.loads(response.choices[0].message.content)
            
            # 2. Log to Database using a fresh background session
            async with async_session() as db:
                audit_entry = {
                    "id": uuid.uuid4(),
                    "action": "ai_query_audit",
                    "entity_type": "message",
                    "user_id": user_id,
                    "ip_address": ip_address,
                    "user_agent": user_agent,
                    "details": {
                        "query_excerpt": query[:100] + "..." if len(query) > 100 else query,
                        "evaluation": audit_result
                    },
                    "created_at": datetime.now(timezone.utc)
                }
                
                stmt = insert(AuditLog).values(**audit_entry)
                await db.execute(stmt)
                await db.commit()
            
            logger.info(f"Audit complete for query. Verdict: {audit_result.get('risk_category')}")
            
        except Exception as e:
            logger.error(f"AuditService Error: {e}")
            # Do not raise - auditing failure should not break the user experience
            # We just log the failure.

# Singleton instance
audit_service = AuditService()
