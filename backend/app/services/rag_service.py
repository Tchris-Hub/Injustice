"""
RAG Service
------------
Retrieval Augmented Generation for the Legal Advisor.
Combines vector search with LLM generation for accurate, cited responses.
"""

import hashlib
import logging
from typing import List, Optional, Tuple
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------
# System Prompts (The "Soul" of the Advisor)
# ---------------------------------------------
SYSTEM_PROMPT = """You are a warm, empathetic, and knowledgeable Legal Assistant for Nigeria. 
Your goal is to help everyday people understand their rights in simple terms.

## Your Core Personality:
1.  **Active Listening**: Start by acknowledging the user's pain. Use phrases like "I hear you," "That sounds incredibly stressful," or "I'm so sorry you're going through this."
2.  **Simple Language**: Speak like a helpful friend, not a lawyer. Use Grade 8 reading level. Avoid "legalese" (e.g., instead of "pursuant to," say "according to").
3.  **Empowerment**: Make the user feel they have options and rights.

## Response Structure:
1.  **Validation**: "I hear you..." (1-2 sentences validating their feelings).
2.  **Your Rights (Simplified)**: Explain what the Nigerian Constitution says about their situation in plain English. Cite sections but explain them simply.
3.  **Practical Steps**: Bullet points of what they can actually DO today (e.g., "Write a letter," "Report to X").
4.  **Safety/Disclaimer**: Gently remind them you are an AI and they should see a real lawyer for court cases.

## Critical Rules:
- NEVER say "I cannot provide legal advice" as your opening. It's cold. Put the disclaimer at the end.
- ALWAYS cite the 1999 Constitution (As Amended) where relevant.
- IF the user mentions violence or immediate danger, prioritize safety instructions immediately.
"""

DOCUMENT_REVIEW_PROMPT = """You are a hawk-eyed Legal Contract Reviewer. Your job is to protect the user from bad deals.
Review the following legal document text and identify any clauses that are:
1.  **Dangerous/Malicious**: Directly harmful to the user.
2.  **Not in Best Interest**: Unfairly weighted against the user.
3.  **Legal Risks**: Hidden liabilities or ambiguous terms.

Document Text:
"{document_text}"

Respond with a JSON object containing:
- "summary": "A 1-sentence summary of what this document is.",
- "risk_score": A number from 1-10 (10 is very dangerous),
- "dangerous_clauses": A list of objects, each with:
    - "clause_text": "The exact text from the document",
    - "category": "Dangerous", "Not in Best Interest", or "Legal Risk",
    - "explanation": "Brief legal explanation of why this is bad",
    - "simplified_explanation": "Break down the legal terms into Grade 8 level English",
    - "long_term_implications": "What does this mean for the user 1 or 5 years from now?",
    - "pros": ["List of any hidden benefits or 'silver linings' (if any)", "..."],
    - "cons": ["Direct disadvantages to the user", "..."],
    - "recommendation": "What they should ask to change"
- "overall_verdict": "Safe to sign", "Proceed with caution", or "DO NOT SIGN"

Only respond with the JSON object."""

DOCUMENT_GENERATION_PROMPT = """You are a Legal Document Generator.
Your goal is to create a professional, legally sound TEMPLATE based on the user's request.

User Request: "{user_request}"
User Details: "{user_details}"

Rules:
1.  Create a clear, formal document.
2.  Use placeholders like [INSERT DATE], [INSERT NAME] where specific info is missing.
3.  Ensure the tone is professional and assertive but polite.
4.  Do NOT invent fake laws. Use general legal principles applicable in Nigeria.

Output ONLY the document text."""

LEGAL_DISCLAIMER_TEXT = """
\n\n---
**⚠️ LEGAL DISCLAIMER**
This information is for educational purposes only and does not constitute legal advice. 
I am an AI, not a lawyer. Laws change and specific situations vary. 
**Please consult a qualified attorney for professional legal advice.**
"""


RISK_ASSESSMENT_PROMPT = """Analyze the following user message and determine the risk level and topic category.

User message: "{message}"

Respond with a JSON object containing:
- "risk_level": "low", "medium", or "high"
- "legal_topic": one of ["arrest_rights", "police_encounter", "tenant_rights", "employment", "consumer_rights", "family_law", "criminal_matter", "civil_matter", "constitutional_rights", "general_info"]
- "escalation_needed": true or false
- "escalation_reason": string explaining why escalation is needed (if applicable)

Risk Levels:
- LOW: General legal information questions, educational queries
- MEDIUM: Active legal situation but not immediately dangerous
- HIGH: Criminal charges, court deadlines within 7 days, violence, detention, imminent harm

Only respond with the JSON object, no other text."""


# ---------------------------------------------
# RAG Service Class
# ---------------------------------------------
class RAGService:
    """
    Handles document retrieval and AI-powered response generation.
    Uses ChromaDB for vector storage, Local Embeddings, and OpenRouter for generation.
    """
    
    def __init__(self):
        """Initialize the RAG service with embeddings and vector store."""
        self.initialization_error = None
        
        logger.info(f"Initializing RAG Service...")
        
        # Step 1: Initialize local embeddings (Default for privacy/reliability)
        try:
            logger.info(f"Step 1/4: Initializing Local Embeddings ({settings.embedding_model})...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"✓ Local HuggingFace Embeddings initialized")
        except Exception as e:
            self.initialization_error = f"Failed to initialize embeddings: {e}"
            logger.error(self.initialization_error)
            raise
        
        # Step 3: Initialize LLM (OpenRouter)
        try:
            model_name = settings.MODEL_CONFIG["default"]
            logger.info(f"Step 2/4: Initializing OpenRouter LLM ({model_name})...")
            
            self.llm = ChatOpenAI(
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=settings.OPENROUTER_API_KEY,
                model=model_name,
                temperature=0.3,
                max_tokens=4000
            )
            logger.info("✓ OpenRouter LLM initialized")
        except Exception as e:
            self.initialization_error = f"Failed to initialize OpenRouter LLM: {type(e).__name__}: {str(e)}"
            logger.error(self.initialization_error)
            raise
        
        # Step 4: Initialize text splitter
        try:
            logger.info("Step 3/4: Initializing text splitter...")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.rag_chunk_size,
                chunk_overlap=settings.rag_chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            logger.info("✓ Text splitter initialized")
        except Exception as e:
            self.initialization_error = f"Failed to initialize text splitter: {type(e).__name__}: {str(e)}"
            logger.error(self.initialization_error)
            raise
        
        # Step 5: Initialize vector store
        try:
            logger.info("Step 4/4: Initializing ChromaDB vector store...")
            self.persist_dir = Path(settings.chroma_persist_dir)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.vector_store = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings,
                collection_name="nigerian_legal_docs"
            )
            logger.info(f"✓ Vector store initialized at: {self.persist_dir}")
        except Exception as e:
            self.initialization_error = f"Failed to initialize ChromaDB: {type(e).__name__}: {str(e)}"
            logger.error(self.initialization_error)
            raise
        
        logger.info("✓ RAG Service fully initialized successfully!")
    
    def ingest_document(
        self,
        content: str,
        title: str,
        document_type: str,
        metadata: Optional[dict] = None
    ) -> int:
        """
        Ingest a legal document into the vector store.
        
        Args:
            content: The full text of the document
            title: Document title (e.g., "Constitution of Nigeria 1999")
            document_type: Type of document ("constitution", "statute", etc.)
            metadata: Additional metadata
            
        Returns:
            Number of chunks created
        """
        # Split document into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                "title": title,
                "document_type": document_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "jurisdiction": "Nigeria",
                "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()[:8]
            }
            if metadata:
                doc_metadata.update(metadata)
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        # Add to vector store (Local embeddings are fast, no need for batching/sleep)
        self.vector_store.add_documents(documents)
        
        logger.info(f"Ingested '{title}': {len(chunks)} chunks created")
        return len(chunks)
    
    def retrieve_relevant_chunks(
        self,
        query: str,
        k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: User's question
            k: Number of chunks to retrieve
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        k = k or settings.rag_top_k
        
        try:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise e
        
        logger.debug(f"Retrieved {len(results)} chunks for query: '{query[:50]}...'")
        return results
    
    def assess_risk(self, message: str) -> dict:
        """
        Assess the risk level and topic of a user message.
        
        Args:
            message: User's message
            
        Returns:
            Dict with risk_level, legal_topic, escalation_needed, escalation_reason
        """
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a risk assessment system. Respond only with valid JSON."),
                HumanMessage(content=RISK_ASSESSMENT_PROMPT.format(message=message))
            ])
            
            import json
            return json.loads(response.content)
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {
                "risk_level": "medium",
                "legal_topic": "general_info",
                "escalation_needed": False,
                "escalation_reason": None
            }
    
    def generate_response(
        self,
        user_message: str,
        conversation_history: List[dict] = None,
        retrieved_chunks: List[Tuple[Document, float]] = None
    ) -> dict:
        """
        Generate an AI response using RAG.
        
        Args:
            user_message: The user's question
            conversation_history: Previous messages in the conversation
            retrieved_chunks: Pre-retrieved chunks (if None, will retrieve)
            
        Returns:
            Dict with response content, sources, and metadata
        """
        # Retrieve relevant chunks if not provided
        if retrieved_chunks is None:
            retrieved_chunks = self.retrieve_relevant_chunks(user_message)
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for doc, score in retrieved_chunks:
            context_parts.append(
                f"[Source: {doc.metadata.get('title', 'Unknown')} - "
                f"{doc.metadata.get('section', 'General')}]\n{doc.page_content}"
            )
            sources.append({
                "title": doc.metadata.get("title", "Unknown"),
                "section": doc.metadata.get("section"),
                "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "document_type": doc.metadata.get("document_type", "unknown"),
                "relevance_score": round(1 - score, 2)  # Convert distance to similarity
            })
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant legal documents found."
        
        # Build messages for LLM
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(SystemMessage(content=f"[Previous response]: {msg['content'][:500]}"))
        
        # Add current query with context
        user_prompt = f"""## Legal Context (from Nigerian law):
{context}

## User's Question:
{user_message}

Please provide a helpful, empathetic response based on the legal context above. Remember to:
1. Acknowledge their situation first
2. Cite specific sections when referencing laws
3. Provide practical next steps
4. Recommend professional help if appropriate"""
        
        messages.append(HumanMessage(content=user_prompt))
        
        # Generate response
        try:
            response = self.llm.invoke(messages)
            
            # Determine confidence based on source relevance
            avg_relevance = sum(s["relevance_score"] for s in sources) / len(sources) if sources else 0
            if avg_relevance > 0.7:
                confidence = "high"
            elif avg_relevance > 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            
            # Append Disclaimer
            final_content = response.content + LEGAL_DISCLAIMER_TEXT
            
            return {
                "content": final_content,
                "sources": sources,
                "confidence_score": confidence,
                "model_version": settings.MODEL_CONFIG["default"]
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "content": (
                    "I apologize, but I'm having trouble processing your question right now. "
                    "This could be a temporary issue. Please try again in a moment, or if this "
                    "is urgent, please contact a local legal aid organization directly."
                ),
                "sources": [],
                "confidence_score": "low",
                "model_version": settings.MODEL_CONFIG["default"],
                "error": str(e)
            }
    
    def analyze_document(self, document_text: str) -> dict:
        """
        Analyze a legal document for dangerous clauses.
        
        Args:
            document_text: The text of the contract/document
            
        Returns:
            Dict with risk analysis
        """
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a strict contract reviewer. Respond only with valid JSON."),
                HumanMessage(content=DOCUMENT_REVIEW_PROMPT.format(document_text=document_text))
            ])
            
            # Clean up response content if it contains markdown code blocks
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            import json
            result = json.loads(content)
            
            # Add disclaimer to analysis as well
            result["disclaimer"] = "This analysis is automated and for informational purposes only. Consult a lawyer."
            return result
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return {
                "error": "Could not analyze document. Please ensure it is text-based.",
                "details": str(e)
            }

    def generate_document(self, doc_type: str, user_details: str) -> str:
        """
        Generate a legal document template.
        
        Args:
            doc_type: Type of document (e.g., "Demand Letter")
            user_details: Specific details to include
            
        Returns:
            String containing the document text
        """
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a professional legal document generator."),
                HumanMessage(content=DOCUMENT_GENERATION_PROMPT.format(
                    user_request=doc_type,
                    user_details=user_details
                ))
            ])
            
            # Prepend strict warning
            warning = "⚠️ **TEMPLATE ONLY - REVIEW WITH A LAWYER** ⚠️\n\n"
            return warning + response.content + LEGAL_DISCLAIMER_TEXT
            
        except Exception as e:
            logger.error(f"Document generation failed: {e}")
            return f"Error generating document: {e}"

    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store."""
        try:
            collection = self.vector_store._collection
            return {
                "total_documents": collection.count(),
                "persist_directory": str(self.persist_dir)
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}


# Singleton instance
_rag_service: Optional[RAGService] = None
_rag_service_error: Optional[str] = None


def get_rag_service() -> RAGService:
    """Get the singleton RAG service instance."""
    global _rag_service, _rag_service_error
    
    # If we already have an error, raise it immediately
    if _rag_service_error:
        logger.error(f"RAG Service previously failed: {_rag_service_error}")
        raise RuntimeError(f"RAG Service initialization failed: {_rag_service_error}")
    
    if _rag_service is None:
        try:
            logger.info("Creating new RAG Service instance...")
            _rag_service = RAGService()
        except Exception as e:
            _rag_service_error = f"{type(e).__name__}: {str(e)}"
            logger.error(f"RAG Service creation failed: {_rag_service_error}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"RAG Service initialization failed: {_rag_service_error}")
    
    return _rag_service


def reset_rag_service():
    """Reset the RAG service (useful for retrying after fixing config)."""
    global _rag_service, _rag_service_error
    _rag_service = None
    _rag_service_error = None
    logger.info("RAG Service reset - will reinitialize on next request")
