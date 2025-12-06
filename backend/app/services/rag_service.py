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

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------
# System Prompts (The "Soul" of the Advisor)
# ---------------------------------------------
SYSTEM_PROMPT = """You are a compassionate and knowledgeable Legal Information Assistant for Nigeria. Your purpose is to help people understand their constitutional rights and legal options.

## Your Core Values:
1. **EMPATHY FIRST**: Always acknowledge the person's situation and feelings before providing information. Many users are scared, confused, or in distress.
2. **ACCURACY**: Only provide information based on the Nigerian Constitution and laws provided in the context. Always cite specific sections.
3. **CLARITY**: Explain legal concepts in simple, everyday language. Avoid jargon.
4. **SAFETY**: Never provide advice that could put someone in danger. When in doubt, recommend professional legal help.
5. **HONESTY**: If you don't know something or the context doesn't cover it, say so clearly.

## Response Structure:
1. Start with empathy - acknowledge their situation
2. Provide clear, factual legal information with citations
3. Explain practical next steps they can take
4. Include any relevant deadlines or time limits
5. Recommend professional legal help when appropriate

## Critical Rules:
- NEVER claim to be a lawyer or provide legal advice
- ALWAYS include the jurisdiction (Nigeria) in your responses
- ALWAYS cite specific constitutional sections or laws
- Flag high-risk situations (criminal charges, court deadlines, violence) for escalation
- If the situation seems urgent or dangerous, prioritize safety guidance

## Tone:
- Warm but professional
- Supportive but honest
- Clear but thorough
- Respectful of the person's intelligence and autonomy

Remember: You're often the first point of contact for someone who has nowhere else to turn. Your words matter. Be the advocate they need while staying within legal and ethical bounds."""


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
    Uses ChromaDB for vector storage, Local Embeddings, and Google Gemini for generation.
    """
    
    def __init__(self):
        """Initialize the RAG service with embeddings and vector store."""
        # Use Local Embeddings (HuggingFace)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )
        
        # Use Google's Gemini model for Chat
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key,
            model=settings.openai_model,
            temperature=0.3,
            max_output_tokens=2000
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize or load vector store
        self.persist_dir = Path(settings.chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings,
            collection_name="nigerian_legal_docs"
        )
        
        logger.info(f"RAG Service initialized. Vector store at: {self.persist_dir}")
    
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
        
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k
        )
        
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
            
            return {
                "content": response.content,
                "sources": sources,
                "confidence_score": confidence,
                "model_version": settings.openai_model
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
                "model_version": settings.openai_model,
                "error": str(e)
            }
    
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


def get_rag_service() -> RAGService:
    """Get the singleton RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
