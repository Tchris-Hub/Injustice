"""
Search Service
--------------
Handles web search integration using SearxNG.
Provides 'Live' real-time results for the RAG pipeline.
"""

import logging
import httpx
from typing import List, Dict, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class SearchService:
    """
    Service for interacting with SearxNG metasearch engine.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.SEARXNG_URL
        self.fallback_urls = [
            "https://searx.be",
            "https://searx.neocities.org",
            "https://searx.work",
            "https://search.md",
            "https://searx.priv.at",
            "https://searx.octopuce.fr",
            "https://xo.smashit.digital",
            "https://search.disroot.org"
        ]
        self.timeout = 10.0  # Slightly lower timeout for faster fallbacks
    
    async def search(
        self, 
        query: str, 
        categories: List[str] = ["general", "news"],
        language: str = "en-US",
        top_k: int = None
    ) -> List[Dict]:
        """
        Execute search on SearxNG with automatic instance fallback.
        """
        top_k = top_k or settings.SEARCH_TOP_K
        
        # Test the provided base URL first, then try others if it fails
        search_urls = [self.base_url] + [url for url in self.fallback_urls if url != self.base_url]
        
        for url in search_urls:
            try:
                results = await self._execute_request(url, query, categories, language, top_k)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"⚠️ Search failed via {url}: {e}. Trying next...")
                continue
                
        logger.error("❌ All search instances failed for query.")
        return []

    async def _execute_request(self, base_url: str, query: str, categories: List[str], language: str, top_k: int) -> List[Dict]:
        params = {
            "q": query,
            "format": "json",
            "categories": ",".join(categories),
            "language": language,
            "safesearch": 1
        }
        
        async with httpx.AsyncClient(follow_redirects=True) as client:
            logger.info(f"🔍 Live Pulse: '{query[:50]}' via {base_url}")
            response = await client.get(
                base_url.rstrip('/') + "/search",
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            results = []
            
            for raw in data.get("results", [])[:top_k]:
                # Extract and clean content
                content = raw.get("content", "") or raw.get("snippet", "")
                if not content and "title" in raw:
                    content = f"Article titled: {raw['title']}"
                    
                results.append({
                    "title": raw.get("title", "No Title"),
                    "content": content,
                    "url": raw.get("url", ""),
                    "score": raw.get("score", 0),
                    "publishedDate": raw.get("publishedDate")
                })
            return results

    def format_results_for_prompt(self, results: List[Dict]) -> str:
        """Helper to format search results into a clean text block for the LLM."""
        if not results:
            return "No recent web results found relating to this query."
            
        formatted = ["### 🌐 Live Web Context (from SearxNG):"]
        for i, res in enumerate(results, 1):
            date_info = f" (Date: {res['publishedDate']})" if res.get('publishedDate') else ""
            formatted.append(
                f"{i}. **{res['title']}** {date_info}\n"
                f"   Snippet: {res['content']}\n"
                f"   URL: {res['url']}"
            )
        return "\n\n".join(formatted)
