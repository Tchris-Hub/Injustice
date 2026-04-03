
import httpx
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class NewsService:
    """
    Fetches real-time legal news from Nigerian RSS feeds.
    Provides the 'Pulse' layer for the Digital Jurist.
    """
    
    FEEDS = {
        "Thenigerianlawyer": "https://thenigerianlawyer.com/feed/",
        "Lawyard": "https://lawyard.ng/feed/",
        "The Jurist": "https://thejurist.ng/feed/"
    }

    async def fetch_latest_news(self, limit: int = 5) -> List[Dict]:
        """
        Fetch latest headlines from all configured feeds.
        """
        all_news = []
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for source, url in self.FEEDS.items():
                try:
                    logger.info(f"📰 Fetching pulse from: {source}")
                    response = await client.get(url)
                    if response.status_code == 200:
                        news_items = self._parse_rss(response.text, source)
                        all_news.extend(news_items[:limit])
                except Exception as e:
                    logger.error(f"❌ Failed to fetch news from {source}: {e}")
        
        # Sort by date (if available) - naturally sorted by feed typically
        return all_news

    def _parse_rss(self, xml_content: str, source: str) -> List[Dict]:
        """Simple XML parsing for RSS feeds."""
        items = []
        try:
            root = ET.fromstring(xml_content)
            # RSS typically has <channel><item> structure
            for item in root.findall(".//item"):
                title = item.find("title").text if item.find("title") is not None else "No Title"
                link = item.find("link").text if item.find("link") is not None else ""
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                description = item.find("description").text if item.find("description") is not None else ""
                
                # Basic cleaning of description (some have HTML)
                import re
                clean_desc = re.sub('<[^<]+?>', '', description)[:200] + "..." if description else ""

                items.append({
                    "title": title,
                    "url": link,
                    "published": pub_date,
                    "summary": clean_desc,
                    "source": source
                })
        except Exception as e:
            logger.error(f"❌ RSS Parsing Error for {source}: {e}")
            
        return items

    def format_news_for_prompt(self, news_items: List[Dict]) -> str:
        """Formats legal news items for inclusion in LLM prompt."""
        if not news_items:
            return "No recent legal news updates found."
            
        formatted = ["### 🏮 Latest Nigerian Legal Pulse (via RSS):"]
        for i, item in enumerate(news_items, 1):
            formatted.append(
                f"{i}. **{item['title']}** ({item['source']})\n"
                f"   Summary: {item['summary']}\n"
                f"   Link: {item['url']}"
            )
        return "\n\n".join(formatted)
