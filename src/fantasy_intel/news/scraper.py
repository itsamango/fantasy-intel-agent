"""Scraper and RSS collector for NFL news feeds."""

from __future__ import annotations
import logging
from typing import Any, Dict, List
import feedparser

logger = logging.getLogger(__name__)

DEFAULT_FEEDS = {
    "ESPN NFL": "https://www.espn.com/espn/rss/nfl/news",
    "ProFootballTalk": "https://profootballtalk.nbcsports.com/feed/",
    "NFL Analysis": "https://www.rotowire.com/rss/news.php?sport=NFL",
}


class NewsCollector:
    """Collects latest NFL news, injury updates, and beat reporter feeds."""

    def __init__(self, feeds: Dict[str, str] = DEFAULT_FEEDS):
        self.feeds = feeds

    def fetch_latest_news(self, limit_per_feed: int = 15) -> List[Dict[str, Any]]:
        """Parses all configured RSS feeds into a unified list of articles."""
        articles = []
        for source_name, url in self.feeds.items():
            try:
                parsed = feedparser.parse(url)
                for entry in parsed.entries[:limit_per_feed]:
                    article = {
                        "source": source_name,
                        "title": getattr(entry, "title", ""),
                        "summary": getattr(entry, "summary", "") or getattr(entry, "description", ""),
                        "link": getattr(entry, "link", ""),
                        "published": getattr(entry, "published", "") or getattr(entry, "updated", ""),
                    }
                    articles.append(article)
            except Exception as e:
                logger.warning(f"Error fetching feed {source_name} ({url}): {e}")

        return articles
