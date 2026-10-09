"""Scraper and collector for NFL news feeds and Sleeper injury alerts."""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import feedparser
from bs4 import BeautifulSoup

from fantasy_intel.data.sleeper import SleeperClient

logger = logging.getLogger(__name__)

DEFAULT_FEEDS = {
    "ESPN NFL": "https://www.espn.com/espn/rss/nfl/news",
    "CBS Fantasy": "https://www.cbssports.com/rss/headlines/fantasy/football/",
    "ProFootballTalk": "https://profootballtalk.nbcsports.com/feed/",
    "NFL Analysis": "https://www.rotowire.com/rss/news.php?sport=NFL",
    "Yahoo NFL": "https://sports.yahoo.com/nfl/rss/",
}


def clean_html(text: str) -> str:
    """Strip HTML markup from RSS summary strings."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    text_with_spaces = soup.get_text(separator=" ")
    import re
    return re.sub(r"\s+", " ", text_with_spaces).strip()


class NewsCollector:
    """Collects latest NFL news, injury updates, beat reporter feeds, and Sleeper injury notes."""

    def __init__(self, feeds: Optional[Dict[str, str]] = None, sleeper_client: Optional[SleeperClient] = None):
        self.feeds = feeds or DEFAULT_FEEDS
        self.sleeper = sleeper_client or SleeperClient()

    def fetch_latest_news(self, limit_per_feed: int = 15) -> List[Dict[str, Any]]:
        """Parses all configured RSS feeds into a unified list of articles."""
        articles = []
        for source_name, url in self.feeds.items():
            try:
                parsed = feedparser.parse(url)
                for entry in parsed.entries[:limit_per_feed]:
                    raw_summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
                    clean_summary_text = clean_html(raw_summary)
                    article = {
                        "source": source_name,
                        "title": getattr(entry, "title", "").strip(),
                        "summary": clean_summary_text,
                        "link": getattr(entry, "link", ""),
                        "published": getattr(entry, "published", "") or getattr(entry, "updated", ""),
                    }
                    if article["title"]:
                        articles.append(article)
            except Exception as e:
                logger.warning(f"Error fetching feed {source_name} ({url}): {e}")

        return articles

    def fetch_sleeper_injury_notes(self, positions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Fetches official beat reporter injury notes directly from Sleeper's player database.
        Returns active skill-position players with injury status updates.
        """
        target_positions = set(positions or ["QB", "RB", "WR", "TE"])
        all_players = self.sleeper.get_all_players()
        notes_list = []

        for pid, p in all_players.items():
            pos = p.get("position")
            injury_status = p.get("injury_status")
            notes = p.get("injury_notes")
            team = p.get("team")

            if pos in target_positions and (injury_status or notes) and team:
                notes_list.append({
                    "source": "Sleeper Injury Wire",
                    "player_id": str(pid),
                    "player_name": p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                    "team": team,
                    "position": pos,
                    "injury_status": injury_status or "Injured",
                    "injury_body_part": p.get("injury_body_part") or "Undisclosed",
                    "depth_chart_order": p.get("depth_chart_order"),
                    "summary": notes or f"{p.get('full_name')} is listed as {injury_status} ({p.get('injury_body_part')}).",
                    "title": f"{p.get('full_name')} ({team} {pos}): {injury_status} - {p.get('injury_body_part') or ''}".strip(),
                })

        return sorted(notes_list, key=lambda x: (x["depth_chart_order"] or 99, x["player_name"]))
