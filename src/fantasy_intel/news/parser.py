"""Context parser for news items: player mentions, injury severity, depth chart changes."""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional

from fantasy_intel.news.llm_analyzer import LLMNewsAnalyzer

INJURY_KEYWORDS = [
    "injured", "injury", "hamstring", "concussion", "ankle", "knee", "torn acl",
    "sprain", "out for season", "placed on ir", "reserve/injured", "questionable", "doubtful", "ruled out"
]

OPPORTUNITY_KEYWORDS = [
    "named starter", "first-team reps", "promoted", "increased role",
    "workhorse", "featured back", "red zone target", "breakout", "target monster"
]


class NewsContextParser:
    """Extracts fantasy context from news headlines and summaries."""

    def __init__(self, llm_analyzer: Optional[LLMNewsAnalyzer] = None):
        self.llm_analyzer = llm_analyzer or LLMNewsAnalyzer()

    def filter_by_player(self, articles: List[Dict[str, Any]], player_name: str) -> List[Dict[str, Any]]:
        """Filters articles that explicitly mention a player's name."""
        pattern = re.compile(re.escape(player_name), re.IGNORECASE)
        matching = []
        for a in articles:
            text = f"{a.get('title', '')} {a.get('summary', '')}"
            if pattern.search(text):
                matching.append(a)
        return matching

    def analyze_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Tags an article with injury and opportunity signals."""
        text = f"{article.get('title', '')} {article.get('summary', '')}".lower()

        injuries = [kw for kw in INJURY_KEYWORDS if kw in text]
        opportunities = [kw for kw in OPPORTUNITY_KEYWORDS if kw in text]

        category = "general"
        if injuries:
            category = "injury"
        elif opportunities:
            category = "role_expansion"

        return {
            **article,
            "category": category,
            "injury_signals": injuries,
            "opportunity_signals": opportunities,
        }

    def analyze_with_ai(self, articles: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Deep analysis extracting structured injury severity, timeframe,
        promoted depth-chart beneficiaries, and actionable fantasy advice.
        """
        return self.llm_analyzer.analyze_news_items(articles, limit=limit)
