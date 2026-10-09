"""Fantasy Intelligence Advisor orchestrating data, news, and league APIs."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd

from fantasy_intel.data.loader import DataLoader
from fantasy_intel.data.sleeper import SleeperClient
from fantasy_intel.analytics.waiver_wire import WaiverWireEngine
from fantasy_intel.analytics.trades import TradeAnalyzer
from fantasy_intel.analytics.depth_chart import DepthChartManager
from fantasy_intel.analytics.matchups import MatchupAdvisor
from fantasy_intel.news.scraper import NewsCollector
from fantasy_intel.news.parser import NewsContextParser


class FantasyAdvisor:
    """The central agent brain for fantasy recommendations."""

    def __init__(self, data_loader: Optional[DataLoader] = None, sleeper_client: Optional[SleeperClient] = None):
        self.loader = data_loader or DataLoader()
        self.sleeper = sleeper_client or SleeperClient()
        self.depth_manager = DepthChartManager(sleeper_client=self.sleeper)
        self.news_collector = NewsCollector(sleeper_client=self.sleeper)
        self.news_parser = NewsContextParser()
        self.matchup_advisor = MatchupAdvisor()

        # Cached analytics engines
        self._waiver_engine: Optional[WaiverWireEngine] = None
        self._trade_analyzer: Optional[TradeAnalyzer] = None
        self._loaded_seasons: List[int] = []

    def _ensure_engines(self, seasons: List[int]):
        if self._waiver_engine is None or self._loaded_seasons != seasons:
            weekly_with_snaps = self.loader.get_weekly_with_snaps(seasons=seasons)
            self._waiver_engine = WaiverWireEngine(weekly_with_snaps)
            self._trade_analyzer = TradeAnalyzer(weekly_with_snaps)
            self._loaded_seasons = seasons

    def get_waiver_sleepers(
        self,
        season: int = 2024,
        week: Optional[int] = None,
        positions: Optional[List[str]] = None,
        league_id: Optional[str] = None,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Finds under-the-radar waiver sleepers, optionally filtering out owned players in a Sleeper league."""
        self._ensure_engines([season])
        candidates = self._waiver_engine.scan_sleepers(
            week=week,
            season=season,
            positions=positions,
            limit=limit * 2 if league_id else limit,
        )

        if candidates.empty:
            return pd.DataFrame()

        # If Sleeper league provided, filter out already rostered players
        if league_id:
            rostered_ids = self.sleeper.get_rostered_player_ids(league_id)
            players_db = self.sleeper.get_all_players()
            # Map player names to sleeper IDs
            rostered_names = set()
            for pid in rostered_ids:
                pinfo = players_db.get(pid, {})
                fname = pinfo.get("full_name", "").lower()
                if fname:
                    rostered_names.add(fname)

            # Filter candidates
            candidates["is_rostered"] = candidates["player_display_name"].apply(
                lambda name: name.lower() in rostered_names
            )
            available = candidates[~candidates["is_rostered"]].head(limit)
            return available.reset_index(drop=True)

        return candidates.head(limit).reset_index(drop=True)

    def get_trade_recommendations(
        self,
        season: int = 2024,
        week: Optional[int] = None,
        positions: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Returns buy-low and sell-high regression trade candidates."""
        self._ensure_engines([season])
        return {
            "buy_low": self._trade_analyzer.get_buy_low_candidates(season=season, week=week, positions=positions),
            "sell_high": self._trade_analyzer.get_sell_high_candidates(season=season, week=week, positions=positions),
        }

    def get_trending_movers(self, lookback_hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """Returns national trending adds and drops across fantasy football."""
        adds = self.sleeper.get_trending_adds(lookback_hours=lookback_hours, limit=10)
        drops = self.sleeper.get_trending_drops(lookback_hours=lookback_hours, limit=10)
        return {"trending_adds": adds, "trending_drops": drops}

    def get_injury_wire(
        self,
        limit: int = 12,
        positions: Optional[List[str]] = None,
        query: Optional[str] = None,
        use_ai: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes real-time injury reports from RSS feeds and Sleeper notes,
        evaluates severity with LLM/AI, and identifies promoted next-man-up beneficiaries.
        """
        # Fetch RSS articles
        rss_articles = self.news_collector.fetch_latest_news(limit_per_feed=10)
        # Fetch Sleeper beat notes
        sleeper_notes = self.news_collector.fetch_sleeper_injury_notes(positions=positions)

        combined_items = rss_articles + sleeper_notes

        if query:
            q = query.lower()
            combined_items = [
                item for item in combined_items
                if q in item.get("title", "").lower() or q in item.get("summary", "").lower()
            ]

        if use_ai:
            return self.news_parser.analyze_with_ai(combined_items, limit=limit)

        return [self.news_parser.analyze_article(a) for a in combined_items[:limit]]

    def get_sidelined_starters_report(self) -> List[Dict[str, Any]]:
        """
        Returns all starter-tier players (depth_order == 1 or top WR) currently sidelined,
        paired with their promoted backup beneficiary.
        """
        return self.depth_manager.get_all_sidelined_starters()

    def get_player_intel(self, player_name: str, season: int = 2024) -> Dict[str, Any]:
        """Deep dive dossier on a specific player: stats trajectory, depth chart, and relevant news."""
        self._ensure_engines([season])
        df = self._waiver_engine.df

        # Filter player stats
        matches = df[df["player_display_name"].str.contains(player_name, case=False, na=False)]
        news = self.news_collector.fetch_latest_news(limit_per_feed=25)
        player_news = self.news_parser.filter_by_player(news, player_name)

        # Depth chart & Next man up
        depth_info = self.depth_manager.find_player(player_name)
        beneficiary = self.depth_manager.get_next_man_up(player_name) if depth_info else None

        # AI-analyzed news
        ai_analyzed_news = self.news_parser.analyze_with_ai(player_news, limit=5) if player_news else []

        return {
            "player_name": player_name,
            "depth_chart": depth_info,
            "next_man_up": beneficiary,
            "weekly_stats": matches.sort_values(by="week", ascending=False).to_dict(orient="records"),
            "news_updates": player_news,
            "ai_analyzed_news": ai_analyzed_news,
        }

    def get_situational_matchup_advice(
        self,
        username_or_id: str,
        league_id: Optional[str] = None,
        week: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fetches the user's active matchup and delivers tailored tactical advice:
        - Injury alerts for active starters
        - Recommended bench/starter swaps
        - Opponent strengths/vulnerabilities
        """
        # Resolve username to user_id
        user_info = self.sleeper.get_user(username_or_id)
        user_id = user_info.get("user_id") if user_info else username_or_id

        # If league_id not provided, find user's leagues
        if not league_id:
            leagues = self.sleeper.get_user_leagues(user_id)
            if not leagues:
                return {"error": f"No active leagues found for user '{username_or_id}'."}
            league_id = leagues[0].get("league_id")

        overview = self.sleeper.get_user_matchup_overview(league_id=league_id, user_id=user_id, week=week)
        if "error" in overview:
            return overview

        eval_result = self.matchup_advisor.evaluate_lineup(overview)
        return {
            "matchup_overview": overview,
            "tactical_advice": eval_result,
        }
