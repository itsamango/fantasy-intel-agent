"""Fantasy Intelligence Advisor orchestrating data, news, and league APIs."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd

from fantasy_intel.data.loader import DataLoader
from fantasy_intel.data.sleeper import SleeperClient
from fantasy_intel.analytics.waiver_wire import WaiverWireEngine
from fantasy_intel.analytics.trades import TradeAnalyzer
from fantasy_intel.news.scraper import NewsCollector
from fantasy_intel.news.parser import NewsContextParser


class FantasyAdvisor:
    """The central agent brain for fantasy recommendations."""

    def __init__(self, data_loader: Optional[DataLoader] = None, sleeper_client: Optional[SleeperClient] = None):
        self.loader = data_loader or DataLoader()
        self.sleeper = sleeper_client or SleeperClient()
        self.news_collector = NewsCollector()
        self.news_parser = NewsContextParser()

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

    def get_player_intel(self, player_name: str, season: int = 2024) -> Dict[str, Any]:
        """Deep dive dossier on a specific player: stats trajectory + relevant news."""
        self._ensure_engines([season])
        df = self._waiver_engine.df

        # Filter player stats
        matches = df[df["player_display_name"].str.contains(player_name, case=False, na=False)]
        news = self.news_collector.fetch_latest_news(limit_per_feed=25)
        player_news = self.news_parser.filter_by_player(news, player_name)

        return {
            "player_name": player_name,
            "weekly_stats": matches.sort_values(by="week", ascending=False).to_dict(orient="records"),
            "news_updates": player_news,
        }
