"""Sleeper API client for live league data, trending adds/drops, and rosters."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import requests

from fantasy_intel.config import CACHE_DIR

logger = logging.getLogger(__name__)

SLEEPER_BASE_URL = "https://api.sleeper.app/v1"


class SleeperClient:
    """Client for fetching fantasy data from Sleeper."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir

    def get_trending_adds(self, lookback_hours: int = 24, limit: int = 25) -> List[Dict[str, Any]]:
        """Fetch players trending up across all Sleeper leagues."""
        url = f"{SLEEPER_BASE_URL}/players/nfl/trending/add"
        params = {"lookback_hours": lookback_hours, "limit": limit}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            trending = resp.json()
            # Enrich with player names if metadata is cached
            players = self.get_all_players()
            for item in trending:
                pid = str(item.get("player_id"))
                if pid in players:
                    p = players[pid]
                    item["full_name"] = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
                    item["position"] = p.get("position")
                    item["team"] = p.get("team")
                    item["injury_status"] = p.get("injury_status")
            return trending
        except Exception as e:
            logger.error(f"Error fetching trending adds from Sleeper: {e}")
            return []

    def get_trending_drops(self, lookback_hours: int = 24, limit: int = 25) -> List[Dict[str, Any]]:
        """Fetch players trending down (drops) across all Sleeper leagues."""
        url = f"{SLEEPER_BASE_URL}/players/nfl/trending/drop"
        params = {"lookback_hours": lookback_hours, "limit": limit}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            trending = resp.json()
            players = self.get_all_players()
            for item in trending:
                pid = str(item.get("player_id"))
                if pid in players:
                    p = players[pid]
                    item["full_name"] = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
                    item["position"] = p.get("position")
                    item["team"] = p.get("team")
            return trending
        except Exception as e:
            logger.error(f"Error fetching trending drops from Sleeper: {e}")
            return []

    def get_all_players(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Fetches and caches the master dictionary of NFL players from Sleeper."""
        cache_path = self.cache_dir / "sleeper_players.json"
        if cache_path.exists() and not force_refresh:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cached sleeper players: {e}")

        url = f"{SLEEPER_BASE_URL}/players/nfl"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return data
        except Exception as e:
            logger.error(f"Failed to download sleeper players: {e}")
            return {}

    def get_league_rosters(self, league_id: str) -> List[Dict[str, Any]]:
        """Fetch all rosters for a specific league."""
        url = f"{SLEEPER_BASE_URL}/league/{league_id}/rosters"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error fetching rosters for league {league_id}: {e}")
            return []

    def get_rostered_player_ids(self, league_id: str) -> Set[str]:
        """Returns set of all player_ids currently owned on a roster in the league."""
        rosters = self.get_league_rosters(league_id)
        owned: Set[str] = set()
        for r in rosters:
            players = r.get("players") or []
            owned.update(str(p) for p in players)
        return owned

    def get_league_info(self, league_id: str) -> Dict[str, Any]:
        """Fetch league settings, name, and total teams."""
        url = f"{SLEEPER_BASE_URL}/league/{league_id}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error fetching league details {league_id}: {e}")
            return {}
