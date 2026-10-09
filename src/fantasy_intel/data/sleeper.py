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

    def get_nfl_state(self) -> Dict[str, Any]:
        """Fetch the current NFL week and season state from Sleeper."""
        url = f"{SLEEPER_BASE_URL}/state/nfl"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error fetching NFL state: {e}")
            return {"season": "2024", "week": 1}

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Fetch user profile by Sleeper username or user_id."""
        url = f"{SLEEPER_BASE_URL}/user/{username}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.error(f"Error fetching user {username}: {e}")
            return None

    def get_user_leagues(self, user_id: str, season: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch all leagues for a user in a given season."""
        if season is None:
            state = self.get_nfl_state()
            season = int(state.get("season", 2024))
        url = f"{SLEEPER_BASE_URL}/user/{user_id}/leagues/nfl/{season}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error fetching leagues for user {user_id}: {e}")
            return []

    def get_league_users(self, league_id: str) -> List[Dict[str, Any]]:
        """Fetch all member users/teams in a league."""
        url = f"{SLEEPER_BASE_URL}/league/{league_id}/users"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error fetching users for league {league_id}: {e}")
            return []

    def get_matchups(self, league_id: str, week: int) -> List[Dict[str, Any]]:
        """Fetch all matchups for a specific week."""
        url = f"{SLEEPER_BASE_URL}/league/{league_id}/matchups/{week}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error fetching matchups for league {league_id} week {week}: {e}")
            return []

    def get_user_matchup_overview(
        self, league_id: str, user_id: str, week: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Gathers full context for the user's current matchup:
        - User's team roster and starters
        - Opponent's team roster and starters
        - Players metadata (names, positions, injury status)
        """
        if week is None:
            state = self.get_nfl_state()
            week = state.get("display_week") or state.get("week", 1)

        rosters = self.get_league_rosters(league_id)
        users = self.get_league_users(league_id)
        user_map = {u.get("user_id"): u.get("display_name") for u in users}

        # Identify user's roster
        my_roster = next((r for r in rosters if str(r.get("owner_id")) == str(user_id)), None)
        if not my_roster:
            return {"error": f"User {user_id} not found in league {league_id} rosters."}

        my_roster_id = my_roster.get("roster_id")

        matchups = self.get_matchups(league_id, week)
        my_matchup = next((m for m in matchups if m.get("roster_id") == my_roster_id), None)
        if not my_matchup:
            return {"error": f"No matchup found for roster {my_roster_id} in week {week}."}

        matchup_id = my_matchup.get("matchup_id")
        # Find opponent matchup with same matchup_id
        opp_matchup = next(
            (m for m in matchups if m.get("matchup_id") == matchup_id and m.get("roster_id") != my_roster_id),
            None,
        )

        opp_roster = None
        opp_user_name = "Opponent"
        if opp_matchup:
            opp_roster_id = opp_matchup.get("roster_id")
            opp_roster = next((r for r in rosters if r.get("roster_id") == opp_roster_id), None)
            if opp_roster:
                opp_owner_id = opp_roster.get("owner_id")
                opp_user_name = user_map.get(opp_owner_id, f"Team {opp_roster_id}")

        all_players = self.get_all_players()

        def enrich_player(pid: str) -> Dict[str, Any]:
            p = all_players.get(str(pid), {})
            return {
                "player_id": str(pid),
                "name": p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or f"Player {pid}",
                "position": p.get("position", "N/A"),
                "team": p.get("team", "FA"),
                "injury_status": p.get("injury_status"),
            }

        # User's players
        my_starter_ids = [str(p) for p in my_matchup.get("starters", []) if p]
        my_all_ids = [str(p) for p in my_roster.get("players", []) if p]
        my_bench_ids = [pid for pid in my_all_ids if pid not in my_starter_ids]

        # Opponent's players
        opp_starter_ids = [str(p) for p in opp_matchup.get("starters", []) if p] if opp_matchup else []
        opp_all_ids = [str(p) for p in opp_roster.get("players", []) if p] if opp_roster else []
        opp_bench_ids = [pid for pid in opp_all_ids if pid not in opp_starter_ids]

        return {
            "week": week,
            "league_id": league_id,
            "my_team": {
                "user_name": user_map.get(str(user_id), "My Team"),
                "roster_id": my_roster_id,
                "points": my_matchup.get("points", 0.0),
                "starters": [enrich_player(pid) for pid in my_starter_ids],
                "bench": [enrich_player(pid) for pid in my_bench_ids],
            },
            "opponent_team": {
                "user_name": opp_user_name,
                "roster_id": opp_roster.get("roster_id") if opp_roster else None,
                "points": opp_matchup.get("points", 0.0) if opp_matchup else 0.0,
                "starters": [enrich_player(pid) for pid in opp_starter_ids],
                "bench": [enrich_player(pid) for pid in opp_bench_ids],
            } if opp_matchup else None,
        }

