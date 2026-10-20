"""Sleeper API client for live league data, trending adds/drops, and rosters."""

from __future__ import annotations
import json
import logging
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import requests

from fantasy_intel.config import CACHE_DIR, SCORING_HALF_PPR, SCORING_PPR, SCORING_STANDARD, SCORING_TEP

logger = logging.getLogger(__name__)

SLEEPER_BASE_URL = "https://api.sleeper.app/v1"


def _percentile(sorted_vals: List[float], p: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo]) * (hi - k) + float(sorted_vals[hi]) * (k - lo)


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

    def get_season_stats(self, season: int = 2026, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Fetches and caches cumulative in-season player stats from Sleeper."""
        cache_path = self.cache_dir / f"sleeper_stats_{season}.json"
        if cache_path.exists() and not force_refresh:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cached sleeper stats: {e}")

        url = f"{SLEEPER_BASE_URL}/stats/nfl/regular/{season}"
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return data
        except Exception as e:
            logger.error(f"Failed to download sleeper season stats for {season}: {e}")
            return {}

    def get_week_projections(
        self, season: int = 2026, week: int = 1, force_refresh: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Fetches and caches weekly player projections from Sleeper."""
        cache_path = self.cache_dir / f"sleeper_projections_{season}_{week}.json"
        if cache_path.exists() and not force_refresh:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cached projections for {season} W{week}: {e}")

        url = f"{SLEEPER_BASE_URL}/projections/nfl/regular/{season}/{week}"
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch sleeper projections for {season} W{week}: {e}")
            return {}

    def get_bye_teams(self, season: int = 2026, week: int = 1) -> Set[str]:
        """Identifies NFL teams on bye for the given week based on zero player projections."""
        projections = self.get_week_projections(season=season, week=week)
        players = self.get_all_players()

        if not projections or not players:
            return set()

        team_projected_pts: Dict[str, float] = {}
        for pid, proj in projections.items():
            if not proj or not isinstance(proj, dict):
                continue
            pinfo = players.get(str(pid), {})
            team = pinfo.get("team")
            if not team:
                continue
            pts = float(proj.get("pts_ppr", 0.0) or 0.0)
            team_projected_pts[team] = team_projected_pts.get(team, 0.0) + pts

        # Teams with 0 projected fantasy points are on bye
        bye_teams = {t for t, pts in team_projected_pts.items() if pts == 0.0}
        return bye_teams



    def get_all_players(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Fetches and caches the master dictionary of NFL players from Sleeper (24h TTL)."""
        cache_path = self.cache_dir / "sleeper_players.json"
        if cache_path.exists() and not force_refresh:
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours < 24:
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

    def get_league_transactions(self, league_id: str, week: int) -> List[Dict[str, Any]]:
        """Fetch transactions for a league in a specific week."""
        url = f"{SLEEPER_BASE_URL}/league/{league_id}/transactions/{week}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error fetching transactions for league {league_id} week {week}: {e}")
            return []

    def get_league_faab_history(self, league_id: str, weeks_back: int = 8) -> Dict[str, Any]:
        """Summarizes completed FAAB waiver bids across recent transaction weeks.

        Returns the league's empirical bid distribution: median, p80, count, and
        the raw sorted winning bids (settings.waiver_bid of type=="waiver" and
        status=="complete" transactions).
        """
        state = self.get_nfl_state()
        current_week = int(state.get("week") or state.get("display_week") or 1)

        weeks = list(range(max(1, current_week - weeks_back + 1), current_week + 1))
        bids: List[int] = []
        for wk in weeks:
            transactions = self.get_league_transactions(league_id, wk)
            for tx in transactions or []:
                if tx.get("type") != "waiver" or tx.get("status") != "complete":
                    continue
                bid = (tx.get("settings") or {}).get("waiver_bid")
                if isinstance(bid, (int, float)) and not isinstance(bid, bool):
                    bids.append(int(bid))

        if not bids:
            return {"league_id": league_id, "median": None, "p80": None, "count": 0, "bids": []}

        sorted_bids = sorted(bids)
        return {
            "league_id": league_id,
            "median": _percentile(sorted_bids, 0.5),
            "p80": _percentile(sorted_bids, 0.8),
            "count": len(sorted_bids),
            "bids": sorted_bids,
        }

    def get_league_context(self, league_id: str) -> Dict[str, Any]:
        """Extracts league settings context for VORP/FAAB/lineup optimization.

        Returns total_rosters, roster position slot counts (QB/RB/WR/TE/FLEX/
        SUPER_FLEX/etc. from league roster_positions), scoring settings (rec,
        bonus_rec_te, pass_td) with a detected scoring_type, and per-roster
        records with remaining FAAB (league waiver budget minus used).
        """
        league = self.get_league_info(league_id)
        rosters = self.get_league_rosters(league_id)
        if not league:
            return {}

        settings = league.get("settings") or {}
        scoring = league.get("scoring_settings") or {}
        roster_positions = league.get("roster_positions") or []

        position_counts = Counter(
            slot for slot in roster_positions if isinstance(slot, str)
        )

        scoring_settings = {
            key: scoring.get(key)
            for key in ("rec", "bonus_rec_te", "pass_td")
            if key in scoring
        }

        rec_weight = scoring.get("rec", 0.0) or 0.0
        te_bonus = scoring.get("bonus_rec_te", 0.0) or 0.0
        if rec_weight == 1.0 and te_bonus > 0:
            scoring_type = SCORING_TEP
        elif rec_weight == 1.0:
            scoring_type = SCORING_PPR
        elif rec_weight > 0:
            scoring_type = SCORING_HALF_PPR
        else:
            scoring_type = SCORING_STANDARD

        faab_budget = None
        for key in ("waiver_budget", "budget_blind_bid"):
            if isinstance(settings.get(key), (int, float)) and not isinstance(settings[key], bool):
                faab_budget = int(settings[key])
                break

        roster_records = []
        for roster in rosters or []:
            roster_settings = roster.get("settings") or {}
            faab_used = None
            for key in ("waiver_budget_used", "budget_blind_bid_used"):
                if isinstance(roster_settings.get(key), (int, float)) and not isinstance(
                    roster_settings[key], bool
                ):
                    faab_used = int(roster_settings[key])
                    break
            faab_remaining = (
                max(0, faab_budget - faab_used)
                if faab_budget is not None and faab_used is not None
                else None
            )
            roster_records.append(
                {
                    "roster_id": roster.get("roster_id"),
                    "owner_id": roster.get("owner_id"),
                    "wins": int(roster_settings.get("wins", 0) or 0),
                    "losses": int(roster_settings.get("losses", 0) or 0),
                    "ties": int(roster_settings.get("ties", 0) or 0),
                    "faab_remaining": faab_remaining,
                }
            )

        return {
            "league_id": league_id,
            "name": league.get("name"),
            "season": league.get("season"),
            "total_rosters": league.get("total_rosters") or len(rosters or []),
            "roster_position_counts": dict(position_counts),
            "scoring_settings": scoring_settings,
            "scoring_type": scoring_type,
            "faab_budget": faab_budget,
            "waiver_type": settings.get("waiver_type"),
            "rosters": roster_records,
        }

    def get_recent_dropped_player_ids(self, league_id: str, week: int, hours: int = 48) -> Set[str]:
        """Identifies player IDs recently dropped within the waiver hold window."""
        tx = self.get_league_transactions(league_id, week)
        now_ms = time.time() * 1000
        cutoff_ms = now_ms - (hours * 3600 * 1000)
        dropped_ids = set()
        for t in tx:
            drops = t.get("drops") or {}
            created = t.get("created", 0)
            if drops and created >= cutoff_ms:
                dropped_ids.update(str(pid) for pid in drops.keys())
        return dropped_ids

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

    def get_user_leagues_summary(self, username_or_id: str, season: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch summary of all leagues the user participates in with records and points."""
        user_info = self.get_user(username_or_id)
        user_id = user_info.get("user_id") if user_info else username_or_id

        leagues = self.get_user_leagues(user_id, season=season)
        summaries = []

        for lg in leagues:
            lid = lg.get("league_id")
            rosters = self.get_league_rosters(lid)
            my_roster = next((r for r in rosters if str(r.get("owner_id")) == str(user_id)), None)

            record_str = "N/A"
            points_for = 0.0
            roster_id = None

            if my_roster:
                roster_id = my_roster.get("roster_id")
                settings = my_roster.get("settings", {})
                wins = settings.get("wins", 0)
                losses = settings.get("losses", 0)
                ties = settings.get("ties", 0)
                record_str = f"{wins}-{losses}" + (f"-{ties}" if ties else "")
                fpts = settings.get("fpts", 0)
                fpts_dec = settings.get("fpts_decimal", 0)
                points_for = round(fpts + (fpts_dec / 100.0), 1)

            summaries.append({
                "league_id": lid,
                "name": lg.get("name", f"League {lid}"),
                "total_teams": lg.get("total_rosters", 12),
                "status": lg.get("status", "in_season"),
                "season": lg.get("season"),
                "record": record_str,
                "points_for": points_for,
                "roster_id": roster_id,
            })

        return summaries

    def get_user_roster_details(self, league_id: str, username_or_id: str) -> Dict[str, Any]:
        """Fetch detailed breakdown of user's team, starters, bench, and player stats."""
        user_info = self.get_user(username_or_id)
        user_id = user_info.get("user_id") if user_info else username_or_id

        rosters = self.get_league_rosters(league_id)
        my_roster = next((r for r in rosters if str(r.get("owner_id")) == str(user_id)), None)
        if not my_roster:
            return {"error": f"User {username_or_id} not found in league {league_id}."}

        league_info = self.get_league_info(league_id)
        all_players = self.get_all_players()

        def enrich(pid: str) -> Dict[str, Any]:
            p = all_players.get(str(pid), {})
            return {
                "player_id": str(pid),
                "name": p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or f"Player {pid}",
                "position": p.get("position", "N/A"),
                "team": p.get("team", "FA"),
                "depth_order": p.get("depth_chart_order"),
                "injury_status": p.get("injury_status"),
                "injury_body_part": p.get("injury_body_part"),
                "injury_notes": p.get("injury_notes"),
            }

        starters = [enrich(pid) for pid in (my_roster.get("starters") or []) if pid]
        starter_ids = {p["player_id"] for p in starters}
        all_p = [enrich(pid) for pid in (my_roster.get("players") or []) if pid]
        bench = [p for p in all_p if p["player_id"] not in starter_ids]
        reserve = [enrich(pid) for pid in (my_roster.get("reserve") or []) if pid]
        taxi = [enrich(pid) for pid in (my_roster.get("taxi") or []) if pid]

        settings = my_roster.get("settings", {})
        wins = settings.get("wins", 0)
        losses = settings.get("losses", 0)
        ties = settings.get("ties", 0)

        return {
            "league_id": league_id,
            "league_name": league_info.get("name", f"League {league_id}"),
            "season": league_info.get("season"),
            "record": f"{wins}-{losses}" + (f"-{ties}" if ties else ""),
            "points_for": round(settings.get("fpts", 0) + (settings.get("fpts_decimal", 0) / 100.0), 1),
            "points_against": round(settings.get("fpts_against", 0) + (settings.get("fpts_against_decimal", 0) / 100.0), 1),
            "waiver_position": settings.get("waiver_position", "N/A"),
            "starters": starters,
            "bench": bench,
            "reserve": reserve,
            "taxi": taxi,
        }


