"""Depth chart tracker and next-man-up beneficiary engine."""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from fantasy_intel.data.sleeper import SleeperClient
from fantasy_intel.data.loader import clean_name

logger = logging.getLogger(__name__)

OFFENSIVE_POSITIONS = {"QB", "RB", "WR", "TE"}


class DepthChartManager:
    """Tracks team depth charts and determines next-man-up beneficiaries when injuries occur."""

    def __init__(self, sleeper_client: Optional[SleeperClient] = None):
        self.sleeper = sleeper_client or SleeperClient()
        self._players: Dict[str, Dict[str, Any]] = {}
        self._depth_by_team: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    def _ensure_depth_chart(self):
        if self._players:
            return

        self._players = self.sleeper.get_all_players()
        self._depth_by_team = {}

        for pid, p in self._players.items():
            team = p.get("team")
            pos = p.get("position")
            order = p.get("depth_chart_order")

            if not team or pos not in OFFENSIVE_POSITIONS:
                continue

            # Normalized position
            team_dict = self._depth_by_team.setdefault(team, {})
            pos_list = team_dict.setdefault(pos, [])
            pos_list.append({
                "player_id": str(pid),
                "name": p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or f"Player {pid}",
                "norm_name": clean_name(p.get("full_name", "")),
                "team": team,
                "position": pos,
                "depth_order": int(order) if order is not None else 99,
                "depth_position": p.get("depth_chart_position") or pos,
                "injury_status": p.get("injury_status"),
                "injury_body_part": p.get("injury_body_part"),
                "injury_notes": p.get("injury_notes"),
                "status": p.get("status", "Active"),
            })

        # Sort each team/position list by depth order
        for team, positions in self._depth_by_team.items():
            for pos, players in positions.items():
                players.sort(key=lambda x: x["depth_order"])

    def get_team_depth_chart(self, team: str, position: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get ordered depth chart for a team, optionally filtered by position."""
        self._ensure_depth_chart()
        team_data = self._depth_by_team.get(team.upper(), {})
        if position:
            return team_data.get(position.upper(), [])

        combined = []
        for pos_players in team_data.values():
            combined.extend(pos_players)
        return sorted(combined, key=lambda x: (x["position"], x["depth_order"]))

    def find_player(self, player_name: str) -> Optional[Dict[str, Any]]:
        """Find player metadata by full name or normalized name."""
        self._ensure_depth_chart()
        norm_target = clean_name(player_name)
        for team_data in self._depth_by_team.values():
            for pos_players in team_data.values():
                for p in pos_players:
                    if p["norm_name"] == norm_target or norm_target in p["norm_name"]:
                        return p
        return None

    def get_next_man_up(self, player_name: str) -> Optional[Dict[str, Any]]:
        """
        Given a player (typically an injured starter), identify the direct
        beneficiary who moves up in the depth chart.
        """
        self._ensure_depth_chart()
        player = self.find_player(player_name)
        if not player:
            return None

        team = player["team"]
        pos = player["position"]
        depth_order = player["depth_order"]

        pos_list = self._depth_by_team.get(team, {}).get(pos, [])
        # Find healthy players ranked below this player
        for candidate in pos_list:
            if candidate["depth_order"] > depth_order:
                # Exclude players also ruled out or on IR
                if candidate.get("injury_status") not in ["Out", "IR", "PUP", "Doubtful"]:
                    return candidate

        return None

    def get_all_sidelined_starters(self) -> List[Dict[str, Any]]:
        """
        Scan all teams for starters (depth_order == 1 or 2 in WR) sidelined with injury,
        and pair them with their promoted beneficiaries.
        """
        self._ensure_depth_chart()
        sidelined_reports = []

        for team, pos_dict in self._depth_by_team.items():
            for pos, plist in pos_dict.items():
                for p in plist:
                    is_starter_tier = (p["depth_order"] == 1) or (pos == "WR" and p["depth_order"] <= 2)
                    is_injured = p.get("injury_status") in ["Out", "IR", "PUP", "Doubtful", "Questionable"]

                    if is_starter_tier and is_injured:
                        beneficiary = self.get_next_man_up(p["name"])
                        sidelined_reports.append({
                            "injured_starter": p,
                            "promoted_beneficiary": beneficiary,
                            "team": team,
                            "position": pos,
                            "severity": "CRITICAL" if p.get("injury_status") in ["Out", "IR", "PUP"] else "MONITOR",
                        })

        return sorted(
            sidelined_reports,
            key=lambda x: (0 if x["severity"] == "CRITICAL" else 1, x["position"]),
        )
