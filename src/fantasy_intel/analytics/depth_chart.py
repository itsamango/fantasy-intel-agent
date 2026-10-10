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

    def find_player(self, player_identifier: str) -> Optional[Dict[str, Any]]:
        """Find player metadata by ID, full name, or normalized name."""
        self._ensure_depth_chart()
        norm_target = clean_name(player_identifier)
        for team_data in self._depth_by_team.values():
            for pos_players in team_data.values():
                for p in pos_players:
                    if p["player_id"] == str(player_identifier):
                        return p
                    if p["norm_name"] == norm_target or norm_target in p["norm_name"]:
                        return p
        return None

    def get_player_elevation_context(self, player_identifier: str) -> Dict[str, Any]:
        """
        Calculates the situational elevation of a player based on upstream teammate injuries.
        Identifies whether a player is an entrenched starter, an elevated starter (due to injuries),
        a potential surge beneficiary, or a standard backup.
        """
        self._ensure_depth_chart()
        player = self.find_player(player_identifier)
        if not player:
            return {
                "player_name": player_identifier,
                "tier": "UNKNOWN",
                "is_elevated": False,
                "ir_eligible": False,
                "summary": "Player not found on depth chart",
                "sidelined_ahead": [],
            }

        pos = player["position"]
        team = player["team"]
        depth_order = player["depth_order"]
        own_injury = player.get("injury_status")
        own_part = player.get("injury_body_part")
        ir_eligible = own_injury in ["Out", "IR", "PUP", "Sus"]

        pos_list = self._depth_by_team.get(team, {}).get(pos, [])
        ahead = [p for p in pos_list if p["depth_order"] < depth_order]
        out_ahead = [p for p in ahead if p.get("injury_status") in ["Out", "IR", "PUP", "Doubtful"]]
        q_ahead = [p for p in ahead if p.get("injury_status") == "Questionable"]

        # Also find any key position peers who are currently Out or on IR
        sidelined_peers = [
            p for p in pos_list
            if p["player_id"] != player["player_id"]
            and p.get("injury_status") in ["Out", "IR", "PUP", "Doubtful"]
        ]

        healthy_role = f"{team} {pos}{depth_order if depth_order <= 4 else 'Depth'}"
        is_temporary = False

        if depth_order == 1 or (pos == "WR" and depth_order <= 2):
            if sidelined_peers:
                tier = "ELEVATED_STARTER"
                is_elevated = True
                is_temporary = True
                peer_desc = ", ".join([f"{p['name']} ({p.get('injury_status')} - {p.get('injury_body_part') or 'Injured'})" for p in sidelined_peers])
                summary = f"ELEVATED STARTER: Starting {pos}1 for {team} while {peer_desc} are sidelined"
            else:
                tier = "ENTRENCHED_STARTER"
                is_elevated = False
                summary = f"Starting {pos}{depth_order} for {team}"
        elif ahead and len(out_ahead) == len(ahead):
            tier = "ELEVATED_STARTER"
            is_elevated = True
            is_temporary = True
            desc = ", ".join([f"{p['name']} ({p.get('injury_status')} - {p.get('injury_body_part') or 'Injured'})" for p in out_ahead])
            summary = f"ELEVATED STARTER: {pos}1 workload while {desc}"
        elif out_ahead or q_ahead:
            tier = "POTENTIAL_SURGE"
            is_elevated = True
            all_inj = out_ahead + q_ahead
            desc = ", ".join([f"{p['name']} ({p.get('injury_status')} - {p.get('injury_body_part') or 'Injured'})" for p in all_inj])
            summary = f"POTENTIAL SURGE: Target/touch boost if injured teammates sit ({desc})"
        else:
            tier = "BENCH_BACKUP"
            is_elevated = False
            starter_names = ", ".join([p["name"] for p in ahead])
            summary = f"Backup behind healthy starter(s): {starter_names}"

        return {
            "player_id": player["player_id"],
            "player_name": player["name"],
            "team": team,
            "position": pos,
            "depth_order": depth_order,
            "tier": tier,
            "is_elevated": is_elevated,
            "is_temporary_starter": is_temporary,
            "ir_eligible": ir_eligible,
            "own_injury": own_injury,
            "own_body_part": own_part,
            "healthy_role": healthy_role,
            "sidelined_ahead": out_ahead + q_ahead or sidelined_peers,
            "summary": summary,
        }


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
