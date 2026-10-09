"""Situational matchup analysis: Start/Sit optimization, injury warnings, and opponent tactics."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd


class MatchupAdvisor:
    """Analyzes user's roster vs their weekly opponent and generates strategic advice."""

    def __init__(self, weekly_stats_df: Optional[pd.DataFrame] = None):
        self.stats = weekly_stats_df

    def evaluate_lineup(self, matchup_overview: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produces detailed situational advice for the current week:
        - Injury alerts in starting lineup
        - Start/Sit swap opportunities between bench and starters
        - Positional strengths/weaknesses vs opponent
        """
        my_team = matchup_overview.get("my_team", {})
        opp_team = matchup_overview.get("opponent_team")

        starters = my_team.get("starters", [])
        bench = my_team.get("bench", [])

        # 1. Injury Alerts for Active Starters
        injury_alerts = []
        for p in starters:
            status = p.get("injury_status")
            if status in ["Out", "IR", "Doubtful"]:
                injury_alerts.append({
                    "player": p["name"],
                    "position": p["position"],
                    "status": status,
                    "severity": "CRITICAL",
                    "action": f"Bench {p['name']} immediately ({status}).",
                })
            elif status == "Questionable":
                injury_alerts.append({
                    "player": p["name"],
                    "position": p["position"],
                    "status": status,
                    "severity": "WARNING",
                    "action": f"Monitor {p['name']} game-time status; prepare bench replacement.",
                })

        # 2. Start / Sit Recommendations
        start_sit_suggestions = []
        # Group bench by position
        bench_by_pos: Dict[str, List[Dict[str, Any]]] = {}
        for b in bench:
            pos = b.get("position", "")
            bench_by_pos.setdefault(pos, []).append(b)

        # Flag healthy bench players when a starter is injured
        for alert in injury_alerts:
            pos = alert["position"]
            replacements = bench_by_pos.get(pos, [])
            healthy_replacements = [r for r in replacements if not r.get("injury_status") or r.get("injury_status") == "Questionable"]
            if healthy_replacements:
                best_sub = healthy_replacements[0]
                start_sit_suggestions.append({
                    "sit": alert["player"],
                    "start": best_sub["name"],
                    "position": pos,
                    "reason": f"Starter is {alert['status']}; sub in {best_sub['name']} ({best_sub['team']}).",
                })

        # 3. Positional Head-to-Head Comparison
        opp_starters = opp_team.get("starters", []) if opp_team else []
        opp_by_pos: Dict[str, int] = {}
        for op in opp_starters:
            pos = op.get("position", "OTHER")
            opp_by_pos[pos] = opp_by_pos.get(pos, 0) + 1

        my_by_pos: Dict[str, int] = {}
        for sp in starters:
            pos = sp.get("position", "OTHER")
            my_by_pos[pos] = my_by_pos.get(pos, 0) + 1

        strategy_notes = []
        if opp_team:
            opp_injuries = [op["name"] for op in opp_starters if op.get("injury_status") in ["Out", "IR", "Doubtful"]]
            if opp_injuries:
                strategy_notes.append(f"Opponent has injured players in active lineup: {', '.join(opp_injuries)}. Capitalize by playing high-floor starters.")
            else:
                strategy_notes.append("Opponent has a healthy starting lineup; prioritize high-ceiling players in FLEX spots.")

        return {
            "week": matchup_overview.get("week"),
            "my_team_name": my_team.get("user_name"),
            "opponent_name": opp_team.get("user_name") if opp_team else "None",
            "injury_alerts": injury_alerts,
            "start_sit_suggestions": start_sit_suggestions,
            "strategy_notes": strategy_notes,
            "lineup_summary": {
                "starter_count": len(starters),
                "bench_count": len(bench),
            },
        }
