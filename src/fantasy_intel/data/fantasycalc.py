"""FantasyCalc client: crowdsourced trade-market player values (keyed by Sleeper id).

Only generic league settings (QB count, team count, PPR) are sent; no user or
league identifiers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import requests

from fantasy_intel.config import (
    CACHE_DIR,
    FANTASYCALC_TTL_HOURS,
    SCORING_HALF_PPR,
    SCORING_STANDARD,
)

logger = logging.getLogger(__name__)

FANTASYCALC_URL = "https://api.fantasycalc.com/values/current"


def market_params_from_league(league_context: Mapping[str, Any]) -> Dict[str, Any]:
    """FantasyCalc query params (numQbs / numTeams / ppr) from a Sleeper league context."""
    slots = {str(k).upper(): int(v) for k, v in (league_context.get("roster_position_counts") or {}).items()}
    num_qbs = 2 if slots.get("SUPER_FLEX", 0) > 0 or slots.get("QB", 1) >= 2 else 1
    scoring = league_context.get("scoring_type")
    ppr = 0.5 if scoring == SCORING_HALF_PPR else 0.0 if scoring == SCORING_STANDARD else 1.0
    return {"numQbs": num_qbs, "numTeams": int(league_context.get("total_rosters") or 12), "ppr": ppr}


class FantasyCalcClient:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir

    def get_values(
        self,
        num_qbs: int = 1,
        num_teams: int = 12,
        ppr: float = 1.0,
        is_dynasty: bool = False,
        force_refresh: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """{sleeper_id: {"value", "overall_rank", "position_rank", "trend_30d", "tier",
        "trade_frequency", "roster_pct", "name", "position", "team"}}; {} on failure."""
        tag = f"{'dyn' if is_dynasty else 'redraft'}_{num_qbs}qb_{num_teams}tm_{ppr:g}ppr"
        cache_path = self.cache_dir / f"fantasycalc_{tag}.json"
        raw = None
        if not force_refresh and cache_path.exists():
            age_h = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_h < FANTASYCALC_TTL_HOURS:
                try:
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"Failed to read FantasyCalc cache: {e}")
        if raw is None:
            params = {
                "isDynasty": str(bool(is_dynasty)).lower(),
                "numQbs": num_qbs,
                "numTeams": num_teams,
                "ppr": ppr,
            }
            try:
                resp = requests.get(FANTASYCALC_URL, params=params, timeout=20)
                resp.raise_for_status()
                raw = resp.json()
                cache_path.write_text(json.dumps(raw), encoding="utf-8")
            except Exception as e:
                logger.warning(f"FantasyCalc values unavailable ({e}).")
                if cache_path.exists():  # stale beats nothing
                    try:
                        raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    except Exception:
                        raw = None
        return self._parse(raw or [])

    @staticmethod
    def _parse(raw: Any) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for row in raw if isinstance(raw, list) else []:
            player = row.get("player") or {}
            sid = player.get("sleeperId")
            if not sid or row.get("value") is None:
                continue
            out[str(sid)] = {
                "value": float(row["value"]),
                "overall_rank": row.get("overallRank"),
                "position_rank": row.get("positionRank"),
                "trend_30d": row.get("trend30Day"),
                "tier": row.get("maybeTier"),
                "trade_frequency": row.get("maybeTradeFrequency"),
                "roster_pct": row.get("maybeRosterPercent"),
                "name": player.get("name"),
                "position": player.get("position"),
                "team": player.get("maybeTeam"),
            }
        return out
