"""Trade evaluator: buy-low/sell-high regression candidates and bilateral league trade proposals."""

from __future__ import annotations
import logging
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd

from fantasy_intel.analytics.metrics import (
    _normalize_ids,
    compute_player_trajectories,
    estimate_expected_fantasy_points,
)
from fantasy_intel.analytics.matchups import OUT_STATUSES, MatchupAdvisor
from fantasy_intel.config import PACKAGE_CONSOLIDATION_ALPHA, SCORING_PPR

logger = logging.getLogger(__name__)

TRADEABLE_POSITIONS = ("QB", "RB", "WR", "TE")

# Heuristic per-team flex starter distribution used for K_pos. FLEX (RB/WR/TE
# eligible) slots are split 45% RB / 45% WR / 10% TE, matching typical league
# usage; WRRB_FLEX / REC_FLEX split evenly across their two eligible positions.
# Each SUPER_FLEX slot contributes 60% toward QB (superflex start rates for QB
# far exceed RB/WR/TE rates) with the remainder split evenly between the two
# most common superflex filler positions, RB and WR.
FLEX_STARTER_DISTRIBUTION: Dict[str, Dict[str, float]] = {
    "FLEX": {"RB": 0.45, "WR": 0.45, "TE": 0.10},
    "WRRB_FLEX": {"RB": 0.50, "WR": 0.50},
    "REC_FLEX": {"WR": 0.50, "TE": 0.50},
}
SUPERFLEX_QB_SHARE = 0.60
SUPERFLEX_REMAINDER_SPLIT = {"RB": 0.20, "WR": 0.20}

# Search caps: only players with positive model or market VORP are trade
# candidates, each side is capped to its top assets by VORP, and swaps whose
# strongest and weakest single-asset market VORPs differ by more than
# MAX_ASSET_VALUE_GAP are skipped as obviously lopsided (pre-decay).
MAX_TRADEABLE_ASSETS_PER_SIDE = 12
MAX_ASSET_VALUE_GAP = 25.0
MAX_SWAP_EVALUATIONS = 20000

# Pareto thresholds
MIN_USER_LINEUP_GAIN = 1.2
MIN_MARKET_DELTA = -1.5

# Acceptance heuristic. No training data for league-mate acceptance exists in
# this repo, so the logistic weights are documented judgment calls:
# P(accept) = logistic(-1.0 + 0.15*dV_market_opp + 0.5*dU_opp + 0.5*need_filled)
ACCEPTANCE_BIAS = -1.0
ACCEPTANCE_MARKET_WEIGHT = 0.15
ACCEPTANCE_LINEUP_WEIGHT = 0.5
ACCEPTANCE_NEED_WEIGHT = 0.5
ACCEPTANCE_CLIP = 15.0

# Rolling buy-low / sell-high signal thresholds
BUY_LOW_MIN_ROLLING_XFP = 12.0
BUY_LOW_MAX_ROLLING_FPOE = -3.0
SELL_HIGH_MIN_RECENT_PPG = 15.0
SELL_HIGH_MIN_ROLLING_FPOE = 4.0

OUTPUT_COLUMNS = [
    "player_display_name",
    "position",
    "recent_team",
    "offense_pct",
    "touches",
    "targets",
    "fantasy_points_ppr",
    "expected_fp_ppr",
    "fp_diff",
    "rationale",
]


def _norm_name(name: Any) -> Optional[str]:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    text = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return text or None


def _has_rolling_signals(df: pd.DataFrame) -> bool:
    return "rolling_xfp_3w" in df.columns and "rolling_fpoe_3w" in df.columns


class TradeAnalyzer:
    """Identifies players whose fantasy output deviates sharply from true opportunity."""

    def __init__(self, weekly_df: pd.DataFrame):
        traj = compute_player_trajectories(weekly_df)
        self.df = estimate_expected_fantasy_points(traj)

    def _latest_frame(
        self,
        week: Optional[int],
        season: Optional[int],
        positions: Optional[List[str]],
        default_positions: List[str],
    ) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame()

        filtered = self.df.copy()
        if season is not None:
            filtered = filtered[filtered["season"] == season]
        if week is not None:
            filtered = filtered[filtered["week"] == week]
        else:
            reg = filtered[filtered["season_type"] == "REG"] if "season_type" in filtered.columns else filtered[filtered["week"] <= 18]
            latest_week = reg["week"].max() if not reg.empty else filtered["week"].max()
            filtered = filtered[filtered["week"] == latest_week]

        if positions:
            filtered = filtered[filtered["position"].isin(positions)]
        else:
            filtered = filtered[filtered["position"].isin(default_positions)]
        return filtered

    def get_buy_low_candidates(
        self,
        week: Optional[int] = None,
        season: Optional[int] = None,
        positions: Optional[List[str]] = None,
        min_snaps: float = 0.50,
        limit: int = 10,
    ) -> pd.DataFrame:
        """
        Players with strong rolling xFP whose recent actual output was heavily
        suppressed by negative variance (rolling FPOE): buy-low trade targets.
        """
        filtered = self._latest_frame(week, season, positions, ["RB", "WR", "TE"])
        if filtered.empty:
            return pd.DataFrame()

        if _has_rolling_signals(filtered):
            buy_low = filtered[
                (filtered["offense_pct"] >= min_snaps)
                & (filtered["rolling_xfp_3w"] >= BUY_LOW_MIN_ROLLING_XFP)
                & (filtered["rolling_fpoe_3w"] <= BUY_LOW_MAX_ROLLING_FPOE)
            ].copy()
            buy_low["rationale"] = buy_low.apply(
                lambda r: (
                    f"Rolling 3-week xFP {r['rolling_xfp_3w']:.1f} masked by "
                    f"{r['rolling_fpoe_3w']:+.1f} FPOE ({int(r['offense_pct']*100)}% snaps, "
                    f"{int(r['touches'])} touches/targets)"
                ),
                axis=1,
            )
            sort_col = "rolling_fpoe_3w"
        else:
            buy_low = filtered[
                (filtered["offense_pct"] >= min_snaps)
                & (filtered["expected_fp_ppr"] >= 10.0)
                & (filtered["fp_diff"] <= -3.0)
            ].copy()
            buy_low["rationale"] = buy_low.apply(
                lambda r: f"Underperformed xFP by {abs(r['fp_diff']):.1f} pts despite {int(r['offense_pct']*100)}% snap share and {int(r['touches'])} touches/targets",
                axis=1,
            )
            sort_col = "fp_diff"

        buy_low = buy_low.sort_values(by=sort_col).head(limit)
        return buy_low[[c for c in OUTPUT_COLUMNS if c in buy_low.columns]].reset_index(drop=True)

    def get_sell_high_candidates(
        self,
        week: Optional[int] = None,
        season: Optional[int] = None,
        positions: Optional[List[str]] = None,
        limit: int = 10,
    ) -> pd.DataFrame:
        """
        Players with bloated recent scoring driven by unsustainable efficiency
        (sustained positive rolling FPOE on modest xFP): sell-high trade targets.
        """
        filtered = self._latest_frame(week, season, positions, ["RB", "WR", "TE"])
        if filtered.empty:
            return pd.DataFrame()

        if "recent_ppr" in filtered.columns:
            filtered["recent_ppg"] = pd.to_numeric(filtered["recent_ppr"], errors="coerce").fillna(0.0)
        else:
            filtered["recent_ppg"] = pd.to_numeric(filtered["fantasy_points_ppr"], errors="coerce").fillna(0.0)

        if _has_rolling_signals(filtered):
            sell_high = filtered[
                (filtered["recent_ppg"] >= SELL_HIGH_MIN_RECENT_PPG)
                & (filtered["rolling_fpoe_3w"] >= SELL_HIGH_MIN_ROLLING_FPOE)
            ].copy()
            sell_high["rationale"] = sell_high.apply(
                lambda r: (
                    f"Recent {r['recent_ppg']:.1f} PPG vs rolling xFP {r['rolling_xfp_3w']:.1f} "
                    f"({r['rolling_fpoe_3w']:+.1f} FPOE) — unsustainable TD conversion"
                ),
                axis=1,
            )
            sort_col = "rolling_fpoe_3w"
        else:
            sell_high = filtered[
                (filtered["fantasy_points_ppr"] >= 15.0)
                & (filtered["fp_diff"] >= 5.0)
            ].copy()
            sell_high["rationale"] = sell_high.apply(
                lambda r: f"Outperformed true usage by +{r['fp_diff']:.1f} pts on limited volume ({int(r['touches'])} touches, {int(r['offense_pct']*100)}% snaps)",
                axis=1,
            )
            sort_col = "fp_diff"

        sell_high = sell_high.sort_values(by=sort_col, ascending=False).head(limit)
        return sell_high[[c for c in OUTPUT_COLUMNS if c in sell_high.columns]].reset_index(drop=True)


def _slot_counts(league_context: Dict[str, Any]) -> Dict[str, int]:
    counts = league_context.get("roster_position_counts") or {}
    return {
        str(slot).upper(): int(count)
        for slot, count in counts.items()
        if isinstance(count, (int, float)) and not isinstance(count, bool)
    }


def estimated_flex_starters(slot_counts: Dict[str, int]) -> Dict[str, float]:
    """Estimated per-team flex starters at each of QB/RB/WR/TE (see FLEX_STARTER_DISTRIBUTION)."""
    flex = {pos: 0.0 for pos in TRADEABLE_POSITIONS}
    for slot, dist in FLEX_STARTER_DISTRIBUTION.items():
        for pos, share in dist.items():
            flex[pos] += share * slot_counts.get(slot, 0)
    superflex = slot_counts.get("SUPER_FLEX", 0)
    if superflex:
        flex["QB"] += SUPERFLEX_QB_SHARE * superflex
        for pos, share in SUPERFLEX_REMAINDER_SPLIT.items():
            flex[pos] += share * superflex
    return flex


def replacement_rank(pos: str, league_context: Dict[str, Any]) -> int:
    """K_pos = N * (S_pos + F_pos) + 1: the rostered-starter depth index at a position."""
    n_teams = int(league_context.get("total_rosters") or 10)
    slot_counts = _slot_counts(league_context)
    flex = estimated_flex_starters(slot_counts)
    starters = float(slot_counts.get(pos, 0))
    return max(1, int(round(n_teams * (starters + flex.get(pos, 0.0)) + 1)))


def compute_replacement_levels(pool: pd.DataFrame, league_context: Dict[str, Any]) -> Dict[str, Any]:
    """Dynamic replacement baselines at each tradeable position for a league-relevant pool.

    replacement_xfp[pos] is the rolling_xfp_3w of the K_pos-th ranked player at
    that position and replacement_market[pos] is the same rank on market_ppg
    (0.7 * recent actual PPG + 0.3 * rolling xFP). Falls back to the worst
    available player when the pool is smaller than K_pos.
    """
    result: Dict[str, Any] = {
        "k_pos": {},
        "replacement_xfp": {},
        "replacement_market": {},
        "flex_starters_per_team": estimated_flex_starters(_slot_counts(league_context)),
    }
    empty = pool is None or pool.empty
    if not empty:
        pool = pool.copy()
        if "market_ppg" not in pool.columns:
            recent = (
                pd.to_numeric(pool["recent_ppr"], errors="coerce").fillna(0.0)
                if "recent_ppr" in pool.columns
                else pd.Series(0.0, index=pool.index)
            )
            xfp = (
                pd.to_numeric(pool["rolling_xfp_3w"], errors="coerce").fillna(0.0)
                if "rolling_xfp_3w" in pool.columns
                else pd.Series(0.0, index=pool.index)
            )
            pool["market_ppg"] = 0.7 * recent + 0.3 * xfp

    for pos in TRADEABLE_POSITIONS:
        k = replacement_rank(pos, league_context)
        result["k_pos"][pos] = k
        if empty or "position" not in pool.columns or "rolling_xfp_3w" not in pool.columns:
            result["replacement_xfp"][pos] = 0.0
            result["replacement_market"][pos] = 0.0
            continue
        sub = pool[pool["position"] == pos]
        if sub.empty:
            result["replacement_xfp"][pos] = 0.0
            result["replacement_market"][pos] = 0.0
            continue
        xfp_ranked = pd.to_numeric(sub["rolling_xfp_3w"], errors="coerce").fillna(0.0).sort_values(ascending=False).tolist()
        market_ranked = pd.to_numeric(sub["market_ppg"], errors="coerce").fillna(0.0).sort_values(ascending=False).tolist()
        idx = min(k, len(xfp_ranked)) - 1
        result["replacement_xfp"][pos] = float(xfp_ranked[idx])
        result["replacement_market"][pos] = float(market_ranked[idx])
    return result


def package_value(
    values: Sequence[float],
    v_drop: float = 0.0,
    alpha: float = PACKAGE_CONSOLIDATION_ALPHA,
) -> float:
    """Multi-player package value with consolidation decay.

    V(1) + alpha*V(2) + alpha^2*V(3) - V_drop, where values are sorted
    descending and V_drop is the forced-cut cost of the receiving side.
    """
    ordered = sorted((max(0.0, float(v)) for v in values), reverse=True)
    decayed = sum(v * (alpha ** i) for i, v in enumerate(ordered))
    return max(0.0, decayed - max(0.0, float(v_drop)))


def estimate_acceptance_probability(
    market_delta: float,
    opp_lineup_gain: float,
    need_filled: bool,
) -> float:
    """Heuristic P(partner accepts): logistic on market parity, their lineup gain, and positional need.

    Untrained heuristic (no acceptance data exists in this repo); inputs are
    clipped to +/- ACCEPTANCE_CLIP to bound the logits.
    """
    d_v = float(np.clip(market_delta, -ACCEPTANCE_CLIP, ACCEPTANCE_CLIP))
    d_u = float(np.clip(opp_lineup_gain, 0.0, ACCEPTANCE_CLIP))
    z = (
        ACCEPTANCE_BIAS
        + ACCEPTANCE_MARKET_WEIGHT * d_v
        + ACCEPTANCE_LINEUP_WEIGHT * d_u
        + ACCEPTANCE_NEED_WEIGHT * (1.0 if need_filled else 0.0)
    )
    return float(1.0 / (1.0 + np.exp(-z)))


def _latest_player_rows(frame: pd.DataFrame) -> pd.DataFrame:
    key = "player_id" if "player_id" in frame.columns else "player_display_name"
    sort_cols = [c for c in ("season", "week") if c in frame.columns]
    if sort_cols:
        return frame.sort_values(sort_cols, kind="stable").groupby(key, sort=False).tail(1).reset_index(drop=True)
    return frame.reset_index(drop=True)


def _prepare_trade_frame(metrics_df: pd.DataFrame, league_context: Dict[str, Any]) -> pd.DataFrame:
    """Latest per-player frame with rolling xFP / FPOE, recent actual PPG, and market PPG."""
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()

    frame = metrics_df.copy()
    if "fantasy_points_ppr" not in frame.columns:
        frame["fantasy_points_ppr"] = 0.0
    if (
        "season" in frame.columns
        and "week" in frame.columns
        and (
            "rolling_xfp_3w" not in frame.columns
            or "rolling_fpoe_3w" not in frame.columns
            or "recent_ppr" not in frame.columns
        )
    ):
        scoring = league_context.get("scoring_type") or SCORING_PPR
        frame = estimate_expected_fantasy_points(compute_player_trajectories(frame), scoring=scoring)
    if frame.empty:
        return pd.DataFrame()

    for col in ("sleeper_id", "player_id"):
        if col in frame.columns:
            frame[col] = _normalize_ids(frame[col])

    frame = _latest_player_rows(frame)
    for col in ("rolling_xfp_3w", "rolling_fpoe_3w", "recent_ppr"):
        if col not in frame.columns:
            frame[col] = frame["fantasy_points_ppr"] if col == "recent_ppr" else 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    if "position" in frame.columns:
        frame["position"] = frame["position"].astype(str).str.upper()

    # Public-perception proxy blend: 70% recent actual PPG + 30% rolling xFP
    frame["market_ppg"] = (0.7 * frame["recent_ppr"] + 0.3 * frame["rolling_xfp_3w"]).round(2)
    return frame


def generate_league_trade_proposals(
    metrics_df: pd.DataFrame,
    league_context: Dict[str, Any],
    rosters: List[List[str]],
    user_roster_index: int = 0,
    roster_ids: Optional[List[Any]] = None,
    manager_names: Optional[List[str]] = None,
    players_db: Optional[Dict[str, Dict[str, Any]]] = None,
    free_agent_ids: Optional[Iterable[str]] = None,
    max_roster_size: Optional[int] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Bilateral league trade generator.

    Combines dynamic VORP, package consolidation decay, bilateral lineup
    improvement (via the Phase-4 ILP lineup optimizer), and a market-parity
    guard into ranked 1-for-1 and 2-for-1 trade proposals between the user's
    roster and every opponent roster.

    - metrics_df: per-player metrics frame; latest row per player is used,
      carrying rolling_xfp_3w (model projection), rolling_fpoe_3w, and
      recent_ppr (recent actual PPG). Rolling columns are derived from the
      raw weekly frame when absent.
    - league_context: Sleeper league context (total_rosters,
      roster_position_counts, scoring_type).
    - rosters: league rosters as lists of player ids (Sleeper ids);
      user_roster_index selects the user's roster.
    - players_db: optional Sleeper player metadata for names/positions/injury
      of roster players missing from the metrics frame.
    - free_agent_ids: optional free-agent ids; when absent the VORP pool is
      the full metrics frame (every player with weekly data).
    - max_roster_size: optional roster limit; when absent, benches are
      assumed full so every 2-for-1 receiving side pays a V_drop cut cost.
    """
    notes: List[str] = []
    base: Dict[str, Any] = {
        "proposals": [],
        "league": {
            "total_rosters": league_context.get("total_rosters"),
            "roster_position_counts": league_context.get("roster_position_counts") or {},
            "scoring_type": league_context.get("scoring_type"),
        },
        "notes": notes,
    }
    frame = _prepare_trade_frame(metrics_df, league_context)
    if frame.empty or not rosters:
        notes.append("No per-player metrics or rosters available; bilateral trade evaluation skipped.")
        return {**base, "vorp": compute_replacement_levels(pd.DataFrame(), league_context)}

    rosters = [[str(p) for p in roster] for roster in rosters]
    roster_ids = list(roster_ids) if roster_ids else list(range(1, len(rosters) + 1))
    manager_names = [str(n) for n in manager_names] if manager_names else []
    user_roster_index = min(max(int(user_roster_index), 0), len(rosters) - 1)

    # --- Dynamic VORP calibration over the league-relevant pool ---
    rostered_ids = {pid for roster in rosters for pid in roster}
    id_cols = [c for c in ("sleeper_id", "player_id") if c in frame.columns]
    if free_agent_ids is not None:
        pool_ids = rostered_ids | {str(p) for p in free_agent_ids}
        in_pool = pd.Series(False, index=frame.index)
        for col in id_cols:
            in_pool |= frame[col].isin(pool_ids)
        pool_frame = frame[in_pool]
        pool_source = "league_rosters_plus_free_agents"
    else:
        pool_frame = frame
        pool_source = "full_metrics_frame"
    repl = compute_replacement_levels(pool_frame, league_context)

    # --- Per-player records for every rostered player ---
    metrics_by_id: Dict[str, pd.Series] = {}
    metrics_by_name: Dict[str, pd.Series] = {}
    for _, row in frame.iterrows():
        for col in id_cols:
            pid = row.get(col)
            if pid is not None and pd.notna(pid):
                metrics_by_id.setdefault(str(pid), row)
        name = _norm_name(row.get("player_display_name"))
        if name:
            metrics_by_name.setdefault(name, row)

    players_db = players_db or {}
    records: Dict[str, Dict[str, Any]] = {}

    def record_for(pid: str) -> Dict[str, Any]:
        if pid in records:
            return records[pid]
        row = metrics_by_id.get(pid)
        meta = players_db.get(pid, {})
        if row is None and meta.get("full_name"):
            row = metrics_by_name.get(_norm_name(meta["full_name"]))
        name = (row.get("player_display_name") if row is not None else None) or meta.get("full_name") or f"Player {pid}"
        position = str((row.get("position") if row is not None else None) or meta.get("position") or "").upper()

        def _num(col: str) -> float:
            if row is None:
                return 0.0
            val = row.get(col)
            try:
                return float(val) if val is not None and pd.notna(val) else 0.0
            except (TypeError, ValueError):
                return 0.0

        rec = {
            "player_id": pid,
            "name": str(name),
            "position": position,
            "team": (row.get("recent_team") if row is not None else None) or meta.get("team"),
            "injury_status": meta.get("injury_status"),
            "rolling_xfp_3w": _num("rolling_xfp_3w"),
            "recent_ppg": _num("recent_ppr"),
            "rolling_fpoe_3w": _num("rolling_fpoe_3w"),
        }
        rec["market_ppg"] = round(0.7 * rec["recent_ppg"] + 0.3 * rec["rolling_xfp_3w"], 2)
        if position in TRADEABLE_POSITIONS:
            rec["model_vorp"] = max(0.0, rec["rolling_xfp_3w"] - repl["replacement_xfp"][position])
            rec["market_vorp"] = max(0.0, rec["market_ppg"] - repl["replacement_market"][position])
        else:
            rec["model_vorp"] = 0.0
            rec["market_vorp"] = 0.0
        records[pid] = rec
        return rec

    for roster in rosters:
        for pid in roster:
            record_for(pid)

    # --- Optimal-lineup sums via the Phase-4 ILP optimizer ---
    slot_counts = _slot_counts(league_context)
    roster_positions = [slot for slot, count in slot_counts.items() for _ in range(count)]
    solver = MatchupAdvisor(roster_positions=roster_positions or None)
    lineup_cache: Dict[frozenset, Tuple[float, List[Dict[str, Any]]]] = {}

    def lineup(pid_set: frozenset) -> Tuple[float, List[Dict[str, Any]]]:
        if pid_set not in lineup_cache:
            lineup_cache[pid_set] = solver.solve_lineup_points([
                {
                    "player_id": records[pid]["player_id"],
                    "name": records[pid]["name"],
                    "position": records[pid]["position"],
                    "value": records[pid]["rolling_xfp_3w"],
                    "injury_status": records[pid]["injury_status"],
                }
                for pid in pid_set
                if pid in records
            ])
        return lineup_cache[pid_set]

    pre_lineups = {idx: lineup(frozenset(roster)) for idx, roster in enumerate(rosters)}

    def need_positions(owner_idx: int) -> Set[str]:
        _, starters = pre_lineups[owner_idx]
        by_pos: Dict[str, List[Dict[str, Any]]] = {}
        for p in starters:
            by_pos.setdefault(p["position"], []).append(p)
        needs: Set[str] = set()
        for pos in TRADEABLE_POSITIONS:
            players_at_pos = by_pos.get(pos, [])
            if not players_at_pos:
                if slot_counts.get(pos, 0) > 0:
                    needs.add(pos)
                continue
            if any(p["y"] < repl["replacement_xfp"][pos] for p in players_at_pos):
                needs.add(pos)
        return needs

    def tradables(owner_idx: int) -> List[Dict[str, Any]]:
        candidates = [
            records[pid]
            for pid in rosters[owner_idx]
            if records[pid]["position"] in TRADEABLE_POSITIONS
            and str(records[pid].get("injury_status") or "") not in OUT_STATUSES
            and (records[pid]["model_vorp"] > 0 or records[pid]["market_vorp"] > 0)
        ]
        candidates.sort(key=lambda r: max(r["model_vorp"], r["market_vorp"]), reverse=True)
        return candidates[:MAX_TRADEABLE_ASSETS_PER_SIDE]

    def drop_cost(post_roster: List[str], post_lineup: List[Dict[str, Any]]) -> Tuple[float, Optional[str]]:
        """Market VORP of the worst bench player the receiving side must cut (0 with an open spot)."""
        if max_roster_size is not None and len(post_roster) <= max_roster_size:
            return 0.0, None
        starter_keys = {p["player_id"] for p in post_lineup}
        bench = [records[pid] for pid in post_roster if pid not in starter_keys]
        if not bench:
            return 0.0, None
        worst = min(bench, key=lambda r: r["market_vorp"])
        return worst["market_vorp"], worst["name"]

    def player_card(rec: Dict[str, Any]) -> Dict[str, Any]:
        tag = None
        if rec["rolling_xfp_3w"] >= BUY_LOW_MIN_ROLLING_XFP and rec["rolling_fpoe_3w"] <= BUY_LOW_MAX_ROLLING_FPOE:
            tag = "buy-low"
        elif rec["recent_ppg"] >= SELL_HIGH_MIN_RECENT_PPG and rec["rolling_fpoe_3w"] >= SELL_HIGH_MIN_ROLLING_FPOE:
            tag = "sell-high"
        return {
            "player_id": rec["player_id"],
            "name": rec["name"],
            "position": rec["position"],
            "team": rec.get("team"),
            "rolling_xfp_3w": rec["rolling_xfp_3w"],
            "recent_ppg": rec["recent_ppg"],
            "rolling_fpoe_3w": rec["rolling_fpoe_3w"],
            "market_ppg": rec["market_ppg"],
            "model_vorp": round(rec["model_vorp"], 2),
            "market_vorp": round(rec["market_vorp"], 2),
            "tag": tag,
        }

    def manager_label(idx: int) -> str:
        if idx < len(manager_names) and manager_names[idx]:
            return manager_names[idx]
        return f"Roster {roster_ids[idx]}" if idx < len(roster_ids) else f"Team {idx + 1}"

    def build_pitch(
        send_cards: List[Dict[str, Any]],
        receive_cards: List[Dict[str, Any]],
        need_hit: Set[str],
        d_user: float,
        opp_drop_name: Optional[str],
        user_drop_name: Optional[str],
    ) -> str:
        parts: List[str] = []
        for c in send_cards:
            if c["tag"] == "sell-high":
                parts.append(
                    f"sell-high: {c['name']} FPOE {c['rolling_fpoe_3w']:+.1f} vs rolling xFP {c['rolling_xfp_3w']:.1f}"
                )
        for c in receive_cards:
            if c["tag"] == "buy-low":
                parts.append(
                    f"buy-low: {c['name']} rolling xFP {c['rolling_xfp_3w']:.1f} masked by {c['rolling_fpoe_3w']:+.1f} FPOE"
                )
        if need_hit:
            parts.append("fills their " + "/".join(sorted(need_hit)) + " hole")
        if len(send_cards) > 1:
            decay_note = f"package decay {PACKAGE_CONSOLIDATION_ALPHA} on the second asset"
            if opp_drop_name:
                decay_note += f"; they cut {opp_drop_name}"
            parts.append(decay_note)
        if len(receive_cards) > 1:
            decay_note = f"package decay {PACKAGE_CONSOLIDATION_ALPHA} on the second asset"
            if user_drop_name:
                decay_note += f"; you cut {user_drop_name}"
            parts.append(decay_note)
        if not parts:
            parts.append(f"upgrades your optimal lineup by +{d_user:.1f} rolling xFP")
        return "; ".join(parts)

    proposals: List[Dict[str, Any]] = []
    evaluations = 0

    def evaluate_swap(opp_idx: int, user_send: List[str], opp_send: List[str], opp_label: str, needs: Set[str]) -> None:
        nonlocal evaluations
        if evaluations >= MAX_SWAP_EVALUATIONS:
            return
        evaluations += 1

        user_sent_market = [records[pid]["market_vorp"] for pid in user_send]
        opp_sent_market = [records[pid]["market_vorp"] for pid in opp_send]
        all_markets = user_sent_market + opp_sent_market
        if max(all_markets) - min(all_markets) > MAX_ASSET_VALUE_GAP:
            return
        # Cheap pre-guard: drop costs only lower the opponent's received value
        if package_value(user_sent_market) - package_value(opp_sent_market) < MIN_MARKET_DELTA:
            return

        user_post = [pid for pid in rosters[user_roster_index] if pid not in user_send] + opp_send
        opp_post = [pid for pid in rosters[opp_idx] if pid not in opp_send] + user_send
        user_pre_total, _ = pre_lineups[user_roster_index]
        opp_pre_total, _ = pre_lineups[opp_idx]
        user_post_total, user_post_lineup = lineup(frozenset(user_post))
        opp_post_total, opp_post_lineup = lineup(frozenset(opp_post))
        d_user = round(user_post_total - user_pre_total, 2)
        d_opp = round(opp_post_total - opp_pre_total, 2)
        if d_user < MIN_USER_LINEUP_GAIN or d_opp <= 0:
            return

        opp_drop, opp_drop_name = (0.0, None)
        if len(user_sent_market) > 1:
            opp_drop, opp_drop_name = drop_cost(opp_post, opp_post_lineup)
        market_delta = round(
            package_value(user_sent_market, v_drop=opp_drop) - package_value(opp_sent_market), 2
        )
        if market_delta < MIN_MARKET_DELTA:
            return

        user_drop, user_drop_name = (0.0, None)
        if len(opp_sent_market) > 1:
            user_drop, user_drop_name = drop_cost(user_post, user_post_lineup)

        need_hit = {records[pid]["position"] for pid in user_send} & needs
        p_accept = estimate_acceptance_probability(market_delta, d_opp, bool(need_hit))

        send_cards = [player_card(records[pid]) for pid in user_send]
        receive_cards = [player_card(records[pid]) for pid in opp_send]
        proposals.append({
            "target_manager": opp_label,
            "target_roster_id": roster_ids[opp_idx] if opp_idx < len(roster_ids) else None,
            "send": send_cards,
            "receive": receive_cards,
            "your_lineup_gain": d_user,
            "their_lineup_gain": d_opp,
            "market_delta": market_delta,
            "need_filled": bool(need_hit),
            "p_accept": round(p_accept, 4),
            "pitch": build_pitch(send_cards, receive_cards, need_hit, d_user, opp_drop_name, user_drop_name),
        })

    user_assets = tradables(user_roster_index)
    for opp_idx in range(len(rosters)):
        if opp_idx == user_roster_index:
            continue
        opp_assets = tradables(opp_idx)
        if not user_assets or not opp_assets:
            continue
        opp_label = manager_label(opp_idx)
        needs = need_positions(opp_idx)
        for u in user_assets:
            for o in opp_assets:
                evaluate_swap(opp_idx, [u["player_id"]], [o["player_id"]], opp_label, needs)
        for u1, u2 in combinations(user_assets, 2):
            for o in opp_assets:
                evaluate_swap(opp_idx, [u1["player_id"], u2["player_id"]], [o["player_id"]], opp_label, needs)
        for o1, o2 in combinations(opp_assets, 2):
            for u in user_assets:
                evaluate_swap(opp_idx, [u["player_id"]], [o1["player_id"], o2["player_id"]], opp_label, needs)
        if evaluations >= MAX_SWAP_EVALUATIONS:
            notes.append(f"Swap search capped at {MAX_SWAP_EVALUATIONS} evaluations.")
            break

    proposals.sort(key=lambda p: (-p["p_accept"], -p["your_lineup_gain"]))
    notes.append(
        "VORP pool = "
        + (
            "league rosters + provided free agents"
            if free_agent_ids is not None
            else "full metrics frame (free-agent pool unavailable; every player with weekly data counts as league-relevant)"
        )
    )
    notes.append(
        "K_pos flex heuristics: FLEX 45% RB / 45% WR / 10% TE per slot; "
        f"SUPER_FLEX {SUPERFLEX_QB_SHARE:.0%} QB with the remainder split RB/WR."
    )
    if max_roster_size is None:
        notes.append("Roster-cut cost assumes full benches (no open spots); pass max_roster_size to model open slots.")
    notes.append(
        "Acceptance probability is an untrained heuristic (logistic on market delta, their lineup gain, positional need)."
    )
    notes.append(
        f"Search caps: top {MAX_TRADEABLE_ASSETS_PER_SIDE} assets per side by VORP; swaps with a "
        f"single-asset market VORP gap > {MAX_ASSET_VALUE_GAP:.0f} are skipped as lopsided."
    )

    return {
        **base,
        "proposals": proposals[:limit],
        "vorp": {**repl, "pool_size": int(len(pool_frame)), "pool_source": pool_source},
    }
