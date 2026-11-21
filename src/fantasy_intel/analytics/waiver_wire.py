"""Waiver wire scanner for discovering under-the-radar breakouts."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from fantasy_intel.analytics.metrics import compute_player_trajectories, estimate_expected_fantasy_points
from fantasy_intel.config import (
    FAAB_GAMMA,
    FAAB_URGENCY_HIGH,
    FAAB_URGENCY_LOW,
    SCORING_PPR,
)


class WaiverWireEngine:
    """Detects under-the-radar waiver pickups using leading volume metrics."""

    def __init__(
        self,
        weekly_df: pd.DataFrame,
        scoring: str = SCORING_PPR,
        fpoe_history: Optional[pd.DataFrame] = None,
    ):
        self.raw_df = weekly_df
        # Enrich dataset (fpoe_history seeds each player's efficiency baseline)
        traj = compute_player_trajectories(weekly_df)
        self.df = estimate_expected_fantasy_points(traj, scoring=scoring, fpoe_history=fpoe_history)

    def scan_sleepers(
        self,
        week: Optional[int] = None,
        season: Optional[int] = None,
        positions: Optional[List[str]] = None,
        max_recent_ppr: float = 14.0,  # Below star threshold (unlikely to already be 100% rostered)
        min_snap_pct: float = 0.35,     # At least 35% snaps
        min_opp_index: float = 40.0,
        limit: int = 15,
    ) -> pd.DataFrame:
        """
        Identifies high-opportunity players flying under the radar.
        """
        if self.df.empty:
            return pd.DataFrame()

        filtered = self.df.copy()

        if season is not None:
            filtered = filtered[filtered["season"] == season]

        if week is not None:
            filtered = filtered[filtered["week"] == week]
        else:
            # Default to latest regular season week
            reg = filtered[filtered["season_type"] == "REG"] if "season_type" in filtered.columns else filtered[filtered["week"] <= 18]
            latest_week = reg["week"].max() if not reg.empty else filtered["week"].max()
            filtered = filtered[filtered["week"] == latest_week]

        if positions:
            filtered = filtered[filtered["position"].isin(positions)]

        # Filter to skill positions
        filtered = filtered[filtered["position"].isin(["QB", "RB", "WR", "TE"])]

        # Filter out players who scored massive points (everyone already sees them on box scores)
        # We want people who have the *volume* without the national attention yet
        filtered = filtered[
            (filtered["offense_pct"] >= min_snap_pct) &
            (filtered["recent_ppr"] <= max_recent_ppr)
        ].copy()

        if filtered.empty:
            return pd.DataFrame()

        # Position-specific composite Opportunity Score (0-100); a single formula
        # locks RBs out of the WOPR component and penalizes WRs on touches.
        # WR/TE: snap share (25) + WOPR (40) + snap surge (20) + unrealized air yards (15)
        # RB:    snap share (25) + high-value touches (35) + touch volume (25) + snap surge (15)
        # QB:    legacy snap/WOPR/touch blend (QBs are not the waiver focus)
        for col in ("hvt", "unrealized_air_yards"):
            if col not in filtered.columns:
                filtered[col] = 0.0
            else:
                filtered[col] = filtered[col].fillna(0)

        snap_score = filtered["offense_pct"].clip(0, 1.0) * 25
        delta_score = filtered["snap_delta"].clip(0, 0.35) / 0.35 * 15

        wr_te_score = (
            snap_score
            + filtered["wopr"].clip(0, 0.75) / 0.75 * 40
            + filtered["snap_delta"].clip(0, 0.35) / 0.35 * 20
            + filtered["unrealized_air_yards"].clip(0, 70) / 70 * 15
        )
        rb_score = (
            snap_score
            + filtered["hvt"].clip(0, 8) / 8 * 35
            + filtered["touches"].clip(0, 18) / 18 * 25
            + delta_score
        )
        qb_score = (
            filtered["offense_pct"].clip(0, 1.0) * 30
            + filtered["wopr"].clip(0, 0.8) / 0.8 * 35
            + filtered["touches"].clip(0, 20) / 20 * 20
            + filtered["snap_delta"].clip(0, 0.4) / 0.4 * 15
        )

        filtered["opportunity_index"] = np.select(
            [
                filtered["position"].isin(["WR", "TE"]),
                filtered["position"] == "RB",
            ],
            [wr_te_score, rb_score],
            default=qb_score,
        ).round(1)

        # Generate actionable catalyst note
        def make_catalyst(row: pd.Series) -> str:
            notes = []
            if row["snap_delta"] >= 0.15:
                notes.append(f"Snap surge (+{int(row['snap_delta']*100)}%)")
            if row["wopr"] >= 0.40:
                notes.append(f"High WOPR ({row['wopr']:.2f})")
            elif row["target_share"] >= 0.18:
                notes.append(f"Target share {int(row['target_share']*100)}%")
            if row["touches"] >= 12 and row["position"] == "RB":
                notes.append(f"{int(row['touches'])} touches")
            if row["position"] == "RB" and row["hvt"] >= 6:
                notes.append(f"{int(row['hvt'])} high-value touches")
            if row["fp_diff"] <= -3.0:
                notes.append("Unlucky TD luck (due for positive regression)")
            return "; ".join(notes) if notes else "Growing weekly snap involvement"

        filtered["catalyst"] = filtered.apply(make_catalyst, axis=1)

        # Enrich with live depth chart and injury context. Sleeper's live
        # payload is the source of truth; nflverse depth/injury data is only
        # a season-checked fallback enrichment inside the manager.
        from fantasy_intel.analytics.depth_chart import DepthChartManager
        dcm = DepthChartManager(season=season)

        elevation_tiers = []
        ir_eligibles = []
        depth_roles = []
        situations = []
        inj_statuses = []

        for name in filtered["player_display_name"]:
            ctx = dcm.get_player_elevation_context(name)
            elevation_tiers.append(ctx.get("tier", "UNKNOWN"))
            ir_eligibles.append(ctx.get("ir_eligible", False))
            depth_roles.append(ctx.get("healthy_role", ""))
            situations.append(ctx.get("summary", ""))
            own_inj = ctx.get("own_injury")
            own_part = ctx.get("own_body_part")
            inj_statuses.append(f"{own_inj} ({own_part})" if own_inj and own_part else (own_inj or "Healthy"))

        filtered["elevation_tier"] = elevation_tiers
        filtered["ir_eligible"] = ir_eligibles
        filtered["depth_role"] = depth_roles
        filtered["situation"] = situations
        filtered["injury_status"] = inj_statuses

        # Bayesian Vacancy & Availability Adjustments
        adjusted_indices = []
        categories = []
        strategic_notes = []

        for _, row in filtered.iterrows():
            base_opp = float(row["opportunity_index"])
            tier = row["elevation_tier"]
            ir_eligible = row["ir_eligible"]
            own_inj = row["injury_status"]
            summary = row["situation"]

            if ir_eligible:
                # Out / IR stash: discount immediate week points, but maintain high rest-of-season stash ranking
                adj_score = round(base_opp * 0.9, 1)
                cat = "IR_STASH_ONLY"
                note = f"STASH ASSET: Ruled {own_inj}. Eligible for open IR slot."
            elif tier == "ELEVATED_STARTER":
                # V2 fix: never cap below base_opp; a high-usage handcuff with
                # base_opp=85 was previously downgraded to 80.0 by the old min().
                adj_score = min(100.0, max(base_opp, round(base_opp + 15.0, 1)))
                cat = "HIGH_PRIORITY_ADD"
                note = f"IMMEDIATE STARTER: {summary}"
            elif tier == "POTENTIAL_SURGE":
                # Starters Questionable
                adj_score = min(95.0, max(base_opp, round(base_opp + 8.0, 1)))
                cat = "CONTINGENCY_ADD"
                note = f"SURGE CANDIDATE: {summary}"
            else:
                adj_score = round(base_opp, 1)
                cat = "STANDARD_ADD"
                note = summary or "Standard depth"

            adjusted_indices.append(adj_score)
            categories.append(cat)
            strategic_notes.append(note)

        filtered["base_opportunity_index"] = filtered["opportunity_index"]
        filtered["opportunity_index"] = adjusted_indices
        filtered["add_category"] = categories
        filtered["strategic_note"] = strategic_notes

        result_cols = [
            "player_id",
            "player_display_name",
            "position",
            "recent_team",
            "week",
            "offense_pct",
            "snap_delta",
            "targets",
            "carries",
            "touches",
            "hvt",
            "wopr",
            "racr",
            "unrealized_air_yards",
            "rolling_xfp_3w",
            "proj_ppg",
            "fpoe_eff",
            "fantasy_points_ppr",
            "expected_fp_ppr",
            "base_opportunity_index",
            "opportunity_index",
            "add_category",
            "strategic_note",
            "catalyst",
            "injury_status",
            "ir_eligible",
            "depth_role",
            "situation",
        ]

        available_cols = [c for c in result_cols if c in filtered.columns]
        sleepers = (
            filtered[available_cols]
            .query(f"opportunity_index >= {min_opp_index}")
            .sort_values(by="opportunity_index", ascending=False)
            .head(limit)
            .reset_index(drop=True)
        )

        return sleepers


def _roster_constant(user_roster: Optional[pd.DataFrame], col: str) -> Optional[Any]:
    """Returns the first non-null value of a constant roster-level column."""
    if user_roster is None or user_roster.empty or col not in user_roster.columns:
        return None
    vals = user_roster[col].dropna()
    return vals.iloc[0] if not vals.empty else None


def _user_roster_record(
    user_roster: Optional[pd.DataFrame], league_context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Finds the calling user's roster record (wins/losses/faab_remaining) in league_context."""
    if user_roster is None or user_roster.empty:
        return None
    rosters = league_context.get("rosters") or []
    for col in ("owner_id", "roster_id"):
        ident = _roster_constant(user_roster, col)
        if ident is None:
            continue
        for rec in rosters:
            if str(rec.get(col)) == str(ident):
                return rec
    return None


def recommend_faab_bids(
    candidates_df: pd.DataFrame,
    user_roster: pd.DataFrame,
    league_context: Dict[str, Any],
    faab_history: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Recommends FAAB bids (or rolling-priority claim flags) for waiver candidates.

    candidates_df: this week's available candidates from WaiverWireEngine.scan_sleepers.
      Uses "position", "rolling_xfp_3w", and "opportunity_index" columns.

    user_roster: one row per rostered player of the calling user. Required columns:
      "position" and "rolling_xfp_3w" (NaN where not computable — the caller should
      join the analytics frame's latest-week rolling_xfp_3w by Sleeper id).
      Optional: "is_starter" (bool), "injury_status", "player_display_name".
      Roster-level context as constant columns: "owner_id" or "roster_id"
      (resolved against league_context["rosters"] for the user's wins/losses and
      faab_remaining) and "waiver_position" (current rolling-waiver priority).

    league_context: SleeperClient.get_league_context(league_id). Uses "waiver_type"
      (0 = rolling priority, no FAAB), "faab_budget" (initial budget, default 100),
      and "rosters".

    faab_history: SleeperClient.get_league_faab_history(league_id) or None. Empirical
      winning-bid distribution (median/p80/count) used for the clearing price; the
      count=0 path falls back to 12% (index >= 70) / 5% of the initial budget.

    FAAB mode (waiver_type != 0), per candidate:
      bid = min(B_rem, round(Bid_clearing * (delta_vorp / VORP_max)
                            * (B_rem / B_init)^FAAB_GAMMA * urgency)) with a $0 floor,
      and $0 whenever delta_vorp <= 0 (no upgrade over the user's worst rostered
      player at that position). urgency = FAAB_URGENCY_HIGH when win% < 0.40 or the
      candidate fills an injured (Out/IR/Doubtful) starter at that position,
      FAAB_URGENCY_LOW at win% >= 0.83, else 1.0.

    Priority mode (waiver_type == 0): no dollar bids. priority_claim_recommended is
      True when delta_vorp > 0 and ((delta_vorp > 2.0 and waiver_position <= 6) or
      opportunity_index >= 75); recommended_faab_bid is None throughout.

    Returns candidates_df with delta_vorp, recommended_faab_bid,
    faab_pct_of_remaining, priority_claim_recommended, and faab_rationale.
    """
    out = candidates_df.copy()
    if out.empty:
        return out

    waiver_type = league_context.get("waiver_type")
    faab_budget = league_context.get("faab_budget")
    b_init = (
        int(faab_budget)
        if isinstance(faab_budget, (int, float)) and not isinstance(faab_budget, bool)
        else 100
    )

    record = _user_roster_record(user_roster, league_context)
    if record is not None:
        b_rem = record.get("faab_remaining")
        b_rem = (
            int(b_rem)
            if isinstance(b_rem, (int, float)) and not isinstance(b_rem, bool)
            else b_init
        )
        wins = int(record.get("wins", 0) or 0)
        losses = int(record.get("losses", 0) or 0)
        ties = int(record.get("ties", 0) or 0)
    else:
        b_rem, wins, losses, ties = b_init, 0, 0, 0
    games = wins + losses + ties
    win_pct = wins / games if games > 0 else None

    waiver_position = _roster_constant(user_roster, "waiver_position")
    try:
        waiver_position = int(waiver_position)
    except (TypeError, ValueError):
        waiver_position = None

    # Value = skill-adjusted projection (proj_ppg) when supplied, else rolling xFP.
    # Candidates and roster must use the same scale.
    def _value_col(frame: Optional[pd.DataFrame]) -> str:
        if frame is not None and "proj_ppg" in frame.columns and frame["proj_ppg"].notna().any():
            return "proj_ppg"
        return "rolling_xfp_3w"

    roster_value_col = _value_col(user_roster)
    cand_value_col = _value_col(out)
    if roster_value_col != cand_value_col:
        roster_value_col = cand_value_col = "rolling_xfp_3w"

    # Replacement baseline: the user's worst rostered player at each position
    baselines: Dict[str, float] = {}
    if (
        user_roster is not None
        and not user_roster.empty
        and {"position", roster_value_col}.issubset(user_roster.columns)
    ):
        by_pos = (
            user_roster.dropna(subset=[roster_value_col])
            .groupby("position")[roster_value_col]
            .min()
        )
        baselines = by_pos.to_dict()

    # Positions where an injured (Out/IR/Doubtful) starter needs replacing
    injured_starter_positions = set()
    if (
        user_roster is not None
        and not user_roster.empty
        and {"is_starter", "injury_status", "position"}.issubset(user_roster.columns)
    ):
        hurt = user_roster[
            user_roster["is_starter"].astype(bool)
            & user_roster["injury_status"].astype(str).str.upper().isin(["OUT", "IR", "DOUBTFUL"])
        ]
        injured_starter_positions = {p for p in hurt["position"].dropna()}

    positions = out["position"] if "position" in out.columns else pd.Series("", index=out.index)
    if cand_value_col in out.columns:
        cand_xfp = pd.to_numeric(out[cand_value_col], errors="coerce").fillna(0.0)
    else:
        cand_xfp = pd.Series(0.0, index=out.index)
    if "opportunity_index" in out.columns:
        opp_index = pd.to_numeric(out["opportunity_index"], errors="coerce").fillna(0.0)
    else:
        opp_index = pd.Series(0.0, index=out.index)

    delta_vorp = (cand_xfp - positions.map(baselines).fillna(0.0)).round(2)
    out["delta_vorp"] = delta_vorp
    vorp_max = float(delta_vorp.max()) if not delta_vorp.empty else 0.0

    history = faab_history or {}
    hist_count = int(history.get("count") or 0)
    hist_median = history.get("median")
    hist_p80 = history.get("p80")
    if hist_count >= 5 and isinstance(hist_p80, (int, float)):
        league_clearing, clearing_src = float(hist_p80), f"p80 of {hist_count} league bids"
    elif hist_count >= 1 and isinstance(hist_median, (int, float)):
        league_clearing, clearing_src = float(hist_median), f"median of {hist_count} league bids"
    else:
        league_clearing, clearing_src = None, ""

    priority_mode = waiver_type == 0
    bids: List[Optional[int]] = []
    pcts: List[Optional[float]] = []
    claims: List[bool] = []
    rationales: List[str] = []

    for pos, delta, opp in zip(positions, delta_vorp, opp_index):
        pos_label = pos if isinstance(pos, str) and pos else "player"

        if delta <= 0:
            bids.append(None if priority_mode else 0)
            pcts.append(None if priority_mode else 0.0)
            claims.append(False)
            rationales.append(f"No upgrade over worst rostered {pos_label} (ΔVORP {delta:.1f})")
            continue

        fills_injury = pos in injured_starter_positions
        if fills_injury or (win_pct is not None and win_pct < 0.40):
            urgency = FAAB_URGENCY_HIGH
            urgency_why = "replacing injured starter" if fills_injury else "record below .400"
        elif win_pct is not None and win_pct >= 0.83:
            urgency, urgency_why = FAAB_URGENCY_LOW, "contending, conserve budget"
        else:
            urgency, urgency_why = 1.0, "neutral"

        if priority_mode:
            top_priority = waiver_position is not None and waiver_position <= 6
            claim = (delta > 2.0 and top_priority) or opp >= 75
            claims.append(claim)
            bids.append(None)
            pcts.append(None)
            if claim:
                why = (
                    f"ΔVORP {delta:.1f} with priority #{waiver_position}"
                    if delta > 2.0 and top_priority
                    else f"elite opportunity index {opp:.0f}"
                )
                rationales.append(f"Priority claim: {why}")
            else:
                rationales.append(
                    f"Hold priority: ΔVORP {delta:.1f} (claim bar: ΔVORP > 2.0 with top-6 priority or index >= 75)"
                )
            continue

        if league_clearing is not None:
            bid_clearing, src = league_clearing, clearing_src
        else:
            bid_clearing = 0.12 * b_init if opp >= 70 else 0.05 * b_init
            src = "fallback 12% of budget" if opp >= 70 else "fallback 5% of budget"

        ratio = delta / vorp_max if vorp_max > 0 else 0.0
        conservation = (b_rem / b_init) ** FAAB_GAMMA if b_init > 0 else 1.0
        bid = min(b_rem, max(0, round(bid_clearing * ratio * conservation * urgency)))
        pct = round(bid / b_rem, 3) if b_rem > 0 else 0.0

        bids.append(bid)
        pcts.append(pct)
        claims.append(False)
        rationales.append(
            f"ΔVORP {delta:.1f}; {src} ${bid_clearing:.0f}; urgency x{urgency:.2f} ({urgency_why}); ${bid} of ${b_rem} left"
        )

    out["recommended_faab_bid"] = bids
    out["faab_pct_of_remaining"] = pcts
    out["priority_claim_recommended"] = claims
    out["faab_rationale"] = rationales
    return out
