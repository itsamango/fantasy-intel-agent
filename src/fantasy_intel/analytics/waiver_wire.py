"""Waiver wire scanner for discovering under-the-radar breakouts."""

from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd

from fantasy_intel.analytics.metrics import compute_player_trajectories, estimate_expected_fantasy_points


class WaiverWireEngine:
    """Detects under-the-radar waiver pickups using leading volume metrics."""

    def __init__(self, weekly_df: pd.DataFrame):
        self.raw_df = weekly_df
        # Enrich dataset
        traj = compute_player_trajectories(weekly_df)
        self.df = estimate_expected_fantasy_points(traj)

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
            # Default to latest week
            latest_week = filtered["week"].max()
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

        # Calculate composite Opportunity Score (0-100)
        # Components:
        # 1. Snap share (weight 30)
        # 2. WOPR / Target share (weight 35)
        # 3. Touch volume (weight 20)
        # 4. Snap delta surge (weight 15)
        snap_score = filtered["offense_pct"].clip(0, 1.0) * 30
        wopr_score = filtered["wopr"].clip(0, 0.8) / 0.8 * 35
        touch_score = (filtered["touches"].clip(0, 20) / 20) * 20
        delta_score = (filtered["snap_delta"].clip(0, 0.4) / 0.4) * 15

        filtered["opportunity_index"] = (snap_score + wopr_score + touch_score + delta_score).round(1)

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
            if row["fp_diff"] <= -3.0:
                notes.append("Unlucky TD luck (due for positive regression)")
            return "; ".join(notes) if notes else "Growing weekly snap involvement"

        filtered["catalyst"] = filtered.apply(make_catalyst, axis=1)

        result_cols = [
            "player_display_name",
            "position",
            "recent_team",
            "week",
            "offense_pct",
            "snap_delta",
            "targets",
            "carries",
            "wopr",
            "fantasy_points_ppr",
            "expected_fp_ppr",
            "opportunity_index",
            "catalyst",
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
