"""Trade evaluator: Buy-low and sell-high regression candidates."""

from __future__ import annotations
from typing import List, Optional
import pandas as pd

from fantasy_intel.analytics.metrics import compute_player_trajectories, estimate_expected_fantasy_points


class TradeAnalyzer:
    """Identifies players whose fantasy output deviates sharply from true opportunity."""

    def __init__(self, weekly_df: pd.DataFrame):
        traj = compute_player_trajectories(weekly_df)
        self.df = estimate_expected_fantasy_points(traj)

    def get_buy_low_candidates(
        self,
        week: Optional[int] = None,
        season: Optional[int] = None,
        positions: Optional[List[str]] = None,
        min_snaps: float = 0.50,
        limit: int = 10,
    ) -> pd.DataFrame:
        """
        Players with strong usage/volume whose fantasy points were heavily suppressed
        by negative variance or missed touchdowns.
        """
        if self.df.empty:
            return pd.DataFrame()

        filtered = self.df.copy()
        if season is not None:
            filtered = filtered[filtered["season"] == season]
        if week is not None:
            filtered = filtered[filtered["week"] == week]
        else:
            filtered = filtered[filtered["week"] == filtered["week"].max()]

        if positions:
            filtered = filtered[filtered["position"].isin(positions)]
        else:
            filtered = filtered[filtered["position"].isin(["RB", "WR", "TE"])]

        # Criteria: Good snaps, good expected fantasy points, but underperformed expected points
        buy_low = filtered[
            (filtered["offense_pct"] >= min_snaps) &
            (filtered["expected_fp_ppr"] >= 10.0) &
            (filtered["fp_diff"] <= -3.0)
        ].copy()

        buy_low["rationale"] = buy_low.apply(
            lambda r: f"Underperformed xFP by {abs(r['fp_diff']):.1f} pts despite {int(r['offense_pct']*100)}% snap share and {int(r['touches'])} touches/targets",
            axis=1,
        )

        cols = [
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
        return buy_low[[c for c in cols if c in buy_low.columns]].sort_values(by="fp_diff").head(limit).reset_index(drop=True)

    def get_sell_high_candidates(
        self,
        week: Optional[int] = None,
        season: Optional[int] = None,
        positions: Optional[List[str]] = None,
        limit: int = 10,
    ) -> pd.DataFrame:
        """
        Players with bloated fantasy points driven by unsustainable TD rates or fragile low volume.
        """
        if self.df.empty:
            return pd.DataFrame()

        filtered = self.df.copy()
        if season is not None:
            filtered = filtered[filtered["season"] == season]
        if week is not None:
            filtered = filtered[filtered["week"] == week]
        else:
            filtered = filtered[filtered["week"] == filtered["week"].max()]

        if positions:
            filtered = filtered[filtered["position"].isin(positions)]
        else:
            filtered = filtered[filtered["position"].isin(["RB", "WR", "TE"])]

        # Criteria: High fantasy score, but low underlying opportunity
        sell_high = filtered[
            (filtered["fantasy_points_ppr"] >= 15.0) &
            (filtered["fp_diff"] >= 5.0)
        ].copy()

        sell_high["rationale"] = sell_high.apply(
            lambda r: f"Outperformed true usage by +{r['fp_diff']:.1f} pts on limited volume ({int(r['touches'])} touches, {int(r['offense_pct']*100)}% snaps)",
            axis=1,
        )

        cols = [
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
        return sell_high[[c for c in cols if c in sell_high.columns]].sort_values(by="fp_diff", ascending=False).head(limit).reset_index(drop=True)
