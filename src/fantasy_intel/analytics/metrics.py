"""Advanced opportunity and efficiency metrics calculation."""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_player_trajectories(df: pd.DataFrame, min_games: int = 2) -> pd.DataFrame:
    """Computes trends (snap share trend, target share trend, WOPR) across recent weeks."""
    if df.empty:
        return df

    # Ensure required columns exist
    df = df.copy()
    if "target_share" not in df.columns:
        df["target_share"] = 0.0
    if "air_yards_share" not in df.columns:
        df["air_yards_share"] = 0.0
    if "wopr" not in df.columns:
        df["wopr"] = 1.5 * df["target_share"] + 0.7 * df["air_yards_share"]
    if "offense_pct" not in df.columns:
        df["offense_pct"] = 0.0

    df["fantasy_points_ppr"] = df["fantasy_points_ppr"].fillna(0.0)
    df["targets"] = df["targets"].fillna(0)
    df["carries"] = df["carries"].fillna(0)
    df["touches"] = df["carries"] + df["targets"]

    # Sort chronologically
    df = df.sort_values(by=["player_display_name", "season", "week"]).reset_index(drop=True)

    # Compute rolling metrics per player
    grouped = df.groupby(["player_display_name", "season"])

    df["recent_snap_avg"] = grouped["offense_pct"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["prev_snap_avg"] = grouped["offense_pct"].transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean()).fillna(df["offense_pct"])
    df["snap_delta"] = df["offense_pct"] - df["prev_snap_avg"]

    df["recent_target_share"] = grouped["target_share"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["recent_wopr"] = grouped["wopr"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["recent_touches"] = grouped["touches"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["recent_ppr"] = grouped["fantasy_points_ppr"].transform(lambda x: x.rolling(3, min_periods=1).mean())

    return df


def estimate_expected_fantasy_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates baseline Expected Fantasy Points (xFP) based on touches,
    target volume, air yards, and yardage to identify regression candidates.
    """
    if df.empty:
        return df

    df = df.copy()
    if "receiving_air_yards" not in df.columns:
        df["receiving_air_yards"] = 0.0
    if "targets" not in df.columns:
        df["targets"] = 0.0
    if "carries" not in df.columns:
        df["carries"] = 0.0

    exp_pass_pts = df["targets"].fillna(0) * 1.3 + df["receiving_air_yards"].fillna(0) * 0.08
    exp_rush_pts = df["carries"].fillna(0) * 0.68

    df["expected_fp_ppr"] = np.round(exp_pass_pts + exp_rush_pts, 2)
    actual_pts = df.get("fantasy_points_ppr", 0).fillna(0)
    df["fp_diff"] = np.round(actual_pts - df["expected_fp_ppr"], 2)

    return df
