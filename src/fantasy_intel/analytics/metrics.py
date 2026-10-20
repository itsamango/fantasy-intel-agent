"""Advanced opportunity and efficiency metrics calculation."""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy import stats

from fantasy_intel.config import SCORING_PPR, SCORING_WEIGHTS

logger = logging.getLogger(__name__)

# PBP-derived opportunity columns required by the situational xFP model
SITUATIONAL_XFP_COLUMNS = ["carries_inside_10", "carries_inside_5", "end_zone_targets"]

# Player-level PBP opportunity feature columns produced by DataLoader.get_pbp_opportunity_features
PBP_FEATURE_COLUMNS = [
    "carries_inside_10",
    "carries_inside_5",
    "targets_inside_10",
    "end_zone_targets",
    "high_value_touches",
]


def _player_group_key(df: pd.DataFrame) -> str:
    """Returns the stable per-player grouping column (player_id / gsis_id, display name as fallback)."""
    for col in ("player_id", "gsis_id"):
        if col in df.columns:
            return col
    return "player_display_name"


def _normalize_ids(series: pd.Series) -> pd.Series:
    """Normalizes an ID column to clean strings (e.g. 1234.0 -> "1234", NaN -> None)."""
    s = series.where(series.notna(), None)
    return s.map(
        lambda v: str(int(v))
        if isinstance(v, (int, float)) and not isinstance(v, bool)
        else (str(v) if v is not None else None)
    )


def _sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts by stable player id, season, week so rolling/ewm stats follow game order."""
    return df.sort_values(
        by=[_player_group_key(df), "season", "week"], kind="stable"
    ).reset_index(drop=True)


def compute_player_trajectories(df: pd.DataFrame, min_games: int = 2) -> pd.DataFrame:
    """Computes trends (snap share trend, target share trend, WOPR) across recent weeks.

    Rows are grouped by the stable player id (player_id / gsis_id) so two players
    sharing a display name cannot collide; player_display_name is carried along
    only for display.
    """
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

    # Diagnostic efficiency / leverage columns (row level, no rolling needed)
    if "receiving_air_yards" in df.columns and "receiving_yards" in df.columns:
        if "racr" not in df.columns:
            df["racr"] = df["receiving_yards"] / df["receiving_air_yards"].clip(lower=1.0)
        df["unrealized_air_yards"] = (
            df["receiving_air_yards"].fillna(0) - df["receiving_yards"].fillna(0)
        ).clip(lower=0)
    if "carries_inside_10" in df.columns:
        df["hvt"] = df["targets"] + df["carries_inside_10"].fillna(0)

    # Sort chronologically
    df = _sort_chronologically(df)

    # Compute rolling metrics per player (grouped by stable player id, not display name)
    player_key = _player_group_key(df)
    grouped = df.groupby([player_key, "season"], sort=False)

    df["recent_snap_avg"] = grouped["offense_pct"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["prev_snap_avg"] = grouped["offense_pct"].transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean()).fillna(df["offense_pct"])
    df["snap_delta"] = df["offense_pct"] - df["prev_snap_avg"]

    df["recent_target_share"] = grouped["target_share"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["recent_wopr"] = grouped["wopr"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["recent_touches"] = grouped["touches"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["recent_ppr"] = grouped["fantasy_points_ppr"].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # High-value-touch share of the team's weekly total (team totals approximated
    # from this frame, so it requires the full weekly player set per team)
    if "hvt" in df.columns and "recent_team" in df.columns:
        team_hvt = df.groupby(["recent_team", "season", "week"], sort=False)["hvt"].transform("sum")
        df["hvt_share"] = (df["hvt"] / team_hvt.where(team_hvt > 0)).fillna(0.0).round(3)

    return df


def estimate_expected_fantasy_points(df: pd.DataFrame, scoring: str = SCORING_PPR) -> pd.DataFrame:
    """
    Estimates Expected Fantasy Points (xFP) from volume and field-position context.

    Primary model: situational xFP stratified by goal-line proximity (regular /
    inside-10 / inside-5 carries) and end-zone targets, valued by the league
    scoring system's reception weight. Falls back to the legacy linear per-touch
    model when the PBP opportunity columns are absent or empty. Also emits the
    FPOE alias of fp_diff and rolling 3-week exponentially weighted xFP / FPOE.
    """
    if df.empty:
        return df

    df = df.copy()
    weights = SCORING_WEIGHTS.get(scoring)
    if weights is None:
        logger.warning(f"Unknown scoring system '{scoring}'; defaulting to '{SCORING_PPR}' weights.")
        weights = SCORING_WEIGHTS[SCORING_PPR]
    rec_weight = weights["rec"]

    if "receiving_air_yards" not in df.columns:
        df["receiving_air_yards"] = 0.0
    if "targets" not in df.columns:
        df["targets"] = 0.0
    if "carries" not in df.columns:
        df["carries"] = 0.0

    targets = df["targets"].fillna(0)
    carries = df["carries"].fillna(0)
    air_yards = df["receiving_air_yards"].fillna(0)

    has_situational = all(c in df.columns for c in SITUATIONAL_XFP_COLUMNS) and df[
        SITUATIONAL_XFP_COLUMNS
    ].notna().any().any()

    if has_situational:
        carries_inside_10 = df["carries_inside_10"].fillna(0)
        carries_inside_5 = df["carries_inside_5"].fillna(0)
        end_zone_targets = df["end_zone_targets"].fillna(0)

        # Rushing xFP stratified by yard-line proximity
        regular_carries = (carries - carries_inside_10).clip(lower=0)
        carries_10_to_6 = (carries_inside_10 - carries_inside_5).clip(lower=0)
        goal_line_carries = carries_inside_5.clip(lower=0)
        exp_rush_pts = (
            regular_carries * 0.45
            + carries_10_to_6 * 1.45
            + goal_line_carries * 2.65
        )

        # Receiving xFP: catch-rate-weighted receptions + air yardage + end-zone target premium
        regular_targets = (targets - end_zone_targets).clip(lower=0)
        exp_rec_pts = (
            regular_targets * (0.65 * rec_weight + 0.52)
            + air_yards.clip(lower=0) * 0.075
            + end_zone_targets.clip(lower=0) * (0.42 * rec_weight + 2.15)
        )
        df["xfp_model"] = "situational"
    else:
        logger.info("PBP situational columns unavailable; using legacy linear xFP model.")
        exp_rec_pts = targets * 1.3 + air_yards * 0.08
        exp_rush_pts = carries * 0.68
        df["xfp_model"] = "linear_fallback"

    df["expected_fp_ppr"] = np.round(exp_rec_pts + exp_rush_pts, 2)
    actual_pts = df["fantasy_points_ppr"].fillna(0) if "fantasy_points_ppr" in df.columns else 0.0
    df["fp_diff"] = np.round(actual_pts - df["expected_fp_ppr"], 2)
    df["fpoe"] = df["fp_diff"]

    # Rolling 3-week exponentially weighted xFP / FPOE per player
    if "season" in df.columns and "week" in df.columns:
        df = _sort_chronologically(df)
        grouped = df.groupby([_player_group_key(df), "season"], sort=False)
        df["rolling_xfp_3w"] = (
            grouped["expected_fp_ppr"].transform(lambda x: x.ewm(span=3, min_periods=1).mean()).round(2)
        )
        df["rolling_fpoe_3w"] = (
            grouped["fp_diff"].transform(lambda x: x.ewm(span=3, min_periods=1).mean()).round(2)
        )

    return df


def attach_pbp_features(df: pd.DataFrame, loader, seasons) -> pd.DataFrame:
    """Merges PBP player-level opportunity features onto a weekly-stats frame.

    Joins on (gsis_id, season, week). Frames keyed by nflverse gsis ids (the
    weekly player_id column) join directly; frames keyed by Sleeper player ids
    (e.g. from get_sleeper_season_data) are detected via the ID crosswalk and
    mapped sleeper_id -> gsis_id first, so both data paths get situational xFP.
    """
    if df.empty:
        return df

    features = loader.get_pbp_opportunity_features(seasons=seasons)
    player_feats = features.get("player") if isinstance(features, dict) else features
    if player_feats is None or player_feats.empty:
        logger.warning("PBP opportunity features unavailable; skipping enrichment.")
        return df

    if "player_id" not in df.columns:
        logger.warning("Frame has no player_id column; cannot attach PBP features.")
        return df

    df = df.copy()
    df["gsis_id"] = _normalize_ids(df["player_id"])

    ids_df = loader.get_player_ids()
    if not ids_df.empty and "sleeper_id" in ids_df.columns and "gsis_id" in ids_df.columns:
        frame_ids = set(df["gsis_id"].dropna())
        gsis_overlap = len(frame_ids & set(ids_df["gsis_id"].dropna()))
        sleeper_overlap = len(frame_ids & set(ids_df["sleeper_id"].dropna()))
        if sleeper_overlap > gsis_overlap:
            # Frame is keyed by Sleeper ids: map sleeper_id -> gsis_id
            id_map = dict(zip(ids_df["sleeper_id"], ids_df["gsis_id"]))
            df["gsis_id"] = df["gsis_id"].map(id_map)

    # Drop any stale feature columns so re-attaching cannot create _x/_y duplicates
    df = df.drop(columns=[c for c in PBP_FEATURE_COLUMNS if c in df.columns])

    feats = player_feats.rename(columns={"player_id": "gsis_id"})
    feats["gsis_id"] = _normalize_ids(feats["gsis_id"])
    df = pd.merge(df, feats, on=["gsis_id", "season", "week"], how="left")

    for col in PBP_FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def add_projection_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Adds weekly floor / median / ceiling projections (P15 / P50 / P85).

    Fits a zero-bounded log-normal to each player's rolling 3-week xFP (mu) and
    trailing weekly fantasy point standard deviation (sigma, defaulted to
    0.42 * mu when fewer than 3 weekly samples exist or the std is NaN/0).
    Players with mu <= 0 (bench/zeros) are floored at 0.
    """
    if df.empty:
        return df

    if "rolling_xfp_3w" not in df.columns and "expected_fp_ppr" not in df.columns:
        logger.warning("No xFP column available; cannot add projection distributions.")
        return df

    df = df.copy()
    if "season" in df.columns and "week" in df.columns:
        df = _sort_chronologically(df)
        grouped = df.groupby([_player_group_key(df), "season"], sort=False)
        if "fantasy_points_ppr" in df.columns:
            trailing_std = grouped["fantasy_points_ppr"].transform(
                lambda x: x.rolling(3, min_periods=1).std()
            )
            n_samples = grouped.cumcount() + 1
        else:
            trailing_std = pd.Series(np.nan, index=df.index)
            n_samples = pd.Series(1, index=df.index)
    else:
        trailing_std = pd.Series(np.nan, index=df.index)
        n_samples = pd.Series(1, index=df.index)

    if "rolling_xfp_3w" in df.columns:
        mu = df["rolling_xfp_3w"]
    else:
        logger.warning("rolling_xfp_3w missing; parameterizing distributions from weekly expected_fp_ppr.")
        mu = df["expected_fp_ppr"]
    mu = mu.astype(float).fillna(0.0)

    # Default sigma = 0.42 * mu unless we have 3+ weeks of real variance
    sigma = trailing_std.where((n_samples >= 3) & trailing_std.notna() & (trailing_std > 0))
    sigma = sigma.fillna(0.42 * mu.clip(lower=0))

    mu_arr = mu.to_numpy(dtype=float)
    sigma_arr = sigma.to_numpy(dtype=float)
    floor = np.zeros(len(df))
    median = np.zeros(len(df))
    ceiling = np.zeros(len(df))

    valid = np.isfinite(mu_arr) & (mu_arr > 0) & np.isfinite(sigma_arr) & (sigma_arr > 0)
    if valid.any():
        m = mu_arr[valid]
        s = sigma_arr[valid]
        # Moment-matched log-normal: E[X] = mu, SD[X] = sigma
        log_var = np.log1p((s / m) ** 2)
        log_sd = np.sqrt(log_var)
        scale = np.exp(np.log(m) - log_var / 2)
        floor[valid] = stats.lognorm.ppf(0.15, s=log_sd, scale=scale)
        median[valid] = stats.lognorm.ppf(0.50, s=log_sd, scale=scale)
        ceiling[valid] = stats.lognorm.ppf(0.85, s=log_sd, scale=scale)

    df["proj_floor_p15"] = np.round(np.clip(floor, 0, None), 2)
    df["proj_median_p50"] = np.round(np.clip(median, 0, None), 2)
    df["proj_ceiling_p85"] = np.round(np.clip(ceiling, 0, None), 2)

    return df
