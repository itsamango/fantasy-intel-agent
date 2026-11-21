"""Advanced opportunity and efficiency metrics calculation."""

from __future__ import annotations
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from fantasy_intel.config import (
    FPOE_HALFLIFE_GAMES,
    FPOE_POSITION_MEAN_MIN_XFP,
    FPOE_PRIOR_STRENGTH,
    FPOE_SIGNAL_WINDOW_GAMES,
    POSITION_CV_PRIORS,
    SCORING_HALF_PPR,
    SCORING_PPR,
    SCORING_STANDARD,
    SCORING_TEP,
    SCORING_WEIGHTS,
)

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
    "expected_receptions",
    "pbp_rec_xfp_0ppr",
    "pbp_rush_xfp",
    "pass_attempts",
    "pass_attempts_inside_10",
    "pbp_pass_xfp",
    "target_air_yards",
]

# Sleeper red-zone volume is "inside the 20". For player-weeks without nflverse
# pbp yet (the newest week), the finer goal-line splits the situational xFP
# model needs are imputed as ratio * inside-20 volume. Ratios are re-calibrated
# from matched rows on every call; these defaults apply only when too few
# matched rows exist (fit on 2026 weeks 1-10: nflverse pbp vs Sleeper, ~1.2-1.4k
# inside-20 plays per column).
RZ_IMPUTATION_DEFAULTS = {
    # target column: (sleeper inside-20 column, default ratio)
    "carries_inside_10": ("rz_carries", 0.536),
    "carries_inside_5": ("rz_carries", 0.341),
    "targets_inside_10": ("rz_targets", 0.441),
    "end_zone_targets": ("rz_targets", 0.518),
    "pass_attempts_inside_10": ("rz_pass_attempts", 0.447),
}
RZ_CALIBRATION_MIN_VOLUME = 200.0


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


def compute_team_shares(df: pd.DataFrame) -> pd.DataFrame:
    """(Re)computes target_share, air_yards_share, and WOPR from team-week totals.

    Team totals are summed over the frame's rows per (recent_team, season, week),
    so the frame must hold every skill player's line for each team-week. Rows
    without a team get 0 shares. WOPR = 1.5 * target_share + 0.7 * air_yards_share.
    """
    if df.empty or not {"recent_team", "season", "week"}.issubset(df.columns):
        return df
    df = df.copy()
    keys = [df["recent_team"], df["season"], df["week"]]
    targets = pd.to_numeric(df.get("targets", 0.0), errors="coerce").fillna(0.0)
    air = pd.to_numeric(df.get("receiving_air_yards", 0.0), errors="coerce").fillna(0.0)
    team_targets = targets.groupby(keys).transform("sum")
    team_air = air.clip(lower=0.0).groupby(keys).transform("sum")
    df["target_share"] = (targets / team_targets.where(team_targets > 0)).fillna(0.0).round(4)
    df["air_yards_share"] = (air / team_air.where(team_air > 0)).fillna(0.0).round(4)
    df["wopr"] = (1.5 * df["target_share"] + 0.7 * df["air_yards_share"]).round(4)
    return df


def _compute_qb_passing_xfp(df: pd.DataFrame, pass_td_weight: float = 4.0) -> pd.Series:
    """Computes expected passing fantasy points for Quarterbacks.

    Uses continuous play-level PBP pass xFP (`pbp_pass_xfp`) when available —
    produced by DataLoader.get_pbp_opportunity_features. Falls back to a
    box-score approximation using `attempts`/`pass_attempts`, `passing_yards`,
    `pass_attempts_inside_10`, and `sacks` when PBP columns are absent.
    Returns 0.0 for rows with zero passing volume (non-QBs, RBs, WRs, TEs).
    """
    td_scale = pass_td_weight / 4.0

    # Per row: play-level PBP xFP where that player-week matched pbp, box-score
    # fallback elsewhere (e.g. the newest week before nflverse publishes it).
    if "pbp_pass_xfp" in df.columns:
        pbp_pass = pd.to_numeric(df["pbp_pass_xfp"], errors="coerce")
        if "pbp_matched" in df.columns:
            use_pbp = df["pbp_matched"].astype(bool) & pbp_pass.notna()
        else:
            use_pbp = pbp_pass.fillna(0.0) > 0
    else:
        pbp_pass = pd.Series(np.nan, index=df.index)
        use_pbp = pd.Series(False, index=df.index)

    # Fallback: box-score passing volume. Prefer box-score "attempts" (always
    # present on weekly frames) over the pbp-derived pass_attempts, which is 0
    # on unmatched rows.
    pass_att = pd.Series(0.0, index=df.index)
    for col in ("attempts", "pass_attempts"):
        if col in df.columns:
            pass_att = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if (pass_att > 0).any():
                break

    pass_yds = (
        pd.to_numeric(df["passing_yards"], errors="coerce").fillna(0.0)
        if "passing_yards" in df.columns
        else pass_att * 7.1
    )
    rz_pass_att = (
        pd.to_numeric(df["pass_attempts_inside_10"], errors="coerce").fillna(0.0)
        if "pass_attempts_inside_10" in df.columns
        else pass_att * 0.11
    )
    sacks = (
        pd.to_numeric(df["sacks"], errors="coerce").fillna(0.0)
        if "sacks" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    reg_pass_att = (pass_att - rz_pass_att).clip(lower=0.0)
    exp_pass = (
        reg_pass_att * 0.34
        + rz_pass_att * (0.95 * td_scale)
        + pass_yds.clip(lower=0.0) * 0.015
        - sacks.clip(lower=0.0) * 0.35
    ).clip(lower=0.0)
    return pd.Series(
        np.where(use_pbp, pbp_pass.fillna(0.0) * td_scale, exp_pass), index=df.index
    )


def _actual_points(df: pd.DataFrame, scoring: str) -> pd.Series:
    """Actual fantasy points under the league's scoring system.

    Uses the provider's per-system point columns when present, else derives
    from PPR points: ppr - (1 - rec_weight) * receptions, and TEP adds +0.5
    per reception for TEs only.
    """
    ppr = pd.to_numeric(df.get("fantasy_points_ppr", 0.0), errors="coerce")
    ppr = pd.Series(ppr, index=df.index).fillna(0.0)
    receptions = pd.to_numeric(df["receptions"], errors="coerce").fillna(0.0) if "receptions" in df.columns else None
    if scoring == SCORING_HALF_PPR:
        if "fantasy_points_half_ppr" in df.columns:
            return pd.to_numeric(df["fantasy_points_half_ppr"], errors="coerce").fillna(0.0)
        return ppr - 0.5 * receptions if receptions is not None else ppr
    if scoring == SCORING_STANDARD:
        if "fantasy_points" in df.columns:
            return pd.to_numeric(df["fantasy_points"], errors="coerce").fillna(0.0)
        return ppr - receptions if receptions is not None else ppr
    if scoring == SCORING_TEP and receptions is not None and "position" in df.columns:
        return ppr + np.where(df["position"].astype(str).str.upper() == "TE", 0.5 * receptions, 0.0)
    return ppr


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


def estimate_expected_fantasy_points(
    df: pd.DataFrame,
    scoring: str = SCORING_PPR,
    fpoe_history: Optional[pd.DataFrame] = None,
    with_baseline: bool = True,
) -> pd.DataFrame:
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
    if scoring == SCORING_TEP and "position" in df.columns:
        # TE premium applies to tight ends only; everyone else is valued at PPR
        rec_weight = pd.Series(
            np.where(
                df["position"].astype(str).str.upper() == "TE",
                weights["rec"],
                SCORING_WEIGHTS[SCORING_PPR]["rec"],
            ),
            index=df.index,
        )
    pass_td_weight = weights.get("pass_td", 4.0)

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
        # Prefer play-level receiving xFP (completion probability, expected YAC,
        # target-yardline TD probability) on rows that matched pbp. Validated on
        # 2026 wks 1-10: corr 0.822 vs 0.791 for the bucket model, bias -0.30 vs
        # +0.63 pts/game. Rushing stays on the buckets (bias -0.11 vs -0.70).
        if {"pbp_matched", "pbp_rec_xfp_0ppr", "expected_receptions"}.issubset(df.columns):
            pbp_rec = (
                pd.to_numeric(df["pbp_rec_xfp_0ppr"], errors="coerce").fillna(0.0)
                + rec_weight * pd.to_numeric(df["expected_receptions"], errors="coerce").fillna(0.0)
            )
            exp_rec_pts = pd.Series(
                np.where(df["pbp_matched"].astype(bool), pbp_rec, exp_rec_pts), index=df.index
            )
        df["xfp_model"] = "situational"
    else:
        logger.info("PBP situational columns unavailable; using legacy linear xFP model.")
        exp_rec_pts = targets * 1.3 + air_yards * 0.08
        exp_rush_pts = carries * 0.68
        df["xfp_model"] = "linear_fallback"

    exp_pass_pts = _compute_qb_passing_xfp(df, pass_td_weight=pass_td_weight)

    df["expected_fp_ppr"] = np.round(exp_rec_pts + exp_rush_pts + exp_pass_pts, 2)
    # Compare like with like: xFP is in league scoring, so actual points must be too
    actual_pts = _actual_points(df, scoring) if "fantasy_points_ppr" in df.columns else 0.0
    df["fantasy_points_league"] = actual_pts
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
        if with_baseline:
            history = None
            if fpoe_history is not None and not fpoe_history.empty:
                history = estimate_expected_fantasy_points(
                    fpoe_history, scoring=scoring, with_baseline=False
                )
            df = add_fpoe_baseline(df, history=history)

    return df


def add_fpoe_baseline(
    df: pd.DataFrame,
    history: Optional[pd.DataFrame] = None,
    prior_strength: Optional[Dict[str, float]] = None,
    halflife_games: float = FPOE_HALFLIFE_GAMES,
    window: int = FPOE_SIGNAL_WINDOW_GAMES,
) -> pd.DataFrame:
    """Per-player efficiency baseline: Empirical-Bayes shrunk FPOE per expected point.

    FPOE is only partly luck: some players (elite WRs, rushing QBs, explosive
    RBs) beat xFP year after year. For each row, in game order per player
    (seeded from `history`, earlier seasons' rows with expected_fp_ppr /
    fp_diff, then the frame itself):

        S_F, S_X = recency-weighted sums of FPOE and xFP (per-game halflife)
        eff      = (S_F + k_pos * m_pos) / (S_X + k_pos)

    m_pos is the position's pooled FPOE / xFP (also absorbing any position-level
    xFP model bias) and k_pos the prior strength from FPOE_PRIOR_STRENGTH.

    Adds:
      fpoe_eff            efficiency through this game (projection input)
      fpoe_eff_pre        efficiency before the last `window` games (signal baseline)
      fpoe_eff_weight     S_X / (S_X + k): share of eff that is player-specific
      proj_ppg            rolling_xfp_3w * (1 + fpoe_eff): skill-adjusted projection
      expected_fpoe_3w    rolling_xfp_3w * fpoe_eff_pre: FPOE this player normally earns
      fpoe_excess_3w      rolling_fpoe_3w - expected_fpoe_3w: over/under-performance
                          beyond his own baseline (~89% regressed away out of
                          sample on 2026 wks 7-10, vs his baseline's r=0.35)
    """
    needed = {"expected_fp_ppr", "fp_diff", "position", "season", "week"}
    if df.empty or not needed.issubset(df.columns):
        return df
    k_by_pos = prior_strength or FPOE_PRIOR_STRENGTH
    key = _player_group_key(df)

    frame = df.copy()
    frame["_is_history"] = False
    if history is not None and not history.empty and needed.issubset(history.columns):
        hist = history.copy()
        hist_key = _player_group_key(hist)
        if hist_key != key and key in hist.columns:
            hist_key = key
        hist = hist.rename(columns={hist_key: key}) if hist_key != key else hist
        hist[key] = _normalize_ids(hist[key]) if key != "player_display_name" else hist[key]
        frame[key] = _normalize_ids(frame[key]) if key != "player_display_name" else frame[key]
        # History strictly precedes the frame: drop overlapping (season, week) rows
        frame_seasons = set(frame["season"].unique())
        hist = hist[~hist["season"].isin(frame_seasons)]
        hist = hist[[key, "season", "week", "position", "expected_fp_ppr", "fp_diff"]].copy()
        hist["_is_history"] = True
        frame = pd.concat([hist, frame], ignore_index=True, sort=False)

    frame = frame.sort_values([key, "season", "week"], kind="stable").reset_index(drop=True)
    x = pd.to_numeric(frame["expected_fp_ppr"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
    f = pd.to_numeric(frame["fp_diff"], errors="coerce").fillna(0.0).to_numpy()
    active = x > 0
    f = np.where(active, f, 0.0)
    pos = frame["position"].astype(str).str.upper().to_numpy()

    # Position prior means from all active rows (history + frame)
    pos_mean: Dict[str, float] = {}
    for p in np.unique(pos):
        sel = (pos == p) & active
        sx = float(x[sel].sum())
        pos_mean[p] = float(f[sel].sum() / sx) if sx >= FPOE_POSITION_MEAN_MIN_XFP else 0.0

    decay = 0.5 ** (1.0 / halflife_games)
    n = len(frame)
    eff = np.zeros(n)
    eff_pre = np.zeros(n)
    weight = np.zeros(n)
    groups = frame.groupby(key, sort=False).indices
    for idx in groups.values():
        idx = np.sort(idx)
        p = pos[idx[-1]]  # most recent listed position
        k = float(k_by_pos.get(p, max(k_by_pos.values())))
        m = pos_mean.get(p, 0.0)
        s_f = s_x = 0.0
        hist_eff: List[float] = []
        for j, i in enumerate(idx):
            if active[i]:
                s_f = decay * s_f + f[i]
                s_x = decay * s_x + x[i]
            e = (s_f + k * m) / (s_x + k)
            hist_eff.append(e)
            eff[i] = e
            weight[i] = s_x / (s_x + k)
            eff_pre[i] = hist_eff[j - window] if j - window >= 0 else m

    frame["fpoe_eff"] = np.round(eff, 4)
    frame["fpoe_eff_pre"] = np.round(eff_pre, 4)
    frame["fpoe_eff_weight"] = np.round(weight, 3)
    frame = frame[~frame["_is_history"].astype(bool)].drop(columns="_is_history").reset_index(drop=True)

    if "rolling_xfp_3w" in frame.columns:
        rx = pd.to_numeric(frame["rolling_xfp_3w"], errors="coerce").fillna(0.0)
        frame["proj_ppg"] = (rx * (1.0 + frame["fpoe_eff"])).clip(lower=0.0).round(2)
        frame["expected_fpoe_3w"] = (rx * frame["fpoe_eff_pre"]).round(2)
        if "rolling_fpoe_3w" in frame.columns:
            frame["fpoe_excess_3w"] = (
                pd.to_numeric(frame["rolling_fpoe_3w"], errors="coerce").fillna(0.0)
                - frame["expected_fpoe_3w"]
            ).round(2)
    frame.attrs.update(df.attrs)
    return frame


def fit_fpoe_prior_strength(df: pd.DataFrame, min_half_xfp: float = 60.0) -> Dict[str, Dict[str, float]]:
    """Fits FPOE_PRIOR_STRENGTH by split-half reliability (frame needs expected_fp_ppr / fp_diff).

    Each player's active games are split odd/even; efficiency (sum FPOE /
    sum xFP) is computed per half for players with >= min_half_xfp in both.
    The xFP-weighted correlation r between halves is the reliability of a
    half-sample of median size n, so k = n (1 - r) / r.
    """
    key = _player_group_key(df)
    d = df[pd.to_numeric(df["expected_fp_ppr"], errors="coerce").fillna(0) > 0].copy()
    d = d.sort_values([key, "season", "week"], kind="stable")
    d["_x"] = d["expected_fp_ppr"].astype(float)
    d["_f"] = d["fp_diff"].astype(float)
    d["_half"] = d.groupby(key).cumcount() % 2
    out: Dict[str, Dict[str, float]] = {}
    for pos, p in d.groupby(d["position"].astype(str).str.upper()):
        g = p.groupby([key, "_half"]).agg(F=("_f", "sum"), X=("_x", "sum")).unstack().dropna()
        if g.empty:
            continue
        g = g[(g[("X", 0)] >= min_half_xfp) & (g[("X", 1)] >= min_half_xfp)]
        if len(g) < 10:
            continue
        e0, e1 = g[("F", 0)] / g[("X", 0)], g[("F", 1)] / g[("X", 1)]
        w = np.minimum(g[("X", 0)], g[("X", 1)])
        a0, a1 = e0 - np.average(e0, weights=w), e1 - np.average(e1, weights=w)
        r = float(np.average(a0 * a1, weights=w) / np.sqrt(np.average(a0**2, weights=w) * np.average(a1**2, weights=w)))
        n = float(np.median(w))
        out[pos] = {"r": r, "n_half_xfp": n, "k": n * (1 - r) / r if r > 0 else float("inf"), "players": float(len(g))}
    return out


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
        if "rz_targets" in df.columns or "rz_carries" in df.columns:
            # Sleeper weekly frame: still derive goal-line splits from inside-20 volume
            logger.warning("PBP opportunity features unavailable; imputing red-zone splits from Sleeper.")
            attrs = dict(df.attrs)
            df = df.copy()
            df["pbp_matched"] = False
            df = _impute_unmatched_red_zone(df)
            df.attrs.update(attrs)
            return df
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

    sleeper_sourced = df.attrs.get("source") == "sleeper_weekly" or "rz_targets" in df.columns
    attrs = dict(df.attrs)

    # Drop any stale feature columns so re-attaching cannot create _x/_y duplicates
    df = df.drop(columns=[c for c in PBP_FEATURE_COLUMNS + ["pbp_matched"] if c in df.columns])
    feats = player_feats.rename(columns={"player_id": "gsis_id"})
    feats["gsis_id"] = _normalize_ids(feats["gsis_id"])
    feats = feats.drop(columns=["pbp_schema_version"], errors="ignore")
    feats = feats.drop_duplicates(subset=["gsis_id", "season", "week"])
    df = pd.merge(df, feats, on=["gsis_id", "season", "week"], how="left", indicator="_pbp_merge")
    df["pbp_matched"] = df.pop("_pbp_merge").eq("both")

    for col in PBP_FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if sleeper_sourced:
        df = _impute_unmatched_red_zone(df)
        # Sleeper air yards are completions-only; use true target air yards where pbp exists
        if "target_air_yards" in df.columns and "receiving_air_yards" in df.columns:
            df["receiving_air_yards"] = np.where(
                df["pbp_matched"], df["target_air_yards"], df["receiving_air_yards"]
            )
            df = compute_team_shares(df)

    df.attrs.update(attrs)
    return df


def _impute_unmatched_red_zone(df: pd.DataFrame) -> pd.DataFrame:
    """Fills goal-line splits for rows without pbp from Sleeper inside-20 volume.

    ratio = sum(pbp split) / sum(Sleeper inside-20) over matched rows when that
    volume is >= RZ_CALIBRATION_MIN_VOLUME, else RZ_IMPUTATION_DEFAULTS.
    Marks imputed rows with pbp_imputed=True.
    """
    unmatched = ~df["pbp_matched"]
    df["pbp_imputed"] = False
    if not unmatched.any():
        return df
    matched = df["pbp_matched"]
    imputed_any = pd.Series(False, index=df.index)
    for target, (source, default_ratio) in RZ_IMPUTATION_DEFAULTS.items():
        if source not in df.columns:
            continue
        src = pd.to_numeric(df[source], errors="coerce").fillna(0.0)
        ratio = default_ratio
        if target in df.columns:
            denom = float(src[matched].sum())
            if denom >= RZ_CALIBRATION_MIN_VOLUME:
                ratio = float(df.loc[matched, target].sum()) / denom
        df.loc[unmatched, target] = (src[unmatched] * ratio).round(3)
        imputed_any |= unmatched & (src > 0)
    if "carries_inside_10" in df.columns and "targets" in df.columns:
        df.loc[unmatched, "high_value_touches"] = (
            pd.to_numeric(df.loc[unmatched, "targets"], errors="coerce").fillna(0.0)
            + df.loc[unmatched, "carries_inside_10"]
        )
    df["pbp_imputed"] = imputed_any
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

    if "proj_ppg" in df.columns:
        # Skill-adjusted projection (rolling xFP x player's shrunk efficiency)
        mu = df["proj_ppg"]
    elif "rolling_xfp_3w" in df.columns:
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
