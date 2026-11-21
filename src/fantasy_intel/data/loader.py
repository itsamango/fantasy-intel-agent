"""Data loader and cache manager for NFL fantasy statistics."""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import nfl_data_py as nfl

from fantasy_intel.config import CACHE_DIR, DEFAULT_SEASON

logger = logging.getLogger(__name__)

# Depth-chart and injury-report caches are season-keyed with a 24h TTL,
# mirroring the Sleeper players cache in sleeper.py. A cache is only served
# when its frame actually contains rows for the requested season, so a stale
# season's depth chart can never be mixed into a current-season roster.
SEASONAL_CACHE_TTL_HOURS = 24.0

# nflverse pbp columns needed for opportunity, play-level xFP, and defensive EPA aggregation
PBP_FEATURE_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "play_id",
    "epa",
    "posteam",
    "defteam",
    "qb_dropback",
    "qb_kneel",
    "qb_spike",
    "qb_scramble",
    "two_point_attempt",
    "rush_attempt",
    "pass_attempt",
    "sack",
    "passer_player_id",
    "rusher_player_id",
    "receiver_player_id",
    "yardline_100",
    "air_yards",
    "cp",
    "xyac_mean_yardage",
]

# PBP-derived caches for the live season are refetched after this many hours
# (nflverse republishes pbp nightly in-season). Past seasons never expire.
LIVE_PBP_CACHE_TTL_HOURS = 24.0
LIVE_SCHEDULE_CACHE_TTL_HOURS = 12.0

# Bump when the meaning of a cached PBP feature changes (not just its columns)
# so stale caches are rebuilt. v2: target_air_yards added, sack deduction removed.
PBP_FEATURE_SCHEMA_VERSION = 2

# Columns the cached player-level PBP feature frame must carry. A cache written
# by an older schema is treated as stale and rebuilt.
PBP_PLAYER_FEATURE_SCHEMA = (
    "carries_inside_10", "carries_inside_5", "targets_inside_10", "end_zone_targets",
    "high_value_touches", "expected_receptions", "pbp_rec_xfp_0ppr", "pbp_rush_xfp",
    "pass_attempts", "pass_attempts_inside_10", "pbp_pass_xfp", "target_air_yards",
)

# Sleeper weekly stat key -> nflverse-compatible weekly column name
SLEEPER_WEEKLY_STAT_MAP = {
    "pass_cmp": "completions",
    "pass_att": "attempts",
    "pass_yd": "passing_yards",
    "pass_td": "passing_tds",
    "pass_int": "interceptions",
    "pass_sack": "sacks",
    "pass_air_yd": "passing_air_yards",
    "pass_rz_att": "rz_pass_attempts",
    "rush_att": "carries",
    "rush_yd": "rushing_yards",
    "rush_td": "rushing_tds",
    "rush_rz_att": "rz_carries",
    "rec": "receptions",
    "rec_tgt": "targets",
    "rec_yd": "receiving_yards",
    "rec_td": "receiving_tds",
    "rec_air_yd": "receiving_air_yards",
    "rec_yar": "receiving_yards_after_catch",
    "rec_rz_tgt": "rz_targets",
    "pts_ppr": "fantasy_points_ppr",
    "pts_half_ppr": "fantasy_points_half_ppr",
    "pts_std": "fantasy_points",
    "off_snp": "offense_snaps",
    "tm_off_snp": "team_offense_snaps",
}

# A week counts as complete when at least this many teams have stat lines
# (the most byes in any week is 6 -> 26 teams) unless the schedule says fewer.
MIN_TEAMS_FOR_COMPLETE_WEEK = 26

SCHEDULE_COLUMNS = [
    "season",
    "week",
    "game_type",
    "gameday",
    "weekday",
    "gametime",
    "home_team",
    "away_team",
    "stadium",
    "roof",
    "temp",
    "wind",
    "spread_line",
    "total_line",
]


def clean_name(name: str) -> str:
    """Normalize player names for robust matching."""
    if not isinstance(name, str):
        return ""
    # Strip suffixes like Jr., III, II, etc.
    s = name.strip().lower()
    for suffix in [" jr.", " jr", " sr.", " sr", " iii", " ii", " iv"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.replace(".", "").replace("'", "").strip()


def _id_to_str(series: pd.Series) -> pd.Series:
    """Normalizes an ID column to clean strings (e.g. 1234.0 -> "1234", NaN -> None)."""
    s = series.astype(object).where(series.notna(), None)
    return s.map(
        lambda v: str(int(v))
        if isinstance(v, (int, float)) and not isinstance(v, bool)
        else (str(v) if v is not None else None)
    )


def _cache_fresh(path: Path, ttl_hours: Optional[float]) -> bool:
    """True when the cache file exists and is younger than ttl_hours (None = never expires)."""
    if not path.exists():
        return False
    if ttl_hours is None:
        return True
    return (time.time() - path.stat().st_mtime) / 3600 < ttl_hours


def _frame_has_season(df: pd.DataFrame, season: int) -> bool:
    """True when the frame actually carries rows for the requested season."""
    return (
        not df.empty
        and "season" in df.columns
        and (df["season"] == season).any()
    )


class DataLoader:
    """Fetches, caches, and joins NFL statistics from nflverse/nfl_data_py."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir

    def get_weekly_data(self, seasons: Optional[List[int]] = None, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch weekly offensive stats including fantasy points, targets, and air yards."""
        if seasons is None:
            seasons = [DEFAULT_SEASON]

        dfs = []
        for season in seasons:
            cache_file = self.cache_dir / f"weekly_{season}.parquet"
            if cache_file.exists() and not force_refresh:
                df = pd.read_parquet(cache_file)
            else:
                # No cross-season fallback: substituting another season's rows
                # silently mislabels data. Callers handle a missing season.
                try:
                    df = nfl.import_weekly_data([season])
                    df.to_parquet(cache_file, index=False)
                except Exception as e:
                    logger.warning(f"Failed to fetch NFL weekly data for {season}: {e}")
                    continue
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        return combined

    def get_snap_counts(self, seasons: Optional[List[int]] = None, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch offensive and defensive snap count percentages per game."""
        if seasons is None:
            seasons = [DEFAULT_SEASON]

        dfs = []
        for season in seasons:
            cache_file = self.cache_dir / f"snaps_{season}.parquet"
            if cache_file.exists() and not force_refresh:
                df = pd.read_parquet(cache_file)
            else:
                try:
                    df = nfl.import_snap_counts([season])
                    df.to_parquet(cache_file, index=False)
                except Exception as e:
                    logger.warning(f"Failed to fetch NFL snap counts for {season}: {e}")
                    continue
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        return combined

    def get_player_ids(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetches and caches the cross-provider player ID crosswalk, keyed on gsis_id.

        Maps gsis_id (weekly/pbp player_id), pfr_id (snap counts pfr_player_id),
        and sleeper_id (Sleeper player_id), with clean_name/position/team retained
        as a secondary fallback for rows missing IDs.
        """
        cache_file = self.cache_dir / "ff_playerids.parquet"
        if cache_file.exists() and not force_refresh:
            return pd.read_parquet(cache_file)

        try:
            ids_df = nfl.import_ids()
        except Exception as e:
            logger.warning(f"Failed to fetch player ID crosswalk: {e}")
            if cache_file.exists():
                return pd.read_parquet(cache_file)
            return pd.DataFrame()

        keep_cols = ["gsis_id", "pfr_id", "sleeper_id", "name", "position", "team"]
        missing = [c for c in keep_cols if c not in ids_df.columns]
        if missing:
            logger.warning(f"Player ID crosswalk missing expected columns: {missing}")
        ids_df = ids_df[[c for c in keep_cols if c in ids_df.columns]].copy()

        for col in ("gsis_id", "pfr_id", "sleeper_id"):
            if col in ids_df.columns:
                ids_df[col] = _id_to_str(ids_df[col])

        if "name" in ids_df.columns:
            ids_df["norm_name"] = ids_df["name"].apply(clean_name)

        ids_df = ids_df.dropna(subset=["gsis_id"]).drop_duplicates(subset=["gsis_id"], keep="first")
        ids_df.to_parquet(cache_file, index=False)
        return ids_df

    def get_weekly_with_snaps(
        self, seasons: Optional[List[int]] = None, force_refresh: bool = False
    ) -> pd.DataFrame:
        """Merges weekly stats with offensive snap counts.

        Primary join: weekly player_id (gsis_id) -> ID crosswalk pfr_id -> snaps
        pfr_player_id, so shared/similar names cannot collide. Secondary fallback
        (rows still missing snaps after the ID join, e.g. players missing pfr_id):
        normalized player name (clean_name) + season + week.
        """
        weekly_df = self.get_weekly_data(seasons=seasons, force_refresh=force_refresh)
        snaps_df = self.get_snap_counts(seasons=seasons, force_refresh=force_refresh)

        if weekly_df.empty:
            return pd.DataFrame()

        weekly_df = weekly_df.copy()
        weekly_df["norm_name"] = weekly_df["player_display_name"].apply(clean_name)

        snap_cols = ["offense_snaps", "offense_pct"]

        if snaps_df.empty:
            for col in snap_cols:
                weekly_df[col] = 0.0
            return weekly_df

        snaps_df = snaps_df.copy()
        snaps_df["norm_name"] = snaps_df["player"].apply(clean_name)

        merged = weekly_df
        if "player_id" in weekly_df.columns and "pfr_player_id" in snaps_df.columns:
            ids_df = self.get_player_ids(force_refresh=force_refresh)
            if not ids_df.empty and "pfr_id" in ids_df.columns:
                crosswalk = (
                    ids_df[["gsis_id", "pfr_id"]]
                    .dropna()
                    .drop_duplicates(subset=["gsis_id"], keep="first")
                )
                snaps_by_id = (
                    snaps_df[["season", "week", "pfr_player_id"] + snap_cols]
                    .dropna(subset=["pfr_player_id"])
                    .drop_duplicates(subset=["season", "week", "pfr_player_id"])
                )
                merged = pd.merge(
                    weekly_df,
                    crosswalk,
                    left_on="player_id",
                    right_on="gsis_id",
                    how="left",
                )
                merged = pd.merge(
                    merged,
                    snaps_by_id,
                    left_on=["pfr_id", "season", "week"],
                    right_on=["pfr_player_id", "season", "week"],
                    how="left",
                )
                merged = merged.drop(
                    columns=[c for c in ("gsis_id", "pfr_id", "pfr_player_id") if c in merged.columns]
                )

        for col in snap_cols:
            if col not in merged.columns:
                merged[col] = None

        # Secondary name-based fallback only for rows the ID join could not resolve
        need_fallback = merged["offense_pct"].isna() | merged["offense_snaps"].isna()
        if need_fallback.any():
            snaps_by_name = (
                snaps_df[["season", "week", "norm_name"] + snap_cols]
                .drop_duplicates(subset=["season", "week", "norm_name"])
            )
            fb_source = merged.loc[need_fallback, [c for c in merged.columns if c not in snap_cols]]
            fb_merged = pd.merge(
                fb_source,
                snaps_by_name,
                on=["season", "week", "norm_name"],
                how="left",
            )
            merged.loc[need_fallback, snap_cols] = fb_merged[snap_cols].values

        # Fill missing snap percentages with 0.0 if player was active but had no recorded snaps
        merged["offense_pct"] = merged["offense_pct"].fillna(0.0)

        return merged

    def _seasonal_frame(
        self,
        cache_prefix: str,
        fetch,
        seasons: Optional[List[int]],
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Loads a per-season cached nflverse frame with a 24h TTL.

        Season-checked: a cached frame is served only when it contains rows
        for the requested season. A cache holding a different season (e.g. a
        2024/2025 depth chart requested for 2026) is ignored and the source
        is refetched; a fetch that does not return the requested season is
        rejected rather than cached or served.
        """
        if seasons is None:
            seasons = [DEFAULT_SEASON]

        frames = []
        for season in seasons:
            cache_file = self.cache_dir / f"{cache_prefix}_{season}.parquet"

            if cache_file.exists() and not force_refresh:
                age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                if age_hours < SEASONAL_CACHE_TTL_HOURS:
                    try:
                        cached = pd.read_parquet(cache_file)
                    except Exception as e:
                        logger.warning(f"Failed to read cached {cache_prefix} for {season}: {e}")
                        cached = pd.DataFrame()
                    if _frame_has_season(cached, season):
                        frames.append(cached)
                        continue
                    logger.warning(
                        f"Cached {cache_prefix} for {season} does not contain the requested season; refetching."
                    )
                else:
                    logger.info(f"Cached {cache_prefix} for {season} older than {SEASONAL_CACHE_TTL_HOURS:.0f}h; refetching.")

            try:
                df = fetch([season])
            except Exception as e:
                logger.warning(f"Failed to fetch {cache_prefix} for {season}: {e}")
                continue

            if not _frame_has_season(df, season):
                logger.warning(f"{cache_prefix} source returned no {season} rows; rejecting as stale.")
                continue

            df.to_parquet(cache_file, index=False)
            frames.append(df)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_injuries(self, seasons: Optional[List[int]] = None, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch official weekly injury reports, cached per season (24h TTL)."""
        return self._seasonal_frame("injuries", nfl.import_injuries, seasons, force_refresh)

    def get_depth_charts(self, seasons: Optional[List[int]] = None, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch official depth charts, cached per season (24h TTL).

        Season-checked fallback source: consumers must only use these rows to
        enrich players whose live team agrees with the row's team.
        """
        return self._seasonal_frame("depth_charts", nfl.import_depth_charts, seasons, force_refresh)

    def get_pbp_opportunity_features(
        self, seasons: Optional[List[int]] = None, force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Aggregates red-zone opportunity and defensive EPA features from play-by-play.

        Loads only the needed pbp columns (regular-season rush/pass plays only) and
        returns a dict of three frames:
        - "player": per (player_id, season, week) carries_inside_10, carries_inside_5,
          targets_inside_10, end_zone_targets, high_value_touches (targets + carries_inside_10)
        - "team": per (posteam, season, week) team_dropbacks
        - "defense": per (defteam, season, week) def_pass_epa_allowed / def_rush_epa_allowed
          (play EPA sums) and def_pass_attempts / def_rush_attempts (play counts)
        """
        if seasons is None:
            seasons = [DEFAULT_SEASON]

        player_frames, team_frames, defense_frames = [], [], []
        for season in seasons:
            player_cache = self.cache_dir / f"pbp_features_{season}.parquet"
            team_cache = self.cache_dir / f"pbp_team_features_{season}.parquet"
            defense_cache = self.cache_dir / f"pbp_def_features_{season}.parquet"
            ttl = LIVE_PBP_CACHE_TTL_HOURS if season >= DEFAULT_SEASON else None
            if (
                not force_refresh
                and all(_cache_fresh(p, ttl) for p in (player_cache, team_cache, defense_cache))
            ):
                cached_player = pd.read_parquet(player_cache)
                schema_ok = all(c in cached_player.columns for c in PBP_PLAYER_FEATURE_SCHEMA) and (
                    "pbp_schema_version" in cached_player.columns
                    and (cached_player["pbp_schema_version"] == PBP_FEATURE_SCHEMA_VERSION).all()
                )
                if schema_ok:
                    player_frames.append(cached_player)
                    team_frames.append(pd.read_parquet(team_cache))
                    defense_frames.append(pd.read_parquet(defense_cache))
                    continue
                logger.info(f"Cached pbp features for {season} predate the current schema; rebuilding.")

            try:
                pbp = nfl.import_pbp_data(
                    [season],
                    columns=PBP_FEATURE_COLUMNS,
                    include_participation=False,
                    downcast=True,
                )
            except Exception as e:
                logger.warning(f"Failed to fetch NFL pbp data for {season}: {e}")
                continue

            pbp = pbp[
                (pbp["season_type"] == "REG")
                & ((pbp["rush_attempt"] == 1) | (pbp["pass_attempt"] == 1))
                # Exclude non-standard plays that pollute opportunity counts:
                # QB kneels / victory formations, spikes, and 2-pt conversions
                & (pbp["qb_kneel"].fillna(0) == 0)
                & (pbp["qb_spike"].fillna(0) == 0)
                & (pbp["two_point_attempt"].fillna(0) == 0)
            ].copy()

            yardline = pbp["yardline_100"].fillna(50.0).clip(1.0, 99.0)
            air_yds = pbp["air_yards"].fillna(0.0)
            pbp["inside_10"] = yardline <= 10
            pbp["inside_5"] = yardline <= 5
            pbp["ez_target"] = (yardline - air_yds) <= 0

            # Separate true pass attempts from sacks so passer/receiver opportunity
            # is not credited for sack plays, and defensive EPA is split correctly.
            is_sack = pbp["sack"].fillna(0) == 1
            rush = pbp[pbp["rush_attempt"] == 1].copy()
            true_pass = pbp[(pbp["pass_attempt"] == 1) & (~is_sack)].copy()
            all_dropbacks = pbp[(pbp["pass_attempt"] == 1) | (pbp["qb_dropback"] == 1)].copy()

            # --- Play-level Rushing xFP ---
            r_yl = rush["yardline_100"].fillna(50.0).clip(1.0, 99.0)
            r_scramble = rush["qb_scramble"].fillna(0.0) if "qb_scramble" in rush.columns else 0.0
            p_rush_td = np.clip(0.58 * np.exp(-0.18 * (r_yl - 1.0)) * (1.0 + 0.30 * r_scramble), 0.004, 0.68)
            exp_rush_yds = np.clip(4.4 - 0.03 * (100.0 - r_yl) + 1.8 * r_scramble, 1.2, 8.5)
            rush["play_rush_xfp"] = 0.1 * exp_rush_yds + 6.0 * p_rush_td

            rush_feats = (
                rush.groupby(["rusher_player_id", "season", "week"])
                .agg(
                    carries=("rush_attempt", "sum"),
                    carries_inside_10=("inside_10", "sum"),
                    carries_inside_5=("inside_5", "sum"),
                    pbp_rush_xfp=("play_rush_xfp", "sum"),
                )
                .reset_index()
                .rename(columns={"rusher_player_id": "player_id"})
            )

            # --- Play-level Receiving & Passing xFP ---
            p_yl = true_pass["yardline_100"].fillna(50.0).clip(1.0, 99.0)
            p_ay = true_pass["air_yards"].fillna(6.0).clip(-10.0, 65.0)
            default_cp = np.clip(0.74 - 0.014 * np.maximum(p_ay, 0.0), 0.25, 0.88)
            cp = true_pass["cp"].fillna(default_cp).clip(0.05, 0.98) if "cp" in true_pass.columns else default_cp
            xyac = (
                true_pass["xyac_mean_yardage"].fillna(4.5).clip(0.0, 25.0)
                if "xyac_mean_yardage" in true_pass.columns
                else 4.5
            )
            target_yardline = np.clip(p_yl - p_ay, 0.0, 99.0)
            p_rec_td_given_catch = np.where(
                target_yardline <= 0,
                0.78,
                np.clip(0.45 * np.exp(-0.16 * target_yardline), 0.005, 0.55),
            )
            p_rec_td = cp * p_rec_td_given_catch
            exp_rec_yds = cp * np.maximum(0.0, p_ay + xyac)

            true_pass["expected_receptions"] = cp
            true_pass["play_rec_xfp_0ppr"] = 0.1 * exp_rec_yds + 6.0 * p_rec_td

            # QB passing xFP: 4-pt pass TD baseline, 0.04/yd, expected INT penalty
            p_int = np.clip(0.015 + 0.0012 * np.maximum(p_ay, 0.0), 0.01, 0.08)
            true_pass["play_pass_xfp"] = 0.04 * exp_rec_yds + 4.0 * p_rec_td - 2.0 * p_int

            rec_feats = (
                true_pass.groupby(["receiver_player_id", "season", "week"])
                .agg(
                    targets=("pass_attempt", "sum"),
                    targets_inside_10=("inside_10", "sum"),
                    end_zone_targets=("ez_target", "sum"),
                    expected_receptions=("expected_receptions", "sum"),
                    pbp_rec_xfp_0ppr=("play_rec_xfp_0ppr", "sum"),
                    target_air_yards=("air_yards", "sum"),
                )
                .reset_index()
                .rename(columns={"receiver_player_id": "player_id"})
            )

            # Passer features: QB passing volume, red-zone pass attempts, and play-level pass xFP.
            # Sacks are counted (sacks_taken) but not deducted: standard fantasy
            # scoring does not penalize sacks, and the deduction biased QB xFP low.
            if "passer_player_id" in true_pass.columns:
                passer_feats = (
                    true_pass.groupby(["passer_player_id", "season", "week"])
                    .agg(
                        pass_attempts=("pass_attempt", "sum"),
                        pass_attempts_inside_10=("inside_10", "sum"),
                        pbp_pass_xfp=("play_pass_xfp", "sum"),
                    )
                    .reset_index()
                    .rename(columns={"passer_player_id": "player_id"})
                )
                sack_plays = pbp[is_sack]
                if not sack_plays.empty and "passer_player_id" in sack_plays.columns:
                    sack_counts = (
                        sack_plays.groupby(["passer_player_id", "season", "week"])["sack"]
                        .sum()
                        .reset_index()
                        .rename(columns={"passer_player_id": "player_id", "sack": "sacks_taken"})
                    )
                    passer_feats = pd.merge(
                        passer_feats, sack_counts, on=["player_id", "season", "week"], how="left"
                    )
                    passer_feats["sacks_taken"] = passer_feats["sacks_taken"].fillna(0.0)
            else:
                passer_feats = pd.DataFrame(
                    columns=["player_id", "season", "week", "pass_attempts", "pass_attempts_inside_10", "pbp_pass_xfp"]
                )

            player_feats = pd.merge(rec_feats, rush_feats, on=["player_id", "season", "week"], how="outer")
            player_feats = pd.merge(player_feats, passer_feats, on=["player_id", "season", "week"], how="outer")

            fill_zero_cols = (
                "carries", "carries_inside_10", "carries_inside_5",
                "targets", "targets_inside_10", "end_zone_targets",
                "expected_receptions", "pbp_rec_xfp_0ppr", "pbp_rush_xfp",
                "pass_attempts", "pass_attempts_inside_10", "pbp_pass_xfp",
                "target_air_yards",
            )
            for col in fill_zero_cols:
                if col in player_feats.columns:
                    player_feats[col] = player_feats[col].fillna(0.0)

            player_feats["high_value_touches"] = (
                player_feats["targets"] + player_feats["carries_inside_10"]
            )
            keep_player_cols = [
                "player_id", "season", "week",
                "carries_inside_10", "carries_inside_5",
                "targets_inside_10", "end_zone_targets",
                "high_value_touches", "expected_receptions",
                "pbp_rec_xfp_0ppr", "pbp_rush_xfp",
                "pass_attempts", "pass_attempts_inside_10", "pbp_pass_xfp",
                "target_air_yards",
            ]
            player_feats = player_feats[[c for c in keep_player_cols if c in player_feats.columns]].copy()
            player_feats["pbp_schema_version"] = PBP_FEATURE_SCHEMA_VERSION

            team_feats = (
                pbp.groupby(["posteam", "season", "week"])["qb_dropback"]
                .sum()
                .reset_index()
                .rename(columns={"qb_dropback": "team_dropbacks"})
            )

            def_pass = (
                all_dropbacks.groupby(["defteam", "season", "week"])["epa"]
                .agg(def_pass_epa_allowed="sum", def_pass_attempts="count")
                .reset_index()
            )
            def_rush = (
                rush.groupby(["defteam", "season", "week"])["epa"]
                .agg(def_rush_epa_allowed="sum", def_rush_attempts="count")
                .reset_index()
            )
            defense_feats = pd.merge(
                def_pass,
                def_rush,
                on=["defteam", "season", "week"],
                how="outer",
            )

            player_feats.to_parquet(player_cache, index=False)
            team_feats.to_parquet(team_cache, index=False)
            defense_feats.to_parquet(defense_cache, index=False)

            player_frames.append(player_feats)
            team_frames.append(team_feats)
            defense_frames.append(defense_feats)

        return {
            "player": pd.concat(player_frames, ignore_index=True) if player_frames else pd.DataFrame(),
            "team": pd.concat(team_frames, ignore_index=True) if team_frames else pd.DataFrame(),
            "defense": pd.concat(defense_frames, ignore_index=True) if defense_frames else pd.DataFrame(),
        }

    def get_schedules(
        self, seasons: Optional[List[int]] = None, force_refresh: bool = False
    ) -> pd.DataFrame:
        """Fetches and caches the weekly game schedule with Vegas lines and weather context.

        Regular-season games only. spread_line in nflverse schedules is relative to
        the home team with positive meaning home is favored (verified against 2024
        home moneylines), so home_itt = (total + spread) / 2 and away_itt =
        (total - spread) / 2. gameday/gametime are stadium-local.
        """
        if seasons is None:
            seasons = [DEFAULT_SEASON]

        dfs = []
        for season in seasons:
            cache_file = self.cache_dir / f"schedules_{season}.parquet"
            # Live-season schedules carry moving Vegas lines; refresh them
            ttl = LIVE_SCHEDULE_CACHE_TTL_HOURS if season >= DEFAULT_SEASON else None
            if _cache_fresh(cache_file, ttl) and not force_refresh:
                dfs.append(pd.read_parquet(cache_file))
                continue

            try:
                df = nfl.import_schedules([season])
            except Exception as e:
                logger.warning(f"Failed to fetch NFL schedules for {season}: {e}")
                continue

            df = df[[c for c in SCHEDULE_COLUMNS if c in df.columns]].copy()

            if "game_type" in df.columns:
                df = df[df["game_type"] == "REG"].copy()

            if "gameday" in df.columns and "gametime" in df.columns:
                df["game_datetime"] = pd.to_datetime(
                    df["gameday"].astype(str) + " " + df["gametime"].astype(str),
                    errors="coerce",
                )

            if "total_line" in df.columns and "spread_line" in df.columns:
                df["home_itt"] = (df["total_line"] + df["spread_line"]) / 2
                df["away_itt"] = (df["total_line"] - df["spread_line"]) / 2

            df.to_parquet(cache_file, index=False)
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _expected_teams_by_week(self, season: int) -> Dict[int, int]:
        """Teams scheduled to play each regular-season week (2 x games), {} if unavailable."""
        try:
            sched = self.get_schedules(seasons=[season])
        except Exception as e:
            logger.info(f"Schedule unavailable for {season} ({e}); using team-count heuristic.")
            return {}
        if sched.empty or "week" not in sched.columns:
            return {}
        return {int(w): int(2 * n) for w, n in sched.groupby("week").size().items()}

    def get_sleeper_weekly_data(
        self,
        season: int = DEFAULT_SEASON,
        through_week: Optional[int] = None,
        include_partial_week: bool = False,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Per-player, per-week stat lines from Sleeper in nflverse weekly column names.

        One row per player-game (QB/RB/WR/TE with any snaps or volume), keyed by
        Sleeper player_id, with team-at-game-time, real snap share
        (off_snp / tm_off_snp), team target / air-yard shares and WOPR, red-zone
        volume (rz_carries / rz_targets / rz_pass_attempts, inside the 20), and
        PPR / half-PPR / standard points. This replaces nflverse weekly player
        stats, which are no longer published at the nfl_data_py URL for 2025+.

        Caveat: Sleeper's receiving_air_yards counts completions only;
        attach_pbp_features overrides it with true target air yards from
        nflverse play-by-play wherever the week is published.

        The latest week is dropped while incomplete (fewer teams with stat lines
        than scheduled, e.g. only TNF played) unless include_partial_week=True,
        so "latest week" consumers never see a half-played week. Settled weeks
        cache forever; the two most recent weeks re-fetch every 6h to pick up
        stat corrections. frame.attrs["data_through_week"] records the last week
        included.
        """
        from fantasy_intel.analytics.metrics import compute_team_shares
        from fantasy_intel.data.sleeper import RECENT_WEEK_STATS_TTL_HOURS, SleeperClient

        client = SleeperClient(cache_dir=self.cache_dir)
        state = client.get_nfl_state() or {}
        try:
            state_season = int(state.get("season") or DEFAULT_SEASON)
            state_week = int(state.get("week") or 1)
        except (TypeError, ValueError):
            state_season, state_week = DEFAULT_SEASON, 1

        if season > state_season:
            return pd.DataFrame()
        live = season == state_season and state.get("season_type") in (None, "regular")
        if through_week is None:
            through_week = min(state_week, 18) if live else 18

        players = client.get_all_players() or {}
        rows: List[Dict] = []
        for week in range(1, int(through_week) + 1):
            recent = live and week >= state_week - 2
            records = client.get_week_stats(
                season,
                week,
                ttl_hours=RECENT_WEEK_STATS_TTL_HOURS if recent else None,
                force_refresh=force_refresh,
            )
            for rec in records:
                pid = str(rec.get("player_id"))
                pinfo = players.get(pid, {})
                position = rec.get("position") or pinfo.get("position")
                if position not in ("QB", "RB", "WR", "TE"):
                    continue
                st = rec.get("stats") or {}

                def num(key: str) -> float:
                    try:
                        return float(st.get(key) or 0.0)
                    except (TypeError, ValueError):
                        return 0.0

                snaps = num("off_snp")
                volume = num("pass_att") + num("rush_att") + num("rec_tgt")
                if snaps <= 0 and volume <= 0 and num("gp") <= 0:
                    continue  # inactive / DNP

                row: Dict = {
                    "player_id": pid,
                    "player_display_name": pinfo.get("full_name")
                    or f"{pinfo.get('first_name', '')} {pinfo.get('last_name', '')}".strip()
                    or pid,
                    "position": position,
                    "recent_team": rec.get("team") or pinfo.get("team"),
                    "opponent_team": rec.get("opponent"),
                    "game_id": rec.get("game_id"),
                    "season": season,
                    "week": week,
                    "season_type": "REG",
                    "games_played": 1,
                }
                for src, dst in SLEEPER_WEEKLY_STAT_MAP.items():
                    row[dst] = num(src)
                team_snaps = row["team_offense_snaps"]
                row["offense_pct"] = round(min(1.0, snaps / team_snaps), 4) if team_snaps > 0 else 0.0
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Drop a half-played latest week
        last_week = int(df["week"].max())
        if not include_partial_week and live:
            teams_seen = df.loc[df["week"] == last_week, "recent_team"].nunique()
            expected = self._expected_teams_by_week(season).get(last_week, MIN_TEAMS_FOR_COMPLETE_WEEK)
            if teams_seen < expected:
                logger.info(
                    f"Week {last_week} incomplete ({teams_seen}/{expected} teams reported); excluding it."
                )
                df = df[df["week"] < last_week].copy()
                last_week -= 1

        df = compute_team_shares(df).reset_index(drop=True)
        df.attrs["data_through_week"] = last_week
        df.attrs["source"] = "sleeper_weekly"
        return df

    def get_sleeper_season_data(self, season: int = DEFAULT_SEASON, week: Optional[int] = None) -> pd.DataFrame:
        """
        Loads and transforms live in-season player stats directly from Sleeper's API
        into opportunity and efficiency metrics. Week defaults to the current NFL
        week from Sleeper's /state/nfl endpoint.

        Note: Sleeper's rec_air_yd covers completed passes only (rec_yd equals
        rec_air_yd + rec_yar exactly), so it is not true target air yards — racr
        runs hot and unrealized_air_yards stays ~0, leaving the WR/TE waiver
        index effectively capped at 85 on this path. The advisor therefore prefers
        the nflverse weekly frame (which carries true air yards) whenever it is
        published for the season and uses this Sleeper frame as the live fallback.
        """
        from fantasy_intel.data.sleeper import SleeperClient

        client = SleeperClient(cache_dir=self.cache_dir)
        if week is None:
            state = client.get_nfl_state()
            week = int(state.get("display_week") or state.get("week") or 1)
        stats = client.get_season_stats(season=season)
        players = client.get_all_players()

        if not stats or not players:
            return pd.DataFrame()

        rows = []
        for pid, st in stats.items():
            if pid.startswith("TEAM_") or not st:
                continue
            pinfo = players.get(str(pid), {})
            pos = pinfo.get("position")
            if pos not in ["QB", "RB", "WR", "TE"]:
                continue

            gp = float(st.get("gp") or 1.0)
            if gp <= 0:
                continue
            pts = float(st.get("pts_ppr", 0.0) or 0.0)
            targets = float(st.get("rec_tgt", 0.0) or 0.0)
            carries = float(st.get("rush_att", 0.0) or 0.0)
            rec_yd = float(st.get("rec_yd", 0.0) or 0.0)
            rush_yd = float(st.get("rush_yd", 0.0) or 0.0)
            snaps = float(st.get("off_snp", 0.0) or 0.0)
            air_yd = float(st.get("rec_air_yd", 0.0) or 0.0)

            snap_pct = min(1.0, round(snaps / (gp * 65.0), 2))
            tgt_share = min(0.40, round(targets / (gp * 32.0), 2))
            wopr = round(1.5 * tgt_share + 0.7 * (rec_yd / (gp * 220.0)), 2)
            per_game_pts = round(pts / gp, 1)
            per_game_tgt = round(targets / gp, 1)
            per_game_car = round(carries / gp, 1)
            per_game_xfp = round(per_game_tgt * 1.3 + per_game_car * 0.68 + ((rec_yd + rush_yd) / gp) * 0.08, 1)

            rows.append({
                "player_id": str(pid),
                "player_display_name": pinfo.get("full_name") or f"{pinfo.get('first_name', '')} {pinfo.get('last_name', '')}".strip() or str(pid),
                "position": pos,
                "recent_team": pinfo.get("team") or "FA",
                "season": season,
                "week": week,
                "season_type": "REG",
                "games_played": int(gp),
                "fantasy_points_ppr": per_game_pts,
                "recent_ppr": per_game_pts,
                "targets": per_game_tgt,
                "carries": per_game_car,
                "touches": per_game_tgt + per_game_car,
                "offense_snaps": round(snaps / gp, 1),
                "offense_pct": snap_pct,
                "snap_delta": 0.05,
                "target_share": tgt_share,
                "receiving_air_yards": round(air_yd / gp, 1),
                "receiving_yards": round(rec_yd / gp, 1),
                "wopr": wopr,
                "expected_fp_ppr": per_game_xfp,
                "fp_diff": round(per_game_pts - per_game_xfp, 1),
            })

        return pd.DataFrame(rows)

