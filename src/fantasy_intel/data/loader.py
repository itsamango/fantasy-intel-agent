"""Data loader and cache manager for NFL fantasy statistics."""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import nfl_data_py as nfl

from fantasy_intel.config import CACHE_DIR, DEFAULT_SEASON

logger = logging.getLogger(__name__)

# Depth-chart and injury-report caches are season-keyed with a 24h TTL,
# mirroring the Sleeper players cache in sleeper.py. A cache is only served
# when its frame actually contains rows for the requested season, so a stale
# season's depth chart can never be mixed into a current-season roster.
SEASONAL_CACHE_TTL_HOURS = 24.0

# nflverse pbp columns needed for opportunity and defensive EPA aggregation
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
    "rush_attempt",
    "pass_attempt",
    "rusher_player_id",
    "receiver_player_id",
    "yardline_100",
    "air_yards",
]

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
                try:
                    df = nfl.import_weekly_data([season])
                    df.to_parquet(cache_file, index=False)
                except Exception as e:
                    logger.warning(f"Failed to fetch NFL weekly data for {season}: {e}. Falling back to season {DEFAULT_SEASON}.")
                    fallback_cache = self.cache_dir / f"weekly_{DEFAULT_SEASON}.parquet"
                    if fallback_cache.exists():
                        df = pd.read_parquet(fallback_cache)
                    else:
                        df = nfl.import_weekly_data([DEFAULT_SEASON])
                        df.to_parquet(fallback_cache, index=False)
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
                    logger.warning(f"Failed to fetch NFL snap counts for {season}: {e}. Falling back to season {DEFAULT_SEASON}.")
                    fallback_cache = self.cache_dir / f"snaps_{DEFAULT_SEASON}.parquet"
                    if fallback_cache.exists():
                        df = pd.read_parquet(fallback_cache)
                    else:
                        df = nfl.import_snap_counts([DEFAULT_SEASON])
                        df.to_parquet(fallback_cache, index=False)
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
            if (
                player_cache.exists()
                and team_cache.exists()
                and defense_cache.exists()
                and not force_refresh
            ):
                player_frames.append(pd.read_parquet(player_cache))
                team_frames.append(pd.read_parquet(team_cache))
                defense_frames.append(pd.read_parquet(defense_cache))
                continue

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
            ].copy()

            pbp["inside_10"] = pbp["yardline_100"] <= 10
            pbp["inside_5"] = pbp["yardline_100"] <= 5
            pbp["ez_target"] = (pbp["yardline_100"] - pbp["air_yards"]) <= 0

            rush = pbp[pbp["rush_attempt"] == 1]
            passing = pbp[pbp["pass_attempt"] == 1]

            rush_feats = (
                rush.groupby(["rusher_player_id", "season", "week"])
                .agg(
                    carries=("rush_attempt", "sum"),
                    carries_inside_10=("inside_10", "sum"),
                    carries_inside_5=("inside_5", "sum"),
                )
                .reset_index()
                .rename(columns={"rusher_player_id": "player_id"})
            )

            rec_feats = (
                passing.groupby(["receiver_player_id", "season", "week"])
                .agg(
                    targets=("pass_attempt", "sum"),
                    targets_inside_10=("inside_10", "sum"),
                    end_zone_targets=("ez_target", "sum"),
                )
                .reset_index()
                .rename(columns={"receiver_player_id": "player_id"})
            )

            player_feats = pd.merge(
                rec_feats,
                rush_feats,
                on=["player_id", "season", "week"],
                how="outer",
            )
            for col in (
                "carries",
                "carries_inside_10",
                "carries_inside_5",
                "targets",
                "targets_inside_10",
                "end_zone_targets",
            ):
                if col in player_feats.columns:
                    player_feats[col] = player_feats[col].fillna(0)
            player_feats["high_value_touches"] = (
                player_feats["targets"] + player_feats["carries_inside_10"]
            )
            player_feats = player_feats[
                [
                    "player_id",
                    "season",
                    "week",
                    "carries_inside_10",
                    "carries_inside_5",
                    "targets_inside_10",
                    "end_zone_targets",
                    "high_value_touches",
                ]
            ]

            team_feats = (
                pbp.groupby(["posteam", "season", "week"])["qb_dropback"]
                .sum()
                .reset_index()
                .rename(columns={"qb_dropback": "team_dropbacks"})
            )

            def_pass = (
                passing.groupby(["defteam", "season", "week"])["epa"]
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
            if cache_file.exists() and not force_refresh:
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

