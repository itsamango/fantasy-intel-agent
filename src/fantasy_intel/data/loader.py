"""Data loader and cache manager for NFL fantasy statistics."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
import nfl_data_py as nfl

from fantasy_intel.config import CACHE_DIR, DEFAULT_SEASON

logger = logging.getLogger(__name__)


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

    def get_weekly_with_snaps(
        self, seasons: Optional[List[int]] = None, force_refresh: bool = False
    ) -> pd.DataFrame:
        """Merges weekly stats with offensive snap counts."""
        weekly_df = self.get_weekly_data(seasons=seasons, force_refresh=force_refresh)
        snaps_df = self.get_snap_counts(seasons=seasons, force_refresh=force_refresh)

        if weekly_df.empty:
            return pd.DataFrame()

        # Prepare normalized join keys
        weekly_df["norm_name"] = weekly_df["player_display_name"].apply(clean_name)
        snaps_df["norm_name"] = snaps_df["player"].apply(clean_name)

        # Subset snaps to relevant offensive columns
        snap_cols = [
            "season",
            "week",
            "norm_name",
            "offense_snaps",
            "offense_pct",
        ]
        snaps_subset = snaps_df[[c for c in snap_cols if c in snaps_df.columns]].drop_duplicates(
            subset=["season", "week", "norm_name"]
        )

        merged = pd.merge(
            weekly_df,
            snaps_subset,
            on=["season", "week", "norm_name"],
            how="left",
        )

        # Fill missing snap percentages with 0.0 if player was active but had no recorded snaps
        if "offense_pct" in merged.columns:
            merged["offense_pct"] = merged["offense_pct"].fillna(0.0)

        return merged

    def get_injuries(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """Fetch official weekly injury reports."""
        if seasons is None:
            seasons = [DEFAULT_SEASON]
        try:
            return nfl.import_injuries(seasons)
        except Exception as e:
            logger.warning(f"Failed to fetch injury reports: {e}")
            return pd.DataFrame()

    def get_depth_charts(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """Fetch official depth charts."""
        if seasons is None:
            seasons = [DEFAULT_SEASON]
        try:
            return nfl.import_depth_charts(seasons)
        except Exception as e:
            logger.warning(f"Failed to fetch depth charts: {e}")
            return pd.DataFrame()

    def get_sleeper_season_data(self, season: int = 2026) -> pd.DataFrame:
        """
        Loads and transforms live in-season player stats directly from Sleeper's API
        into opportunity and efficiency metrics.
        """
        from fantasy_intel.data.sleeper import SleeperClient

        client = SleeperClient(cache_dir=self.cache_dir)
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
                "week": 5,
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
                "wopr": wopr,
                "expected_fp_ppr": per_game_xfp,
                "fp_diff": round(per_game_pts - per_game_xfp, 1),
            })

        return pd.DataFrame(rows)

