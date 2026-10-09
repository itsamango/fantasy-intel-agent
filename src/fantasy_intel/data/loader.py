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
                df = nfl.import_weekly_data([season])
                df.to_parquet(cache_file, index=False)
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
                df = nfl.import_snap_counts([season])
                df.to_parquet(cache_file, index=False)
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
