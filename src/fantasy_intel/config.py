"""Configuration and constants for fantasy-intel."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE_DIR / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default season settings
DEFAULT_SEASON = 2024
DEFAULT_SEASONS = [2023, 2024]

# Scoring systems
SCORING_PPR = "ppr"
SCORING_HALF_PPR = "half_ppr"
SCORING_STANDARD = "standard"

# Key fantasy positions
SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]
