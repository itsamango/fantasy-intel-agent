"""Configuration, user settings, and constants for fantasy-intel."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE_DIR / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USER_CONFIG_DIR = Path.home() / ".fantasy_intel"
USER_CONFIG_FILE = USER_CONFIG_DIR / "config.json"

# Default season settings
DEFAULT_SEASON = 2026
DEFAULT_SEASONS = [2024, 2025, 2026]

# Scoring systems
SCORING_PPR = "ppr"
SCORING_HALF_PPR = "half_ppr"
SCORING_STANDARD = "standard"
SCORING_TEP = "tep"

# Per-event scoring weights by scoring system. The "tep" rec weight is the
# effective per-reception value for tight ends (PPR base + 0.5 TE premium);
# non-TE positions in TEP leagues are valued with the "ppr" weights.
SCORING_WEIGHTS: Dict[str, Dict[str, float]] = {
    SCORING_PPR: {"rec": 1.0, "rush_yd": 0.1, "rec_yd": 0.1, "td": 6.0},
    SCORING_HALF_PPR: {"rec": 0.5, "rush_yd": 0.1, "rec_yd": 0.1, "td": 6.0},
    SCORING_STANDARD: {"rec": 0.0, "rush_yd": 0.1, "rec_yd": 0.1, "td": 6.0},
    SCORING_TEP: {"rec": 1.5, "rush_yd": 0.1, "rec_yd": 0.1, "td": 6.0},
}

# Trade package valuation
PACKAGE_CONSOLIDATION_ALPHA = 0.72

# Matchup / lineup optimization
WIND_PENALTY_THRESHOLD_MPH = 15
MATCHUP_POSTURE_MARGIN = 10.0

# FAAB bid sizing
FAAB_GAMMA = 0.65
FAAB_URGENCY_HIGH = 1.25
FAAB_URGENCY_LOW = 0.85

# Key fantasy positions
SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]


def load_user_config() -> Dict[str, Any]:
    """Loads persistent user configuration, checking env vars first."""
    config: Dict[str, Any] = {
        "sleeper_username": os.environ.get("SLEEPER_USERNAME"),
        "default_league_id": os.environ.get("SLEEPER_LEAGUE_ID"),
        "season": DEFAULT_SEASON,
    }

    if USER_CONFIG_FILE.exists():
        try:
            with open(USER_CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if not config["sleeper_username"]:
                    config["sleeper_username"] = saved.get("sleeper_username")
                if not config["default_league_id"]:
                    config["default_league_id"] = saved.get("default_league_id")
                if "season" in saved:
                    config["season"] = saved.get("season")
        except Exception:
            pass

    return config


def save_user_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Saves user configuration to ~/.fantasy_intel/config.json."""
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_user_config()
    existing.update(updates)

    with open(USER_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    return existing
