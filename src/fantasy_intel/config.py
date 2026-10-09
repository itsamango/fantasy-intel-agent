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
DEFAULT_SEASON = 2024
DEFAULT_SEASONS = [2023, 2024]

# Scoring systems
SCORING_PPR = "ppr"
SCORING_HALF_PPR = "half_ppr"
SCORING_STANDARD = "standard"

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
