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

# LLM Default Model — overridable via GEMINI_MODEL env var
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Scoring systems
SCORING_PPR = "ppr"
SCORING_HALF_PPR = "half_ppr"
SCORING_STANDARD = "standard"
SCORING_TEP = "tep"

# Per-event scoring weights by scoring system. The "tep" rec weight is the
# effective per-reception value for tight ends (PPR base + 0.5 TE premium);
# non-TE positions in TEP leagues are valued with the "ppr" weights.
# pass_yd / pass_td / int are used by the QB passing xFP model.
SCORING_WEIGHTS: Dict[str, Dict[str, float]] = {
    SCORING_PPR:      {"rec": 1.0, "rush_yd": 0.1, "rec_yd": 0.1, "td": 6.0, "pass_yd": 0.04, "pass_td": 4.0, "int": -2.0},
    SCORING_HALF_PPR: {"rec": 0.5, "rush_yd": 0.1, "rec_yd": 0.1, "td": 6.0, "pass_yd": 0.04, "pass_td": 4.0, "int": -2.0},
    SCORING_STANDARD: {"rec": 0.0, "rush_yd": 0.1, "rec_yd": 0.1, "td": 6.0, "pass_yd": 0.04, "pass_td": 4.0, "int": -2.0},
    SCORING_TEP:      {"rec": 1.5, "rush_yd": 0.1, "rec_yd": 0.1, "td": 6.0, "pass_yd": 0.04, "pass_td": 4.0, "int": -2.0},
}

# Positional Coefficient of Variation priors for Empirical Bayes variance shrinkage
# (used in add_projection_distributions to prevent sample std from collapsing to ~0).
POSITION_CV_PRIORS: Dict[str, float] = {
    "QB": 0.32,
    "RB": 0.40,
    "WR": 0.48,
    "TE": 0.52,
}

# FPOE efficiency baseline (Empirical-Bayes shrinkage of FPOE / xFP per player).
# Prior strength k is in xFP points: a player's own efficiency gets weight
# S_X / (S_X + k) where S_X is his recency-weighted xFP sample. Fit by split-half
# reliability on 2024-2026 wk10 Sleeper weekly + nflverse pbp (k = n(1-r)/r):
# QB r=0.39, RB r=0.36, WR r=0.35, TE r=0.21. Refit with fit_fpoe_prior_strength.
FPOE_PRIOR_STRENGTH: Dict[str, float] = {
    "QB": 340.0,
    "RB": 230.0,
    "WR": 250.0,
    "TE": 450.0,
}
# Per-game recency decay for the baseline. Halflives of 8-32 games tested within
# noise of each other out of sample; 16 (~one season) chosen.
FPOE_HALFLIFE_GAMES = 16.0
# Games excluded from the pre-window baseline when scoring recent FPOE "excess"
FPOE_SIGNAL_WINDOW_GAMES = 3
# Minimum xFP volume before a position-level mean efficiency replaces 0 as the prior
FPOE_POSITION_MEAN_MIN_XFP = 500.0
# Prior seasons loaded for the baseline in live (2025+) seasons
FPOE_HISTORY_SEASONS = 2

# Rest-of-season (ROS) trade valuation.
# P(a regular plays and records volume in his team's next game), measured on
# 2024-2026 wk10 Sleeper weekly data (players with >= 50% snaps in the prior game).
ROS_AVAILABILITY_BY_POS: Dict[str, float] = {
    "QB": 0.915,
    "RB": 0.960,
    "WR": 0.911,
    "TE": 0.885,
}
# Next-week availability by current injury tag (later weeks use the base rate,
# or 0 while Sleeper publishes no projection for an IR/Out player).
ROS_NEXT_WEEK_STATUS_AVAILABILITY: Dict[str, float] = {
    "Questionable": 0.80,
    "Doubtful": 0.25,
    "Out": 0.0,
    "IR": 0.0,
    "PUP": 0.0,
    "Sus": 0.0,
}
ROS_NO_PROJECTION_STATUSES = ("IR", "Out", "PUP", "Sus", "NA")
ROS_SIMS = 300
ROS_FA_CANDIDATES_PER_POS = 5
# Waiver-optimized baseline: before valuing a trade, each team makes its best free
# drop->add swaps (a trade must not get credit for pickups available anyway).
ROS_WAIVER_MAX_SWAPS = 3
ROS_WAIVER_MIN_GAIN_PPW = 0.1
# Standings-aware verdicts: a trade changing your title odds by less than this
# (absolute probability, 0.005 = 0.5 percentage points) is called neutral.
ROS_TITLE_VERDICT_MARGIN = 0.005
# A trade within +/- this many points per remaining week is called roughly even
ROS_VERDICT_MARGIN_PPW = 0.5

# Trade-market layer (FantasyCalc crowdsourced redraft values from real trades).
# FantasyCalc fits values so traded packages sum equal (e.g. A = B + C), so the
# consolidation premium is already inside each value: sides are compared by plain
# sums, with no KTC-style stud adjustment on top (that would double count it).
FANTASYCALC_TTL_HOURS = 6.0
# |market gap| = |get - send| / (get + send). KTC's default fairness band is 5%.
MARKET_FAIR_BAND = 0.05
# Acceptance tiers from the partner's market change m = (their get - their send) /
# total and their ROS lineup change L (pts/week). Untrained judgment calls:
# tune from what actually gets accepted in your league.
MARKET_LIKELY_MIN = -0.025           # LIKELY: m >= -2.5% and L >= 0
MARKET_PLAUSIBLE_MIN = -0.05         # PLAUSIBLE: m >= -5% ...
MARKET_PLAUSIBLE_LINEUP_MIN = -0.10  # ... or m >= -10% with L >= MARKET_LINEUP_OFFSET_PPW
MARKET_LINEUP_OFFSET_PPW = 1.0
# Below this combined value neither side has meaningful market data
MARKET_MIN_TOTAL_VALUE = 300.0
# Fallers: managers anchor on recent box scores, so a player whose FantasyCalc
# value fell >= MARKET_FALLER_THRESHOLD over 30 days is perceived below even his
# current value: perceived = value * (1 - MARKET_FALLER_DISCOUNT * decline).
# Judgment calls (e.g. Kyren Williams -40% in 30d -> counted at 80%).
MARKET_FALLER_THRESHOLD = 0.25
MARKET_FALLER_DISCOUNT = 0.5
# Waiver-moves: a suggested drop with at least this FantasyCalc value still has
# trade value; flag it so the user shops him before cutting.
MARKET_TRADEABLE_MIN_VALUE = 500.0
# Sweetener search: players whose value is within this multiple of the gap to close
MARKET_SWEETENER_MAX_MULTIPLE = 2.5
MARKET_SWEETENER_OPTIONS = 2

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
