"""Unit tests for Discord Bot embeds and bot initialization."""

import pytest
import pandas as pd
from unittest.mock import MagicMock

from fantasy_intel.bot.embeds import (
    create_roster_embed,
    create_matchup_embed,
    create_waivers_embed,
    create_injuries_embed,
    create_leagues_embed,
    create_ai_advice_embed,
    get_status_emoji,
)
from fantasy_intel.bot.discord_bot import create_bot


def test_status_emoji():
    assert get_status_emoji(None) == "🟢"
    assert get_status_emoji("Out") == "🚨"
    assert get_status_emoji("Questionable") == "⚠️"


def test_roster_embed():
    sample_roster = {
        "league_name": "VRA 26-27",
        "record": "3-1",
        "points_for": 487.0,
        "points_against": 472.5,
        "waiver_position": 9,
        "starters": [
            {"name": "Matthew Stafford", "position": "QB", "team": "LAR", "injury_status": None},
            {"name": "Jonathan Taylor", "position": "RB", "team": "IND", "injury_status": None},
            {"name": "Jordan Addison", "position": "WR", "team": "MIN", "injury_status": "Questionable"},
        ],
        "bench": [
            {"name": "Patrick Mahomes", "position": "QB", "team": "KC", "injury_status": None},
        ],
        "reserve": [
            {"name": "DeVonta Smith", "position": "WR", "team": "PHI", "injury_status": "Out"},
        ],
        "taxi": [],
    }

    embed = create_roster_embed(sample_roster, "lockedoutagain")
    assert "lockedoutagain" in embed.title
    assert "VRA 26-27" in embed.description
    assert len(embed.fields) == 2  # Starters, Bench
    assert "Matthew Stafford" in embed.fields[0].value
    assert "DeVonta Smith" in embed.fields[1].value


def test_matchup_embed():
    sample_matchup = {
        "matchup_overview": {
            "week": 5,
            "my_team": {
                "user_name": "lockedoutagain",
                "points": 45.2,
                "starters": [{"name": "Matthew Stafford", "position": "QB", "injury_status": None}],
            },
            "opponent_team": {
                "user_name": "EuanMorgan",
                "points": 38.0,
                "starters": [{"name": "Kyler Murray", "position": "QB", "injury_status": None}],
            },
        },
        "tactical_advice": {
            "injury_alerts": [{"player": "Jordan Addison", "action": "Bench immediately"}],
            "start_sit_suggestions": [{"sit": "Jordan Addison", "start": "Brian Thomas", "reason": "Injury"}],
            "strategy_notes": ["Opponent has healthy lineup"],
        },
    }

    embed = create_matchup_embed(sample_matchup)
    assert "lockedoutagain vs EuanMorgan" in embed.title
    assert any("Starters" in f.name for f in embed.fields)
    assert any("Warnings" in f.name for f in embed.fields)


def test_waivers_embed():
    df = pd.DataFrame([
        {
            "player_display_name": "Jahan Dotson",
            "position": "WR",
            "recent_team": "PHI",
            "offense_pct": 0.94,
            "snap_delta": 0.51,
            "opportunity_index": 84.2,
            "catalyst": "Snap surge",
            "fantasy_points_ppr": 16.4,
            "expected_fp_ppr": 23.0,
        }
    ])
    embed = create_waivers_embed(df, "VRA 26-27")
    assert "VRA 26-27" in embed.description
    assert "Jahan Dotson" in embed.fields[0].name


def test_create_bot():
    mock_advisor = MagicMock()
    bot = create_bot(advisor=mock_advisor)
    assert bot is not None
    assert bot.command_prefix == "!"
