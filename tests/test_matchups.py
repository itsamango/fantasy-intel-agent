"""Unit tests for matchup analysis and situational team advice."""

from fantasy_intel.analytics.matchups import MatchupAdvisor


def test_matchup_lineup_evaluation():
    mock_overview = {
        "week": 6,
        "league_id": "test_league_123",
        "my_team": {
            "user_name": "My Team",
            "roster_id": 1,
            "starters": [
                {"name": "Christian McCaffrey", "position": "RB", "team": "SF", "injury_status": "Out"},
                {"name": "Justin Jefferson", "position": "WR", "team": "MIN", "injury_status": None},
            ],
            "bench": [
                {"name": "Jordan Mason", "position": "RB", "team": "SF", "injury_status": None},
                {"name": "Rashod Bateman", "position": "WR", "team": "BAL", "injury_status": None},
            ],
        },
        "opponent_team": {
            "user_name": "Rival Team",
            "roster_id": 2,
            "starters": [
                {"name": "Saquon Barkley", "position": "RB", "team": "PHI", "injury_status": "Out"},
                {"name": "CeeDee Lamb", "position": "WR", "team": "DAL", "injury_status": None},
            ],
            "bench": [],
        },
    }

    advisor = MatchupAdvisor()
    evaluation = advisor.evaluate_lineup(mock_overview)

    # 1. Should flag CMC as critical injury alert
    alerts = evaluation["injury_alerts"]
    assert len(alerts) == 1
    assert alerts[0]["player"] == "Christian McCaffrey"
    assert alerts[0]["severity"] == "CRITICAL"

    # 2. Should suggest Jordan Mason as replacement
    suggestions = evaluation["start_sit_suggestions"]
    assert len(suggestions) == 1
    assert suggestions[0]["sit"] == "Christian McCaffrey"
    assert suggestions[0]["start"] == "Jordan Mason"

    # 3. Should flag opponent injured starter
    notes = evaluation["strategy_notes"]
    assert any("Saquon Barkley" in n for n in notes)
