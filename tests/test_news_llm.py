"""Unit tests for DepthChartManager and LLMNewsAnalyzer."""

import pytest
from unittest.mock import MagicMock, patch

from fantasy_intel.analytics.depth_chart import DepthChartManager
from fantasy_intel.news.llm_analyzer import LLMNewsAnalyzer
from fantasy_intel.news.scraper import NewsCollector, clean_html


@pytest.fixture
def mock_sleeper_client():
    client = MagicMock()
    # Mock player database
    client.get_all_players.return_value = {
        "101": {
            "full_name": "Christian McCaffrey",
            "team": "SF",
            "position": "RB",
            "depth_chart_order": 1,
            "depth_chart_position": "RB",
            "injury_status": "Out",
            "injury_body_part": "Calf",
            "injury_notes": "Sidelined on IR with bilateral Achilles tendinitis.",
            "status": "Injured Reserve",
        },
        "102": {
            "full_name": "Jordan Mason",
            "team": "SF",
            "position": "RB",
            "depth_chart_order": 2,
            "depth_chart_position": "RB",
            "injury_status": None,
            "status": "Active",
        },
        "103": {
            "full_name": "Isaac Guerendo",
            "team": "SF",
            "position": "RB",
            "depth_chart_order": 3,
            "depth_chart_position": "RB",
            "injury_status": None,
            "status": "Active",
        },
        "201": {
            "full_name": "CeeDee Lamb",
            "team": "DAL",
            "position": "WR",
            "depth_chart_order": 1,
            "depth_chart_position": "LWR",
            "injury_status": "Questionable",
            "injury_body_part": "Quadricep",
            "status": "Active",
        },
        "202": {
            "full_name": "Brandin Cooks",
            "team": "DAL",
            "position": "WR",
            "depth_chart_order": 2,
            "depth_chart_position": "RWR",
            "injury_status": None,
            "status": "Active",
        },
    }
    return client


def test_depth_chart_manager(mock_sleeper_client):
    mgr = DepthChartManager(sleeper_client=mock_sleeper_client)

    # Test team depth chart
    sf_rbs = mgr.get_team_depth_chart("SF", position="RB")
    assert len(sf_rbs) == 3
    assert sf_rbs[0]["name"] == "Christian McCaffrey"
    assert sf_rbs[1]["name"] == "Jordan Mason"

    # Test next man up
    beneficiary = mgr.get_next_man_up("Christian McCaffrey")
    assert beneficiary is not None
    assert beneficiary["name"] == "Jordan Mason"

    # Test sidelined starters
    sidelined = mgr.get_all_sidelined_starters()
    assert len(sidelined) >= 1
    cmc_report = next((r for r in sidelined if r["injured_starter"]["name"] == "Christian McCaffrey"), None)
    assert cmc_report is not None
    assert cmc_report["promoted_beneficiary"]["name"] == "Jordan Mason"
    assert cmc_report["severity"] == "CRITICAL"


def test_llm_news_analyzer_heuristic(mock_sleeper_client):
    mgr = DepthChartManager(sleeper_client=mock_sleeper_client)
    analyzer = LLMNewsAnalyzer(depth_manager=mgr)

    # 1. Headline involving injured starter
    article = {
        "title": "49ers star Christian McCaffrey ruled out for next 4 weeks with calf strain",
        "summary": "McCaffrey will not play against the Rams as he recovers from Achilles injury.",
        "source": "ESPN NFL",
    }
    result = analyzer.analyze_single(article)
    assert result is not None
    assert result["player_name"] == "Christian McCaffrey"
    assert result["status"] == "Out"
    assert result["promoted_beneficiary"] == "Jordan Mason"
    assert "CRITICAL" in result["waiver_urgency"]
    assert "Jordan Mason" in result["fantasy_action"]

    # 2. General non-injury article should be ignored
    gen_article = {
        "title": "NFL power rankings: Chiefs remain on top following week 5",
        "summary": "Reviewing offensive and defensive team standings across the league.",
        "source": "CBS Sports",
    }
    assert analyzer.analyze_single(gen_article) is None


def test_clean_html_and_sleeper_notes(mock_sleeper_client):
    dirty_html = "<p>Player is <b>questionable</b> with knee swelling. <a href='#'>Read more</a></p>"
    cleaned = clean_html(dirty_html)
    assert "Player is questionable with knee swelling." in cleaned
    assert "<p>" not in cleaned

    collector = NewsCollector(sleeper_client=mock_sleeper_client)
    notes = collector.fetch_sleeper_injury_notes(positions=["RB"])
    assert len(notes) >= 1
    assert notes[0]["player_name"] == "Christian McCaffrey"
    assert notes[0]["injury_status"] == "Out"
