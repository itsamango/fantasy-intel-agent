"""Unit tests for fantasy-intel analytics engine."""

import pandas as pd
from fantasy_intel.analytics.metrics import compute_player_trajectories, estimate_expected_fantasy_points
from fantasy_intel.analytics.waiver_wire import WaiverWireEngine
from fantasy_intel.analytics.trades import TradeAnalyzer


def test_metrics_calculation():
    data = pd.DataFrame([
        {
            "player_display_name": "Test WR",
            "position": "WR",
            "recent_team": "KC",
            "season": 2024,
            "week": 1,
            "offense_pct": 0.40,
            "targets": 4,
            "carries": 0,
            "target_share": 0.12,
            "air_yards_share": 0.15,
            "fantasy_points_ppr": 6.0,
        },
        {
            "player_display_name": "Test WR",
            "position": "WR",
            "recent_team": "KC",
            "season": 2024,
            "week": 2,
            "offense_pct": 0.75,
            "targets": 9,
            "carries": 0,
            "target_share": 0.25,
            "air_yards_share": 0.35,
            "fantasy_points_ppr": 8.5,
        },
    ])

    trajectories = compute_player_trajectories(data)
    assert "snap_delta" in trajectories.columns
    # Week 2 should show a positive snap delta surge
    assert trajectories.iloc[1]["snap_delta"] > 0

    xfp_df = estimate_expected_fantasy_points(trajectories)
    assert "expected_fp_ppr" in xfp_df.columns
    assert "fp_diff" in xfp_df.columns


def test_waiver_engine_filtering():
    data = pd.DataFrame([
        {
            "player_display_name": "Rising Sleeper",
            "position": "WR",
            "recent_team": "LAR",
            "season": 2024,
            "week": 1,
            "offense_pct": 0.65,
            "snap_delta": 0.20,
            "targets": 7,
            "carries": 0,
            "target_share": 0.20,
            "air_yards_share": 0.25,
            "wopr": 0.45,
            "fantasy_points_ppr": 8.0,
            "recent_ppr": 8.0,
        }
    ])

    engine = WaiverWireEngine(data)
    sleepers = engine.scan_sleepers(season=2024, week=1, min_opp_index=30.0)
    assert not sleepers.empty
    assert sleepers.iloc[0]["player_display_name"] == "Rising Sleeper"
    assert sleepers.iloc[0]["opportunity_index"] > 30.0


def test_depth_chart_elevation():
    from fantasy_intel.analytics.depth_chart import DepthChartManager

    class DummySleeper:
        def get_all_players(self):
            return {
                "1": {"player_id": "1", "full_name": "Star RB", "team": "PHI", "position": "RB", "depth_chart_order": 1, "injury_status": "Out", "injury_body_part": "Hamstring"},
                "2": {"player_id": "2", "full_name": "Backup RB", "team": "PHI", "position": "RB", "depth_chart_order": 2, "injury_status": None},
                "3": {"player_id": "3", "full_name": "Injured Stash WR", "team": "NYJ", "position": "WR", "depth_chart_order": 3, "injury_status": "Out", "injury_body_part": "Finger"},
            }

    dcm = DepthChartManager(sleeper_client=DummySleeper())
    # Backup RB with Star RB Out should be elevated
    ctx = dcm.get_player_elevation_context("Backup RB")
    assert ctx["is_elevated"] is True
    assert ctx["tier"] == "ELEVATED_STARTER"

    # Injured WR should be marked IR eligible
    ctx_wr = dcm.get_player_elevation_context("Injured Stash WR")
    assert ctx_wr["ir_eligible"] is True
    assert ctx_wr["own_injury"] == "Out"
