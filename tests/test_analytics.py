"""Unit tests for fantasy-intel analytics engine."""

import numpy as np
import pandas as pd
from fantasy_intel.analytics.metrics import (
    add_projection_distributions,
    attach_pbp_features,
    compute_player_trajectories,
    estimate_expected_fantasy_points,
)
from fantasy_intel.analytics.waiver_wire import WaiverWireEngine
from fantasy_intel.analytics.trades import TradeAnalyzer


def test_metrics_calculation():
    data = pd.DataFrame([
        {
            "player_id": "00-0000001",
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
            "player_id": "00-0000001",
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


def _rb_row(player_id, name, carries, carries_inside_10, carries_inside_5):
    return {
        "player_id": player_id,
        "player_display_name": name,
        "position": "RB",
        "recent_team": "KC",
        "season": 2024,
        "week": 1,
        "offense_pct": 0.50,
        "targets": 0,
        "carries": carries,
        "receiving_air_yards": 0.0,
        "receiving_yards": 0.0,
        "fantasy_points_ppr": 10.0,
        "carries_inside_10": carries_inside_10,
        "carries_inside_5": carries_inside_5,
        "end_zone_targets": 0,
    }


def _wr_row(player_id, name, targets, end_zone_targets):
    return {
        "player_id": player_id,
        "player_display_name": name,
        "position": "WR",
        "recent_team": "KC",
        "season": 2024,
        "week": 1,
        "offense_pct": 0.80,
        "targets": targets,
        "carries": 0,
        "receiving_air_yards": 50.0,
        "receiving_yards": 30.0,
        "fantasy_points_ppr": 12.0,
        "carries_inside_10": 0,
        "carries_inside_5": 0,
        "end_zone_targets": end_zone_targets,
    }


def test_situational_xfp_ordering():
    data = pd.DataFrame([
        _rb_row("00-0000001", "Goal Line RB", carries=2, carries_inside_10=2, carries_inside_5=2),
        _rb_row("00-0000002", "Red Zone RB", carries=2, carries_inside_10=2, carries_inside_5=0),
        _rb_row("00-0000003", "Regular RB", carries=2, carries_inside_10=0, carries_inside_5=0),
        _wr_row("00-0000004", "EZ Target WR", targets=5, end_zone_targets=1),
        _wr_row("00-0000005", "Regular WR", targets=5, end_zone_targets=0),
    ])

    xfp = estimate_expected_fantasy_points(data)
    by_name = xfp.set_index("player_display_name")["expected_fp_ppr"]

    # Goal-line carry > red-zone carry > regular carry
    assert by_name["Goal Line RB"] > by_name["Red Zone RB"] > by_name["Regular RB"]
    assert abs(by_name["Goal Line RB"] - 2 * 2.65) < 1e-6
    assert abs(by_name["Red Zone RB"] - 2 * 1.45) < 1e-6
    assert abs(by_name["Regular RB"] - 2 * 0.45) < 1e-6

    # End-zone target premium applied on top of the regular target valuation
    assert by_name["EZ Target WR"] > by_name["Regular WR"]
    assert abs(by_name["Regular WR"] - (5 * 1.17 + 50 * 0.075)) < 1e-6
    assert abs(by_name["EZ Target WR"] - (4 * 1.17 + 50 * 0.075 + 2.57)) < 1e-6

    assert (xfp["xfp_model"] == "situational").all()
    assert (xfp["fpoe"] == xfp["fp_diff"]).all()
    assert "rolling_xfp_3w" in xfp.columns
    assert "rolling_fpoe_3w" in xfp.columns


def test_xfp_scoring_weights():
    data = pd.DataFrame([_wr_row("00-0000005", "Regular WR", targets=5, end_zone_targets=0)])

    ppr = estimate_expected_fantasy_points(data, scoring="ppr").iloc[0]["expected_fp_ppr"]
    half = estimate_expected_fantasy_points(data, scoring="half_ppr").iloc[0]["expected_fp_ppr"]
    std = estimate_expected_fantasy_points(data, scoring="standard").iloc[0]["expected_fp_ppr"]

    assert std < half < ppr


def test_xfp_fallback_without_pbp_columns():
    data = pd.DataFrame([
        {
            "player_id": "00-0000001",
            "player_display_name": "Volume WR",
            "position": "WR",
            "recent_team": "KC",
            "season": 2024,
            "week": 1,
            "offense_pct": 0.50,
            "targets": 5,
            "carries": 1,
            "receiving_air_yards": 100.0,
            "fantasy_points_ppr": 10.0,
        }
    ])

    xfp = estimate_expected_fantasy_points(data)
    assert (xfp["xfp_model"] == "linear_fallback").all()
    assert abs(xfp.iloc[0]["expected_fp_ppr"] - (5 * 1.3 + 100 * 0.08 + 1 * 0.68)) < 1e-6

    # All-NaN situational columns also trigger the fallback
    data["carries_inside_10"] = np.nan
    data["carries_inside_5"] = np.nan
    data["end_zone_targets"] = np.nan
    xfp_nan = estimate_expected_fantasy_points(data)
    assert (xfp_nan["xfp_model"] == "linear_fallback").all()


def test_diagnostic_metrics_math():
    data = pd.DataFrame([
        # Two different players sharing a display name must not collide
        {"player_id": "00-0000001", "player_display_name": "Same Name", "position": "WR", "recent_team": "KC", "season": 2024, "week": 1, "offense_pct": 0.50, "targets": 6, "carries": 0, "receiving_air_yards": 100.0, "receiving_yards": 70.0, "fantasy_points_ppr": 10.0, "carries_inside_10": 0},
        {"player_id": "00-0000002", "player_display_name": "Same Name", "position": "WR", "recent_team": "BUF", "season": 2024, "week": 1, "offense_pct": 0.50, "targets": 8, "carries": 0, "receiving_air_yards": 120.0, "receiving_yards": 60.0, "fantasy_points_ppr": 20.0, "carries_inside_10": 0},
        {"player_id": "00-0000001", "player_display_name": "Same Name", "position": "WR", "recent_team": "KC", "season": 2024, "week": 2, "offense_pct": 0.60, "targets": 6, "carries": 0, "receiving_air_yards": 80.0, "receiving_yards": 40.0, "fantasy_points_ppr": 12.0, "carries_inside_10": 0},
        {"player_id": "00-0000002", "player_display_name": "Same Name", "position": "WR", "recent_team": "BUF", "season": 2024, "week": 2, "offense_pct": 0.60, "targets": 8, "carries": 0, "receiving_air_yards": 100.0, "receiving_yards": 50.0, "fantasy_points_ppr": 22.0, "carries_inside_10": 0},
        {"player_id": "00-0000003", "player_display_name": "HVT RB", "position": "RB", "recent_team": "KC", "season": 2024, "week": 2, "offense_pct": 0.50, "targets": 4, "carries": 10, "receiving_air_yards": 20.0, "receiving_yards": 15.0, "fantasy_points_ppr": 18.0, "carries_inside_10": 3},
    ])

    traj = compute_player_trajectories(data)
    p1 = traj[traj["player_id"] == "00-0000001"].sort_values("week")
    p2 = traj[traj["player_id"] == "00-0000002"].sort_values("week")

    # Rolling stats stay within each same-named player's own weeks
    assert p1.iloc[1]["recent_ppr"] == 11.0
    assert p2.iloc[1]["recent_ppr"] == 21.0

    # racr and unrealized air yards
    assert abs(p1.iloc[0]["racr"] - 0.7) < 1e-9
    assert abs(p1.iloc[0]["unrealized_air_yards"] - 30.0) < 1e-9

    # hvt = targets + carries inside the 10
    hvt_row = traj[traj["player_id"] == "00-0000003"].iloc[0]
    assert hvt_row["hvt"] == 7

    # hvt_share = player hvt / team hvt that week (KC week 2: 6 + 7 = 13)
    assert abs(hvt_row["hvt_share"] - 7.0 / 13.0) < 1e-3


class _DummyLoader:
    def __init__(self, features=None):
        self._features = features

    def get_pbp_opportunity_features(self, seasons=None):
        return {"player": self._features if self._features is not None else pd.DataFrame()}

    def get_player_ids(self):
        return pd.DataFrame([
            {"gsis_id": "00-0000001", "pfr_id": "TestBa01", "sleeper_id": "9999", "name": "Test Back", "position": "RB", "team": "KC", "norm_name": "test back"},
        ])


_PBP_FEATURES = pd.DataFrame([
    {"player_id": "00-0000001", "season": 2024, "week": 1, "carries_inside_10": 2, "carries_inside_5": 1, "targets_inside_10": 1, "end_zone_targets": 1, "high_value_touches": 3},
])


def test_attach_pbp_features_gsis_keyed():
    weekly = pd.DataFrame([
        {"player_id": "00-0000001", "player_display_name": "Test Back", "position": "RB", "recent_team": "KC", "season": 2024, "week": 1, "targets": 4, "carries": 8, "receiving_air_yards": 30.0, "fantasy_points_ppr": 15.0},
    ])

    enriched = attach_pbp_features(weekly, _DummyLoader(_PBP_FEATURES), seasons=[2024])
    assert enriched.iloc[0]["carries_inside_10"] == 2
    assert enriched.iloc[0]["high_value_touches"] == 3

    xfp = estimate_expected_fantasy_points(enriched)
    assert (xfp["xfp_model"] == "situational").all()


def test_attach_pbp_features_sleeper_keyed():
    sleeper_frame = pd.DataFrame([
        {"player_id": "9999", "player_display_name": "Test Back", "position": "RB", "recent_team": "KC", "season": 2024, "week": 1, "targets": 4, "carries": 8, "receiving_air_yards": 30.0, "fantasy_points_ppr": 15.0},
    ])

    enriched = attach_pbp_features(sleeper_frame, _DummyLoader(_PBP_FEATURES), seasons=[2024])
    assert enriched.iloc[0]["gsis_id"] == "00-0000001"
    assert enriched.iloc[0]["carries_inside_10"] == 2
    assert enriched.iloc[0]["end_zone_targets"] == 1


def test_attach_pbp_features_noop_without_features():
    weekly = pd.DataFrame([
        {"player_id": "00-0000001", "player_display_name": "Test Back", "position": "RB", "recent_team": "KC", "season": 2024, "week": 1, "targets": 4, "carries": 8, "receiving_air_yards": 30.0, "fantasy_points_ppr": 15.0},
    ])

    enriched = attach_pbp_features(weekly, _DummyLoader(None), seasons=[2024])
    assert "carries_inside_10" not in enriched.columns

    xfp = estimate_expected_fantasy_points(enriched)
    assert (xfp["xfp_model"] == "linear_fallback").all()


def test_projection_distributions():
    rows = []
    # Starter: 3 weeks of real weekly variance -> trailing sigma used from week 3 on
    for week, pts in [(1, 10.0), (2, 16.0), (3, 8.0)]:
        rows.append({**_wr_row("00-0000001", "Starter WR", targets=8, end_zone_targets=1), "week": week, "fantasy_points_ppr": pts})
    # Bench player with zero volume -> mu = 0 -> floored outputs
    rows.append({**_rb_row("00-0000002", "Bench RB", carries=0, carries_inside_10=0, carries_inside_5=0), "fantasy_points_ppr": 0.0, "offense_pct": 0.05})

    dist = add_projection_distributions(estimate_expected_fantasy_points(pd.DataFrame(rows)))

    # Floor <= median <= ceiling ordering and zero floor everywhere
    assert (dist["proj_floor_p15"] <= dist["proj_median_p50"]).all()
    assert (dist["proj_median_p50"] <= dist["proj_ceiling_p85"]).all()
    assert (dist[["proj_floor_p15", "proj_median_p50", "proj_ceiling_p85"]] >= 0).all().all()

    bench = dist[dist["player_id"] == "00-0000002"].iloc[0]
    assert bench["proj_floor_p15"] == 0.0
    assert bench["proj_median_p50"] == 0.0
    assert bench["proj_ceiling_p85"] == 0.0

    starter = dist[dist["player_id"] == "00-0000001"].sort_values("week")
    # Default sigma = 0.42 * mu on a single sample puts the log-normal median below the mean
    assert starter.iloc[0]["proj_median_p50"] < starter.iloc[0]["rolling_xfp_3w"]
    assert starter.iloc[0]["proj_ceiling_p85"] > starter.iloc[0]["proj_median_p50"]
    # Trailing weekly std (10, 16, 8 -> 4.0) is used from week 3 on: ceiling stays above the mean
    assert starter.iloc[2]["proj_ceiling_p85"] > starter.iloc[2]["rolling_xfp_3w"]
