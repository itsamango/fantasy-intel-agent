"""Tests for the Sleeper per-week data path and its pbp enrichment."""

import pandas as pd
import pytest

import fantasy_intel.data.sleeper as sleeper_mod
from fantasy_intel.analytics.metrics import (
    RZ_IMPUTATION_DEFAULTS,
    attach_pbp_features,
    estimate_expected_fantasy_points,
)
from fantasy_intel.config import SCORING_HALF_PPR, SCORING_TEP
from fantasy_intel.data.loader import DataLoader


def _line(pid, team, pos, **stats):
    return {"player_id": pid, "team": team, "opponent": None, "game_id": None, "position": pos, "stats": stats}


class _WeeklySleeper:
    """SleeperClient stub serving two full weeks (2 teams) plus a partial week 3 (1 team)."""

    weeks = {
        1: [
            _line("1", "KC", "WR", off_snp=60, tm_off_snp=60, rec_tgt=9, rec_air_yd=60, pts_ppr=20, pts_half_ppr=16, rec=8),
            _line("2", "KC", "RB", off_snp=30, tm_off_snp=60, rec_tgt=3, rush_att=15, rush_rz_att=4, pts_ppr=12),
            _line("3", "BUF", "WR", off_snp=50, tm_off_snp=70, rec_tgt=10, rec_air_yd=100, pts_ppr=15),
            _line("4", "BUF", "TE", off_snp=0, tm_off_snp=70),  # inactive -> dropped
        ],
        2: [
            _line("1", "KC", "WR", off_snp=55, tm_off_snp=55, rec_tgt=6, pts_ppr=9),
            _line("3", "BUF", "WR", off_snp=60, tm_off_snp=60, rec_tgt=7, pts_ppr=11),
        ],
        3: [
            _line("1", "KC", "WR", off_snp=50, tm_off_snp=50, rec_tgt=5, pts_ppr=8),
        ],
    }

    def __init__(self, cache_dir=None):
        self.ttls = {}

    def get_nfl_state(self):
        return {"season": "2026", "week": 3, "season_type": "regular"}

    def get_all_players(self):
        return {
            "1": {"full_name": "Kc Wr", "position": "WR", "team": "KC"},
            "2": {"full_name": "Kc Rb", "position": "RB", "team": "KC"},
            "3": {"full_name": "Buf Wr", "position": "WR", "team": "BUF"},
            "4": {"full_name": "Buf Te", "position": "TE", "team": "BUF"},
        }

    def get_week_stats(self, season, week, ttl_hours=None, force_refresh=False):
        return self.weeks.get(week, [])


@pytest.fixture
def weekly_loader(monkeypatch, tmp_path):
    monkeypatch.setattr(sleeper_mod, "SleeperClient", _WeeklySleeper)
    loader = DataLoader(cache_dir=tmp_path)
    # 2 teams scheduled per week in this toy league
    monkeypatch.setattr(loader, "_expected_teams_by_week", lambda season: {1: 2, 2: 2, 3: 2})
    return loader


def test_sleeper_weekly_frame_shape_and_shares(weekly_loader):
    df = weekly_loader.get_sleeper_weekly_data(season=2026)

    # Partial week 3 (1 of 2 teams reported) is excluded; inactive TE dropped
    assert sorted(df["week"].unique()) == [1, 2]
    assert df.attrs["data_through_week"] == 2
    assert "4" not in set(df["player_id"])

    w1 = df[df["week"] == 1].set_index("player_id")
    assert w1.loc["2", "offense_pct"] == pytest.approx(0.5)  # real snaps / team snaps
    assert w1.loc["1", "target_share"] == pytest.approx(0.75)  # 9 of KC's 12 targets
    assert w1.loc["3", "target_share"] == pytest.approx(1.0)
    assert w1.loc["1", "wopr"] == pytest.approx(1.5 * 0.75 + 0.7 * 1.0)
    assert w1.loc["2", "rz_carries"] == 4
    assert w1.loc["1", "fantasy_points_half_ppr"] == 16


def test_sleeper_weekly_can_include_partial_week(weekly_loader):
    df = weekly_loader.get_sleeper_weekly_data(season=2026, include_partial_week=True)
    assert sorted(df["week"].unique()) == [1, 2, 3]


class _PbpLoader:
    def __init__(self, player_feats):
        self._feats = player_feats

    def get_pbp_opportunity_features(self, seasons=None):
        return {"player": self._feats}

    def get_player_ids(self):
        return pd.DataFrame([
            {"gsis_id": "00-0000001", "sleeper_id": "1"},
            {"gsis_id": "00-0000002", "sleeper_id": "2"},
            {"gsis_id": "00-0000009", "sleeper_id": "9"},
        ])


def _sleeper_rows():
    base = {"season": 2026, "season_type": "REG", "recent_team": "KC", "receptions": 0.0,
            "attempts": 0.0, "passing_yards": 0.0, "rz_targets": 0.0, "rz_pass_attempts": 0.0}
    return pd.DataFrame([
        # Week 1 matched in pbp; Sleeper air yards (completions only) = 20
        {**base, "player_id": "1", "player_display_name": "Kc Wr", "position": "WR", "week": 1,
         "targets": 8.0, "carries": 0.0, "receiving_air_yards": 20.0, "rz_carries": 0.0,
         "fantasy_points_ppr": 10.0},
        # Week 2 not yet in pbp: goal-line splits imputed from inside-20 carries
        {**base, "player_id": "2", "player_display_name": "Kc Rb", "position": "RB", "week": 2,
         "targets": 2.0, "carries": 18.0, "receiving_air_yards": 0.0, "rz_carries": 10.0,
         "fantasy_points_ppr": 14.0},
    ])


def test_attach_pbp_overrides_air_yards_and_imputes_red_zone():
    feats = pd.DataFrame([{
        "player_id": "00-0000001", "season": 2026, "week": 1,
        "carries_inside_10": 0, "carries_inside_5": 0, "targets_inside_10": 1, "end_zone_targets": 1,
        "high_value_touches": 8, "expected_receptions": 5.5, "pbp_rec_xfp_0ppr": 9.0,
        "pbp_rush_xfp": 0.0, "pass_attempts": 0, "pass_attempts_inside_10": 0, "pbp_pass_xfp": 0.0,
        "target_air_yards": 95.0, "pbp_schema_version": 2,
    }])
    out = attach_pbp_features(_sleeper_rows(), _PbpLoader(feats), seasons=[2026]).set_index("player_id")

    assert bool(out.loc["1", "pbp_matched"]) and not bool(out.loc["2", "pbp_matched"])
    # True target air yards replace completions-only Sleeper air yards on matched rows
    assert out.loc["1", "receiving_air_yards"] == 95.0
    # Unmatched row: too little matched volume to calibrate -> default ratios
    assert out.loc["2", "carries_inside_10"] == pytest.approx(10 * RZ_IMPUTATION_DEFAULTS["carries_inside_10"][1])
    assert out.loc["2", "carries_inside_5"] == pytest.approx(10 * RZ_IMPUTATION_DEFAULTS["carries_inside_5"][1])
    assert bool(out.loc["2", "pbp_imputed"])

    xfp = estimate_expected_fantasy_points(out.reset_index())
    assert (xfp["xfp_model"] == "situational").all()
    wr = xfp[xfp["player_id"] == "1"].iloc[0]
    # Matched receiving uses play-level xFP: 9.0 + 1.0 * 5.5 expected receptions
    assert wr["expected_fp_ppr"] == pytest.approx(14.5)


def test_qb_passing_xfp_is_per_row():
    rows = pd.DataFrame([
        {"player_id": "q", "position": "QB", "season": 2026, "week": 1, "targets": 0, "carries": 0,
         "attempts": 30, "passing_yards": 250, "pbp_pass_xfp": 15.0, "pbp_matched": True, "fantasy_points_ppr": 18.0},
        {"player_id": "q", "position": "QB", "season": 2026, "week": 2, "targets": 0, "carries": 0,
         "attempts": 30, "passing_yards": 250, "pbp_pass_xfp": 0.0, "pbp_matched": False, "fantasy_points_ppr": 18.0},
    ])
    out = estimate_expected_fantasy_points(rows).sort_values("week")
    assert out.iloc[0]["expected_fp_ppr"] == pytest.approx(15.0)
    # Unmatched week falls back to box score instead of collapsing to 0
    assert out.iloc[1]["expected_fp_ppr"] > 5.0


def test_fpoe_uses_league_scoring_points():
    rows = pd.DataFrame([
        {"player_id": "w", "position": "WR", "season": 2026, "week": 1, "targets": 8, "carries": 0,
         "receptions": 6, "fantasy_points_ppr": 16.0, "fantasy_points_half_ppr": 13.0},
    ])
    out = estimate_expected_fantasy_points(rows, scoring=SCORING_HALF_PPR).iloc[0]
    assert out["fantasy_points_league"] == 13.0
    assert out["fp_diff"] == pytest.approx(13.0 - out["expected_fp_ppr"], abs=0.01)


def test_tep_premium_applies_to_tight_ends_only():
    base = {"season": 2026, "week": 1, "targets": 8, "carries": 0, "receptions": 6,
            "fantasy_points_ppr": 12.0, "carries_inside_10": 0, "carries_inside_5": 0, "end_zone_targets": 0}
    rows = pd.DataFrame([{**base, "player_id": "te", "position": "TE"}, {**base, "player_id": "wr", "position": "WR"}])
    out = estimate_expected_fantasy_points(rows, scoring=SCORING_TEP).set_index("player_id")
    assert out.loc["te", "expected_fp_ppr"] > out.loc["wr", "expected_fp_ppr"]
    assert out.loc["te", "fantasy_points_league"] == pytest.approx(15.0)
    assert out.loc["wr", "fantasy_points_league"] == pytest.approx(12.0)
