"""Targeted tests for Sleeper-first depth context and season/team-guarded enrichment."""

import os
import time

import pandas as pd

import fantasy_intel.data.loader as loader_mod
from fantasy_intel.analytics.depth_chart import DepthChartManager


class _StubSleeper:
    def __init__(self, players, week=6):
        self._players = players
        self._week = week

    def get_all_players(self):
        return self._players

    def get_nfl_state(self):
        return {"season": 2026, "week": self._week, "display_week": self._week}


class _StubNFLverseLoader:
    def __init__(self, depth_df=None, injuries_df=None):
        self.depth_df = depth_df if depth_df is not None else pd.DataFrame()
        self.injuries_df = injuries_df if injuries_df is not None else pd.DataFrame()

    def get_depth_charts(self, seasons=None, force_refresh=False):
        return self.depth_df.copy()

    def get_injuries(self, seasons=None, force_refresh=False):
        return self.injuries_df.copy()


def _raise(seasons):
    raise RuntimeError("nflverse down")


# ---------------------------------------------------------------------------
# Elevation narrative: an established starter is never "elevated"
# ---------------------------------------------------------------------------

def test_established_starter_not_elevated_by_injured_backup():
    """Diggs-style bug: a starter-tier WR must not be called elevated because a
    backup ranked behind him (order 7) went on IR."""
    players = {
        "1": {"full_name": "Top Wr", "team": "WAS", "position": "WR", "depth_chart_order": 1, "depth_chart_position": "LWR", "injury_status": None},
        "2": {"full_name": "Stefon Diggs", "team": "WAS", "position": "WR", "depth_chart_order": 2, "depth_chart_position": "SWR", "injury_status": None},
        "3": {"full_name": "Jaylin Lane", "team": "WAS", "position": "WR", "depth_chart_order": 7, "injury_status": "IR", "injury_body_part": "Ankle"},
    }
    mgr = DepthChartManager(sleeper_client=_StubSleeper(players))

    ctx = mgr.get_player_elevation_context("Stefon Diggs")
    assert ctx["tier"] == "ENTRENCHED_STARTER"
    assert ctx["is_elevated"] is False
    assert "ELEVATED" not in ctx["summary"]
    assert "Jaylin Lane" not in ctx["summary"]
    assert ctx["healthy_role"] == "WAS WR2 (SWR)"
    assert "WR2 (SWR) for WAS" in ctx["summary"]
    assert "full-time starter role" in ctx["summary"]


def test_wr2_elevated_only_when_wr_ahead_is_out():
    """Elevation fires only for a teammate ranked AHEAD being ruled out; an
    injured backup ranked behind stays out of the narrative."""
    players = {
        "1": {"full_name": "Top Wr", "team": "PHI", "position": "WR", "depth_chart_order": 1, "injury_status": "Out", "injury_body_part": "Hamstring"},
        "2": {"full_name": "Dontayvion Wicks", "team": "PHI", "position": "WR", "depth_chart_order": 2, "depth_chart_position": "RWR", "injury_status": None},
        "3": {"full_name": "Deep Wr", "team": "PHI", "position": "WR", "depth_chart_order": 7, "injury_status": "Out", "injury_body_part": "Coach's Decision"},
    }
    mgr = DepthChartManager(sleeper_client=_StubSleeper(players))

    ctx = mgr.get_player_elevation_context("Dontayvion Wicks")
    assert ctx["tier"] == "ELEVATED_STARTER"
    assert ctx["is_elevated"] is True
    assert "Top Wr" in ctx["summary"]
    assert "Deep Wr" not in ctx["summary"]
    assert [p["name"] for p in ctx["sidelined_ahead"]] == ["Top Wr"]


def test_unlisted_depth_gets_muted_narrative():
    """A player no source lists a slot for gets a muted role note instead of an
    invented vacancy claim (old logic fabricated 'WR1 workload' for him)."""
    players = {
        "1": {"full_name": "Top Wr", "team": "KC", "position": "WR", "depth_chart_order": 1, "injury_status": "Out", "injury_body_part": "Knee"},
        "2": {"full_name": "Mystery Wr", "team": "KC", "position": "WR", "injury_status": None},
    }
    mgr = DepthChartManager(sleeper_client=_StubSleeper(players))

    ctx = mgr.get_player_elevation_context("Mystery Wr")
    assert ctx["tier"] == "DEPTH_UNLISTED"
    assert ctx["is_elevated"] is False
    assert "ELEVATED" not in ctx["summary"]
    assert ctx["sidelined_ahead"] == []


# ---------------------------------------------------------------------------
# Season-checked seasonal caches in DataLoader
# ---------------------------------------------------------------------------

def test_stale_season_depth_cache_rejected_and_refetched(tmp_path, monkeypatch):
    """A cache holding another season's rows must never be served for the
    requested season: it is ignored, refetched, and only a valid current-season
    frame is cached and served afterwards."""
    stale = pd.DataFrame([
        {"season": 2024, "club_code": "PHI", "week": 1, "full_name": "Johnny Wilson", "position": "WR", "depth_team": 2},
    ])
    stale.to_parquet(tmp_path / "depth_charts_2026.parquet", index=False)

    calls = []

    def failing_fetch(seasons):
        calls.append(list(seasons))
        raise RuntimeError("nflverse down")

    monkeypatch.setattr(loader_mod.nfl, "import_depth_charts", failing_fetch)
    loader = loader_mod.DataLoader(cache_dir=tmp_path)

    out = loader.get_depth_charts([2026])
    assert out.empty
    assert calls == [[2026]]  # stale cache forced a refetch attempt

    fresh = pd.DataFrame([
        {"season": 2026, "club_code": "KC", "week": 5, "full_name": "Live Wr", "position": "WR", "depth_team": 1},
    ])
    monkeypatch.setattr(loader_mod.nfl, "import_depth_charts", lambda seasons: fresh.copy())
    out = loader.get_depth_charts([2026])
    assert not out.empty
    assert (out["season"] == 2026).all()

    # Fresh cache is now served without refetching
    monkeypatch.setattr(loader_mod.nfl, "import_depth_charts", failing_fetch)
    out = loader.get_depth_charts([2026])
    assert not out.empty
    assert (out["season"] == 2026).all()
    assert calls == [[2026]]


def test_injuries_stale_season_cache_rejected(tmp_path, monkeypatch):
    stale = pd.DataFrame([
        {"season": 2025, "team": "PHI", "week": 3, "full_name": "Marquise Brown", "position": "WR", "report_status": "Out"},
    ])
    stale.to_parquet(tmp_path / "injuries_2026.parquet", index=False)
    monkeypatch.setattr(loader_mod.nfl, "import_injuries", _raise)

    loader = loader_mod.DataLoader(cache_dir=tmp_path)
    assert loader.get_injuries([2026]).empty


def test_seasonal_cache_ttl_expired_refetches(tmp_path, monkeypatch):
    fresh = pd.DataFrame([
        {"season": 2026, "team": "KC", "week": 5, "full_name": "Live Wr", "position": "WR", "report_status": "Out"},
    ])
    cache = tmp_path / "injuries_2026.parquet"
    fresh.to_parquet(cache, index=False)
    expired = time.time() - 49 * 3600
    os.utime(cache, (expired, expired))

    monkeypatch.setattr(loader_mod.nfl, "import_injuries", _raise)
    loader = loader_mod.DataLoader(cache_dir=tmp_path)
    assert loader.get_injuries([2026]).empty


# ---------------------------------------------------------------------------
# Team-change guard and injury enrichment inside DepthChartManager
# ---------------------------------------------------------------------------

def test_team_mismatch_depth_rows_discarded():
    """nflverse rows that contradict a player's live Sleeper team are stale and
    must be discarded; matching-team rows may fill unlisted slots only."""
    players = {
        "1": {"full_name": "Mover Wr", "team": "KC", "position": "WR", "injury_status": None},
        "2": {"full_name": "Loyal Wr", "team": "KC", "position": "WR", "injury_status": None},
        "3": {"full_name": "Anchor Wr", "team": "KC", "position": "WR", "depth_chart_order": 1, "injury_status": None},
    }
    depth_df = pd.DataFrame([
        {"season": 2026, "club_code": "PHI", "week": 6, "formation": "Offense", "position": "WR", "depth_position": "WR", "depth_team": 1, "full_name": "Mover Wr"},
        {"season": 2026, "club_code": "KC", "week": 6, "formation": "Offense", "position": "WR", "depth_position": "WR", "depth_team": 3, "full_name": "Loyal Wr"},
        {"season": 2026, "club_code": "KC", "week": 6, "formation": "Offense", "position": "WR", "depth_position": "WR", "depth_team": 5, "full_name": "Anchor Wr"},
    ])
    loader = _StubNFLverseLoader(depth_df=depth_df)
    mgr = DepthChartManager(sleeper_client=_StubSleeper(players), season=2026, loader=loader)

    mover = mgr.get_player_elevation_context("Mover Wr")
    assert mover["tier"] == "DEPTH_UNLISTED"  # PHI row discarded as stale
    assert mover["is_elevated"] is False
    assert "ELEVATED" not in mover["summary"]

    loyal = mgr.get_player_elevation_context("Loyal Wr")
    assert loyal["depth_order"] == 3  # matching-team row fills the gap
    assert loyal["tier"] == "BENCH_BACKUP"

    anchor = mgr.find_player("Anchor Wr")
    assert anchor["depth_order"] == 1  # Sleeper-provided order is never overridden


def test_injury_enrichment_fills_gaps_only_current_week_right_team():
    players = {
        "1": {"full_name": "Set Wr", "team": "KC", "position": "WR", "depth_chart_order": 1, "injury_status": "Questionable", "injury_body_part": "Undisclosed"},
        "2": {"full_name": "Gap Wr", "team": "KC", "position": "WR", "depth_chart_order": 2, "injury_status": None},
        "3": {"full_name": "Else Wr", "team": "DAL", "position": "WR", "depth_chart_order": 1, "injury_status": None},
    }
    injuries = pd.DataFrame([
        {"season": 2026, "team": "KC", "week": 5, "position": "WR", "full_name": "Gap Wr", "report_primary_injury": "Ankle", "report_status": "Out"},
        {"season": 2026, "team": "KC", "week": 6, "position": "WR", "full_name": "Gap Wr", "report_primary_injury": "Knee", "report_status": "Out"},
        {"season": 2026, "team": "KC", "week": 6, "position": "WR", "full_name": "Set Wr", "report_primary_injury": "Ankle", "report_status": "Out"},
        {"season": 2026, "team": "PHI", "week": 6, "position": "WR", "full_name": "Else Wr", "report_primary_injury": "Foot", "report_status": "Out"},
    ])
    loader = _StubNFLverseLoader(injuries_df=injuries)
    mgr = DepthChartManager(sleeper_client=_StubSleeper(players, week=6), season=2026, loader=loader)

    mgr.get_player_elevation_context("Gap Wr")  # triggers the one-shot enrichment

    gap = mgr.find_player("Gap Wr")
    assert gap["injury_status"] == "Out"  # filled from the live week's report
    assert gap["injury_body_part"] == "Knee"  # last week's Ankle row ignored

    set_wr = mgr.find_player("Set Wr")
    assert set_wr["injury_status"] == "Questionable"  # Sleeper value wins

    else_wr = mgr.find_player("Else Wr")
    assert else_wr["injury_status"] is None  # wrong-team row discarded


# ---------------------------------------------------------------------------
# Practice-squad players never count as sidelined starters
# ---------------------------------------------------------------------------

def test_practice_squad_teammate_does_not_elevate_anyone():
    """A PS player (Sleeper status 'Practice Squad') ranked ahead and ruled Out
    does not elevate the player behind him, and is not listed as a sidelined
    starter himself."""
    players = {
        "1": {"full_name": "Ps Wr", "team": "KC", "position": "WR", "depth_chart_order": 1, "status": "Practice Squad", "injury_status": "Out", "injury_body_part": "Coach's Decision"},
        "2": {"full_name": "Real Wr", "team": "KC", "position": "WR", "depth_chart_order": 2, "depth_chart_position": "RWR", "injury_status": None},
    }
    mgr = DepthChartManager(sleeper_client=_StubSleeper(players))

    ctx = mgr.get_player_elevation_context("Real Wr")
    assert ctx["tier"] == "ENTRENCHED_STARTER"
    assert ctx["is_elevated"] is False
    assert "ELEVATED" not in ctx["summary"]
    assert "Ps Wr" not in ctx["summary"]
    assert ctx["sidelined_ahead"] == []
    assert "WR2 (RWR) for KC" in ctx["summary"]

    # The PS player himself is never reported as a sidelined starter
    assert mgr.get_all_sidelined_starters() == []


def test_next_man_up_skips_practice_squad_beneficiary():
    """The direct beneficiary skips a PS player ranked in between."""
    players = {
        "1": {"full_name": "Star Wr", "team": "KC", "position": "WR", "depth_chart_order": 1, "injury_status": "Out", "injury_body_part": "Hamstring"},
        "2": {"full_name": "Ps Wr", "team": "KC", "position": "WR", "depth_chart_order": 2, "status": "Practice Squad"},
        "3": {"full_name": "Real Backup", "team": "KC", "position": "WR", "depth_chart_order": 3},
    }
    mgr = DepthChartManager(sleeper_client=_StubSleeper(players))

    beneficiary = mgr.get_next_man_up("Star Wr")
    assert beneficiary is not None
    assert beneficiary["name"] == "Real Backup"


def test_practice_squad_candidate_gets_muted_narrative():
    """A PS waiver candidate himself is never an immediate/elevated starter:
    he must be activated to play at all."""
    players = {
        "1": {"full_name": "Real Wr", "team": "KC", "position": "WR", "depth_chart_order": 1, "injury_status": "Out", "injury_body_part": "Knee"},
        "2": {"full_name": "Ps Wr", "team": "KC", "position": "WR", "depth_chart_order": 3, "status": "Practice Squad"},
    }
    mgr = DepthChartManager(sleeper_client=_StubSleeper(players))

    ctx = mgr.get_player_elevation_context("Ps Wr")
    assert ctx["tier"] == "PRACTICE_SQUAD"
    assert ctx["is_elevated"] is False
    assert "ELEVATED" not in ctx["summary"]
    assert "practice-squad" in ctx["summary"]
    assert ctx["sidelined_ahead"] == []
