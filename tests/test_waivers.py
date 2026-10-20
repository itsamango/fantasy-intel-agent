"""Unit tests for the waiver wire engine and FAAB bid sizing."""

import pandas as pd

from fantasy_intel.agent.advisor import FantasyAdvisor
from fantasy_intel.analytics.waiver_wire import WaiverWireEngine, recommend_faab_bids


def _skill_row(
    pid,
    name,
    pos,
    week,
    offense_pct,
    targets,
    carries,
    target_share,
    air_yards_share,
    air_yards,
    rec_yards,
    fppg,
    carries_inside_10=0,
    carries_inside_5=0,
    end_zone_targets=0,
    team="KC",
):
    return {
        "player_id": pid,
        "player_display_name": name,
        "position": pos,
        "recent_team": team,
        "season": 2024,
        "week": week,
        "offense_pct": offense_pct,
        "targets": targets,
        "carries": carries,
        "target_share": target_share,
        "air_yards_share": air_yards_share,
        "receiving_air_yards": air_yards,
        "receiving_yards": rec_yards,
        "fantasy_points_ppr": fppg,
        "carries_inside_10": carries_inside_10,
        "carries_inside_5": carries_inside_5,
        "end_zone_targets": end_zone_targets,
    }


def _engine_frame():
    rows = []
    # RB and WR with identical raw counting stats: the RB scores via the
    # HVT formula despite a near-zero WOPR, the WR scores via WOPR
    for week, off in [(1, 0.5), (2, 0.9)]:
        rows.append(_skill_row("00-0000002", "Hvt Back", "RB", week, off, 4, 14, 0.10, 0.05, 10, 5, 10, carries_inside_10=6, carries_inside_5=2))
        rows.append(_skill_row("00-0000001", "Air Wr", "WR", week, off, 4, 14, 0.30, 0.43, 100, 30, 10, team="BUF"))
    # QB keeps a sensible snap/touch blend
    rows.append(_skill_row("00-0000003", "Pocket Passer", "QB", 1, 0.75, 8, 2, 0.10, 0.10, 30, 20, 12, team="PHI"))
    rows.append(_skill_row("00-0000003", "Pocket Passer", "QB", 2, 0.95, 8, 2, 0.10, 0.10, 30, 20, 12, team="PHI"))
    # Ceiling players: every component above its cap must stop at 100
    rows.append(_skill_row("00-0000004", "Maxed Wr", "WR", 1, 0.5, 5, 0, 0.30, 0.43, 100, 30, 12, team="LAR"))
    rows.append(_skill_row("00-0000004", "Maxed Wr", "WR", 2, 1.0, 5, 0, 0.50, 0.50, 150, 50, 12, team="LAR"))
    rows.append(_skill_row("00-0000005", "Maxed Rb", "RB", 1, 0.5, 6, 24, 0.05, 0.02, 10, 5, 12, carries_inside_10=20, carries_inside_5=10, team="PHI"))
    rows.append(_skill_row("00-0000005", "Maxed Rb", "RB", 2, 1.0, 6, 24, 0.05, 0.02, 10, 5, 12, carries_inside_10=20, carries_inside_5=10, team="PHI"))
    return pd.DataFrame(rows)


def test_position_specific_opportunity_index():
    engine = WaiverWireEngine(_engine_frame())
    sleepers = engine.scan_sleepers(season=2024, week=2)
    by_name = sleepers.set_index("player_display_name")

    # Same touches/targets/carries: RB scores high via HVT despite WOPR 0.19,
    # WR scores high via WOPR — a single formula would lock one of them out
    assert by_name.loc["Hvt Back", "wopr"] < 0.20
    assert by_name.loc["Hvt Back", "hvt"] == 10
    assert by_name.loc["Hvt Back", "opportunity_index"] == 97.5
    assert by_name.loc["Air Wr", "opportunity_index"] == 97.5

    # QBs are not broken by the position split
    assert 50 < by_name.loc["Pocket Passer", "opportunity_index"] < 60

    # Values above every cap clip at exactly 100
    assert by_name.loc["Maxed Wr", "opportunity_index"] == 100.0
    assert by_name.loc["Maxed Rb", "opportunity_index"] == 100.0
    assert (sleepers["opportunity_index"] <= 100).all()

    # Engine output carries the columns the FAAB/CLI layers need
    for col in ("player_id", "hvt", "racr", "unrealized_air_yards", "rolling_xfp_3w"):
        assert col in sleepers.columns
    assert by_name.loc["Air Wr", "unrealized_air_yards"] == 70
    assert by_name.loc["Hvt Back", "rolling_xfp_3w"] > 0


def _user_roster(rows=None, owner="u1", waiver_position=None):
    base = {
        "player_id": "p1",
        "player_display_name": "Roster WR",
        "position": "WR",
        "rolling_xfp_3w": 6.0,
        "is_starter": False,
        "injury_status": None,
        "owner_id": owner,
        "roster_id": 1,
        "waiver_position": waiver_position,
    }
    return pd.DataFrame([base] if rows is None else rows)


def _league(waiver_type=1, budget=100, wins=5, losses=5, b_rem=100, owner="u1"):
    return {
        "league_id": "L1",
        "waiver_type": waiver_type,
        "faab_budget": budget,
        "rosters": [
            {
                "roster_id": 1,
                "owner_id": owner,
                "wins": wins,
                "losses": losses,
                "ties": 0,
                "faab_remaining": b_rem,
            }
        ],
    }


def _candidates(rows):
    return pd.DataFrame(
        [{"player_display_name": f"Cand {i}", **r} for i, r in enumerate(rows)]
    )


def test_faab_bid_monotonic_in_vorp():
    candidates = _candidates(
        [
            {"position": "WR", "rolling_xfp_3w": 12.0, "opportunity_index": 60.0},
            {"position": "WR", "rolling_xfp_3w": 10.0, "opportunity_index": 60.0},
            {"position": "WR", "rolling_xfp_3w": 8.0, "opportunity_index": 60.0},
        ]
    )
    out = recommend_faab_bids(
        candidates,
        _user_roster(),
        _league(),
        {"count": 10, "median": 20, "p80": 30, "bids": [30] * 10},
    )

    # Delta VORP over the worst rostered WR (6.0) and VORP_max normalization
    assert list(out["delta_vorp"]) == [6.0, 4.0, 2.0]
    bids = list(out["recommended_faab_bid"])
    assert bids == sorted(bids, reverse=True)
    assert bids[0] == 30  # top target ratio = 1.0 at full budget, neutral urgency
    assert bids[1] == 20
    assert bids[2] == 10
    assert list(out["faab_pct_of_remaining"]) == [0.3, 0.2, 0.1]


def test_faab_bid_capped_by_remaining():
    candidates = _candidates(
        [{"position": "WR", "rolling_xfp_3w": 12.0, "opportunity_index": 60.0}]
    )
    out = recommend_faab_bids(
        candidates,
        _user_roster(),
        _league(b_rem=50),
        {"count": 10, "median": 200, "p80": 200, "bids": [200] * 10},
    )
    assert out.iloc[0]["recommended_faab_bid"] == 50


def test_faab_urgency_multipliers():
    candidates = _candidates(
        [{"position": "WR", "rolling_xfp_3w": 12.0, "opportunity_index": 60.0}]
    )
    history = {"count": 10, "median": 20, "p80": 30, "bids": [30] * 10}

    neutral = recommend_faab_bids(candidates, _user_roster(), _league(), history)
    assert neutral.iloc[0]["recommended_faab_bid"] == 30

    injured = recommend_faab_bids(
        candidates,
        _user_roster([{
            "player_id": "p1", "player_display_name": "Start WR", "position": "WR",
            "rolling_xfp_3w": 6.0, "is_starter": True, "injury_status": "Out",
            "owner_id": "u1", "roster_id": 1, "waiver_position": None,
        }]),
        _league(),
        history,
    )
    assert injured.iloc[0]["recommended_faab_bid"] == 38
    assert "injured starter" in injured.iloc[0]["faab_rationale"]

    losing = recommend_faab_bids(candidates, _user_roster(), _league(wins=2, losses=8), history)
    assert losing.iloc[0]["recommended_faab_bid"] == 38

    contending = recommend_faab_bids(candidates, _user_roster(), _league(wins=10, losses=2), history)
    assert contending.iloc[0]["recommended_faab_bid"] == 26


def test_faab_fallback_when_no_history():
    candidates = _candidates(
        [
            {"position": "WR", "rolling_xfp_3w": 10.0, "opportunity_index": 78.0},
            {"position": "WR", "rolling_xfp_3w": 8.0, "opportunity_index": 60.0},
        ]
    )
    out = recommend_faab_bids(candidates, _user_roster(), _league(), None)
    # No league bids: 12% of the initial budget for index >= 70, 5% otherwise,
    # still scaled by the delta VORP ratio (4.0 / 4.0 and 2.0 / 4.0)
    assert out.iloc[0]["recommended_faab_bid"] == 12
    assert out.iloc[1]["recommended_faab_bid"] == 2


def test_faab_zero_delta_bids_zero():
    candidates = _candidates(
        [{"position": "WR", "rolling_xfp_3w": 4.0, "opportunity_index": 60.0}]
    )
    out = recommend_faab_bids(
        candidates, _user_roster(), _league(),
        {"count": 10, "median": 20, "p80": 30, "bids": [30] * 10},
    )
    assert out.iloc[0]["delta_vorp"] == -2.0
    assert out.iloc[0]["recommended_faab_bid"] == 0
    assert out.iloc[0]["faab_pct_of_remaining"] == 0.0
    assert "No upgrade" in out.iloc[0]["faab_rationale"]


def test_priority_claim_mode():
    candidates = _candidates(
        [
            {"position": "WR", "rolling_xfp_3w": 11.0, "opportunity_index": 60.0},  # dVORP 5.0
            {"position": "WR", "rolling_xfp_3w": 7.0, "opportunity_index": 78.0},   # dVORP 1.0
            {"position": "WR", "rolling_xfp_3w": 7.0, "opportunity_index": 60.0},   # dVORP 1.0
            {"position": "WR", "rolling_xfp_3w": 5.0, "opportunity_index": 80.0},   # dVORP -1.0
        ]
    )
    out = recommend_faab_bids(
        candidates, _user_roster(waiver_position=3), _league(waiver_type=0), None
    )

    assert out["recommended_faab_bid"].isna().all()
    assert out["faab_pct_of_remaining"].isna().all()
    # dVORP 5.0 with priority #3 -> claim; index 78 -> claim; weak and non-upgrades -> no
    assert list(out["priority_claim_recommended"]) == [True, True, False, False]

    # Without a known waiver position only the elite-index rule fires
    no_priority = recommend_faab_bids(
        candidates, _user_roster(), _league(waiver_type=0), None
    )
    assert list(no_priority["priority_claim_recommended"]) == [False, True, False, False]
    assert no_priority["recommended_faab_bid"].isna().all()


class _StubSleeper:
    def __init__(self, league_context):
        self._league_context = league_context
        self.faab_history_called = False

    def get_league_context(self, league_id):
        return self._league_context

    def get_nfl_state(self):
        return {"season": 2024, "week": 2}

    def get_league_rosters(self, league_id):
        return [
            {
                "roster_id": 1, "owner_id": "u1", "players": ["5555"], "starters": ["5555"],
                "settings": {"waiver_position": 3, "wins": 2, "losses": 8},
            },
            {"roster_id": 2, "owner_id": "u2", "players": ["7777", "8888"], "starters": ["7777"], "settings": {}},
        ]

    def get_recent_dropped_player_ids(self, league_id, week):
        return {"1111"}

    def get_all_players(self):
        return {
            "1111": {"full_name": "Alpha WR", "position": "WR", "team": "KC"},
            "7777": {"full_name": "Gamma WR", "position": "WR", "team": "BUF"},
            "8888": {"full_name": "Beta RB", "position": "RB", "team": "KC"},
            "5555": {"full_name": "Depth WR", "position": "WR", "team": "LAR", "injury_status": "Out"},
        }

    def get_user(self, username):
        return {"user_id": "u1", "display_name": username}

    def get_league_faab_history(self, league_id, weeks_back=8):
        self.faab_history_called = True
        return {"count": 0, "median": None, "p80": None, "bids": []}


class _StubLoader:
    def get_weekly_with_snaps(self, seasons=None, force_refresh=False):
        rows = []
        for week, off in [(1, 0.5), (2, 0.9)]:
            rows.append(_skill_row("00-0000001", "Alpha WR", "WR", week, off, 8, 0, 0.26, 0.30, 100, 30, 10))
            rows.append(_skill_row("00-0000002", "Beta RB", "RB", week, off, 3, 12, 0.05, 0.02, 10, 5, 10))
            rows.append(_skill_row("00-0000003", "Gamma WR", "WR", week, off, 8, 0, 0.26, 0.30, 100, 30, 10, team="BUF"))
            rows.append(_skill_row("00-0000004", "Depth WR", "WR", week, 0.4 if week == 1 else 0.5, 5, 0, 0.20, 0.15, 40, 20, 6, team="LAR"))
        return pd.DataFrame(rows)

    def get_player_ids(self):
        return pd.DataFrame([
            {"gsis_id": "00-0000001", "sleeper_id": "1111", "name": "Alpha WR", "position": "WR", "team": "KC"},
            {"gsis_id": "00-0000003", "sleeper_id": "7777", "name": "Gamma WR", "position": "WR", "team": "BUF"},
            {"gsis_id": "00-0000004", "sleeper_id": "5555", "name": "Depth WR", "position": "WR", "team": "LAR"},
        ])

    def get_pbp_opportunity_features(self, seasons=None):
        return {"player": pd.DataFrame()}

    def get_sleeper_season_data(self, season=2026, week=None):
        return pd.DataFrame()


def test_get_waiver_sleepers_sleeper_id_filtering_and_faab():
    sleeper = _StubSleeper(_league(wins=2, losses=8, b_rem=40))
    advisor = FantasyAdvisor(data_loader=_StubLoader(), sleeper_client=sleeper)
    result = advisor.get_waiver_sleepers(
        season=2024, week=2, league_id="L1", limit=5, username_or_id="me"
    )

    # Gamma WR and the user's own Depth WR filtered via sleeper_id; Beta RB is
    # unmapped in the crosswalk and filtered via the name fallback
    assert list(result["player_display_name"]) == ["Alpha WR"]
    assert result.iloc[0]["sleeper_id"] == "1111"
    assert result.iloc[0]["availability"] == "WAIVER CLAIM"
    assert not result.iloc[0]["is_rostered"]

    # FAAB: worst rostered WR (Depth WR, xFP 9.7) baseline, fallback clearing
    # (no league bids) 12% of $100, urgency 1.25 (2-8 record + injured starter)
    row = result.iloc[0]
    assert row["delta_vorp"] > 0
    assert 0 < row["recommended_faab_bid"] <= 40
    assert 0 < row["faab_pct_of_remaining"] <= 1.0
    assert not row["priority_claim_recommended"]
    assert "injured starter" in row["faab_rationale"]
    assert sleeper.faab_history_called


def test_get_waiver_sleepers_priority_league_skips_faab_history():
    sleeper = _StubSleeper(_league(waiver_type=0, wins=2, losses=8, b_rem=40))
    advisor = FantasyAdvisor(data_loader=_StubLoader(), sleeper_client=sleeper)
    result = advisor.get_waiver_sleepers(
        season=2024, week=2, league_id="L1", limit=5, username_or_id="me"
    )

    row = result.iloc[0]
    assert row["recommended_faab_bid"] is None
    assert bool(row["priority_claim_recommended"])  # elite index 89.5
    assert not sleeper.faab_history_called


def test_get_waiver_sleepers_without_league_context(monkeypatch):
    import fantasy_intel.agent.advisor as advisor_mod

    # No configured league either: pure analytics path, no Sleeper calls
    monkeypatch.setattr(advisor_mod, "load_user_config", lambda: {})
    advisor = FantasyAdvisor(data_loader=_StubLoader(), sleeper_client=_StubSleeper(_league()))
    result = advisor.get_waiver_sleepers(season=2024, week=2, limit=5)
    assert not result.empty
    assert "availability" in result.columns
    # No league in play: no FAAB columns attached
    assert "recommended_faab_bid" not in result.columns


class _PathStubLoader:
    """Loader stub recording which live-path data source the advisor chose."""

    def __init__(self, weekly_frame=None, sleeper_frame=None):
        self.weekly_frame = weekly_frame
        self.sleeper_frame = sleeper_frame
        self.weekly_calls = []
        self.sleeper_calls = []

    def get_weekly_with_snaps(self, seasons=None, force_refresh=False):
        self.weekly_calls.append(list(seasons or []))
        if self.weekly_frame is None:
            raise RuntimeError("nflverse weekly unavailable")
        return self.weekly_frame

    def get_sleeper_season_data(self, season=2026, week=None):
        self.sleeper_calls.append(season)
        return self.sleeper_frame

    def get_player_ids(self):
        return pd.DataFrame()

    def get_pbp_opportunity_features(self, seasons=None):
        return {"player": pd.DataFrame()}


def _sleeper_frame_row(pid="9", name="Live Wr", week=6, season=2026):
    return {
        "player_id": pid,
        "player_display_name": name,
        "position": "WR",
        "recent_team": "KC",
        "season": season,
        "week": week,
        "season_type": "REG",
        "fantasy_points_ppr": 10.0,
        "recent_ppr": 10.0,
        "targets": 8.0,
        "carries": 0.0,
        "touches": 8.0,
        "offense_snaps": 58.0,
        "offense_pct": 0.9,
        "snap_delta": 0.05,
        "target_share": 0.26,
        "receiving_air_yards": 40.0,
        "receiving_yards": 70.0,
        "wopr": 0.5,
        "expected_fp_ppr": 8.0,
        "fp_diff": 2.0,
    }


def test_ensure_engines_prefers_nflverse_weekly_for_current_season():
    row = _skill_row("00-0000001", "Alpha WR", "WR", 6, 0.9, 8, 0, 0.26, 0.30, 100, 30, 10)
    row["season"] = 2026
    loader = _PathStubLoader(weekly_frame=pd.DataFrame([row]), sleeper_frame=pd.DataFrame([_sleeper_frame_row()]))
    advisor = FantasyAdvisor(data_loader=loader, sleeper_client=_StubSleeper(_league()))

    advisor._ensure_engines([2026])

    assert loader.weekly_calls == [[2026]]
    assert loader.sleeper_calls == []
    assert advisor._waiver_engine.df["player_display_name"].tolist() == ["Alpha WR"]


def test_ensure_engines_falls_back_to_sleeper_when_nflverse_unpublished():
    loader = _PathStubLoader(weekly_frame=None, sleeper_frame=pd.DataFrame([_sleeper_frame_row()]))
    advisor = FantasyAdvisor(data_loader=loader, sleeper_client=_StubSleeper(_league()))

    advisor._ensure_engines([2026])

    assert loader.sleeper_calls == [2026]
    assert advisor._waiver_engine.df["player_display_name"].tolist() == ["Live Wr"]


def test_ensure_engines_rejects_stale_fallback_season_frame():
    # Loader's internal fallback can return DEFAULT_SEASON (older) rows for a
    # current-season request; those must not satisfy the current-season path
    row = _skill_row("00-0000001", "Alpha WR", "WR", 6, 0.9, 8, 0, 0.26, 0.30, 100, 30, 10)
    loader = _PathStubLoader(weekly_frame=pd.DataFrame([row]), sleeper_frame=pd.DataFrame([_sleeper_frame_row()]))
    advisor = FantasyAdvisor(data_loader=loader, sleeper_client=_StubSleeper(_league()))

    advisor._ensure_engines([2026])

    assert loader.sleeper_calls == [2026]
    assert advisor._waiver_engine.df["player_display_name"].tolist() == ["Live Wr"]


def test_get_sleeper_season_data_emits_receiving_yards(monkeypatch):
    import fantasy_intel.data.loader as loader_mod
    import fantasy_intel.data.sleeper as sleeper_mod

    class _StatsSleeper:
        def __init__(self, cache_dir=None):
            pass

        def get_nfl_state(self):
            return {"week": 6, "display_week": 6}

        def get_season_stats(self, season=2026):
            return {
                "1234": {
                    "gp": 2, "pts_ppr": 20.0, "rec_tgt": 16.0, "rec_yd": 140.0,
                    "rec_air_yd": 80.0, "rush_att": 4.0, "rush_yd": 20.0, "off_snp": 120.0,
                }
            }

        def get_all_players(self):
            return {"1234": {"full_name": "Air Wr", "position": "WR", "team": "KC"}}

    monkeypatch.setattr(sleeper_mod, "SleeperClient", _StatsSleeper)
    loader = loader_mod.DataLoader()
    df = loader.get_sleeper_season_data(season=2026, week=6)

    assert not df.empty
    row = df.iloc[0]
    assert row["receiving_yards"] == 70.0
    assert row["receiving_air_yards"] == 40.0

    engine = WaiverWireEngine(df)
    out = engine.df.iloc[0]
    # rec_yd / rec_air_yd both flow through so racr / unrealized compute on the live path
    assert abs(out["racr"] - 70.0 / 40.0) < 1e-9
    assert out["unrealized_air_yards"] == 0.0
