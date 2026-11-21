"""Tests for rest-of-season roster valuation and trade evaluation."""

import pytest

from fantasy_intel.analytics.ros_value import (
    LeagueShape,
    PlayerOutlook,
    RosterState,
    RosValuator,
    build_outlooks,
    evaluate_trade,
    projection_points,
)
from fantasy_intel.config import ROS_AVAILABILITY_BY_POS, SCORING_HALF_PPR, SCORING_TEP

WEEKS = [11, 12, 13, 14, 15, 16, 17]


def _shape(max_active=6, reserve=0, flex=1):
    return LeagueShape(
        dedicated={"QB": 1, "RB": 1, "WR": 1, "TE": 0},
        flex={"FLEX": flex},
        max_active=max_active,
        reserve_slots=reserve,
        weeks=WEEKS,
        playoff_weeks={15, 16, 17},
    )


def _o(pid, pos, pts, avail=1.0, byes=(), status=None):
    weekly = {w: (0.0 if w in byes else pts) for w in WEEKS}
    availability = {w: (0.0 if w in byes else avail) for w in WEEKS}
    source = {w: ("bye" if w in byes else "sleeper") for w in WEEKS}
    return PlayerOutlook(pid, pid.upper(), pos, "KC", status, weekly, availability, source)


def test_league_shape_from_sleeper_settings():
    shape = LeagueShape.from_sleeper(
        ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF", "BN", "BN", "BN", "BN", "BN"],
        {"playoff_week_start": 15, "playoff_teams": 6, "reserve_slots": 2},
        start_week=11,
    )
    assert shape.dedicated == {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    assert shape.flex["FLEX"] == 2
    assert shape.max_active == 15
    assert shape.weeks == [11, 12, 13, 14, 15, 16, 17]  # 6 teams -> 3 playoff rounds
    assert shape.playoff_weeks == {15, 16, 17}


def test_projection_points_scoring_keys():
    rec = {"pts_ppr": 15.0, "pts_half_ppr": 12.0, "pts_std": 9.0, "rec": 6}
    assert projection_points(rec, SCORING_HALF_PPR, "WR") == 12.0
    assert projection_points(rec, SCORING_TEP, "TE") == 18.0
    assert projection_points(rec, SCORING_TEP, "WR") == 15.0
    assert projection_points({}, SCORING_HALF_PPR, "WR") is None


def test_build_outlooks_byes_injuries_and_fallback():
    players = {
        "a": {"position": "WR", "team": "KC", "full_name": "A", "injury_status": "Questionable"},
        "b": {"position": "RB", "team": "BUF", "full_name": "B", "injury_status": "IR"},
        "c": {"position": "TE", "team": "KC", "full_name": "C", "injury_status": None},
    }
    proj = {w: {"a": {"pts_ppr": 14.0}} for w in WEEKS}
    proj[16]["b"] = {"pts_ppr": 18.0}  # Sleeper expects B back in week 16
    playing = {w: {"KC", "BUF"} for w in WEEKS}
    playing[13] = {"BUF"}  # KC bye
    out = build_outlooks(players, players, proj, playing, WEEKS, "ppr", fallback_ppg={"c": 7.5})

    a, b, c = out["a"], out["b"], out["c"]
    assert a.availability[11] == pytest.approx(0.80)            # questionable, next week
    assert a.availability[12] == pytest.approx(ROS_AVAILABILITY_BY_POS["WR"])
    assert a.weekly[13] == 0.0 and a.source[13] == "bye"
    assert b.weekly[12] == 0.0 and b.source[12] == "injured"     # IR, no projection
    assert b.weekly[16] == 18.0 and b.availability[16] == ROS_AVAILABILITY_BY_POS["RB"]
    assert c.weekly[12] == 7.5 and c.source[12] == "model"       # healthy, Sleeper silent


def test_expected_lineup_respects_slots_and_flex():
    v = RosValuator(_shape(), {
        "q": _o("q", "QB", 20), "r1": _o("r1", "RB", 15), "r2": _o("r2", "RB", 12),
        "w1": _o("w1", "WR", 14), "w2": _o("w2", "WR", 5), "t": _o("t", "TE", 9),
    }, n_sims=0)
    # QB + RB + WR + FLEX(best remaining RB/WR/TE = r2 12) = 20+15+14+12
    assert v.team_value(["q", "r1", "r2", "w1", "w2", "t"]).per_week[11] == pytest.approx(61.0)


def test_depth_has_value_only_under_simulated_availability():
    outlooks = {
        "q": _o("q", "QB", 20, 0.9), "r1": _o("r1", "RB", 15, 0.9), "w1": _o("w1", "WR", 14, 0.9),
        "f": _o("f", "RB", 10, 0.9), "bench": _o("bench", "WR", 9, 0.9),
    }
    v = RosValuator(_shape(), outlooks, n_sims=400)
    starters = ["q", "r1", "w1", "f"]
    det_gain = v.team_value(starters + ["bench"], "expected").total - v.team_value(starters, "expected").total
    sim_gain = v.team_value(starters + ["bench"], "sim").total - v.team_value(starters, "sim").total
    assert det_gain == pytest.approx(0.0, abs=1e-9)   # never starts when everyone is "90% healthy"
    assert sim_gain > 3.0                              # but covers ~10% injury weeks: ~0.3 x 9 x 7 wks ~ 5-6


def test_bye_coverage_counts_in_expected_mode():
    outlooks = {
        "q": _o("q", "QB", 20), "r1": _o("r1", "RB", 15), "w1": _o("w1", "WR", 14, byes=(12,)),
        "f": _o("f", "RB", 10), "bench": _o("bench", "WR", 9),
    }
    v = RosValuator(_shape(), outlooks, n_sims=0)
    starters = ["q", "r1", "w1", "f"]
    gain = v.team_value(starters + ["bench"]).per_week[12] - v.team_value(starters).per_week[12]
    assert gain == pytest.approx(9.0)


def test_two_for_one_consolidation_forces_drop_and_values_both_sides():
    outlooks = {
        # Team A (user): two mid WRs; Team B: one stud WR + thin roster
        "aq": _o("aq", "QB", 18), "ar": _o("ar", "RB", 14), "aw1": _o("aw1", "WR", 11),
        "aw2": _o("aw2", "WR", 10), "ab": _o("ab", "RB", 6), "ac": _o("ac", "TE", 4),
        "bq": _o("bq", "QB", 18), "br": _o("br", "RB", 14), "stud": _o("stud", "WR", 22),
        "bw": _o("bw", "WR", 3), "bb": _o("bb", "RB", 4), "bx": _o("bx", "TE", 2),
        "fa": _o("fa", "WR", 7),
    }
    v = RosValuator(_shape(max_active=6), outlooks, n_sims=0)
    team_a = RosterState(active=["aq", "ar", "aw1", "aw2", "ab", "ac"], reserve=[])
    team_b = RosterState(active=["bq", "br", "stud", "bw", "bb", "bx"], reserve=[])  # full
    res = evaluate_trade(v, team_a, team_b, ["aw1", "aw2"], ["stud"], free_agents=["fa"], mode="expected")

    # A consolidates: WR slot 11 -> 22, FLEX loses aw2 (10) and falls to the free agent WR (7)
    assert res["you"]["delta_per_week"] == pytest.approx((22 - 11) + (7 - 10), abs=0.01)
    assert "FA" in res["you"]["adds"]  # opened roster spot filled from waivers
    # B (full roster): WR 22 -> 11 and FLEX 4 -> 10, and must cut someone (6 + 1 > 6)
    assert res["partner"]["delta_per_week"] == pytest.approx((11 - 22) + (10 - 4), abs=0.01)
    assert res["partner"]["drops"]
    assert res["verdict"] == "FAVORS YOU"
    assert res["you"]["delta_playoffs"] == pytest.approx(3 * 8.0, abs=0.01)


def test_common_random_numbers_make_identical_rosters_identical():
    outlooks = {p: _o(p, pos, pts, 0.9) for p, pos, pts in
                [("q", "QB", 20), ("r", "RB", 15), ("w", "WR", 14), ("f", "WR", 9)]}
    v = RosValuator(_shape(), outlooks, n_sims=200)
    res = evaluate_trade(
        v, RosterState(["q", "r", "w", "f"], []), RosterState([], []), [], [], free_agents=[], mode="sim"
    )
    assert res["you"]["delta_total"] == 0.0


def test_schedule_team_codes_normalized_to_sleeper():
    import pandas as pd
    from fantasy_intel.analytics.ros_value import teams_playing_from_schedule

    sched = pd.DataFrame([
        {"week": 12, "home_team": "LA", "away_team": "SEA"},
        {"week": 12, "home_team": "KC", "away_team": "WSH"},
    ])
    playing = teams_playing_from_schedule(sched, [12, 13])
    assert playing == {12: {"LAR", "SEA", "KC", "WAS"}}  # week 13 absent -> not "everyone on bye"


def test_find_trades_discovers_planted_win_win():
    from fantasy_intel.analytics.ros_value import find_trades

    shape = LeagueShape(
        dedicated={"QB": 1, "RB": 2, "WR": 2, "TE": 0}, flex={}, max_active=6,
        reserve_slots=0, weeks=WEEKS, playoff_weeks={15, 16, 17},
    )
    outlooks = {
        # You: three good RBs (one stuck on the bench), weak WR2
        "uq": _o("uq", "QB", 18), "ur1": _o("ur1", "RB", 16), "ur2": _o("ur2", "RB", 15),
        "ur3": _o("ur3", "RB", 14), "uw1": _o("uw1", "WR", 15), "uw2": _o("uw2", "WR", 4),
        # Them: three good WRs (one benched), weak RB2
        "tq": _o("tq", "QB", 18), "tr1": _o("tr1", "RB", 16), "tr2": _o("tr2", "RB", 3),
        "tw1": _o("tw1", "WR", 16), "tw2": _o("tw2", "WR", 15), "tw3": _o("tw3", "WR", 14),
    }
    v = RosValuator(shape, outlooks, n_sims=50)
    rosters = [
        RosterState(["uq", "ur1", "ur2", "ur3", "uw1", "uw2"], []),
        RosterState(["tq", "tr1", "tr2", "tw1", "tw2", "tw3"], []),
    ]
    found = find_trades(v, rosters, 0, free_agents=[], manager_names=["You", "Them"], limit=5)
    assert found["proposals"], found
    best = found["proposals"][0]
    sent = {c["player_id"] for c in best["you_send"]}
    got = {c["player_id"] for c in best["you_get"]}
    assert sent & {"ur1", "ur2", "ur3"} and got & {"tw1", "tw2", "tw3"}
    assert best["you"]["delta_per_week"] > 0.5 and best["partner"]["delta_per_week"] >= 0


def test_vectorized_simulation_matches_reference_greedy():
    import numpy as np
    from fantasy_intel.analytics.ros_value import _solve_week

    rng = np.random.default_rng(3)
    shape = LeagueShape({"QB": 1, "RB": 2, "WR": 2, "TE": 1}, {"FLEX": 2, "SUPER_FLEX": 1}, 15, 0, WEEKS, {15, 16, 17})
    outlooks = {}
    for i in range(14):
        pos = ["QB", "RB", "WR", "TE"][i % 4]
        outlooks[f"p{i}"] = PlayerOutlook(
            f"p{i}", f"p{i}", pos, "KC", None,
            {w: float(rng.uniform(0, 25)) * (w != 13 or i % 3) for w in WEEKS},
            {w: float(rng.uniform(0.5, 1.0)) for w in WEEKS},
        )
    v = RosValuator(shape, outlooks, n_sims=200)
    got = v.team_value(list(outlooks)).per_week
    for wi, w in enumerate(WEEKS):
        acc = 0.0
        for s in range(200):
            by_pos = {}
            for o in sorted(outlooks.values(), key=lambda o: o.weekly[w], reverse=True):
                if o.weekly[w] > 0 and v._draws(o.player_id)[s, wi] < o.availability[w]:
                    by_pos.setdefault(o.position, []).append(o.weekly[w])
            acc += _solve_week(by_pos, shape.dedicated, shape.flex)
        assert got[w] == pytest.approx(acc / 200, rel=1e-9)


def test_waiver_baseline_removes_credit_for_free_pickups():
    """Regression: a 2-for-1 opened a roster spot and the trade got credit for a
    free agent the user could have added anyway."""
    from fantasy_intel.analytics.ros_value import optimize_waivers

    shape = LeagueShape({"QB": 0, "RB": 1, "WR": 1, "TE": 0}, {"FLEX": 2}, 5, 0, WEEKS, {15, 16, 17})
    outlooks = {
        # You: two real WRs go out (w2, w3); "junk" is the bench player you'd cut anyway
        "r1": _o("r1", "RB", 12), "w1": _o("w1", "WR", 12), "w2": _o("w2", "WR", 10),
        "w3": _o("w3", "WR", 8), "junk": _o("junk", "RB", 1),
        "pr": _o("pr", "RB", 13), "hall": _o("hall", "RB", 11), "pw": _o("pw", "WR", 6),
        "pw2": _o("pw2", "WR", 5), "pte": _o("pte", "TE", 2),
        "miller": _o("miller", "RB", 9),  # free agent
    }
    v = RosValuator(shape, outlooks, n_sims=50)
    you = RosterState(["r1", "w1", "w2", "w3", "junk"], [])
    them = RosterState(["pr", "hall", "pw", "pw2", "pte"], [])

    naive = evaluate_trade(v, you, them, ["w2", "w3"], ["hall"], ["miller"], waiver_baseline=False)
    fair = evaluate_trade(v, you, them, ["w2", "w3"], ["hall"], ["miller"])
    # Naive: the open spot gets Miller and the trade is credited with him
    assert "MILLER" in naive["you"]["adds"]
    assert naive["you"]["delta_per_week"] == pytest.approx((12 + 12 + 11 + 9) - (12 + 12 + 10 + 8), abs=0.01)
    # Fair: junk -> Miller is a free move you'd make anyway; reported, not credited
    assert [(m["drop_name"], m["add_name"]) for m in fair["you"]["waiver_moves"]] == [("JUNK", "MILLER")]
    assert fair["you"]["delta_per_week"] == pytest.approx((12 + 12 + 11 + 9) - (12 + 12 + 10 + 9), abs=0.01)

    # Traded players are never the waiver drop
    _, moves = optimize_waivers(v, you, ["miller"], keep=["junk"])
    assert all(m["drop"] != "junk" for m in moves)
    assert fair["you"]["lineup_after"]["starters"][0] == {"slot": "RB", "name": "R1", "position": "RB", "ppw": 12.0}


def test_plan_waiver_moves_reports_steps_odds_and_alternatives():
    import numpy as np
    from fantasy_intel.analytics.ros_value import plan_waiver_moves
    from fantasy_intel.analytics.standings import SeasonSimulator, SeasonState

    shape = LeagueShape({"QB": 0, "RB": 1, "WR": 1, "TE": 0}, {"FLEX": 1}, 4, 0, WEEKS, {15, 16, 17})
    outlooks = {
        "r1": _o("r1", "RB", 12, 0.9), "w1": _o("w1", "WR", 12, 0.9), "f": _o("f", "WR", 6, 0.9),
        "junk": _o("junk", "RB", 1, 0.9),
        "star_fa": _o("star_fa", "RB", 10, 0.9), "meh_fa": _o("meh_fa", "WR", 2, 0.9),
    }
    v = RosValuator(shape, outlooks, n_sims=200)
    roster = RosterState(["r1", "w1", "f", "junk"], [])
    state = SeasonState(
        team_keys=[0, 1], wins=np.array([5.0, 5.0]), points_for=np.zeros(2),
        schedule={11: [(0, 1)], 12: [(0, 1)], 13: [(0, 1)], 14: [(0, 1)]},
        regular_weeks=[11, 12, 13, 14], playoff_rounds=[[15], [16, 17]], playoff_teams=2, weekly_sd=10.0,
    )
    season = {"sim": SeasonSimulator(state, n_sims=2000), "base_means": np.full((2, 7), 30.0), "user_index": 0}
    season["base_means"][0] = [v.team_value(roster.all_ids()).per_week[w] for w in WEEKS]

    plan = plan_waiver_moves(v, roster, ["star_fa", "meh_fa"], season=season)
    first = plan["moves"][0]
    assert first["add"]["player_id"] == "star_fa"
    assert first["drop"]["player_id"] in ("junk", "f")
    assert first["cumulative_ppw"] == pytest.approx(first["gain_ppw"], abs=0.01)
    assert plan["after"]["ppw"] > plan["before"]["ppw"]
    assert plan["after"]["odds"]["title"] > plan["before"]["odds"]["title"]
    assert all(a["add"]["player_id"] != "star_fa" for a in plan["alternatives"])
