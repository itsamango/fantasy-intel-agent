"""Accuracy tests for the ROS engine: league scoring, the in-progress week, waiver
drop search, exact flex lineups, two-week playoff rounds, and noise-aware verdicts."""

import datetime as dt
import itertools

import numpy as np
import pandas as pd
import pytest

from fantasy_intel.agent.advisor import FantasyAdvisor
from fantasy_intel.analytics.ros_value import (
    LeagueShape,
    PlayerOutlook,
    RosterState,
    RosValuator,
    _solve_week,
    build_outlooks,
    evaluate_trade,
    optimize_waivers,
    plan_waiver_moves,
    projection_points,
    season_means,
    teams_kicked_off,
    title_verdict,
)
from fantasy_intel.analytics.standings import SeasonSimulator, SeasonState, playoff_round_weeks
from fantasy_intel.config import ROS_TITLE_VERDICT_MARGIN, SCORING_HALF_PPR

WEEKS = [11, 12, 13, 14, 15, 16, 17]


def _o(pid, pos, pts, avail=1.0, weeks=WEEKS, team="KC"):
    return PlayerOutlook(pid, pid.upper(), pos, team, None, {w: pts for w in weeks}, {w: avail for w in weeks},
                         {w: "sleeper" for w in weeks})


# --------------------------------------------------------------------------- #
# League scoring
# --------------------------------------------------------------------------- #
def test_projection_scored_with_league_settings():
    qb = {"pass_td": 2.0, "pass_yd": 250.0, "pass_int": 1.0, "rush_yd": 10.0, "pts_half_ppr": 18.0}
    four = {"pass_td": 4, "pass_yd": 0.04, "pass_int": -1, "rush_yd": 0.1}
    six = dict(four, pass_td=6)
    assert projection_points(qb, SCORING_HALF_PPR, "QB", four) == pytest.approx(18.0)
    # 6-pt passing TDs add 2 x projected passing TDs
    assert projection_points(qb, SCORING_HALF_PPR, "QB", six) == pytest.approx(18.0 + 2 * 2.0)
    # first-down and TE-premium bonuses come from the projected stat keys
    rb = {"rush_yd": 60.0, "rush_fd": 4.0, "pts_half_ppr": 6.0}
    assert projection_points(rb, SCORING_HALF_PPR, "RB", {"rush_yd": 0.1, "rush_fd": 0.5}) == pytest.approx(8.0)
    te = {"rec": 5.0, "rec_yd": 50.0, "bonus_rec_te": 5.0}
    assert projection_points(te, "tep", "TE", {"rec": 1, "rec_yd": 0.1, "bonus_rec_te": 0.5}) == pytest.approx(12.5)
    # no settings, or no stat keys the league scores -> Sleeper's precomputed total
    assert projection_points(qb, SCORING_HALF_PPR, "QB") == 18.0
    assert projection_points({"pts_half_ppr": 7.0}, SCORING_HALF_PPR, "WR", six) == 7.0


# --------------------------------------------------------------------------- #
# In-progress week
# --------------------------------------------------------------------------- #
def _schedule():
    return pd.DataFrame([
        {"week": 11, "gameday": "2026-11-19", "gametime": "20:15", "home_team": "HOU", "away_team": "IND"},
        {"week": 11, "gameday": "2026-11-22", "gametime": "13:00", "home_team": "LA", "away_team": "SEA"},
        {"week": 11, "gameday": "2026-11-23", "gametime": "20:15", "home_team": "KC", "away_team": "DEN"},
        {"week": 12, "gameday": "2026-11-26", "gametime": "13:00", "home_team": "HOU", "away_team": "KC"},
    ])


def test_teams_kicked_off_uses_eastern_kickoff_times():
    sched = _schedule()
    sunday_night = dt.datetime(2026, 11, 22, 21, 0)  # naive = Eastern
    assert teams_kicked_off(sched, 11, sunday_night) == {"HOU", "IND", "LAR", "SEA"}
    # 20:00 Eastern on Monday = 01:00 UTC Tuesday: MNF (20:15 ET) not yet started
    from zoneinfo import ZoneInfo
    utc = dt.datetime(2026, 11, 24, 1, 0, tzinfo=ZoneInfo("UTC"))
    assert "KC" not in teams_kicked_off(sched, 11, utc)
    assert teams_kicked_off(sched, 11, dt.datetime(2026, 11, 19, 12, 0)) == set()
    assert teams_kicked_off(None, 11) == set()


def test_kicked_off_players_count_zero_this_week_only():
    players = {
        "hou": {"position": "WR", "team": "HOU", "full_name": "Hou"},
        "kc": {"position": "WR", "team": "KC", "full_name": "Kc"},
    }
    proj = {w: {"hou": {"pts_ppr": 15.0}, "kc": {"pts_ppr": 12.0}} for w in (11, 12)}
    out = build_outlooks(players, players, proj, {}, [11, 12], "ppr", kicked_off={"HOU", "IND"})
    assert out["hou"].weekly[11] == 0.0 and out["hou"].availability[11] == 0.0
    assert out["hou"].source[11] == "played"
    assert out["hou"].weekly[12] == 15.0  # next week unaffected
    assert out["kc"].weekly[11] == 12.0


def test_locked_starter_slot_is_unavailable_to_the_bench_this_week():
    shape = LeagueShape({"QB": 0, "RB": 1, "WR": 1, "TE": 0}, {"FLEX": 1}, 5, 0, [11, 12], set())
    weeks = [11, 12]
    outs = {
        "r_played": PlayerOutlook("r_played", "R", "RB", "HOU", None, {11: 0.0, 12: 14.0}, {11: 0.0, 12: 1.0}),
        "w1": _o("w1", "WR", 12, weeks=weeks), "w2": _o("w2", "WR", 10, weeks=weeks),
        "r_bench": _o("r_bench", "RB", 8, weeks=weeks),
    }
    ids = list(outs)
    free = RosValuator(shape, outs, n_sims=0).team_value(ids).per_week
    locked = RosValuator(shape, outs, n_sims=0, locked_slots={"r_played": "RB"}).team_value(ids).per_week
    # Without the lock the bench RB would fill the played starter's RB slot
    assert free[11] == pytest.approx(8 + 12 + 10)
    assert locked[11] == pytest.approx(12 + 10)  # RB slot used: WR + FLEX only
    assert locked[12] == free[12] == pytest.approx(14 + 12 + 10)


def test_banked_points_added_back_to_current_week_in_season_means():
    state = SeasonState([0, 1], np.zeros(2), np.zeros(2), {11: [(0, 1)]}, [11], [[12]], 2, 10.0)
    season = {
        "sim": SeasonSimulator(state, n_sims=10), "base_means": np.array([[50.0, 90.0], [60.0, 95.0]]),
        "banked_points": np.array([30.0, 5.0]), "current_week": 11, "user_index": 0,
    }
    m = season_means(season, {0: {11: 40.0, 12: 100.0}})
    assert m[0].tolist() == [70.0, 100.0]  # 40 remaining + 30 banked
    assert m[1].tolist() == [60.0, 95.0]


class _FakeSleeper:
    SETTINGS = {"playoff_week_start": 14, "playoff_teams": 2, "playoff_round_type": 0}

    def __init__(self, week=12, settings=None):
        self.week = week
        self.settings = settings or self.SETTINGS

    def get_league_info(self, league_id):
        return {
            "season": "2026", "name": "Test", "settings": self.settings,
            "roster_positions": ["QB", "RB", "WR", "FLEX", "BN", "BN"],
            "scoring_settings": {"pass_td": 6, "pass_yd": 0.04, "rush_yd": 0.1, "rec": 0.5, "rec_yd": 0.1},
        }

    def get_league_context(self, league_id):
        return {"scoring_type": "half_ppr"}

    def get_nfl_state(self):
        return {"season": "2026", "week": self.week, "season_type": "regular"}

    def get_user(self, username):
        return {"user_id": "u1"}

    def get_league_rosters(self, league_id):
        return [
            {"roster_id": 1, "owner_id": "u1", "players": ["qa", "ra", "wa", "wb"], "settings": {"wins": 6}},
            {"roster_id": 2, "owner_id": "u2", "players": ["qb", "rb", "wc", "wd"], "settings": {"wins": 5}},
        ]

    def get_league_users(self, league_id):
        return [{"user_id": "u1", "display_name": "me"}, {"user_id": "u2", "display_name": "them"}]

    def get_all_players(self):
        team = {"qa": "KC", "ra": "KC", "wa": "BUF", "wb": "BUF", "qb": "BUF", "rb": "BUF", "wc": "KC", "wd": "KC"}
        pos = {"q": "QB", "r": "RB", "w": "WR"}
        return {p: {"full_name": p.upper(), "position": pos[p[0]], "team": t} for p, t in team.items()}

    def get_week_projections(self, season, week):
        line = {"q": {"pass_td": 2.0, "pass_yd": 250.0, "pts_half_ppr": 18.0},
                "r": {"rush_yd": 80.0, "pts_half_ppr": 8.0},
                "w": {"rec": 5.0, "rec_yd": 70.0, "pts_half_ppr": 9.5}}
        return {p: dict(line[p[0]]) for p in self.get_all_players()}

    def get_matchups(self, league_id, week):
        if week == self.week:
            return [
                {"roster_id": 1, "matchup_id": 1, "starters": ["qa", "ra", "wa", "wb"],
                 "players_points": {"qa": 25.0, "ra": 9.0}},
                {"roster_id": 2, "matchup_id": 1, "starters": ["qb", "rb", "wc", "wd"],
                 "players_points": {"wc": 11.0, "wd": 9.0}},
            ]
        if week < self.week:
            return [{"roster_id": 1, "matchup_id": 1, "points": 100.0 + week},
                    {"roster_id": 2, "matchup_id": 1, "points": 95.0 - week}]
        return [{"roster_id": 1, "matchup_id": 1}, {"roster_id": 2, "matchup_id": 1}]


class _FakeLoader:
    def get_schedules(self, seasons):
        return pd.DataFrame([
            {"week": 12, "gameday": "2000-01-01", "gametime": "20:15", "home_team": "KC", "away_team": "DEN"},
            {"week": 12, "gameday": "2999-01-01", "gametime": "13:00", "home_team": "BUF", "away_team": "MIA"},
        ] + [
            {"week": w, "gameday": "2999-01-01", "gametime": "13:00", "home_team": h, "away_team": a}
            for w in (13, 14, 15) for h, a in (("KC", "DEN"), ("BUF", "MIA"))
        ])


def test_ros_context_banks_played_games_and_uses_league_scoring():
    adv = FantasyAdvisor(data_loader=_FakeLoader(), sleeper_client=_FakeSleeper())
    ctx = adv.build_ros_context(league_id="L", username="me", n_sims=50, use_model_fallback=False)
    v = ctx["valuator"]
    assert v.shape.weeks == [12, 13, 14]
    # 6-pt passing TDs: 2 TD x 6 + 250 x .04 = 22 (Sleeper's 4-pt total says 18)
    assert v.outlooks["qa"].weekly[13] == pytest.approx(22.0)
    assert v.outlooks["qa"].weekly[12] == 0.0 and v.outlooks["qa"].source[12] == "played"
    assert v.outlooks["wa"].weekly[12] == pytest.approx(5 * 0.5 + 7.0)  # BUF not kicked off
    assert v.locked_slots == {"qa": "QB", "ra": "RB", "wc": "WR", "wd": "FLEX"}
    season = ctx["season"]
    assert season["banked_points"].tolist() == [34.0, 20.0]
    j = season["sim"].state.weeks.index(12)
    remaining = v.team_value(ctx["rosters"][0].all_ids()).per_week[12]
    assert season["base_means"][0, j] == pytest.approx(remaining + 34.0)
    assert any("in progress" in n for n in ctx["notes"])


def test_no_title_odds_once_playoffs_are_under_way():
    settings = {"playoff_week_start": 14, "playoff_teams": 4, "playoff_round_type": 0}
    adv = FantasyAdvisor(data_loader=_FakeLoader(), sleeper_client=_FakeSleeper(week=15, settings=settings))
    ctx = adv.build_ros_context(league_id="L", username="me", n_sims=20, use_model_fallback=False)
    assert ctx["season"] is None and ctx["current_odds"] is None
    assert any("Playoffs under way" in n for n in ctx["notes"])


# --------------------------------------------------------------------------- #
# Waiver drop search
# --------------------------------------------------------------------------- #
def _backup_qb_league():
    """A redundant backup QB with a high projection is the right drop, though
    seven teammates have lower ROS projections (bench depth with real insurance value)."""
    shape = LeagueShape({"QB": 1, "RB": 1, "WR": 1, "TE": 0}, {}, 9, 0, WEEKS, {15, 16, 17})
    outs = {
        "qb1": _o("qb1", "QB", 20), "qb2": _o("qb2", "QB", 15),
        "r1": _o("r1", "RB", 12, 0.5), "r2": _o("r2", "RB", 10, 0.5), "r3": _o("r3", "RB", 10, 0.5),
        "r4": _o("r4", "RB", 10, 0.5),
        "w1": _o("w1", "WR", 12, 0.5), "w2": _o("w2", "WR", 9, 0.5), "w3": _o("w3", "WR", 9, 0.5),
        "rb_fa": _o("rb_fa", "RB", 14, 0.6), "wr_fa": _o("wr_fa", "WR", 13, 0.6),
        "p1": _o("p1", "WR", 11), "p2": _o("p2", "RB", 6),
    }
    roster = RosterState(["qb1", "qb2", "r1", "r2", "r3", "r4", "w1", "w2", "w3"], [])
    return shape, outs, roster


def test_waivers_drop_a_redundant_high_projection_backup():
    shape, outs, roster = _backup_qb_league()
    v = RosValuator(shape, outs, n_sims=200)
    assert sorted(roster.active, key=lambda p: outs[p].ros_expected()).index("qb2") >= 6
    _, moves = optimize_waivers(v, roster, ["rb_fa", "wr_fa"], max_swaps=1)
    assert moves[0]["drop"] == "qb2"
    # The old 6-lowest restriction never considered the backup QB
    _, limited = optimize_waivers(v, roster, ["rb_fa", "wr_fa"], max_swaps=1, drop_candidates=6)
    assert limited[0]["drop"] != "qb2" and limited[0]["gain_ppw"] < moves[0]["gain_ppw"]
    plan = plan_waiver_moves(v, roster, ["rb_fa", "wr_fa"], max_swaps=1)
    assert plan["moves"][0]["drop"]["player_id"] == "qb2"


def test_waiver_cache_respects_swap_count():
    shape, outs, roster = _backup_qb_league()
    v = RosValuator(shape, outs, n_sims=100)
    _, one = optimize_waivers(v, roster, ["rb_fa", "wr_fa"], max_swaps=1)
    _, three = optimize_waivers(v, roster, ["rb_fa", "wr_fa"], max_swaps=3)
    assert len(one) == 1 and len(three) == 2


def test_trade_gets_no_credit_for_the_backup_qb_swap():
    """2-for-1: the open spot must not be credited with the free agent the user
    could already have added by dropping the backup QB."""
    shape, outs, roster = _backup_qb_league()
    v = RosValuator(shape, outs, n_sims=200)
    them = RosterState(["p1", "p2"], [])
    res = evaluate_trade(v, roster, them, ["w2", "w3"], ["p1"], ["rb_fa", "wr_fa"])
    baseline = [(m["drop_name"], m["add_name"]) for m in res["you"]["waiver_moves"]]
    assert "QB2" in [d for d, _ in baseline]
    # both free agents are already in the baseline: the open spot adds nobody
    assert {a for _, a in baseline} == {"RB_FA", "WR_FA"}
    assert res["you"]["adds"] == []


# --------------------------------------------------------------------------- #
# Exact lineups
# --------------------------------------------------------------------------- #
def _brute_force(by_pos, dedicated, flex):
    from fantasy_intel.analytics.ros_value import FLEX_ELIGIBILITY

    slots = [({p}) for p, n in dedicated.items() for _ in range(n)]
    slots += [FLEX_ELIGIBILITY[f] for f, n in flex.items() for _ in range(n)]
    players = [(p, v) for p, vals in by_pos.items() for v in vals]
    best = 0.0
    for perm in itertools.permutations(range(len(players) + len(slots)), len(slots)):
        total, ok = 0.0, True
        for slot, pi in zip(slots, perm):
            if pi >= len(players):
                continue  # slot left empty
            pos, val = players[pi]
            if pos not in slot:
                ok = False
                break
            total += val
        if ok:
            best = max(best, total)
    return best


def test_lineup_is_optimal_with_rec_flex_and_wrrb_flex():
    # Greedy (REC_FLEX first) takes the WR here and leaves WRRB_FLEX with a weak RB
    by_pos = {"WR": [10.0], "TE": [9.0], "RB": [2.0]}
    assert _solve_week(by_pos, {}, {"REC_FLEX": 1, "WRRB_FLEX": 1}) == pytest.approx(19.0)
    rng = np.random.default_rng(5)
    flex = {"REC_FLEX": 1, "WRRB_FLEX": 1, "FLEX": 1}
    for _ in range(40):
        by_pos = {p: sorted(rng.uniform(0, 20, rng.integers(0, 4)).round(1).tolist(), reverse=True)
                  for p in ("RB", "WR", "TE")}
        ded = {"RB": 1, "WR": 1, "TE": 0}
        assert _solve_week(by_pos, ded, flex) == pytest.approx(_brute_force(by_pos, ded, flex))


def test_simulated_lineup_matches_exact_solver_with_mixed_flex():
    rng = np.random.default_rng(9)
    weeks = [12, 13]
    shape = LeagueShape({"QB": 1, "RB": 1, "WR": 1, "TE": 1}, {"REC_FLEX": 1, "WRRB_FLEX": 1, "SUPER_FLEX": 1},
                        12, 0, weeks, set())
    outs = {f"p{i}": PlayerOutlook(f"p{i}", f"p{i}", ["QB", "RB", "WR", "TE"][i % 4], "KC", None,
                                   {w: float(rng.uniform(0, 20)) for w in weeks},
                                   {w: float(rng.uniform(0.4, 1.0)) for w in weeks}) for i in range(12)}
    v = RosValuator(shape, outs, n_sims=150)
    got = v.team_value(list(outs)).per_week
    for wi, w in enumerate(weeks):
        acc = 0.0
        for s in range(150):
            by_pos = {}
            for o in sorted(outs.values(), key=lambda o: -o.weekly[w]):
                if v._draws(o.player_id)[s, wi] < o.availability[w]:
                    by_pos.setdefault(o.position, []).append(o.weekly[w])
            acc += _solve_week(by_pos, shape.dedicated, shape.flex)
        assert got[w] == pytest.approx(acc / 150, rel=1e-9)


def test_tied_values_give_identical_results_in_any_order():
    shape = LeagueShape({"QB": 0, "RB": 0, "WR": 1, "TE": 0}, {"REC_FLEX": 1, "WRRB_FLEX": 1}, 6, 0, WEEKS, set())
    outs = {"w": _o("w", "WR", 10, 0.7), "t": _o("t", "TE", 10, 0.7), "w2": _o("w2", "WR", 10, 0.7),
            "r": _o("r", "RB", 3, 0.7)}
    a = RosValuator(shape, outs, n_sims=100).team_value(["w", "t", "w2", "r"]).total
    b = RosValuator(shape, dict(reversed(list(outs.items()))), n_sims=100).team_value(["r", "w2", "t", "w"]).total
    assert a == b


# --------------------------------------------------------------------------- #
# Playoff rounds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rtype,expected_rounds", [
    (0, [[15], [16], [17]]),
    (1, [[15], [16], [17, 18]]),
    (2, [[15, 16], [17, 18]]),  # third round would be weeks 19-20: cut
])
def test_shape_weeks_follow_playoff_rounds(rtype, expected_rounds):
    settings = {"playoff_week_start": 15, "playoff_teams": 6, "playoff_round_type": rtype}
    assert playoff_round_weeks(settings) == expected_rounds
    shape = LeagueShape.from_sleeper(["QB", "RB", "WR"], settings, start_week=12)
    assert shape.weeks == list(range(12, expected_rounds[-1][-1] + 1))
    assert max(shape.weeks) <= 18
    assert shape.playoff_weeks == {w for r in expected_rounds for w in r}


# --------------------------------------------------------------------------- #
# Noise-aware verdict
# --------------------------------------------------------------------------- #
def test_title_verdict_is_too_close_when_noise_spans_the_margin():
    m = ROS_TITLE_VERDICT_MARGIN
    assert title_verdict(0.020, 0.002, 0.01) == "WIN-WIN (raises both teams' title odds)"
    assert title_verdict(0.020, 0.002, -0.01) == "FAVORS YOU"
    assert title_verdict(-0.020, 0.002, 0.0) == "HURTS YOU"
    assert title_verdict(0.0, 0.001, 0.0) == "ROUGHLY NEUTRAL FOR YOU"
    # reviewer's case: +0.45 pp with SE 0.17 pp straddles the 0.5 pp margin
    assert title_verdict(0.0045, 0.0017, 0.0).startswith("TOO CLOSE")
    assert title_verdict(m + 0.001, 0.001, 0.0).startswith("TOO CLOSE")


def test_paired_standard_error_from_common_random_numbers():
    state = SeasonState(list(range(4)), np.zeros(4), np.zeros(4), {12: [(0, 1), (2, 3)]}, [12], [[13], [14]], 4, 20.0)
    sim = SeasonSimulator(state, n_sims=2000)
    base = np.full((4, 3), 100.0)
    o0 = sim.run(base)
    assert o0.paired_se(sim.run(base), 0)["title"] == 0.0  # identical means -> no noise in the change
    bumped = base.copy()
    bumped[0] += 5.0
    se = o0.paired_se(sim.run(bumped), 0)["title"]
    assert 0.0 < se < 0.02
