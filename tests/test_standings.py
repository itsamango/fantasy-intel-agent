"""Tests for the season / playoff simulator."""

import numpy as np
import pytest

from fantasy_intel.analytics.standings import (
    SeasonSimulator,
    SeasonState,
    bracket_order,
    estimate_weekly_sd,
    playoff_round_weeks,
)


def test_bracket_order_and_round_weeks():
    assert bracket_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]
    assert playoff_round_weeks({"playoff_week_start": 15, "playoff_teams": 6}) == [[15], [16], [17]]
    assert playoff_round_weeks({"playoff_week_start": 15, "playoff_teams": 4, "playoff_round_type": 1}) == [[15], [16, 17]]


def test_weekly_sd_pooled():
    sd = estimate_weekly_sd({1: [100, 110, 120, 130], 2: [90, 100, 110, 120], 3: [80, 90, 100, 110], 4: [95, 105, 115, 125]})
    assert sd == pytest.approx(np.std([100, 110, 120, 130], ddof=1))


def _league(n=6, wins=None, playoff_teams=4, sd=20.0):
    weeks = [12, 13]
    sched = {12: [(0, 1), (2, 3), (4, 5)], 13: [(0, 2), (1, 4), (3, 5)]}
    return SeasonState(
        team_keys=list(range(n)),
        wins=np.array(wins if wins is not None else [5.0] * n),
        points_for=np.zeros(n),
        schedule=sched,
        regular_weeks=weeks,
        playoff_rounds=[[14], [15]],
        playoff_teams=playoff_teams,
        weekly_sd=sd,
    )


def test_odds_are_probabilities_and_conserve():
    sim = SeasonSimulator(_league(), n_sims=3000)
    odds = sim.run(np.full((6, 4), 100.0))
    assert odds.title.sum() == pytest.approx(1.0)
    assert odds.playoffs.sum() == pytest.approx(4.0)
    assert np.allclose(odds.title, 1 / 6, atol=0.03)  # equal teams


def test_locked_team_gains_little_from_regular_season_points():
    # Team 0 has a 4-game cushion: regular-season points barely change its odds,
    # but playoff-week points do. Bubble team 3 is the reverse on regular weeks.
    st = _league(wins=[9, 5, 5, 5, 5, 5])
    sim = SeasonSimulator(st, n_sims=6000)
    base = np.full((6, 4), 100.0)
    b = sim.run(base)

    def bump(team, cols):
        m = base.copy()
        m[team, cols] += 10.0
        return sim.run(m)

    lock_reg = bump(0, [0, 1]).playoffs[0] - b.playoffs[0]
    bubble_reg = bump(3, [0, 1]).playoffs[3] - b.playoffs[3]
    assert lock_reg < 0.01 < bubble_reg
    lock_po = bump(0, [2, 3]).title[0] - b.title[0]
    lock_reg_title = bump(0, [0, 1]).title[0] - b.title[0]
    assert lock_po > 5 * max(lock_reg_title, 1e-3)


def test_byes_go_to_top_seeds_with_six_team_playoffs():
    st = SeasonState(
        team_keys=list(range(8)), wins=np.array([10, 9, 8, 7, 6, 5, 4, 3], dtype=float),
        points_for=np.zeros(8), schedule={}, regular_weeks=[],
        playoff_rounds=[[15], [16], [17]], playoff_teams=6, weekly_sd=10.0,
    )
    odds = SeasonSimulator(st, n_sims=2000).run(np.full((8, 3), 100.0))
    assert odds.bye[0] == 1.0 and odds.bye[1] == 1.0 and odds.bye[2:].sum() == 0
    assert odds.playoffs[:6].sum() == pytest.approx(6.0) and odds.playoffs[6:].sum() == 0
    # Equal strength: a bye means 2 wins needed vs 3 -> ~25% vs ~12.5%
    assert odds.title[0] == pytest.approx(0.25, abs=0.04)
    assert odds.title[2] == pytest.approx(0.125, abs=0.03)
