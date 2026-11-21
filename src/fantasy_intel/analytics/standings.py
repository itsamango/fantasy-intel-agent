"""Season simulator: playoff / bye / title odds from standings, schedule, and team strength.

A trade's real value depends on the standings: a lock for the playoffs barely
benefits from regular-season points, a bubble team needs them now, and every
contender cares about the playoff weeks only if they get there. Instead of
hand-weighting weeks, we simulate the rest of the season and the bracket, and
compare title odds before / after a trade with common random numbers (the same
draws), so the difference isolates the trade.

Model per simulated week: team score ~ Normal(projected lineup mean, sigma),
sigma estimated from this league's completed weeks (pooled within-team SD).
Regular season: head-to-head per the league schedule (+ optional extra game vs
the league median). Seeding: wins (ties = half), then points for. Bracket:
standard fixed bracket, byes to top seeds when playoff_teams is not a power of
two (6 teams: 1-2 bye, 3v6, 4v5; no reseeding).

Pure numpy; the advisor gathers inputs from Sleeper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_WEEKLY_SD = 25.0
SEASON_SIMS = 4000


def bracket_order(slots: int) -> List[int]:
    """Standard seeding order for a power-of-two bracket: 8 -> [1, 8, 4, 5, 2, 7, 3, 6]."""
    order = [1]
    while len(order) < slots:
        n = len(order) * 2
        order = [x for s in order for x in (s, n + 1 - s)]
    return order


def playoff_round_weeks(settings: Mapping[str, Any]) -> List[List[int]]:
    """Weeks belonging to each playoff round (round_type 0: one week each; 1: two-week
    championship; 2: two weeks per round)."""
    start = int(settings.get("playoff_week_start") or 15)
    teams = int(settings.get("playoff_teams") or 6)
    rounds = max(1, math.ceil(math.log2(max(2, teams))))
    rtype = int(settings.get("playoff_round_type") or 0)
    out: List[List[int]] = []
    w = start
    for r in range(rounds):
        span = 2 if rtype == 2 or (rtype == 1 and r == rounds - 1) else 1
        out.append(list(range(w, w + span)))
        w += span
    return out


def estimate_weekly_sd(scores_by_team: Mapping[Any, Sequence[float]], default: float = DEFAULT_WEEKLY_SD) -> float:
    """Pooled within-team SD of weekly scores (needs >= 3 games per team)."""
    ss, df = 0.0, 0
    for vals in scores_by_team.values():
        v = np.asarray([x for x in vals if x and x > 0], dtype=float)
        if len(v) >= 3:
            ss += float(((v - v.mean()) ** 2).sum())
            df += len(v) - 1
    return math.sqrt(ss / df) if df >= 10 else default


@dataclass
class SeasonState:
    team_keys: List[Any]                       # e.g. Sleeper roster_ids, index = team index
    wins: np.ndarray                           # float (ties count 0.5)
    points_for: np.ndarray
    schedule: Dict[int, List[Tuple[int, int]]]  # regular week -> pairs of team indices
    regular_weeks: List[int]                    # remaining regular-season weeks
    playoff_rounds: List[List[int]]
    playoff_teams: int
    weekly_sd: float = DEFAULT_WEEKLY_SD
    median_game: bool = False                   # league_average_match

    @property
    def weeks(self) -> List[int]:
        return list(self.regular_weeks) + [w for r in self.playoff_rounds for w in r]


@dataclass
class SeasonOdds:
    playoffs: np.ndarray
    bye: np.ndarray
    title: np.ndarray

    def for_team(self, i: int) -> Dict[str, float]:
        return {"playoffs": float(self.playoffs[i]), "bye": float(self.bye[i]), "title": float(self.title[i])}


class SeasonSimulator:
    """Monte Carlo of the remaining season; reuse one instance for before/after comparisons."""

    def __init__(self, state: SeasonState, n_sims: int = SEASON_SIMS, seed: int = 11):
        self.state = state
        self.n_sims = int(n_sims)
        n_teams = len(state.team_keys)
        rng = np.random.default_rng(seed)
        self.week_index = {w: i for i, w in enumerate(state.weeks)}
        # Common random numbers: standard-normal noise per (sim, week, team)
        self.noise = rng.standard_normal((self.n_sims, len(state.weeks), n_teams))

    def run(self, means: np.ndarray) -> SeasonOdds:
        """means: (teams, weeks) projected lineup points aligned with state.weeks."""
        st = self.state
        n_teams = len(st.team_keys)
        scores = means.T[None, :, :] + st.weekly_sd * self.noise  # (sims, weeks, teams)
        wins = np.broadcast_to(np.asarray(st.wins, dtype=float), (self.n_sims, n_teams)).copy()
        pf = np.broadcast_to(np.asarray(st.points_for, dtype=float), (self.n_sims, n_teams)).copy()
        for w in st.regular_weeks:
            s = scores[:, self.week_index[w], :]
            pf += s
            for a, b in st.schedule.get(w, []):
                wins[:, a] += (s[:, a] > s[:, b]) + 0.5 * (s[:, a] == s[:, b])
                wins[:, b] += (s[:, b] > s[:, a]) + 0.5 * (s[:, a] == s[:, b])
            if st.median_game:
                med = np.median(s, axis=1, keepdims=True)
                wins += s > med

        # Seeding: wins, then points for
        key = wins * 1e6 + pf
        seeds = np.argsort(-key, axis=1)            # (sims, teams): seed k -> team index
        n_po = min(st.playoff_teams, n_teams)
        sims = np.arange(self.n_sims)
        playoffs = np.zeros(n_teams)
        for k in range(n_po):
            np.add.at(playoffs, seeds[:, k], 1.0)

        slots = 2 ** max(1, math.ceil(math.log2(max(2, n_po))))
        order = bracket_order(slots)
        # alive[:, j] = team index occupying bracket slot j (-1 = bye/empty)
        alive = np.full((self.n_sims, slots), -1, dtype=int)
        for j, seed_no in enumerate(order):
            if seed_no <= n_po:
                alive[:, j] = seeds[:, seed_no - 1]
        bye = np.zeros(n_teams)
        for r, round_weeks in enumerate(st.playoff_rounds):
            idx = [self.week_index[w] for w in round_weeks if w in self.week_index]
            if not idx:
                break
            round_pts = scores[:, idx, :].sum(axis=1)  # (sims, teams)
            nxt = np.full((self.n_sims, alive.shape[1] // 2), -1, dtype=int)
            for j in range(0, alive.shape[1], 2):
                a, b = alive[:, j], alive[:, j + 1]
                pa = np.where(a >= 0, round_pts[sims, np.maximum(a, 0)], -np.inf)
                pb = np.where(b >= 0, round_pts[sims, np.maximum(b, 0)], -np.inf)
                if r == 0:  # a team facing an empty slot has a first-round bye
                    for x, y in ((a, b), (b, a)):
                        got_bye = (x >= 0) & (y < 0)
                        np.add.at(bye, x[got_bye], 1.0)
                nxt[:, j // 2] = np.where(pa >= pb, a, b)
            alive = nxt
            if alive.shape[1] == 1:
                break
        title = np.zeros(n_teams)
        champ = alive[:, 0]
        np.add.at(title, champ[champ >= 0], 1.0)
        n = float(self.n_sims)
        return SeasonOdds(playoffs / n, bye / n, title / n)
