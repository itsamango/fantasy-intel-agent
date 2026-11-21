"""Rest-of-season (ROS) roster valuation and trade evaluation.

A trade's value to a team is the change in its expected optimal-lineup points
summed over every remaining week (regular season and playoffs reported
separately), not the sum of the traded players' values. Per week:

  * each player has a projection if he plays (Sleeper weekly projections,
    our skill-adjusted proj_ppg as fallback; 0 on byes) and a probability of
    playing (position base rate; next week adjusted by injury tag);
  * availability is simulated, and in each scenario the team starts its best
    available lineup for the league's starting slots;
  * the same random draws are reused for a player on every roster (common
    random numbers), so before/after comparisons isolate the trade itself.

This prices what single-number calculators miss: consolidation (2-for-1
costs a roster spot but may raise the starting lineup), bench depth as bye /
injury insurance, schedule (byes, matchups via weekly projections), and the
roster-limit consequences (forced drop / free-agent pickup).

Pure module: no network or file I/O. The advisor gathers inputs.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from fantasy_intel.config import (
    ROS_AVAILABILITY_BY_POS,
    ROS_FA_CANDIDATES_PER_POS,
    ROS_NEXT_WEEK_STATUS_AVAILABILITY,
    ROS_NO_PROJECTION_STATUSES,
    ROS_SIMS,
    ROS_TITLE_VERDICT_MARGIN,
    ROS_VERDICT_MARGIN_PPW,
    ROS_WAIVER_DROP_CANDIDATES,
    ROS_WAIVER_MAX_SWAPS,
    ROS_WAIVER_MIN_GAIN_PPW,
    SCORING_HALF_PPR,
    SCORING_STANDARD,
    SCORING_TEP,
)

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
FLEX_ELIGIBILITY: Dict[str, Set[str]] = {
    "REC_FLEX": {"WR", "TE"},
    "WRRB_FLEX": {"WR", "RB"},
    "FLEX": {"RB", "WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
}
# Most restrictive first: greedy fill is optimal for nested eligibility sets
FLEX_ORDER = ("REC_FLEX", "WRRB_FLEX", "FLEX", "SUPER_FLEX")
NON_ACTIVE_SLOTS = {"IR", "TAXI"}


# --------------------------------------------------------------------------- #
# League shape
# --------------------------------------------------------------------------- #
@dataclass
class LeagueShape:
    """Starting slots, roster limits, and the remaining week schedule."""

    dedicated: Dict[str, int]          # QB/RB/WR/TE starting slots
    flex: Dict[str, int]               # flex slot type -> count
    max_active: int                    # active roster limit (excludes IR/taxi)
    reserve_slots: int
    weeks: List[int]                   # remaining weeks, ascending
    playoff_weeks: Set[int]

    @classmethod
    def from_sleeper(
        cls,
        roster_positions: Sequence[str],
        settings: Mapping[str, Any],
        start_week: int,
    ) -> "LeagueShape":
        dedicated = {p: 0 for p in SKILL_POSITIONS}
        flex = {f: 0 for f in FLEX_ORDER}
        active = 0
        for slot in roster_positions:
            slot = str(slot).upper()
            if slot in NON_ACTIVE_SLOTS:
                continue
            active += 1
            if slot in dedicated:
                dedicated[slot] += 1
            elif slot in flex:
                flex[slot] += 1
        playoff_start = int(settings.get("playoff_week_start") or 15)
        playoff_teams = int(settings.get("playoff_teams") or 6)
        rounds = max(1, math.ceil(math.log2(max(2, playoff_teams))))
        if int(settings.get("playoff_round_type") or 0) == 2:
            rounds *= 2  # two-week championship round variants
        last_week = min(18, playoff_start + rounds - 1)
        weeks = list(range(max(1, int(start_week)), last_week + 1))
        return cls(
            dedicated=dedicated,
            flex=flex,
            max_active=active,
            reserve_slots=int(settings.get("reserve_slots") or 0),
            weeks=weeks,
            playoff_weeks={w for w in weeks if w >= playoff_start},
        )


# nflverse schedule team codes -> Sleeper team codes. Unmapped codes pass
# through. Without this, every Rams player ("LA" vs "LAR") reads as on bye.
TEAM_CODE_ALIASES = {"LA": "LAR", "WSH": "WAS", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LAR"}


def sleeper_team_code(code: Any) -> Optional[str]:
    if code is None:
        return None
    code = str(code).upper()
    return TEAM_CODE_ALIASES.get(code, code)


def teams_playing_from_schedule(schedule: Any, weeks: Sequence[int]) -> Dict[int, Set[str]]:
    """{week: Sleeper team codes with a game} from an nflverse schedule frame.

    Weeks missing from the schedule are omitted (callers then treat every team
    as playing rather than everyone as on bye)."""
    out: Dict[int, Set[str]] = {}
    if schedule is None or getattr(schedule, "empty", True):
        return out
    for w in weeks:
        games = schedule[schedule["week"] == w]
        if games.empty:
            continue
        out[w] = {sleeper_team_code(t) for t in list(games["home_team"]) + list(games["away_team"])}
    return out


# --------------------------------------------------------------------------- #
# Player outlooks
# --------------------------------------------------------------------------- #
@dataclass
class PlayerOutlook:
    player_id: str
    name: str
    position: str
    team: Optional[str]
    injury_status: Optional[str]
    weekly: Dict[int, float]                 # points if he plays
    availability: Dict[int, float]           # P(plays)
    source: Dict[int, str] = field(default_factory=dict)  # sleeper / model / bye / injured

    def expected(self, week: int) -> float:
        return self.weekly.get(week, 0.0) * self.availability.get(week, 0.0)

    def ros_expected(self) -> float:
        return sum(self.expected(w) for w in self.weekly)


def projection_points(proj: Mapping[str, Any], scoring: str, position: str) -> Optional[float]:
    """League-scoring points from a Sleeper projection record (None when absent)."""
    if not proj:
        return None
    key = {SCORING_HALF_PPR: "pts_half_ppr", SCORING_STANDARD: "pts_std"}.get(scoring, "pts_ppr")
    val = proj.get(key)
    if val is None:
        return None
    pts = float(val)
    if scoring == SCORING_TEP and position == "TE":
        pts += 0.5 * float(proj.get("rec") or 0.0)
    return pts


def build_outlooks(
    player_ids: Iterable[str],
    players_db: Mapping[str, Mapping[str, Any]],
    projections_by_week: Mapping[int, Mapping[str, Mapping[str, Any]]],
    teams_playing: Mapping[int, Set[str]],
    weeks: Sequence[int],
    scoring: str,
    fallback_ppg: Optional[Mapping[str, float]] = None,
) -> Dict[str, PlayerOutlook]:
    """Weekly projection + availability per skill player.

    Projection: Sleeper's weekly projection (out-of-sample on 2026 wks 5-10 it
    beat our model: MAE 5.02 vs 5.47, corr 0.633 vs 0.548, and blending did not
    help). Players Sleeper does not project fall back to our proj_ppg unless
    they carry an IR/Out-type tag (no projection then means not expected back).
    Byes (team absent from the schedule) are 0.
    Availability: ROS_AVAILABILITY_BY_POS base rate; the first remaining week
    uses ROS_NEXT_WEEK_STATUS_AVAILABILITY for tagged players.
    """
    fallback_ppg = fallback_ppg or {}
    first_week = min(weeks) if weeks else None
    out: Dict[str, PlayerOutlook] = {}
    for pid in player_ids:
        pid = str(pid)
        meta = players_db.get(pid) or {}
        pos = str(meta.get("position") or "").upper()
        if pos not in SKILL_POSITIONS:
            continue
        team = meta.get("team")
        status = meta.get("injury_status")
        base = ROS_AVAILABILITY_BY_POS.get(pos, 0.9)
        weekly: Dict[int, float] = {}
        avail: Dict[int, float] = {}
        source: Dict[int, str] = {}
        for w in weeks:
            playing = teams_playing.get(w)
            if not team or (playing is not None and team not in playing):
                weekly[w], avail[w], source[w] = 0.0, 0.0, "bye" if team else "free_agent_team"
                continue
            pts = projection_points((projections_by_week.get(w) or {}).get(pid) or {}, scoring, pos)
            if pts is None:
                if status in ROS_NO_PROJECTION_STATUSES:
                    weekly[w], avail[w], source[w] = 0.0, 0.0, "injured"
                    continue
                pts, src = float(fallback_ppg.get(pid, 0.0) or 0.0), "model"
            else:
                src = "sleeper"
            p = base
            if w == first_week and status in ROS_NEXT_WEEK_STATUS_AVAILABILITY:
                p = ROS_NEXT_WEEK_STATUS_AVAILABILITY[status]
            weekly[w], avail[w], source[w] = max(0.0, pts), p, src
        out[pid] = PlayerOutlook(
            player_id=pid,
            name=meta.get("full_name") or pid,
            position=pos,
            team=team,
            injury_status=status,
            weekly=weekly,
            availability=avail,
            source=source,
        )
    return out


# --------------------------------------------------------------------------- #
# Valuation
# --------------------------------------------------------------------------- #
@dataclass
class TeamValue:
    per_week: Dict[int, float]
    regular: float
    playoffs: float

    @property
    def total(self) -> float:
        return self.regular + self.playoffs


def _solve_week(by_pos: Dict[str, List[float]], dedicated: Dict[str, int], flex: Dict[str, int]) -> float:
    """Best lineup total for one week given each position's available values sorted desc."""
    total = 0.0
    left: Dict[str, List[float]] = {}
    for pos in SKILL_POSITIONS:
        vals = by_pos.get(pos, [])
        n = dedicated.get(pos, 0)
        total += sum(vals[:n])
        left[pos] = vals[n:]
    heads = {pos: 0 for pos in SKILL_POSITIONS}
    for ftype in FLEX_ORDER:
        for _ in range(flex.get(ftype, 0)):
            best_pos, best_val = None, 0.0
            for pos in FLEX_ELIGIBILITY[ftype]:
                i = heads[pos]
                if i < len(left[pos]) and left[pos][i] > best_val:
                    best_pos, best_val = pos, left[pos][i]
            if best_pos is None:
                break
            heads[best_pos] += 1
            total += best_val
    return total


class RosValuator:
    """Expected ROS optimal-lineup points for any set of players."""

    def __init__(
        self,
        shape: LeagueShape,
        outlooks: Mapping[str, PlayerOutlook],
        n_sims: int = ROS_SIMS,
        seed: int = 7,
    ):
        self.shape = shape
        self.outlooks = dict(outlooks)
        self.n_sims = int(n_sims)
        self.seed = int(seed)
        self._draw_cache: Dict[str, np.ndarray] = {}
        self._value_cache: Dict[Tuple[frozenset, str], TeamValue] = {}
        self._waiver_cache: Dict[Tuple, Tuple["RosterState", List[Dict[str, Any]]]] = {}

    def _draws(self, pid: str) -> np.ndarray:
        """Per-player uniforms (sims x weeks), identical on every roster (common random numbers)."""
        if pid not in self._draw_cache:
            rng = np.random.default_rng((zlib.crc32(pid.encode()) ^ self.seed) & 0xFFFFFFFF)
            self._draw_cache[pid] = rng.random((self.n_sims, len(self.shape.weeks)))
        return self._draw_cache[pid]

    def team_value(self, pids: Iterable[str], mode: str = "sim") -> TeamValue:
        """mode="sim": simulated availability (prices depth); "expected": deterministic
        availability-weighted values (fast, for search; byes/known injuries only)."""
        key = (frozenset(str(p) for p in pids), mode)
        if key in self._value_cache:
            return self._value_cache[key]
        players = [self.outlooks[p] for p in key[0] if p in self.outlooks]
        weeks = self.shape.weeks
        per_week: Dict[int, float] = {}
        for wi, w in enumerate(weeks):
            # pre-sort each position by projection so filtered lists stay sorted
            ranked = sorted(players, key=lambda o: o.weekly.get(w, 0.0), reverse=True)
            if mode == "expected" or self.n_sims <= 0:
                by_pos: Dict[str, List[float]] = {}
                for o in sorted(players, key=lambda o: o.expected(w), reverse=True):
                    v = o.expected(w)
                    if v > 0:
                        by_pos.setdefault(o.position, []).append(v)
                per_week[w] = _solve_week(by_pos, self.shape.dedicated, self.shape.flex)
                continue
            per_week[w] = self._simulate_week(ranked, w, wi)
        regular = sum(v for w, v in per_week.items() if w not in self.shape.playoff_weeks)
        playoffs = sum(v for w, v in per_week.items() if w in self.shape.playoff_weeks)
        tv = TeamValue(per_week=per_week, regular=regular, playoffs=playoffs)
        self._value_cache[key] = tv
        return tv

    def _simulate_week(self, ranked: Sequence[PlayerOutlook], week: int, wi: int) -> float:
        """Mean best-lineup points over all availability simulations for one week.

        Vectorized greedy (identical picks to _solve_week): players are sorted by
        projection, so within any eligibility group the top-n *available* players
        are the available ones whose running available-count is <= n. Dedicated
        slots fill first, then flex types from most to least restrictive."""
        if not ranked:
            return 0.0
        vals = np.array([o.weekly.get(week, 0.0) for o in ranked])
        probs = np.array([o.availability.get(week, 0.0) for o in ranked])
        draws = np.column_stack([self._draws(o.player_id)[:, wi] for o in ranked])  # (sims, players)
        left = (draws < probs) & (vals > 0)
        pos = np.array([o.position for o in ranked])
        total = np.zeros(self.n_sims)

        def fill(cols: np.ndarray, n: int) -> None:
            nonlocal total
            if n <= 0 or cols.size == 0:
                return
            sub = left[:, cols]
            take = sub & (np.cumsum(sub, axis=1) <= n)
            total += take @ vals[cols]
            left[:, cols] = sub & ~take

        for p in SKILL_POSITIONS:
            fill(np.flatnonzero(pos == p), self.shape.dedicated.get(p, 0))
        for f in FLEX_ORDER:
            fill(np.flatnonzero(np.isin(pos, list(FLEX_ELIGIBILITY[f]))), self.shape.flex.get(f, 0))
        return float(total.mean())

    # ------------------------------------------------------------------ #
    # Roster limits
    # ------------------------------------------------------------------ #
    def settle_roster(
        self,
        roster: "RosterState",
        free_agents: Sequence[str],
        mode: str = "expected",
    ) -> Tuple["RosterState", List[str], List[str]]:
        """Applies roster limits: IR-eligible players to open reserve slots,
        cut the least valuable droppable players when over the active limit,
        and fill open active spots with the best free agents.
        Returns (settled roster, dropped ids, added ids)."""
        roster = roster.copy()
        # Move IR-tagged active players into open reserve slots
        for pid in list(roster.active):
            if len(roster.reserve) >= self.shape.reserve_slots:
                break
            o = self.outlooks.get(pid)
            if o is not None and o.injury_status == "IR":
                roster.active.remove(pid)
                roster.reserve.append(pid)

        dropped: List[str] = []
        protected = set(roster.protected)
        while len(roster.active) > self.shape.max_active:
            candidates = [
                p for p in roster.active
                if p in self.outlooks and p not in protected
            ]
            if not candidates:
                break
            base = self.team_value(roster.all_ids(), mode).total
            loss = {
                p: base - self.team_value([q for q in roster.all_ids() if q != p], mode).total
                for p in candidates
            }
            # tie-break: lowest ROS expected points
            victim = min(candidates, key=lambda p: (round(loss[p], 3), self.outlooks[p].ros_expected()))
            roster.active.remove(victim)
            dropped.append(victim)

        added: List[str] = []
        pool = [p for p in free_agents if p in self.outlooks and p not in roster.all_ids()]
        while len(roster.active) < self.shape.max_active and pool:
            base = self.team_value(roster.all_ids(), mode).total
            gains = {p: self.team_value(roster.all_ids() + [p], mode).total - base for p in pool}
            best = max(pool, key=lambda p: gains[p])
            if gains[best] <= 0:
                break
            roster.active.append(best)
            added.append(best)
            pool.remove(best)
        return roster, dropped, added


@dataclass
class RosterState:
    active: List[str]
    reserve: List[str]
    protected: List[str] = field(default_factory=list)  # never auto-dropped (e.g. K/DEF placeholders)

    def all_ids(self) -> List[str]:
        return list(self.active) + list(self.reserve)

    def copy(self) -> "RosterState":
        return RosterState(list(self.active), list(self.reserve), list(self.protected))


def optimize_waivers(
    valuator: RosValuator,
    roster: RosterState,
    free_agents: Sequence[str],
    keep: Iterable[str] = (),
    max_swaps: int = ROS_WAIVER_MAX_SWAPS,
    min_gain_ppw: float = ROS_WAIVER_MIN_GAIN_PPW,
    drop_candidates: int = ROS_WAIVER_DROP_CANDIDATES,
) -> Tuple[RosterState, List[Dict[str, Any]]]:
    """The roster after its best free waiver moves (greedy drop->add swaps, simulated value).

    This is the fair "before" for a trade: a deal must not get credit for a
    pickup the team could make anyway (e.g. a 2-for-1 opening a spot that is
    then filled by a free agent who was always available). Players in `keep`
    (the ones being traded) are never dropped. Open spots are filled first.
    """
    keep = set(keep)
    pool = tuple(p for p in free_agents if p in valuator.outlooks and p not in roster.all_ids())
    key = (frozenset(roster.active), frozenset(roster.reserve), frozenset(keep), pool)
    if key in valuator._waiver_cache:
        r, moves = valuator._waiver_cache[key]
        return r.copy(), list(moves)

    n_weeks = max(1, len(valuator.shape.weeks))
    r, _, added = valuator.settle_roster(roster, pool, mode="sim")
    moves: List[Dict[str, Any]] = [
        {"drop": None, "add": p, "drop_name": None, "add_name": valuator.outlooks[p].name, "gain_ppw": None}
        for p in added
    ]
    remaining = [p for p in pool if p not in r.all_ids()]
    for _ in range(max_swaps):
        base = valuator.team_value(r.all_ids(), "sim").total
        droppable = sorted(
            (p for p in r.active if p in valuator.outlooks and p not in keep),
            key=lambda p: valuator.outlooks[p].ros_expected(),
        )[:drop_candidates]
        best = (min_gain_ppw * n_weeks, None, None)
        for f in remaining:
            for d in droppable:
                gain = valuator.team_value([q for q in r.all_ids() if q != d] + [f], "sim").total - base
                if gain > best[0]:
                    best = (gain, d, f)
        gain, d, f = best
        if d is None:
            break
        r.active = [q for q in r.active if q != d] + [f]
        remaining.remove(f)
        moves.append({
            "drop": d, "add": f,
            "drop_name": valuator.outlooks[d].name, "add_name": valuator.outlooks[f].name,
            "gain_ppw": round(gain / n_weeks, 2),
        })
    valuator._waiver_cache[key] = (r.copy(), list(moves))
    return r, moves


def plan_waiver_moves(
    valuator: RosValuator,
    roster: RosterState,
    free_agents: Sequence[str],
    season: Optional[Mapping[str, Any]] = None,
    max_swaps: int = ROS_WAIVER_MAX_SWAPS,
    n_alternatives: int = 5,
) -> Dict[str, Any]:
    """Best free drop->add moves for one roster, with cumulative value and odds.

    Moves come from optimize_waivers (greedy, simulated ROS value). Each step
    reports its own gain and the running totals; with `season` (as in
    evaluate_trade, partner_index unused) playoff / bye / title odds are
    re-simulated after every step. `alternatives` lists the next-best single
    swaps from the final roster (the runner-up claims if a top target is taken).
    """
    n_weeks = max(1, len(valuator.shape.weeks))
    _, moves = optimize_waivers(valuator, roster, free_agents, max_swaps=max_swaps)

    def card(pid: Optional[str]) -> Optional[Dict[str, Any]]:
        if pid is None or pid not in valuator.outlooks:
            return None
        o = valuator.outlooks[pid]
        return {
            "player_id": pid, "name": o.name, "position": o.position, "team": o.team,
            "injury_status": o.injury_status, "ros_ppw": round(o.ros_expected() / n_weeks, 2),
            "byes": [w for w in valuator.shape.weeks if o.source.get(w) == "bye"],
        }

    def odds_for(per_week: Mapping[int, float]) -> Optional[Dict[str, float]]:
        if not season:
            return None
        sim, ui = season["sim"], season["user_index"]
        means = np.array(season["base_means"], dtype=float, copy=True)
        for j, w in enumerate(sim.state.weeks):
            if w in per_week:
                means[ui, j] = per_week[w]
        return sim.run(means).for_team(ui)

    cur = roster.copy()
    v0 = valuator.team_value(cur.all_ids(), "sim")
    before = {"ppw": round(v0.total / n_weeks, 2), "odds": odds_for(v0.per_week)}
    steps: List[Dict[str, Any]] = []
    for m in moves:
        if m["drop"] is not None:
            cur.active = [p for p in cur.active if p != m["drop"]]
        cur.active.append(m["add"])
        v = valuator.team_value(cur.all_ids(), "sim")
        steps.append({
            "drop": card(m["drop"]),
            "add": card(m["add"]),
            "gain_ppw": m["gain_ppw"],
            "cumulative_ppw": round((v.total - v0.total) / n_weeks, 2),
            "cumulative_playoffs": round(v.playoffs - v0.playoffs, 2),
            "odds_after": odds_for(v.per_week),
        })

    # Runner-up single swaps from the final roster (best drop for each remaining FA)
    base = valuator.team_value(cur.all_ids(), "sim").total
    droppable = sorted(
        (p for p in cur.active if p in valuator.outlooks),
        key=lambda p: valuator.outlooks[p].ros_expected(),
    )[:ROS_WAIVER_DROP_CANDIDATES]
    alts: List[Dict[str, Any]] = []
    for f in free_agents:
        if f not in valuator.outlooks or f in cur.all_ids():
            continue
        best_gain, best_drop = None, None
        for d in droppable:
            g = valuator.team_value([q for q in cur.all_ids() if q != d] + [f], "sim").total - base
            if best_gain is None or g > best_gain:
                best_gain, best_drop = g, d
        if best_gain is not None:
            alts.append({"add": card(f), "drop": card(best_drop), "gain_ppw": round(best_gain / n_weeks, 2)})
    alts.sort(key=lambda a: a["gain_ppw"], reverse=True)

    after_v = valuator.team_value(cur.all_ids(), "sim")
    return {
        "weeks": list(valuator.shape.weeks),
        "before": before,
        "after": {"ppw": round(after_v.total / n_weeks, 2), "odds": odds_for(after_v.per_week)},
        "moves": steps,
        "alternatives": alts[:n_alternatives],
        "min_gain_ppw": ROS_WAIVER_MIN_GAIN_PPW,
    }


def representative_lineup(valuator: RosValuator, ids: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Typical starting lineup by average ROS points per week (for display; weekly
    lineups vary with byes and injuries)."""
    n_weeks = max(1, len(valuator.shape.weeks))
    players = sorted(
        (valuator.outlooks[p] for p in ids if p in valuator.outlooks),
        key=lambda o: o.ros_expected(), reverse=True,
    )
    used: Set[str] = set()
    starters: List[Dict[str, Any]] = []

    def take(slot: str, eligible: Set[str]) -> None:
        for o in players:
            if o.player_id not in used and o.position in eligible:
                used.add(o.player_id)
                starters.append({"slot": slot, "name": o.name, "position": o.position,
                                 "ppw": round(o.ros_expected() / n_weeks, 1)})
                return

    for pos in SKILL_POSITIONS:
        for _ in range(valuator.shape.dedicated.get(pos, 0)):
            take(pos, {pos})
    for ftype in FLEX_ORDER:
        for _ in range(valuator.shape.flex.get(ftype, 0)):
            take(ftype, FLEX_ELIGIBILITY[ftype])
    bench = [
        {"slot": "BN", "name": o.name, "position": o.position, "ppw": round(o.ros_expected() / n_weeks, 1)}
        for o in players if o.player_id not in used
    ]
    return {"starters": starters, "bench": bench}


def top_free_agents(
    outlooks: Mapping[str, PlayerOutlook],
    rostered: Set[str],
    per_position: int = ROS_FA_CANDIDATES_PER_POS,
) -> List[str]:
    """Best unrostered players per position by ROS expected points (pickup candidates)."""
    out: List[str] = []
    for pos in SKILL_POSITIONS:
        pool = [o for pid, o in outlooks.items() if o.position == pos and pid not in rostered]
        pool.sort(key=lambda o: o.ros_expected(), reverse=True)
        out.extend(o.player_id for o in pool[:per_position])
    return out


def _side_result(
    valuator: RosValuator,
    before: RosterState,
    gives: Sequence[str],
    gets: Sequence[str],
    free_agents: Sequence[str],
    mode: str,
) -> Dict[str, Any]:
    settled_before, drop_b, add_b = valuator.settle_roster(before, free_agents, mode="expected")
    after = before.copy()
    for p in gives:
        if p in after.active:
            after.active.remove(p)
        elif p in after.reserve:
            after.reserve.remove(p)
    after.active.extend(gets)
    settled_after, drop_a, add_a = valuator.settle_roster(after, free_agents, mode="expected")
    v0 = valuator.team_value(settled_before.all_ids(), mode)
    v1 = valuator.team_value(settled_after.all_ids(), mode)
    lineup_before = representative_lineup(valuator, settled_before.all_ids())
    lineup_after = representative_lineup(valuator, settled_after.all_ids())
    weeks = valuator.shape.weeks
    delta_week = {w: round(v1.per_week[w] - v0.per_week[w], 2) for w in weeks}
    n_weeks = max(1, len(weeks))
    d_total = v1.total - v0.total
    return {
        "before_total": round(v0.total, 1),
        "after_total": round(v1.total, 1),
        "delta_total": round(d_total, 2),
        "delta_regular": round(v1.regular - v0.regular, 2),
        "delta_playoffs": round(v1.playoffs - v0.playoffs, 2),
        "delta_per_week": round(d_total / n_weeks, 2),
        "weekly_delta": delta_week,
        "drops": [valuator.outlooks[p].name for p in drop_a if p not in drop_b],
        "adds": [valuator.outlooks[p].name for p in add_a if p not in add_b],
        "lineup_before": lineup_before,
        "lineup_after": lineup_after,
        "per_week_before": {w: v0.per_week[w] for w in weeks},
        "per_week_after": {w: v1.per_week[w] for w in weeks},
    }


def evaluate_trade(
    valuator: RosValuator,
    team_a: RosterState,
    team_b: RosterState,
    a_gives: Sequence[str],
    b_gives: Sequence[str],
    free_agents: Sequence[str],
    mode: str = "sim",
    waiver_baseline: bool = True,
    season: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """ROS evaluation of a trade from both sides (A = the user).

    With waiver_baseline (default), both teams are first moved to their
    waiver-optimized rosters (optimize_waivers; A picks first, B from what is
    left), and the trade is measured against those. The free moves are
    reported separately as "waiver_moves": they are worth making regardless.

    With `season` ({"sim": SeasonSimulator, "base_means": (teams x weeks)
    array aligned with sim.state.weeks, "user_index", "partner_index"}), both
    teams' before/after weekly lineup means are run through the season
    simulator (same draws) and the verdict uses the change in YOUR TITLE ODDS:
    standings decide whether regular-season or playoff points matter more.
    """
    a_moves: List[Dict[str, Any]] = []
    b_moves: List[Dict[str, Any]] = []
    pool = list(free_agents)
    if waiver_baseline and mode == "sim":
        team_a, a_moves = optimize_waivers(valuator, team_a, pool, keep=a_gives)
        pool = [p for p in pool if p not in team_a.all_ids()]
        team_b, b_moves = optimize_waivers(valuator, team_b, pool, keep=b_gives)
        pool = [p for p in pool if p not in team_b.all_ids()]
    a = _side_result(valuator, team_a, a_gives, b_gives, pool, mode)
    b = _side_result(valuator, team_b, b_gives, a_gives, pool, mode)
    a["waiver_moves"] = a_moves
    b["waiver_moves"] = b_moves
    odds = _season_odds(season, a, b) if season else None
    if odds is not None:
        d_you, d_them = odds["you"]["delta"]["title"], odds["partner"]["delta"]["title"]
        m = ROS_TITLE_VERDICT_MARGIN
        if d_you >= m and d_them >= 0:
            verdict = "WIN-WIN (raises both teams' title odds)"
        elif d_you >= m:
            verdict = "FAVORS YOU"
        elif d_you <= -m:
            verdict = "HURTS YOU"
        else:
            verdict = "ROUGHLY NEUTRAL FOR YOU"
        verdict_basis = f"title odds (|Δ| < {m:.1%} = neutral)"
    else:
        margin = ROS_VERDICT_MARGIN_PPW
        if a["delta_per_week"] >= margin and b["delta_per_week"] >= 0:
            verdict = "WIN-WIN (helps both lineups)"
        elif a["delta_per_week"] >= margin:
            verdict = "FAVORS YOU"
        elif a["delta_per_week"] <= -margin:
            verdict = "HURTS YOU"
        else:
            verdict = "ROUGHLY NEUTRAL FOR YOU"
        verdict_basis = f"lineup pts/week (|Δ| < {margin} = neutral)"
    weeks = valuator.shape.weeks

    def card(pid: str) -> Dict[str, Any]:
        o = valuator.outlooks.get(pid)
        if o is None:
            return {"player_id": pid, "name": pid}
        return {
            "player_id": pid,
            "name": o.name,
            "position": o.position,
            "team": o.team,
            "injury_status": o.injury_status,
            "ros_expected_points": round(o.ros_expected(), 1),
            "byes": [w for w in weeks if o.source.get(w) == "bye"],
            "weeks_without_projection": [w for w in weeks if o.source.get(w) == "injured"],
        }

    return {
        "weeks": weeks,
        "playoff_weeks": sorted(valuator.shape.playoff_weeks),
        "you_send": [card(p) for p in a_gives],
        "you_get": [card(p) for p in b_gives],
        "you": a,
        "partner": b,
        "verdict": verdict,
        "verdict_basis": verdict_basis,
        "verdict_margin_ppw": ROS_VERDICT_MARGIN_PPW,
        "odds": odds,
        "n_sims": valuator.n_sims if mode == "sim" else 0,
    }


def _season_odds(season: Mapping[str, Any], a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Playoff / bye / title odds for both teams before and after (common random numbers)."""
    sim = season["sim"]
    weeks = sim.state.weeks
    ui, pi = season["user_index"], season["partner_index"]
    before = np.array(season["base_means"], dtype=float, copy=True)
    after = before.copy()
    for team, side in ((ui, a), (pi, b)):
        for j, w in enumerate(weeks):
            if w in side["per_week_before"]:
                before[team, j] = side["per_week_before"][w]
                after[team, j] = side["per_week_after"][w]
    o0, o1 = sim.run(before), sim.run(after)

    def pack(i: int) -> Dict[str, Any]:
        b0, b1 = o0.for_team(i), o1.for_team(i)
        return {"before": b0, "after": b1, "delta": {k: b1[k] - b0[k] for k in b0}}

    return {"you": pack(ui), "partner": pack(pi), "n_sims": sim.n_sims, "weekly_sd": sim.state.weekly_sd}


def title_leverage(season: Mapping[str, Any], team: int, bump: float = 10.0) -> Dict[int, float]:
    """Title-odds change per projected point in each week for one team (finite
    difference with common random numbers). Shows which weeks matter given the standings."""
    sim = season["sim"]
    base = np.asarray(season["base_means"], dtype=float)
    t0 = sim.run(base).title[team]
    out: Dict[int, float] = {}
    for j, w in enumerate(sim.state.weeks):
        m = base.copy()
        m[team, j] += bump
        out[w] = float((sim.run(m).title[team] - t0) / bump)
    return out


# --------------------------------------------------------------------------- #
# Trade finder
# --------------------------------------------------------------------------- #
def find_trades(
    valuator: RosValuator,
    rosters: Sequence[RosterState],
    user_index: int,
    free_agents: Sequence[str],
    manager_names: Optional[Sequence[str]] = None,
    roster_ids: Optional[Sequence[Any]] = None,
    max_assets: int = 10,
    rescore_top: int = 25,
    limit: int = 10,
    min_user_gain_ppw: float = ROS_VERDICT_MARGIN_PPW,
    screen_partner_tolerance_ppw: float = 0.5,
    max_per_target_set: int = 2,
    max_per_opponent: int = 4,
    market_values: Optional[Mapping[str, Mapping[str, Any]]] = None,
    min_partner_market_pct: float = -0.10,
    min_market_total: float = 300.0,
    season: Optional[Mapping[str, Any]] = None,
    min_user_title_gain: float = ROS_TITLE_VERDICT_MARGIN,
) -> Dict[str, Any]:
    """Searches 1-for-1, 2-for-1, and 1-for-2 swaps with every opponent.

    Stage 1 (screen): deterministic "expected" mode, no roster settling, so
    thousands of swaps stay cheap. Keeps swaps where you gain and the partner
    loses at most screen_partner_tolerance_ppw (the screen undervalues the depth
    a 2-for-1 receiver gets, so it is deliberately loose).
    Stage 2: the best `rescore_top` by screened user gain (at most
    max_per_target_set per received-player set and max_per_opponent per
    opponent, so one hot target cannot crowd out every idea) are re-evaluated with
    simulated availability and roster limits. With `market_values`
    (FantasyCalc), the screen also drops swaps where the partner would give up
    more than -min_partner_market_pct of the combined market value: lineup
    win-wins the partner perceives as lopsided rarely get accepted, so the
    re-score budget goes to realistic offers. Then they are re-evaluated with
    simulated availability and roster limits (evaluate_trade). Survivors must
    gain you >= min_user_gain_ppw per week and not hurt the partner (>= 0):
    trades a rational partner could accept on lineup grounds alone.
    """
    weeks = max(1, len(valuator.shape.weeks))
    manager_names = list(manager_names or [])
    roster_ids = list(roster_ids or [])

    def assets(r: RosterState) -> List[str]:
        pids = [p for p in r.all_ids() if p in valuator.outlooks and valuator.outlooks[p].ros_expected() > 0]
        pids.sort(key=lambda p: valuator.outlooks[p].ros_expected(), reverse=True)
        return pids[:max_assets]

    def quick(r: RosterState, gives: Sequence[str], gets: Sequence[str], weights: Optional[Mapping[int, float]] = None) -> float:
        """Fast lineup change vs the team's waiver-optimized roster, enforcing the
        roster limit (over-limit -> cut the lowest-ROS players not just acquired).
        Returns pts/week, or with `weights` (title leverage per week) the
        estimated title-odds change."""
        before_v = valuator.team_value(r.all_ids(), "expected")
        before = before_v.total
        active = [p for p in r.active if p not in set(gives)] + list(gets)
        excess = len(active) - valuator.shape.max_active
        if excess > 0:
            cuttable = sorted(
                (p for p in active if p in valuator.outlooks and p not in set(gets)),
                key=lambda p: valuator.outlooks[p].ros_expected(),
            )
            cut = set(cuttable[:excess])
            active = [p for p in active if p not in cut]
        after_ids = active + [p for p in r.reserve if p not in set(gives)]
        after_v = valuator.team_value(after_ids, "expected")
        if weights is not None:
            return sum(weights.get(w, 0.0) * (after_v.per_week[w] - before_v.per_week[w]) for w in after_v.per_week)
        return (after_v.total - before) / weeks

    def partner_market_pct(give: Sequence[str], get: Sequence[str]) -> Optional[float]:
        if not market_values:
            return None
        send = sum(float((market_values.get(p) or {}).get("value", 0.0)) for p in give)
        recv = sum(float((market_values.get(p) or {}).get("value", 0.0)) for p in get)
        total = send + recv
        return None if total < min_market_total else (send - recv) / total

    # Screen against waiver-optimized rosters (as evaluate_trade does) so gains that
    # are really free pickups do not crowd the re-score budget
    pool = list(free_agents)
    user, _ = optimize_waivers(valuator, rosters[user_index], pool)
    pool = [p for p in pool if p not in user.all_ids()]
    screen_rosters = {
        oi: optimize_waivers(valuator, opp, pool)[0] for oi, opp in enumerate(rosters) if oi != user_index
    }
    mine = assets(user)
    market_rejected = 0
    # Standings-aware screen: weight your weekly gains by title leverage
    user_weights = title_leverage(season, season["user_index"]) if season else None
    screened: List[Tuple[float, int, Tuple[str, ...], Tuple[str, ...]]] = []
    evaluated = 0
    for oi in screen_rosters:
        opp = screen_rosters[oi]
        theirs = assets(opp)
        swaps: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
        swaps += [((u,), (o,)) for u in mine for o in theirs]
        swaps += [((u1, u2), (o,)) for i, u1 in enumerate(mine) for u2 in mine[i + 1:] for o in theirs]
        swaps += [((u,), (o1, o2)) for u in mine for i, o1 in enumerate(theirs) for o2 in theirs[i + 1:]]
        for give, get in swaps:
            evaluated += 1
            m = partner_market_pct(give, get)
            if m is not None and m < min_partner_market_pct:
                market_rejected += 1
                continue
            du = quick(user, give, get, user_weights)
            if du <= 0:
                continue
            do = quick(opp, get, give)
            if do < -screen_partner_tolerance_ppw:
                continue
            screened.append((du, oi, give, get))

    screened.sort(key=lambda t: t[0], reverse=True)
    picked: List[Tuple[float, int, Tuple[str, ...], Tuple[str, ...]]] = []
    per_set: Dict[frozenset, int] = {}
    per_opp: Dict[int, int] = {}
    for cand in screened:
        if len(picked) >= rescore_top:
            break
        _, oi, _, get = cand
        key = frozenset(get)
        if per_set.get(key, 0) >= max_per_target_set or per_opp.get(oi, 0) >= max_per_opponent:
            continue
        per_set[key] = per_set.get(key, 0) + 1
        per_opp[oi] = per_opp.get(oi, 0) + 1
        picked.append(cand)

    proposals: List[Dict[str, Any]] = []
    for _, oi, give, get in picked:
        trade_season = {**season, "partner_index": oi} if season else None
        res = evaluate_trade(
            valuator, rosters[user_index], rosters[oi], list(give), list(get), free_agents, mode="sim",
            season=trade_season,
        )
        if res["partner"]["delta_per_week"] < 0:
            continue
        if res.get("odds"):
            if res["odds"]["you"]["delta"]["title"] < min_user_title_gain:
                continue
        elif res["you"]["delta_per_week"] < min_user_gain_ppw:
            continue
        res["target_index"] = oi
        res["target_manager"] = manager_names[oi] if oi < len(manager_names) else f"Team {oi + 1}"
        res["target_roster_id"] = roster_ids[oi] if oi < len(roster_ids) else None
        proposals.append(res)

    proposals.sort(
        key=lambda r: (
            r["odds"]["you"]["delta"]["title"] if r.get("odds") else r["you"]["delta_per_week"],
            r["partner"]["delta_per_week"],
        ),
        reverse=True,
    )
    return {
        "user_title_leverage": user_weights,
        "proposals": proposals[:limit],
        "swaps_screened": evaluated,
        "market_rejected": market_rejected,
        "passed_screen": len(screened),
        "rescored": len(picked),
    }
