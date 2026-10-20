"""Situational matchup analysis: Start/Sit optimization, injury warnings, and opponent tactics."""

from __future__ import annotations
import logging
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from fantasy_intel.config import MATCHUP_POSTURE_MARGIN, WIND_PENALTY_THRESHOLD_MPH

logger = logging.getLogger(__name__)

LEAGUE_AVG_TEAM_TOTAL = 22.5
ITT_EXPONENT = 0.6
WIND_PASS_MULTIPLIER = 0.88
WIND_RB_MULTIPLIER = 1.05
DEF_FUNNEL_MULTIPLIER = 1.08
DEF_FUNNEL_TOP_N = 8
QUESTIONABLE_PENALTY = 0.9
DEEP_THREAT_ADOT = 12.0
OUT_STATUSES = ("Out", "IR", "Doubtful")
OUTDOOR_ROOFS = ("outdoors", "open")

SLOT_ELIGIBILITY: Dict[str, Tuple[str, ...]] = {
    "QB": ("QB",),
    "RB": ("RB",),
    "WR": ("WR",),
    "TE": ("TE",),
    "FLEX": ("RB", "WR", "TE"),
    "SUPER_FLEX": ("QB", "RB", "WR", "TE"),
    "WRRB_FLEX": ("WR", "RB"),
    "REC_FLEX": ("WR", "TE"),
    "IDP_FLEX": ("LB", "DB", "DL"),
    "K": ("K",),
    "DEF": ("DEF",),
    "DL": ("DL",),
    "LB": ("LB",),
    "DB": ("DB",),
}

SLOT_PRIORITY: Dict[str, int] = {
    "QB": 0,
    "RB": 0,
    "WR": 0,
    "TE": 0,
    "K": 0,
    "DEF": 0,
    "DL": 0,
    "LB": 0,
    "DB": 0,
    "FLEX": 1,
    "WRRB_FLEX": 1,
    "REC_FLEX": 1,
    "IDP_FLEX": 1,
    "SUPER_FLEX": 2,
}

DEFAULT_ROSTER_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]

POSTURE_WEIGHTS: Dict[str, Tuple[float, float, float]] = {
    "FAVORITE": (0.65, 0.35, 0.0),
    "UNDERDOG": (0.0, 0.35, 0.65),
    "NEUTRAL": (0.25, 0.50, 0.25),
}


def _norm_name(name: Any) -> Optional[str]:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    text = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return text or None


def _player_key(player: Dict[str, Any]) -> str:
    pid = player.get("player_id")
    if pid:
        return f"id:{pid}"
    return f"name:{_norm_name(player.get('name'))}"


def compute_defensive_epa_rates(
    defense_df: Optional[pd.DataFrame],
    season: Optional[int] = None,
    week: Optional[int] = None,
    window: int = 4,
) -> pd.DataFrame:
    """Aggregates the weekly PBP defense frame into trailing per-play EPA-allowed rates.

    Input is the "defense" frame from DataLoader.get_pbp_opportunity_features:
    per (defteam, season, week) EPA sums (def_pass_epa_allowed /
    def_rush_epa_allowed) and play counts (def_pass_attempts /
    def_rush_attempts). The trailing window covers the `window` weeks before
    `week` in `season`; when week is None the window trails the latest week in
    the frame. Returns one row per defense with def_pass_epa_rate /
    def_rush_epa_rate (EPA allowed per play) and the window play counts.
    """
    required = {
        "defteam",
        "season",
        "week",
        "def_pass_epa_allowed",
        "def_pass_attempts",
        "def_rush_epa_allowed",
        "def_rush_attempts",
    }
    if defense_df is None or defense_df.empty or not required.issubset(defense_df.columns):
        return pd.DataFrame()

    df = defense_df.copy()
    for col in required - {"defteam"}:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if season is None:
        season = df["season"].max()
    df = df[df["season"] == season]
    if df.empty:
        return pd.DataFrame()
    if week is None:
        week = int(df["week"].max()) + 1
    window_df = df[(df["week"] < week) & (df["week"] >= week - window)]
    if window_df.empty:
        return pd.DataFrame()

    rates = (
        window_df.groupby("defteam", sort=False)
        .agg(
            def_pass_epa_sum=("def_pass_epa_allowed", "sum"),
            def_pass_plays=("def_pass_attempts", "sum"),
            def_rush_epa_sum=("def_rush_epa_allowed", "sum"),
            def_rush_plays=("def_rush_attempts", "sum"),
        )
        .reset_index()
    )
    rates["def_pass_epa_rate"] = np.where(
        rates["def_pass_plays"] > 0,
        rates["def_pass_epa_sum"] / rates["def_pass_plays"],
        np.nan,
    )
    rates["def_rush_epa_rate"] = np.where(
        rates["def_rush_plays"] > 0,
        rates["def_rush_epa_sum"] / rates["def_rush_plays"],
        np.nan,
    )
    return rates[
        ["defteam", "def_pass_epa_rate", "def_rush_epa_rate", "def_pass_plays", "def_rush_plays"]
    ]


class MatchupAdvisor:
    """Analyzes user's roster vs their weekly opponent and generates strategic advice.

    A bare advisor (no frames) still delivers injury alerts and injury-driven
    bench swaps. Supplied frames enable projection-driven start/sit optimization:

    - metrics_df: latest per-player metrics rows, NFL-wide (not just the user's
      roster, so opponent rosters resolve too). Columns:
      * id: "sleeper_id" (matches the Sleeper overview player_id) or
        "player_id" (gsis or sleeper keyed); a normalized
        "player_display_name" fallback lookup also applies
      * "position", "recent_team"
      * projections: "proj_floor_p15" / "proj_median_p50" / "proj_ceiling_p85";
        per-player fallback uses "rolling_xfp_3w" as p50 with p15 = 0.75x,
        p85 = 1.25x
      * depth: "aDOT" (any casing) optional; else derived from
        receiving_air_yards / targets when both exist
    - schedules_df: games for the target week with home_team / away_team,
      home_itt / away_itt (positive spread = home favored), roof, wind
      (post-game values only), game_datetime (stadium-local)
    - def_epa_df: either the raw weekly PBP defense frame (defteam / season /
      week / def_pass_epa_allowed / def_pass_attempts / def_rush_epa_allowed /
      def_rush_attempts) or the aggregated output of compute_defensive_epa_rates
    - weather_fn: callable(home_team, game_datetime) -> forecast wind mph or
      None; defaults to fantasy_intel.data.weather.get_game_wind_forecast
    - roster_positions: league slot list (QB/RB/WR/TE/FLEX/SUPER_FLEX/K/DEF);
      may also arrive via matchup_overview["roster_positions"]
    """

    def __init__(
        self,
        weekly_stats_df: Optional[pd.DataFrame] = None,
        metrics_df: Optional[pd.DataFrame] = None,
        schedules_df: Optional[pd.DataFrame] = None,
        def_epa_df: Optional[pd.DataFrame] = None,
        weather_fn: Optional[Callable[[str, Any], Optional[float]]] = None,
        roster_positions: Optional[List[str]] = None,
    ):
        self.stats = weekly_stats_df
        self.metrics_df = metrics_df
        self.schedules_df = schedules_df
        self.def_epa_df = def_epa_df
        if weather_fn is None:
            from fantasy_intel.data.weather import get_game_wind_forecast

            weather_fn = get_game_wind_forecast
        self.weather_fn = weather_fn
        self.roster_positions = list(roster_positions) if roster_positions else None
        self._metrics_by_id: Dict[str, pd.Series] = {}
        self._metrics_by_name: Dict[str, pd.Series] = {}
        self._index_metrics(metrics_df)

    def _index_metrics(self, metrics_df: Optional[pd.DataFrame]) -> None:
        if metrics_df is None or metrics_df.empty:
            return
        id_cols = [c for c in ("sleeper_id", "player_id") if c in metrics_df.columns]
        for _, row in metrics_df.iterrows():
            for col in id_cols:
                pid = row.get(col)
                if pid is not None and pd.notna(pid):
                    self._metrics_by_id.setdefault(str(pid), row)
            name = _norm_name(row.get("player_display_name"))
            if name:
                self._metrics_by_name.setdefault(name, row)

    def _lookup_metrics(self, player: Dict[str, Any]) -> Optional[pd.Series]:
        pid = player.get("player_id")
        if pid is not None:
            row = self._metrics_by_id.get(str(pid))
            if row is not None:
                return row
        return self._metrics_by_name.get(_norm_name(player.get("name")))

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def evaluate_lineup(self, matchup_overview: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produces detailed situational advice for the current week:
        - Injury alerts in starting lineup
        - Context-adjusted floor/median/ceiling projections (Vegas ITT, wind,
          defensive funnel multipliers) for both rosters
        - Matchup posture (favorite vs underdog) from median-optimal lineups
        - ILP-optimized start/sit swaps with catalyst rationales
        """
        my_team = matchup_overview.get("my_team", {})
        opp_team = matchup_overview.get("opponent_team")
        week = matchup_overview.get("week")

        starters = my_team.get("starters", [])
        bench = my_team.get("bench", [])
        opp_starters = opp_team.get("starters", []) if opp_team else []
        opp_bench = opp_team.get("bench", []) if opp_team else []

        # 1. Injury Alerts for Active Starters
        injury_alerts = []
        for p in starters:
            status = p.get("injury_status")
            if status in ["Out", "IR", "Doubtful"]:
                injury_alerts.append({
                    "player": p["name"],
                    "position": p["position"],
                    "status": status,
                    "severity": "CRITICAL",
                    "action": f"Bench {p['name']} immediately ({status}).",
                })
            elif status == "Questionable":
                injury_alerts.append({
                    "player": p["name"],
                    "position": p["position"],
                    "status": status,
                    "severity": "WARNING",
                    "action": f"Monitor {p['name']} game-time status; prepare bench replacement.",
                })

        # 2. Context-adjusted projections for every rostered player on both sides
        games = self._target_games(week)
        season = None
        if games:
            game_seasons = [self._to_float(g.get("season")) for g in games.values()]
            game_seasons = [s for s in game_seasons if s is not None]
            season = int(max(game_seasons)) if game_seasons else None
        def_ranks = self._defense_ranks(season, week)
        wind_cache: Dict[Tuple, Optional[float]] = {}
        projections: List[Dict[str, Any]] = []
        for side, players in (
            ("user", list(starters) + list(bench)),
            ("opponent", list(opp_starters) + list(opp_bench)),
        ):
            for p in players:
                proj = self._project_player(p, games, def_ranks, wind_cache)
                proj["side"] = side
                projections.append(proj)

        user_pool = [p for p in projections if p["side"] == "user" and not p["excluded"]]
        opp_pool = [p for p in projections if p["side"] == "opponent" and not p["excluded"]]

        # 3. Matchup posture from median-optimal lineups on both sides
        slots = self._resolve_slots(matchup_overview)
        for p in user_pool + opp_pool:
            p["y"] = p["median"]
        user_median_lineup, _ = self._optimize_lineup(user_pool, slots)
        opp_median_lineup, _ = self._optimize_lineup(opp_pool, slots)
        user_median_total = sum(p["median"] for p in user_median_lineup)
        opp_median_total = sum(p["median"] for p in opp_median_lineup)
        delta_median_raw = user_median_total - opp_median_total
        delta_median = round(delta_median_raw, 1)

        if delta_median_raw >= MATCHUP_POSTURE_MARGIN:
            posture = "FAVORITE"
        elif delta_median_raw <= -MATCHUP_POSTURE_MARGIN:
            posture = "UNDERDOG"
        else:
            posture = "NEUTRAL"
        w15, w50, w85 = POSTURE_WEIGHTS[posture]
        for p in projections:
            p["y"] = w15 * p["floor"] + w50 * p["median"] + w85 * p["ceiling"]

        # 4. Optimal starting lineup under the posture-weighted objective
        optimal_lineup, opt_diag = self._optimize_lineup(user_pool, slots)
        optimal_objective = sum(p["y"] for p in optimal_lineup)

        # 5. Start / Sit swap recommendations
        current_keys = {_player_key(p) for p in starters}
        optimal_keys = {p["key"] for p in optimal_lineup}
        incoming = sorted(
            (p for p in optimal_lineup if p["key"] not in current_keys),
            key=lambda p: p["y"],
            reverse=True,
        )
        proj_by_key = {p["key"]: p for p in projections if p["side"] == "user"}
        outgoing = [proj_by_key[_player_key(s)] for s in starters if _player_key(s) not in optimal_keys]

        start_sit_suggestions = []
        for inc in incoming:
            if not outgoing:
                break
            same_pos = [o for o in outgoing if o["position"] == inc["position"]]
            out = min(same_pos or outgoing, key=lambda p: p["y"])
            outgoing.remove(out)
            gain = round(inc["y"] - out["y"], 1)
            has_catalyst = any(d == "penalty" for d, _ in out["catalysts"]) or any(
                d == "boost" for d, _ in inc["catalysts"]
            )
            if gain <= 0 and not has_catalyst:
                continue
            start_sit_suggestions.append({
                "sit": out["name"],
                "start": inc["name"],
                "position": inc["position"],
                "reason": self._swap_rationale(inc, out, posture),
                "projected_gain": gain,
                "sit_status": out["injury_status"],
                "in_projection": {"floor": inc["floor"], "median": inc["median"], "ceiling": inc["ceiling"]},
                "out_projection": {"floor": out["floor"], "median": out["median"], "ceiling": out["ceiling"]},
            })

        # 6. Positional / tactical notes
        strategy_notes = []
        if opp_team:
            opp_injuries = [op["name"] for op in opp_starters if op.get("injury_status") in ["Out", "IR", "Doubtful"]]
            if opp_injuries:
                strategy_notes.append(f"Opponent has injured players in active lineup: {', '.join(opp_injuries)}. Capitalize by playing high-floor starters.")
            else:
                strategy_notes.append("Opponent has a healthy starting lineup; prioritize high-ceiling players in FLEX spots.")

        if posture == "FAVORITE":
            strategy_notes.append(f"Matchup posture: favored by {delta_median:+.1f} median pts — objective protects the floor (0.65*P15 + 0.35*P50).")
        elif posture == "UNDERDOG":
            strategy_notes.append(f"Matchup posture: underdog by {delta_median:+.1f} median pts — objective chases the ceiling (0.65*P85 + 0.35*P50).")

        user_byes = [p["name"] for p in projections if p["side"] == "user" and p["exclude_reason"] == "bye"]
        if user_byes:
            strategy_notes.append(f"On bye this week (excluded from the optimization pool): {', '.join(user_byes)}.")
        if opt_diag.get("fallback_reason"):
            strategy_notes.append(f"Lineup optimizer: {opt_diag['fallback_reason']}.")
        for shortfall in opt_diag.get("shortfalls", []):
            strategy_notes.append(f"Lineup slot shortfall: {shortfall}.")

        return {
            "week": week,
            "my_team_name": my_team.get("user_name"),
            "opponent_name": opp_team.get("user_name") if opp_team else "None",
            "injury_alerts": injury_alerts,
            "start_sit_suggestions": start_sit_suggestions,
            "strategy_notes": strategy_notes,
            "lineup_summary": {
                "starter_count": len(starters),
                "bench_count": len(bench),
            },
            "posture": posture,
            "delta_median": delta_median,
            "objective_weights": {"floor": w15, "median": w50, "ceiling": w85},
            "player_projections": [
                {
                    "player_id": p["player_id"],
                    "name": p["name"],
                    "position": p["position"],
                    "team": p["team"],
                    "side": p["side"],
                    "status": p["status"],
                    "floor": p["floor"],
                    "median": p["median"],
                    "ceiling": p["ceiling"],
                    "multiplier": p["multiplier"],
                    "notes": p["notes"],
                }
                for p in projections
            ],
            "optimizer": {
                "method": opt_diag.get("method"),
                "fallback_reason": opt_diag.get("fallback_reason"),
                "slots": opt_diag.get("slots", {}),
                "objective_value": round(optimal_objective, 1),
                "optimal_lineup": [p["name"] for p in optimal_lineup],
                "median_lineup": [p["name"] for p in user_median_lineup],
                "user_median_total": round(user_median_total, 1),
                "opponent_median_total": round(opp_median_total, 1),
                "excluded": [
                    {"name": p["name"], "side": p["side"], "reason": p["exclude_reason"]}
                    for p in projections
                    if p["excluded"]
                ],
                "shortfalls": opt_diag.get("shortfalls", []),
            },
        }

    def _target_games(self, week: Optional[int]) -> Dict[str, pd.Series]:
        games: Dict[str, pd.Series] = {}
        if self.schedules_df is None or self.schedules_df.empty or week is None:
            return games
        df = self.schedules_df
        if "week" in df.columns:
            filtered = df[df["week"] == week]
            if filtered.empty:
                return games
            df = filtered
        if "season" in df.columns and df["season"].nunique() > 1:
            df = df[df["season"] == df["season"].max()]
        for _, row in df.iterrows():
            for team_col in ("home_team", "away_team"):
                team = row.get(team_col)
                if team is not None and pd.notna(team):
                    games.setdefault(str(team).upper(), row)
        return games

    def _defense_ranks(self, season: Optional[int], week: Optional[int]) -> Dict[str, Dict[str, Tuple[int, float]]]:
        rates = self._defense_rates(season, week)
        ranks: Dict[str, Dict[str, Tuple[int, float]]] = {"pass": {}, "rush": {}}
        if rates is None or rates.empty or "defteam" not in rates.columns:
            return ranks
        for kind, col in (("pass", "def_pass_epa_rate"), ("rush", "def_rush_epa_rate")):
            if col not in rates.columns:
                continue
            ranked = rates.dropna(subset=[col]).sort_values(col, ascending=False)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                team = str(row.get("defteam")).upper()
                ranks[kind][team] = (rank, float(row[col]))
        return ranks

    def _defense_rates(self, season: Optional[int], week: Optional[int]) -> pd.DataFrame:
        if self.def_epa_df is None or self.def_epa_df.empty or "defteam" not in self.def_epa_df.columns:
            return pd.DataFrame()
        df = self.def_epa_df
        if "def_pass_epa_rate" in df.columns and "week" not in df.columns:
            return df
        if {"season", "week"}.issubset(df.columns):
            return compute_defensive_epa_rates(df, season=season, week=week)
        return pd.DataFrame()

    def _project_player(
        self,
        player: Dict[str, Any],
        games: Dict[str, pd.Series],
        def_ranks: Dict[str, Dict[str, Tuple[int, float]]],
        wind_cache: Dict[Tuple, Optional[float]],
    ) -> Dict[str, Any]:
        metrics = self._lookup_metrics(player)
        position = str(player.get("position") or "").upper()
        team = player.get("team") or (metrics.get("recent_team") if metrics is not None else None)
        team = str(team).upper() if team else None

        p15, p50, p85 = self._base_percentiles(metrics)
        notes: List[str] = []
        catalysts: List[Tuple[str, str]] = []
        multiplier = 1.0
        excluded = False
        exclude_reason: Optional[str] = None

        game = games.get(team) if team else None
        opponent_def: Optional[str] = None
        if game is None:
            if games and team and team != "FA":
                excluded = True
                exclude_reason = "bye"
                notes.append("on bye this week; excluded from lineup pool")
                catalysts.append(("penalty", "starter is on bye"))
        else:
            home = str(game.get("home_team") or "").upper()
            away = str(game.get("away_team") or "").upper()
            team_itt = self._to_float(game.get("home_itt") if team == home else game.get("away_itt"))
            if team_itt is not None:
                itt_mult = (team_itt / LEAGUE_AVG_TEAM_TOTAL) ** ITT_EXPONENT
                multiplier *= itt_mult
                if itt_mult > 1.01:
                    catalysts.append(("boost", f"high implied team total ({team_itt:.1f})"))
                elif itt_mult < 0.99:
                    catalysts.append(("penalty", f"low implied team total ({team_itt:.1f})"))
                notes.append(f"vegas ITT {team_itt:.1f} (x{itt_mult:.2f})")
            opponent_def = away if team == home else home

            roof = str(game.get("roof") or "").lower()
            if roof in OUTDOOR_ROOFS:
                wind = self._game_wind(game, wind_cache)
                if wind is not None and wind >= WIND_PENALTY_THRESHOLD_MPH:
                    if position == "QB":
                        multiplier *= WIND_PASS_MULTIPLIER
                        catalysts.append(("penalty", f"high wind ({wind:.0f} mph) downgrades passing game"))
                        notes.append(f"high wind ({wind:.0f} mph): passing downgrade (x{WIND_PASS_MULTIPLIER})")
                    elif position == "WR":
                        adot = self._adot(metrics)
                        if adot is not None and adot > DEEP_THREAT_ADOT:
                            multiplier *= WIND_PASS_MULTIPLIER
                            catalysts.append(("penalty", f"high wind ({wind:.0f} mph) downgrades deep-threat WR (aDOT {adot:.1f})"))
                            notes.append(f"high wind ({wind:.0f} mph): deep-threat downgrade (x{WIND_PASS_MULTIPLIER})")
                    elif position == "RB":
                        multiplier *= WIND_RB_MULTIPLIER
                        catalysts.append(("boost", f"wind game ({wind:.0f} mph) favors rushing volume"))
                        notes.append(f"high wind ({wind:.0f} mph): rushing volume boost (x{WIND_RB_MULTIPLIER})")

        if opponent_def and def_ranks:
            pass_rank = def_ranks["pass"].get(opponent_def)
            if pass_rank and pass_rank[0] <= DEF_FUNNEL_TOP_N and position in ("QB", "WR", "TE"):
                multiplier *= DEF_FUNNEL_MULTIPLIER
                catalysts.append(("boost", f"opp defense top-{DEF_FUNNEL_TOP_N} pass funnel ({opponent_def})"))
                notes.append(f"{opponent_def} defense top-{DEF_FUNNEL_TOP_N} in pass EPA/play allowed (x{DEF_FUNNEL_MULTIPLIER})")
            rush_rank = def_ranks["rush"].get(opponent_def)
            if rush_rank and rush_rank[0] <= DEF_FUNNEL_TOP_N and position == "RB":
                multiplier *= DEF_FUNNEL_MULTIPLIER
                catalysts.append(("boost", f"opp defense top-{DEF_FUNNEL_TOP_N} rush funnel ({opponent_def})"))
                notes.append(f"{opponent_def} defense top-{DEF_FUNNEL_TOP_N} in rush EPA/play allowed (x{DEF_FUNNEL_MULTIPLIER})")

        injury = player.get("injury_status")
        injury_mult = 1.0
        if injury in OUT_STATUSES:
            excluded = True
            exclude_reason = injury
            catalysts.append(("penalty", f"starter is {injury}"))
            notes.append(f"injury status {injury}; excluded from lineup pool")
        elif injury == "Questionable":
            injury_mult = QUESTIONABLE_PENALTY
            notes.append(f"questionable tag: x{QUESTIONABLE_PENALTY} on median/ceiling")

        floor = p15 * multiplier
        median = p50 * multiplier * injury_mult
        ceiling = p85 * multiplier * injury_mult

        if excluded:
            status = f"OUT ({exclude_reason})" if exclude_reason == "bye" else injury
        elif injury == "Questionable":
            status = injury
        else:
            status = None

        return {
            "key": _player_key(player),
            "player_id": player.get("player_id"),
            "name": player.get("name") or f"Player {player.get('player_id')}",
            "position": position,
            "team": team,
            "side": None,
            "status": status,
            "floor": round(floor, 2),
            "median": round(median, 2),
            "ceiling": round(ceiling, 2),
            "multiplier": round(multiplier * injury_mult, 3),
            "notes": notes,
            "catalysts": catalysts,
            "excluded": excluded,
            "exclude_reason": exclude_reason,
            "injury_status": injury,
        }

    def _base_percentiles(self, metrics: Optional[pd.Series]) -> Tuple[float, float, float]:
        if metrics is None:
            return 0.0, 0.0, 0.0
        p15 = self._to_float(metrics.get("proj_floor_p15"))
        p50 = self._to_float(metrics.get("proj_median_p50"))
        p85 = self._to_float(metrics.get("proj_ceiling_p85"))
        if p15 is not None and p50 is not None and p85 is not None:
            return max(p15, 0.0), max(p50, 0.0), max(p85, 0.0)
        xfp = self._to_float(metrics.get("rolling_xfp_3w"))
        if xfp is None:
            xfp = self._to_float(metrics.get("expected_fp_ppr"))
        if xfp is None or xfp <= 0:
            return 0.0, 0.0, 0.0
        return 0.75 * xfp, xfp, 1.25 * xfp

    def _adot(self, metrics: Optional[pd.Series]) -> Optional[float]:
        if metrics is None:
            return None
        for col in metrics.index:
            if str(col).lower() == "adot":
                value = self._to_float(metrics.get(col))
                if value is not None and value > 0:
                    return value
        air_yards = self._to_float(metrics.get("receiving_air_yards"))
        targets = self._to_float(metrics.get("targets"))
        if air_yards is not None and targets is not None and targets > 0:
            return air_yards / targets
        return None

    def _game_wind(self, game: pd.Series, wind_cache: Dict[Tuple, Optional[float]]) -> Optional[float]:
        key = (
            self._to_float(game.get("season")),
            self._to_float(game.get("week")),
            str(game.get("home_team") or "").upper(),
            str(game.get("away_team") or "").upper(),
        )
        if key in wind_cache:
            return wind_cache[key]
        wind = self._to_float(game.get("wind"))
        if wind is None:
            game_dt = game.get("game_datetime")
            if game_dt is not None and pd.notna(game_dt):
                if isinstance(game_dt, pd.Timestamp):
                    game_dt = game_dt.to_pydatetime()
                try:
                    wind = self._to_float(self.weather_fn(str(game.get("home_team") or "").upper(), game_dt))
                except Exception as e:
                    logger.warning(f"Weather lookup failed for {key}: {e}")
                    wind = None
        wind_cache[key] = wind
        return wind

    def _resolve_slots(self, matchup_overview: Dict[str, Any]) -> Optional[List[Tuple[str, int]]]:
        positions = self.roster_positions or matchup_overview.get("roster_positions")
        if not positions:
            return None
        counts = Counter(str(slot).upper() for slot in positions if isinstance(slot, str))
        slot_counts = [
            (slot, int(count))
            for slot, count in counts.items()
            if slot in SLOT_ELIGIBILITY and count > 0
        ]
        if not slot_counts:
            return None
        slot_counts.sort(key=lambda item: (SLOT_PRIORITY.get(item[0], len(SLOT_PRIORITY)), item[0]))
        return slot_counts

    def _optimize_lineup(
        self, pool: List[Dict[str, Any]], slots: Optional[List[Tuple[str, int]]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "method": "greedy",
            "fallback_reason": None,
            "shortfalls": [],
            "slots": {},
        }
        if not pool:
            diagnostics["fallback_reason"] = "no eligible players in the lineup pool"
            return [], diagnostics
        if slots is None:
            slot_counts = _default_slot_counts()
            diagnostics["fallback_reason"] = (
                "league roster positions unknown; greedy fill against default slots (1 QB / 2 RB / 2 WR / 1 TE / 1 FLEX)"
            )
        else:
            slot_counts = slots
            selected, shortfalls = self._ilp_lineup(pool, slot_counts)
            if selected is not None:
                diagnostics["method"] = "ilp"
                diagnostics["shortfalls"] = shortfalls
                diagnostics["slots"] = dict(slot_counts)
                return selected, diagnostics
            diagnostics["fallback_reason"] = "ILP solver failed or infeasible; greedy fill"
        diagnostics["slots"] = dict(slot_counts)
        selected, shortfalls = self._greedy_lineup(pool, slot_counts)
        diagnostics["shortfalls"] = shortfalls
        return selected, diagnostics

    def _ilp_lineup(
        self, pool: List[Dict[str, Any]], slot_counts: List[Tuple[str, int]]
    ) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
        var_index: List[Tuple[int, str]] = []
        for i, p in enumerate(pool):
            for slot, _ in slot_counts:
                if p["position"] in SLOT_ELIGIBILITY.get(slot, ()):
                    var_index.append((i, slot))
        if not var_index:
            return None, []
        n = len(var_index)
        c = np.zeros(n)
        player_vars: Dict[int, List[int]] = {}
        slot_vars: Dict[str, List[int]] = {slot: [] for slot, _ in slot_counts}
        for v, (i, slot) in enumerate(var_index):
            c[v] = -float(pool[i]["y"])
            player_vars.setdefault(i, []).append(v)
            slot_vars[slot].append(v)

        rows: List[np.ndarray] = []
        lb: List[float] = []
        ub: List[float] = []
        for _, vars_ in player_vars.items():
            row = np.zeros(n)
            row[vars_] = 1.0
            rows.append(row)
            lb.append(-np.inf)
            ub.append(1.0)

        shortfalls: List[str] = []
        for slot, count in slot_counts:
            eligible_vars = slot_vars[slot]
            effective = min(count, len(eligible_vars))
            if effective < count:
                shortfalls.append(f"{slot}: only {len(eligible_vars)} eligible player(s) for {count} required")
                if effective == 0:
                    continue
            row = np.zeros(n)
            row[eligible_vars] = 1.0
            rows.append(row)
            lb.append(float(effective))
            ub.append(float(effective))

        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(np.array(rows), lb, ub),
                integrality=np.ones(n),
                bounds=Bounds(0, 1),
            )
        except Exception as e:
            logger.warning(f"ILP lineup optimization failed: {e}")
            return None, shortfalls
        if res is None or res.status != 0 or res.x is None:
            return None, shortfalls
        selected = [pool[i] for v, (i, _) in enumerate(var_index) if res.x[v] > 0.5]
        return selected, shortfalls

    def _greedy_lineup(
        self, pool: List[Dict[str, Any]], slot_counts: List[Tuple[str, int]]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        remaining = list(pool)
        selected: List[Dict[str, Any]] = []
        shortfalls: List[str] = []
        for slot, count in slot_counts:
            eligible = sorted(
                (p for p in remaining if p["position"] in SLOT_ELIGIBILITY.get(slot, ())),
                key=lambda p: p["y"],
                reverse=True,
            )
            picked = eligible[:count]
            for p in picked:
                selected.append(p)
                remaining.remove(p)
            if len(picked) < count:
                shortfalls.append(f"{slot}: only {len(picked)} eligible player(s) for {count} required")
        return selected, shortfalls

    def solve_lineup_points(
        self, players: List[Dict[str, Any]]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Optimal starting-lineup projection sum and the selected players.

        Reusable entry point for non-matchup consumers (e.g. bilateral trade
        evaluation). `players` are dicts with "position" and "value" (the
        weekly projection to maximize) plus optional "player_id", "name", and
        "injury_status" (Out/IR/Doubtful players are excluded from the pool).
        Optimizes against self.roster_positions when known, else the default
        slots, via the ILP lineup optimizer.
        """
        pool: List[Dict[str, Any]] = []
        for p in players:
            status = str(p.get("injury_status") or "")
            if status in OUT_STATUSES:
                continue
            position = str(p.get("position") or "").upper()
            if not position:
                continue
            value = self._to_float(p.get("value")) or 0.0
            pool.append({
                "player_id": p.get("player_id"),
                "name": str(p.get("name") or p.get("player_id") or "?"),
                "position": position,
                "y": max(0.0, value),
            })
        lineup, _ = self._optimize_lineup(pool, self._resolve_slots({}))
        return round(sum(p["y"] for p in lineup), 2), lineup

    def _swap_rationale(self, inc: Dict[str, Any], out: Dict[str, Any], posture: str) -> str:
        parts = [text for direction, text in inc["catalysts"] if direction == "boost"]
        parts.extend(text for direction, text in out["catalysts"] if direction == "penalty")
        if posture == "FAVORITE":
            parts.append("floor protection as favorite")
        elif posture == "UNDERDOG":
            parts.append("underdog posture: chasing ceiling")
        parts.append(f"adjusted projection {inc['y']:.1f} vs {out['y']:.1f} pts")
        return "; ".join(parts)


def _default_slot_counts() -> List[Tuple[str, int]]:
    counts = Counter(DEFAULT_ROSTER_POSITIONS)
    slot_counts = [(slot, int(count)) for slot, count in counts.items()]
    slot_counts.sort(key=lambda item: (SLOT_PRIORITY.get(item[0], len(SLOT_PRIORITY)), item[0]))
    return slot_counts
