"""Unit tests for matchup analysis and situational team advice."""

from datetime import datetime

import pandas as pd
import pytest

from fantasy_intel.analytics.matchups import MatchupAdvisor, compute_defensive_epa_rates

WEEK = 6
SEASON = 2026


def metrics_row(player_id, name, position, team, p15=None, p50=None, p85=None, air_yards=None, targets=None):
    return {
        "player_id": str(player_id),
        "player_display_name": name,
        "position": position,
        "recent_team": team,
        "season": SEASON,
        "week": WEEK,
        "proj_floor_p15": p15 if p15 is not None else round(0.75 * (p50 or 0.0), 2),
        "proj_median_p50": p50 or 0.0,
        "proj_ceiling_p85": p85 if p85 is not None else round(1.25 * (p50 or 0.0), 2),
        "rolling_xfp_3w": p50 or 0.0,
        "receiving_air_yards": air_yards,
        "targets": targets,
    }


def schedule_row(home, away, spread=0.0, total=45.0, roof="dome", wind=None, dt=None):
    return {
        "season": SEASON,
        "week": WEEK,
        "home_team": home,
        "away_team": away,
        "spread_line": spread,
        "total_line": total,
        "home_itt": (total + spread) / 2,
        "away_itt": (total - spread) / 2,
        "roof": roof,
        "wind": wind,
        "game_datetime": dt or datetime(2026, 10, 15, 13, 0),
    }


def player(pid, name, position, team, injury=None):
    return {
        "player_id": str(pid),
        "name": name,
        "position": position,
        "team": team,
        "injury_status": injury,
    }


def make_overview(my_starters, my_bench, opp_starters=(), opp_bench=(), roster_positions=None, week=WEEK):
    has_opp = bool(opp_starters or opp_bench)
    return {
        "week": week,
        "league_id": "test_league_123",
        "my_team": {
            "user_name": "My Team",
            "roster_id": 1,
            "starters": list(my_starters),
            "bench": list(my_bench),
        },
        "opponent_team": {
            "user_name": "Rival Team",
            "roster_id": 2,
            "starters": list(opp_starters),
            "bench": list(opp_bench),
        } if has_opp else None,
        "roster_positions": roster_positions,
    }


def projections_by_name(result):
    return {p["name"]: p for p in result["player_projections"]}


def test_matchup_lineup_evaluation():
    mock_overview = {
        "week": 6,
        "league_id": "test_league_123",
        "my_team": {
            "user_name": "My Team",
            "roster_id": 1,
            "starters": [
                {"name": "Christian McCaffrey", "position": "RB", "team": "SF", "injury_status": "Out"},
                {"name": "Justin Jefferson", "position": "WR", "team": "MIN", "injury_status": None},
            ],
            "bench": [
                {"name": "Jordan Mason", "position": "RB", "team": "SF", "injury_status": None},
                {"name": "Rashod Bateman", "position": "WR", "team": "BAL", "injury_status": None},
            ],
        },
        "opponent_team": {
            "user_name": "Rival Team",
            "roster_id": 2,
            "starters": [
                {"name": "Saquon Barkley", "position": "RB", "team": "PHI", "injury_status": "Out"},
                {"name": "CeeDee Lamb", "position": "WR", "team": "DAL", "injury_status": None},
            ],
            "bench": [],
        },
    }

    advisor = MatchupAdvisor()
    evaluation = advisor.evaluate_lineup(mock_overview)

    # 1. Should flag CMC as critical injury alert
    alerts = evaluation["injury_alerts"]
    assert len(alerts) == 1
    assert alerts[0]["player"] == "Christian McCaffrey"
    assert alerts[0]["severity"] == "CRITICAL"

    # 2. Should suggest Jordan Mason as replacement
    suggestions = evaluation["start_sit_suggestions"]
    assert len(suggestions) == 1
    assert suggestions[0]["sit"] == "Christian McCaffrey"
    assert suggestions[0]["start"] == "Jordan Mason"

    # 3. Should flag opponent injured starter
    notes = evaluation["strategy_notes"]
    assert any("Saquon Barkley" in n for n in notes)


def test_itt_multiplier_boosts_high_team_total_players():
    metrics = pd.DataFrame([
        metrics_row(1, "High ITT RB", "RB", "KC", p50=10.0),
        metrics_row(2, "Low ITT RB", "RB", "JAX", p50=10.0),
    ])
    schedules = pd.DataFrame([
        schedule_row("KC", "LV", spread=10.0, total=55.0),
        schedule_row("JAX", "TEN", spread=-10.0, total=35.0),
    ])
    advisor = MatchupAdvisor(metrics_df=metrics, schedules_df=schedules)
    result = advisor.evaluate_lineup(make_overview(
        [player(1, "High ITT RB", "RB", "KC"), player(2, "Low ITT RB", "RB", "JAX")],
        [],
    ))

    projections = projections_by_name(result)
    assert projections["High ITT RB"]["median"] == pytest.approx(10.0 * (32.5 / 22.5) ** 0.6, abs=0.01)
    assert projections["Low ITT RB"]["median"] == pytest.approx(10.0 * (12.5 / 22.5) ** 0.6, abs=0.01)
    assert projections["High ITT RB"]["median"] > 10.0 > projections["Low ITT RB"]["median"]


def test_bye_player_excluded_with_note():
    metrics = pd.DataFrame([metrics_row(1, "Bye Week RB", "RB", "SF", p50=15.0)])
    schedules = pd.DataFrame([schedule_row("KC", "LV")])
    advisor = MatchupAdvisor(metrics_df=metrics, schedules_df=schedules)
    result = advisor.evaluate_lineup(make_overview(
        [player(1, "Bye Week RB", "RB", "SF")],
        [],
        roster_positions=["RB"],
    ))

    projections = projections_by_name(result)
    assert projections["Bye Week RB"]["status"] == "OUT (bye)"
    assert any("bye" in note for note in projections["Bye Week RB"]["notes"])
    assert result["optimizer"]["excluded"] == [
        {"name": "Bye Week RB", "side": "user", "reason": "bye"}
    ]
    assert result["optimizer"]["optimal_lineup"] == []


def test_wind_penalty_applies_to_qb_and_deep_wr_and_boosts_rb():
    def fake_weather(team, dt):
        return 18.0

    metrics = pd.DataFrame([
        metrics_row(1, "Forecast QB", "QB", "BUF", p50=20.0),
        metrics_row(2, "Recorded QB", "QB", "CHI", p50=20.0),
        metrics_row(3, "Deep WR", "WR", "BUF", p50=12.0, air_yards=100.0, targets=8.0),
        metrics_row(4, "Short WR", "WR", "BUF", p50=12.0, air_yards=40.0, targets=8.0),
        metrics_row(5, "Windy RB", "RB", "BUF", p50=12.0),
        metrics_row(6, "Dome QB", "QB", "NO", p50=20.0),
    ])
    schedules = pd.DataFrame([
        schedule_row("BUF", "MIA", roof="outdoors", wind=None),
        schedule_row("CHI", "GB", roof="outdoors", wind=18.0),
        schedule_row("NO", "ATL", roof="dome"),
    ])
    advisor = MatchupAdvisor(metrics_df=metrics, schedules_df=schedules, weather_fn=fake_weather)
    result = advisor.evaluate_lineup(make_overview(
        [
            player(1, "Forecast QB", "QB", "BUF"),
            player(2, "Recorded QB", "QB", "CHI"),
            player(6, "Dome QB", "QB", "NO"),
        ],
        [
            player(3, "Deep WR", "WR", "BUF"),
            player(4, "Short WR", "WR", "BUF"),
            player(5, "Windy RB", "RB", "BUF"),
        ],
    ))

    projections = projections_by_name(result)
    assert projections["Forecast QB"]["median"] == pytest.approx(20.0 * 0.88)
    assert projections["Recorded QB"]["median"] == pytest.approx(20.0 * 0.88)
    assert projections["Deep WR"]["median"] == pytest.approx(12.0 * 0.88)
    assert projections["Short WR"]["median"] == pytest.approx(12.0)
    assert projections["Windy RB"]["median"] == pytest.approx(12.0 * 1.05)
    assert projections["Dome QB"]["median"] == pytest.approx(20.0)


def test_wind_skipped_for_wr_without_adot_and_for_domes_without_forecast():
    def no_weather(team, dt):
        raise AssertionError("weather lookup should not run when wind is known or roof is closed")

    metrics = pd.DataFrame([
        metrics_row(1, "No aDOT WR", "WR", "BUF", p50=12.0),
        metrics_row(2, "Dome QB", "QB", "NO", p50=20.0),
    ])
    schedules = pd.DataFrame([
        schedule_row("BUF", "MIA", roof="outdoors", wind=18.0),
        schedule_row("NO", "ATL", roof="dome"),
    ])
    advisor = MatchupAdvisor(metrics_df=metrics, schedules_df=schedules, weather_fn=no_weather)
    result = advisor.evaluate_lineup(make_overview(
        [player(1, "No aDOT WR", "WR", "BUF"), player(2, "Dome QB", "QB", "NO")],
        [],
    ))

    projections = projections_by_name(result)
    assert projections["No aDOT WR"]["median"] == pytest.approx(12.0)
    assert projections["Dome QB"]["median"] == pytest.approx(20.0)


def test_defensive_funnel_multiplier():
    filler = [("T01", 0.17), ("T02", 0.16), ("T03", 0.15), ("T04", 0.14), ("T05", 0.13), ("T06", 0.12), ("T07", 0.11), ("T08", 0.10)]
    entries = [("LV", 0.30, 0.00), ("SEA", 0.00, 0.28), ("NE", 0.05, 0.05)]
    entries += [(team, rate, 0.05) for team, rate in filler]

    defense_rows = []
    for team, pass_rate, rush_rate in entries:
        for wk in range(2, 6):
            defense_rows.append({
                "defteam": team,
                "season": SEASON,
                "week": wk,
                "def_pass_epa_allowed": pass_rate * 30.0,
                "def_pass_attempts": 30,
                "def_rush_epa_allowed": rush_rate * 20.0,
                "def_rush_attempts": 20,
            })
    defense_rows.append({
        "defteam": "LV",
        "season": SEASON,
        "week": 1,
        "def_pass_epa_allowed": -150.0,
        "def_pass_attempts": 30,
        "def_rush_epa_allowed": 0.0,
        "def_rush_attempts": 20,
    })
    defense = pd.DataFrame(defense_rows)

    rates = compute_defensive_epa_rates(defense, season=SEASON, week=WEEK)
    lv_rate = rates.loc[rates["defteam"] == "LV", "def_pass_epa_rate"].iloc[0]
    assert lv_rate == pytest.approx(0.30)

    metrics = pd.DataFrame([
        metrics_row(1, "Funnel WR", "WR", "KC", p50=12.0),
        metrics_row(2, "Control WR", "WR", "JAX", p50=12.0),
        metrics_row(3, "Funnel RB", "RB", "HOU", p50=12.0),
        metrics_row(4, "Control RB", "RB", "KC", p50=12.0),
    ])
    schedules = pd.DataFrame([
        schedule_row("LV", "KC"),
        schedule_row("NE", "JAX"),
        schedule_row("SEA", "HOU"),
    ])
    advisor = MatchupAdvisor(metrics_df=metrics, schedules_df=schedules, def_epa_df=defense)
    result = advisor.evaluate_lineup(make_overview(
        [player(1, "Funnel WR", "WR", "KC"), player(2, "Control WR", "WR", "JAX")],
        [player(3, "Funnel RB", "RB", "HOU"), player(4, "Control RB", "RB", "KC")],
    ))

    projections = projections_by_name(result)
    assert projections["Funnel WR"]["median"] == pytest.approx(12.0 * 1.08)
    assert projections["Control WR"]["median"] == pytest.approx(12.0)
    assert projections["Funnel RB"]["median"] == pytest.approx(12.0 * 1.08)
    assert projections["Control RB"]["median"] == pytest.approx(12.0)


def test_posture_thresholds():
    def build(opp_qb_median, opp_rb_median):
        metrics = pd.DataFrame([
            metrics_row(1, "My QB", "QB", "KC", p50=20.0),
            metrics_row(2, "My RB", "RB", "KC", p50=10.0),
            metrics_row(3, "Opp QB", "QB", "BUF", p50=opp_qb_median),
            metrics_row(4, "Opp RB", "RB", "BUF", p50=opp_rb_median),
        ])
        schedules = pd.DataFrame([schedule_row("KC", "LV"), schedule_row("BUF", "NE")])
        advisor = MatchupAdvisor(metrics_df=metrics, schedules_df=schedules)
        return advisor.evaluate_lineup(make_overview(
            [player(1, "My QB", "QB", "KC"), player(2, "My RB", "RB", "KC")],
            [],
            [player(3, "Opp QB", "QB", "BUF"), player(4, "Opp RB", "RB", "BUF")],
            roster_positions=["QB", "RB"],
        ))

    favored = build(15.0, 5.0)
    assert favored["delta_median"] == pytest.approx(10.0)
    assert favored["posture"] == "FAVORITE"
    assert favored["objective_weights"] == {"floor": 0.65, "median": 0.35, "ceiling": 0.0}

    neutral = build(15.1, 5.0)
    assert neutral["delta_median"] == pytest.approx(9.9)
    assert neutral["posture"] == "NEUTRAL"
    assert neutral["objective_weights"] == {"floor": 0.25, "median": 0.5, "ceiling": 0.25}

    underdog = build(25.0, 15.0)
    assert underdog["delta_median"] == pytest.approx(-10.0)
    assert underdog["posture"] == "UNDERDOG"
    assert underdog["objective_weights"] == {"floor": 0.0, "median": 0.35, "ceiling": 0.65}


def test_ilp_respects_slot_counts_and_never_starts_player_twice():
    metrics = pd.DataFrame([
        metrics_row(1, "Starter QB", "QB", "KC", p15=14.0, p50=20.0, p85=26.0),
        metrics_row(2, "Backup QB", "QB", "KC", p15=10.0, p50=14.0, p85=18.0),
        metrics_row(3, "Elite RB", "RB", "KC", p15=8.0, p50=12.0, p85=16.0),
        metrics_row(4, "Bench RB", "RB", "KC", p15=5.0, p50=8.0, p85=10.0),
        metrics_row(5, "Top WR", "WR", "KC", p15=6.0, p50=10.0, p85=12.0),
        metrics_row(6, "Bench TE", "TE", "KC", p15=1.0, p50=2.0, p85=3.0),
    ])
    schedules = pd.DataFrame([schedule_row("KC", "LV"), schedule_row("BUF", "NE")])
    advisor = MatchupAdvisor(metrics_df=metrics, schedules_df=schedules)
    result = advisor.evaluate_lineup(make_overview(
        [
            player(1, "Starter QB", "QB", "KC"),
            player(3, "Elite RB", "RB", "KC"),
            player(5, "Top WR", "WR", "KC"),
            player(7, "Kicker", "K", "KC"),
            player(8, "Defense", "DEF", "KC"),
        ],
        [player(2, "Backup QB", "QB", "KC"), player(4, "Bench RB", "RB", "KC"), player(6, "Bench TE", "TE", "KC")],
        [player(9, "Opp QB", "QB", "BUF", None)],
        roster_positions=["QB", "RB", "FLEX", "SUPER_FLEX", "K", "DEF"],
    ))

    optimizer = result["optimizer"]
    assert optimizer["method"] == "ilp"
    assert optimizer["fallback_reason"] is None
    lineup = optimizer["optimal_lineup"]
    assert len(lineup) == len(set(lineup))
    assert set(lineup) == {"Starter QB", "Backup QB", "Elite RB", "Top WR", "Kicker", "Defense"}
    assert optimizer["shortfalls"] == []


def test_injured_players_excluded_from_optimal_lineup():
    metrics = pd.DataFrame([
        metrics_row(1, "Star RB", "RB", "KC", p50=20.0),
        metrics_row(2, "Questionable RB", "RB", "KC", p50=12.0),
        metrics_row(3, "Backup RB", "RB", "KC", p50=8.0),
        metrics_row(4, "IR RB", "RB", "KC", p50=15.0),
    ])
    schedules = pd.DataFrame([schedule_row("KC", "LV")])
    advisor = MatchupAdvisor(metrics_df=metrics, schedules_df=schedules)
    result = advisor.evaluate_lineup(make_overview(
        [player(1, "Star RB", "RB", "KC", injury="Out")],
        [
            player(2, "Questionable RB", "RB", "KC", injury="Questionable"),
            player(3, "Backup RB", "RB", "KC"),
            player(4, "IR RB", "RB", "KC", injury="IR"),
        ],
        roster_positions=["RB", "RB"],
    ))

    optimizer = result["optimizer"]
    assert set(optimizer["optimal_lineup"]) == {"Questionable RB", "Backup RB"}
    excluded = {e["name"]: e["reason"] for e in optimizer["excluded"]}
    assert excluded["Star RB"] == "Out"
    assert excluded["IR RB"] == "IR"

    projections = projections_by_name(result)
    assert projections["Questionable RB"]["median"] == pytest.approx(12.0 * 0.9)
    assert projections["Questionable RB"]["floor"] == pytest.approx(0.75 * 12.0)

    suggestions = result["start_sit_suggestions"]
    assert len(suggestions) == 1
    assert suggestions[0]["sit"] == "Star RB"
    assert suggestions[0]["start"] == "Questionable RB"


def test_swap_suggestion_with_rationale_for_better_bench_player():
    metrics = pd.DataFrame([
        metrics_row(1, "Windy Deep WR", "WR", "BUF", p50=12.0, air_yards=120.0, targets=8.0),
        metrics_row(2, "Dome WR", "WR", "NO", p50=12.0),
        metrics_row(3, "Steady WR", "WR", "KC", p50=14.0),
    ])
    schedules = pd.DataFrame([
        schedule_row("BUF", "MIA", roof="outdoors", wind=18.0),
        schedule_row("NO", "ATL", roof="dome"),
        schedule_row("KC", "LV", roof="dome"),
    ])
    advisor = MatchupAdvisor(
        metrics_df=metrics,
        schedules_df=schedules,
        weather_fn=lambda team, dt: None,
    )
    result = advisor.evaluate_lineup(make_overview(
        [player(1, "Windy Deep WR", "WR", "BUF"), player(3, "Steady WR", "WR", "KC")],
        [player(2, "Dome WR", "WR", "NO")],
        roster_positions=["WR", "WR"],
    ))

    suggestions = result["start_sit_suggestions"]
    assert len(suggestions) == 1
    swap = suggestions[0]
    assert swap["sit"] == "Windy Deep WR"
    assert swap["start"] == "Dome WR"
    assert swap["projected_gain"] > 0
    assert "high wind" in swap["reason"]
    assert "deep-threat WR" in swap["reason"]
    assert "adjusted projection" in swap["reason"]


def test_greedy_fallback_without_roster_positions():
    metrics = pd.DataFrame([
        metrics_row(1, "QB", "QB", "KC", p50=18.0),
        metrics_row(2, "RB1", "RB", "KC", p50=14.0),
        metrics_row(3, "RB2", "RB", "KC", p50=10.0),
        metrics_row(4, "RB3", "RB", "KC", p50=6.0),
        metrics_row(5, "WR1", "WR", "KC", p50=12.0),
        metrics_row(6, "WR2", "WR", "KC", p50=9.0),
        metrics_row(7, "WR3", "WR", "KC", p50=7.0),
        metrics_row(8, "TE1", "TE", "KC", p50=5.0),
    ])
    schedules = pd.DataFrame([schedule_row("KC", "LV")])
    advisor = MatchupAdvisor(metrics_df=metrics, schedules_df=schedules)
    overview = make_overview(
        [player(1, "QB", "QB", "KC"), player(2, "RB1", "RB", "KC"), player(5, "WR1", "WR", "KC")],
        [player(3, "RB2", "RB", "KC"), player(4, "RB3", "RB", "KC"), player(6, "WR2", "WR", "KC"), player(7, "WR3", "WR", "KC"), player(8, "TE1", "TE", "KC")],
    )
    overview.pop("roster_positions")
    result = advisor.evaluate_lineup(overview)

    optimizer = result["optimizer"]
    assert optimizer["method"] == "greedy"
    assert optimizer["fallback_reason"]
    assert "roster positions" in optimizer["fallback_reason"]
    assert set(optimizer["optimal_lineup"]) == {"QB", "RB1", "RB2", "WR1", "WR2", "TE1", "WR3"}
