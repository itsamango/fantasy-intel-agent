"""Unit tests for the bilateral trade engine and rolling buy-low/sell-high signals."""

import math

import pandas as pd
import pytest

from fantasy_intel.analytics.matchups import MatchupAdvisor
from fantasy_intel.analytics.trades import (
    TradeAnalyzer,
    compute_replacement_levels,
    estimate_acceptance_probability,
    generate_league_trade_proposals,
    package_value,
)

SEASON = 2026
WEEK = 6

LEAGUE_CONTEXT = {
    "league_id": "L1",
    "name": "Test League",
    "total_rosters": 4,
    "roster_position_counts": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
    "scoring_type": "ppr",
}


def metrics_row(pid, name, position, xfp, recent, team="KC", fpoe=None):
    return {
        "player_id": str(pid),
        "player_display_name": name,
        "position": position,
        "recent_team": team,
        "season": SEASON,
        "week": WEEK,
        "rolling_xfp_3w": float(xfp),
        "rolling_fpoe_3w": float(fpoe if fpoe is not None else round(recent - xfp, 2)),
        "recent_ppr": float(recent),
        "expected_fp_ppr": float(xfp),
        "fantasy_points_ppr": float(recent),
        "offense_pct": 0.80,
        "targets": 6,
        "carries": 6,
    }


def build_synthetic_league(user_wr2_recent=16.0):
    rows = [
        metrics_row("u_qb", "User QB", "QB", 18.0, 18.0),
        metrics_row("u_rb1", "User RB1", "RB", 20.0, 20.0),
        metrics_row("u_rb2", "User RB2", "RB", 8.0, 8.0),
        metrics_row("u_rb3", "User RB3", "RB", 5.0, 5.0),
        metrics_row("u_wr1", "User WR1", "WR", 14.0, 14.0),
        metrics_row("u_wr2", "User WR2", "WR", 11.5, user_wr2_recent),
        metrics_row("u_wr3", "User WR3", "WR", 8.5, 8.5),
        metrics_row("u_te1", "User TE1", "TE", 6.0, 6.0),
        metrics_row("u_te2", "User TE2", "TE", 4.5, 4.5),
        metrics_row("r_qb", "Rival QB", "QB", 12.0, 12.0),
        metrics_row("r_rb1", "Rival RB1", "RB", 17.0, 17.0),
        metrics_row("r_rb2", "Rival RB2", "RB", 15.0, 11.5),
        metrics_row("r_rb3", "Rival RB3", "RB", 9.0, 9.0),
        metrics_row("r_rb4", "Rival RB4", "RB", 8.0, 8.0),
        metrics_row("r_wr1", "Rival WR1", "WR", 15.0, 15.0),
        metrics_row("r_wr2", "Rival WR2", "WR", 2.5, 2.5),
        metrics_row("r_te1", "Rival TE1", "TE", 5.0, 5.0),
        metrics_row("t3_qb", "T3 QB", "QB", 13.0, 13.0),
        metrics_row("t3_rb1", "T3 RB1", "RB", 12.0, 12.0),
        metrics_row("t3_rb2", "T3 RB2", "RB", 6.0, 6.0),
        metrics_row("t3_wr1", "T3 WR1", "WR", 12.0, 12.0),
        metrics_row("t3_wr2", "T3 WR2", "WR", 10.0, 10.0),
        metrics_row("t3_te1", "T3 TE1", "TE", 4.5, 4.5),
        metrics_row("t4_qb", "T4 QB", "QB", 12.5, 12.5),
        metrics_row("t4_rb1", "T4 RB1", "RB", 11.0, 11.0),
        metrics_row("t4_rb2", "T4 RB2", "RB", 4.0, 4.0),
        metrics_row("t4_wr1", "T4 WR1", "WR", 10.5, 10.5),
        metrics_row("t4_wr2", "T4 WR2", "WR", 9.5, 9.5),
        metrics_row("t4_te1", "T4 TE1", "TE", 4.0, 4.0),
        metrics_row("fa_qb1", "FA QB1", "QB", 15.0, 15.0),
        metrics_row("fa_qb2", "FA QB2", "QB", 14.0, 14.0),
        metrics_row("fa_qb3", "FA QB3", "QB", 11.0, 11.0),
        metrics_row("fa_qb4", "FA QB4", "QB", 10.0, 10.0),
        metrics_row("fa_rb1", "FA RB1", "RB", 13.0, 13.0),
        metrics_row("fa_rb2", "FA RB2", "RB", 7.0, 7.0),
        metrics_row("fa_wr1", "FA WR1", "WR", 13.0, 13.0),
        metrics_row("fa_wr2", "FA WR2", "WR", 11.0, 11.0),
        metrics_row("fa_wr3", "FA WR3", "WR", 9.0, 9.0),
        metrics_row("fa_te1", "FA TE1", "TE", 5.5, 5.5),
        metrics_row("fa_te2", "FA TE2", "TE", 3.0, 3.0),
    ]
    rosters = [
        ["u_qb", "u_rb1", "u_rb2", "u_rb3", "u_wr1", "u_wr2", "u_wr3", "u_te1", "u_te2"],
        ["r_qb", "r_rb1", "r_rb2", "r_rb3", "r_rb4", "r_wr1", "r_wr2", "r_te1"],
        ["t3_qb", "t3_rb1", "t3_rb2", "t3_wr1", "t3_wr2", "t3_te1"],
        ["t4_qb", "t4_rb1", "t4_rb2", "t4_wr1", "t4_wr2", "t4_te1"],
    ]
    return pd.DataFrame(rows), rosters


def run_engine(frame, rosters, **kwargs):
    return generate_league_trade_proposals(
        metrics_df=frame,
        league_context=LEAGUE_CONTEXT,
        rosters=rosters,
        user_roster_index=0,
        roster_ids=[1, 2, 3, 4],
        manager_names=["My Team", "Rival Team", "Bulwark", "Crew"],
        **kwargs,
    )


def test_replacement_rank_math():
    # 12-team, 1 QB / 2 RB / 2 WR / 1 TE / 1 FLEX league:
    # flex per team = 0.45 RB / 0.45 WR / 0.10 TE
    context = {
        "total_rosters": 12,
        "roster_position_counts": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
    }
    rows = [metrics_row(f"rb{i}", f"RB{i}", "RB", 36 - i, 36 - i) for i in range(35)]
    pool = pd.DataFrame(rows)

    repl = compute_replacement_levels(pool, context)

    assert repl["k_pos"] == {"QB": 13, "RB": 30, "WR": 30, "TE": 14}
    assert repl["flex_starters_per_team"] == pytest.approx({"QB": 0.0, "RB": 0.45, "WR": 0.45, "TE": 0.10})
    # replacement = the K-th ranked player: RB pool ranked 36..2, so RB30 = 7.0
    assert repl["replacement_xfp"]["RB"] == pytest.approx(7.0)
    assert repl["replacement_market"]["RB"] == pytest.approx(7.0)


def test_superflex_inflates_qb_values():
    one_qb = {
        "total_rosters": 12,
        "roster_position_counts": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
    }
    superflex = {
        "total_rosters": 12,
        "roster_position_counts": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "SUPER_FLEX": 1},
    }
    pool = pd.DataFrame([metrics_row(f"qb{i}", f"QB{i}", "QB", 26 - i, 26 - i) for i in range(25)])

    repl_one_qb = compute_replacement_levels(pool, one_qb)
    repl_superflex = compute_replacement_levels(pool, superflex)

    # K_QB: 12*(1+0)+1 = 13 vs 12*(1+0.6)+1 = 20 (60% of superflex slots count toward QB)
    assert repl_one_qb["k_pos"]["QB"] == 13
    assert repl_superflex["k_pos"]["QB"] == 20
    # QB pool ranked 26..2: QB13 = 14.0 and QB20 = 7.0
    assert repl_one_qb["replacement_xfp"]["QB"] == pytest.approx(14.0)
    assert repl_superflex["replacement_xfp"]["QB"] == pytest.approx(7.0)
    # deeper replacement baseline inflates the QB1 model VORP from 12 to 19
    qb1 = 26.0
    assert (qb1 - repl_superflex["replacement_xfp"]["QB"]) > (qb1 - repl_one_qb["replacement_xfp"]["QB"])
    assert (qb1 - repl_one_qb["replacement_xfp"]["QB"]) == pytest.approx(12.0)
    assert (qb1 - repl_superflex["replacement_xfp"]["QB"]) == pytest.approx(19.0)


def test_package_value_decay():
    # second asset retains 72%, third retains 0.72^2 = 51.84%
    assert package_value([20.0, 10.0]) == pytest.approx(20.0 + 0.72 * 10.0)
    assert package_value([10.0, 20.0]) == pytest.approx(27.2)
    assert package_value([20.0, 10.0, 10.0]) == pytest.approx(20.0 + 7.2 + 0.72**2 * 10.0)
    # roster-cut cost subtracted once off the decayed package
    assert package_value([20.0, 10.0], v_drop=3.0) == pytest.approx(24.2)
    # negative inputs are floored at zero
    assert package_value([-5.0, 10.0]) == pytest.approx(10.0)


def test_acceptance_probability_need_and_ordering():
    base = estimate_acceptance_probability(2.0, 3.0, need_filled=False)
    with_need = estimate_acceptance_probability(2.0, 3.0, need_filled=True)
    assert with_need > base

    assert estimate_acceptance_probability(5.0, 3.0, need_filled=False) > base
    assert estimate_acceptance_probability(0.0, 3.0, need_filled=False) < base
    assert estimate_acceptance_probability(-5.0, 3.0, need_filled=False) < estimate_acceptance_probability(0.0, 3.0, need_filled=False)
    assert estimate_acceptance_probability(2.0, 6.0, need_filled=False) > base
    assert estimate_acceptance_probability(2.0, 1.0, need_filled=False) < base

    # exact logistic value: logistic(-1.0) with all inputs neutral
    assert estimate_acceptance_probability(0.0, 0.0, need_filled=False) == pytest.approx(1.0 / (1.0 + math.exp(1.0)))
    # inputs are clipped so extreme deltas stay bounded in (0, 1)
    assert estimate_acceptance_probability(-1000.0, 0.0, need_filled=False) == pytest.approx(
        estimate_acceptance_probability(-15.0, 0.0, need_filled=False)
    )
    for market_delta in (-100.0, 0.0, 100.0):
        p = estimate_acceptance_probability(market_delta, 5.0, need_filled=True)
        assert 0.0 < p < 1.0


def test_solve_lineup_points_excludes_injured():
    advisor = MatchupAdvisor(roster_positions=["QB", "RB", "FLEX"])
    players = [
        {"player_id": "a", "name": "A", "position": "QB", "value": 20.0},
        {"player_id": "b", "name": "B", "position": "RB", "value": 15.0},
        {"player_id": "c", "name": "C", "position": "RB", "value": 10.0},
        {"player_id": "d", "name": "D", "position": "RB", "value": 30.0, "injury_status": "Out"},
        {"player_id": "e", "name": "E", "position": "RB", "value": 25.0, "injury_status": "IR"},
    ]
    points, lineup = advisor.solve_lineup_points(players)
    names = {p["name"] for p in lineup}
    assert points == pytest.approx(45.0)
    assert names == {"A", "B", "C"}


def test_end_to_end_bilateral_trade_proposals():
    frame, rosters = build_synthetic_league()
    result = run_engine(frame, rosters)

    proposals = result["proposals"]
    assert len(proposals) >= 2

    # every card is Pareto-valid and coherent
    for card in proposals:
        assert card["your_lineup_gain"] >= 1.2
        assert card["their_lineup_gain"] > 0
        assert card["market_delta"] >= -1.5
        assert 0.0 < card["p_accept"] < 1.0
        assert card["send"] and card["receive"]
        assert card["target_manager"]
        assert card["pitch"]

    # ranked by acceptance probability, then user lineup gain
    ranked = sorted(proposals, key=lambda p: (-p["p_accept"], -p["your_lineup_gain"]))
    assert [p["p_accept"] for p in proposals] == [p["p_accept"] for p in ranked]

    by_send = {tuple(p["player_id"] for p in card["send"]): card for card in proposals}
    by_receive = {tuple(p["player_id"] for p in card["receive"]): card for card in proposals}

    # the flagship 1-for-1: sell-high WR surplus for the rival's buy-low RB2
    key = by_send.get(("u_wr2",))
    assert key is not None
    assert [p["player_id"] for p in key["receive"]] == ["r_rb2"]
    assert key["target_manager"] == "Rival Team"
    assert key["your_lineup_gain"] == pytest.approx(3.5, abs=0.05)
    assert key["their_lineup_gain"] == pytest.approx(2.0, abs=0.05)
    assert key["market_delta"] == pytest.approx(-0.4, abs=0.05)
    assert key["send"][0]["tag"] == "sell-high"
    assert key["receive"][0]["tag"] == "buy-low"
    assert "fills their WR hole" in key["pitch"]
    assert key["p_accept"] == pytest.approx(
        estimate_acceptance_probability(key["market_delta"], key["their_lineup_gain"], key["need_filled"]),
        abs=0.01,
    )

    # the 2-for-1 variant sends the sell-high WR plus the spare TE and ranks first
    two_for_one = by_send.get(("u_wr2", "u_te1"))
    assert two_for_one is not None
    assert two_for_one["your_lineup_gain"] == pytest.approx(2.0, abs=0.05)
    assert two_for_one["their_lineup_gain"] == pytest.approx(3.0, abs=0.05)
    assert two_for_one["market_delta"] == pytest.approx(0.68, abs=0.05)
    assert two_for_one["p_accept"] > key["p_accept"]
    assert proposals[0]["p_accept"] >= proposals[-1]["p_accept"]

    # diagnostics: K_pos and replacement baselines for the 4-team league
    vorp = result["vorp"]
    assert vorp["k_pos"] == {"QB": 5, "RB": 11, "WR": 11, "TE": 5}
    assert vorp["replacement_xfp"] == pytest.approx({"QB": 12.5, "RB": 6.0, "WR": 8.5, "TE": 4.5})
    assert vorp["replacement_market"] == pytest.approx({"QB": 12.5, "RB": 6.0, "WR": 8.5, "TE": 4.5})
    assert vorp["pool_source"] == "full_metrics_frame"
    assert vorp["pool_size"] == len(frame)


def test_market_delta_guard_blocks_undervalued_sends():
    # identical league, but the sell-high WR's recent PPG collapses to 9:
    # his market VORP (1.25) now badly undervalues the rival's RB2 (6.55),
    # so the surface-unfair swap is rejected by the -1.5 market-delta guard
    frame, rosters = build_synthetic_league(user_wr2_recent=9.0)
    result = run_engine(frame, rosters)

    proposals = result["proposals"]
    assert all("u_wr2" not in [p["player_id"] for p in card["send"]] for card in proposals)
    assert proposals == []


def test_one_sided_trades_rejected():
    # two-team league of lopsided swaps (stud-for-scrub both ways): every
    # candidate fails the market guard or one side's lineup gain, so nothing
    # is proposed even though studs are mutually available
    rows = [
        metrics_row("u_qb", "User QB", "QB", 18.0, 18.0),
        metrics_row("u_rb1", "User RB1", "RB", 20.0, 20.0),
        metrics_row("u_rb2", "User RB2", "RB", 8.0, 8.0),
        metrics_row("u_wr1", "User WR1", "WR", 14.0, 14.0),
        metrics_row("u_wr2", "User WR2", "WR", 11.5, 11.5),
        metrics_row("u_te1", "User TE1", "TE", 6.0, 6.0),
        metrics_row("o_qb", "Opp QB", "QB", 12.0, 12.0),
        metrics_row("o_rb1", "Opp RB1", "RB", 15.0, 15.0),
        metrics_row("o_rb2", "Opp RB2", "RB", 7.0, 7.0),
        metrics_row("o_wr1", "Opp WR1", "WR", 15.0, 15.0),
        metrics_row("o_wr2", "Opp WR2", "WR", 2.5, 2.5),
        metrics_row("o_te1", "Opp TE1", "TE", 5.0, 5.0),
    ]
    rosters = [
        ["u_qb", "u_rb1", "u_rb2", "u_wr1", "u_wr2", "u_te1"],
        ["o_qb", "o_rb1", "o_rb2", "o_wr1", "o_wr2", "o_te1"],
    ]
    result = run_engine(pd.DataFrame(rows), rosters)

    assert result["proposals"] == []


def _weekly_wr_row(pid, name, week, targets, air_yards, actual, snaps=0.75):
    return {
        "player_id": pid,
        "player_display_name": name,
        "position": "WR",
        "recent_team": "KC",
        "season": 2024,
        "week": week,
        "offense_pct": snaps,
        "targets": targets,
        "carries": 0,
        "receiving_air_yards": air_yards,
        "receiving_yards": air_yards * 0.6,
        "fantasy_points_ppr": actual,
    }


def test_buy_low_sell_high_rolling_thresholds():
    data = pd.DataFrame([
        # high-volume WR scoring far below xFP for three straight weeks
        _weekly_wr_row("00-1", "Buy Low WR", 3, 8, 90.0, 3.0),
        _weekly_wr_row("00-1", "Buy Low WR", 4, 8, 90.0, 4.0),
        _weekly_wr_row("00-1", "Buy Low WR", 5, 8, 90.0, 3.0),
        # efficient steady WR: no signal either way
        _weekly_wr_row("00-2", "Steady WR", 3, 8, 24.0, 12.0),
        _weekly_wr_row("00-2", "Steady WR", 4, 8, 24.0, 12.0),
        _weekly_wr_row("00-2", "Steady WR", 5, 8, 24.0, 12.0),
        # bloated scoring on modest volume: sustained positive FPOE
        _weekly_wr_row("00-3", "Sell High WR", 3, 4, 25.0, 16.0),
        _weekly_wr_row("00-3", "Sell High WR", 4, 4, 25.0, 18.0),
        _weekly_wr_row("00-3", "Sell High WR", 5, 4, 25.0, 16.0),
        # quiet low-volume WR: no signal
        _weekly_wr_row("00-4", "Quiet WR", 3, 3, 20.0, 6.0),
        _weekly_wr_row("00-4", "Quiet WR", 4, 3, 20.0, 6.0),
        _weekly_wr_row("00-4", "Quiet WR", 5, 3, 20.0, 6.0),
    ])

    analyzer = TradeAnalyzer(data)
    buy_low = analyzer.get_buy_low_candidates(week=5)
    sell_high = analyzer.get_sell_high_candidates(week=5)

    assert buy_low["player_display_name"].tolist() == ["Buy Low WR"]
    assert sell_high["player_display_name"].tolist() == ["Sell High WR"]

    latest = analyzer.df[analyzer.df["week"] == 5]
    buy_frame = latest[latest["player_display_name"] == "Buy Low WR"].iloc[0]
    sell_frame = latest[latest["player_display_name"] == "Sell High WR"].iloc[0]

    # buy-low: rolling xFP >= 12 paired with rolling FPOE <= -3
    assert buy_frame["rolling_xfp_3w"] == pytest.approx(17.6, abs=0.1)
    assert buy_frame["rolling_fpoe_3w"] <= -3.0
    assert buy_frame["fpoe_excess_3w"] <= -3.0
    assert "efficiency baseline" in buy_low.iloc[0]["rationale"]

    # sell-high: recent PPG >= 15 with rolling FPOE >= +4 on modest xFP
    assert sell_frame["recent_ppr"] >= 15.0
    assert sell_frame["rolling_fpoe_3w"] >= 4.0
    assert sell_frame["rolling_xfp_3w"] == pytest.approx(7.2, abs=0.1)
    assert sell_frame["fpoe_excess_3w"] >= 4.0
    assert "efficiency baseline" in sell_high.iloc[0]["rationale"]

    # backward-compatible output shape
    for col in ("player_display_name", "position", "recent_team", "offense_pct", "touches",
                "targets", "fantasy_points_ppr", "expected_fp_ppr", "fp_diff", "rationale"):
        assert col in buy_low.columns
        assert col in sell_high.columns
