"""Tests for the FantasyCalc market layer."""

import pytest

from fantasy_intel.analytics.market import acceptance_tier, assess_trade, find_sweeteners
from fantasy_intel.analytics.ros_value import LeagueShape, PlayerOutlook, RosterState, RosValuator, find_trades
from fantasy_intel.config import MARKET_LIKELY_MIN
from fantasy_intel.data.fantasycalc import FantasyCalcClient, market_params_from_league


def _v(**vals):
    return {pid: {"value": float(v)} for pid, v in vals.items()}


def test_market_params_from_league():
    sf = {"roster_position_counts": {"QB": 1, "SUPER_FLEX": 1}, "scoring_type": "half_ppr", "total_rosters": 10}
    assert market_params_from_league(sf) == {"numQbs": 2, "numTeams": 10, "ppr": 0.5}
    std = {"roster_position_counts": {"QB": 1}, "scoring_type": "standard", "total_rosters": 12}
    assert market_params_from_league(std) == {"numQbs": 1, "numTeams": 12, "ppr": 0.0}


def test_parse_keys_by_sleeper_id_and_skips_unmapped():
    raw = [
        {"player": {"sleeperId": "9221", "name": "Gibbs", "position": "RB"}, "value": 10771, "overallRank": 1},
        {"player": {"name": "No Sleeper Id"}, "value": 50},
    ]
    parsed = FantasyCalcClient._parse(raw)
    assert list(parsed) == ["9221"] and parsed["9221"]["value"] == 10771.0


def test_assess_trade_gap_fairness_and_tiers():
    values = _v(a=5000, b=4800, c=3000, d=1000)
    fair = assess_trade(values, ["a"], ["b"], partner_lineup_ppw=0.2)
    assert fair["fair"] and fair["gap_pct"] == pytest.approx(-200 / 9800, abs=1e-4)
    assert fair["tier"] == "LIKELY"  # partner gains market value and lineup

    lopsided = assess_trade(values, ["d"], ["a"], partner_lineup_ppw=0.5)  # you get 5000 for 1000
    assert lopsided["gap_pct"] > 0.5 and lopsided["tier"] == "UNLIKELY"
    # Closing the gap: adding value_to_add to your side brings the partner to MARKET_LIKELY_MIN
    add = lopsided["value_to_add"]
    send, get = 1000 + add, 5000
    assert (send - get) / (send + get) == pytest.approx(MARKET_LIKELY_MIN, abs=1e-3)

    assert assess_trade(_v(x=10), ["x"], ["y"], 0.0)["tier"] == "UNKNOWN"


def test_acceptance_tier_lineup_offset():
    assert acceptance_tier(-0.08, 0.2) == "UNLIKELY"
    assert acceptance_tier(-0.08, 1.5) == "PLAUSIBLE"  # real lineup gain offsets a modest overpay
    assert acceptance_tier(-0.04, 0.0) == "PLAUSIBLE"
    assert acceptance_tier(0.01, -0.3) == "PLAUSIBLE"  # market win but lineup loss is not LIKELY


def test_find_sweeteners_keeps_only_options_that_still_help_you():
    values = _v(stud=6000, mine=3000, cheap=1500, pricey=3200, tiny=100)
    need = assess_trade(values, ["mine"], ["stud"], 0.5)["value_to_add"]  # ~2700

    def evaluate(send):
        # adding "pricey" kills your gain; "cheap" is below the needed value window
        you = 0.2 if "pricey" in send else 1.5
        return {"you": {"delta_per_week": you}, "partner": {"delta_per_week": 0.3}}

    opts = find_sweeteners(values, ["mine", "cheap", "pricey", "tiny"], ["mine"], ["stud"], need, evaluate)
    assert opts == []  # cheap < need, pricey hurts you, tiny too small

    def evaluate_ok(send):
        return {"you": {"delta_per_week": 1.2}, "partner": {"delta_per_week": 0.4}}

    opts = find_sweeteners(values, ["mine", "cheap", "pricey"], ["mine"], ["stud"], need, evaluate_ok, names={"pricey": "Pricey"})
    assert [o["add_name"] for o in opts] == ["Pricey"]
    assert opts[0]["market"]["tier"] in ("LIKELY", "PLAUSIBLE")


def test_finder_market_screen_drops_lopsided_offers():
    weeks = [12, 13, 14, 15]
    def o(pid, pos, pts):
        return PlayerOutlook(pid, pid, pos, "KC", None, {w: pts for w in weeks}, {w: 1.0 for w in weeks})
    shape = LeagueShape({"QB": 0, "RB": 1, "WR": 1, "TE": 0}, {}, 3, 0, weeks, {15})
    outs = {"u1": o("u1", "RB", 15), "u2": o("u2", "RB", 14), "uw": o("uw", "WR", 4),
            "t1": o("t1", "WR", 15), "t2": o("t2", "WR", 14), "tr": o("tr", "RB", 4)}
    v = RosValuator(shape, outs, n_sims=0)
    rosters = [RosterState(["u1", "u2", "uw"], []), RosterState(["t1", "t2", "tr"], [])]
    free = find_trades(v, rosters, 0, [], rescore_top=10)
    assert free["proposals"]
    # Market says the partner's WRs are worth far more than your RBs -> nothing realistic
    lop = find_trades(v, rosters, 0, [], rescore_top=10,
                      market_values=_v(u1=500, u2=500, uw=10, t1=5000, t2=5000, tr=10))
    assert lop["proposals"] == [] and lop["market_rejected"] > 0


def test_perceived_value_discounts_steep_fallers_only():
    from fantasy_intel.analytics.market import market_decline, perceived_value
    values = {
        "kyren": {"value": 3600.0, "trend_30d": -2454},   # was 6054 -> down 40.5%
        "dip": {"value": 3000.0, "trend_30d": -300},      # down 9%: not a faller
        "riser": {"value": 3000.0, "trend_30d": 800},
    }
    assert market_decline(values, "kyren") == pytest.approx(2454 / 6054, rel=1e-6)
    assert perceived_value(values, "kyren") == pytest.approx(3600 * (1 - 0.5 * 2454 / 6054))
    assert perceived_value(values, "dip") == 3000.0
    assert perceived_value(values, "riser") == 3000.0
    m = assess_trade(values, ["kyren"], ["dip"], 0.0)
    assert [f["player_id"] for f in m["fallers"]] == ["kyren"]


def test_sweeteners_never_sell_low():
    values = {
        "mine": {"value": 3000.0}, "stud": {"value": 6000.0},
        "kyren": {"value": 3600.0, "trend_30d": -2454},  # crashed: perceived ~2870, in window
        "cold": {"value": 3100.0},                        # running cold per our model
        "ok": {"value": 3300.0},
    }
    need = assess_trade(values, ["mine"], ["stud"], 0.5)["value_to_add"]
    ok_eval = lambda send: {"you": {"delta_per_week": 1.0}, "partner": {"delta_per_week": 0.5}}
    skipped = []
    opts = find_sweeteners(
        values, ["mine", "kyren", "cold", "ok"], ["mine"], ["stud"], need, ok_eval,
        exclude={"cold": "running cold"}, skipped=skipped,
    )
    assert [o["add_player_id"] for o in opts] == ["ok"]
    reasons = {s["player_id"]: s["reason"] for s in skipped}
    assert reasons["cold"] == "running cold" and "selling low" in reasons["kyren"]
