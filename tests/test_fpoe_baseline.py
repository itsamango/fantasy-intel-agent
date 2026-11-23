"""Tests for the FPOE efficiency baseline (Empirical-Bayes shrunk FPOE / xFP)."""

import numpy as np
import pandas as pd
import pytest

from fantasy_intel.analytics.metrics import (
    add_fpoe_baseline,
    estimate_expected_fantasy_points,
    fit_fpoe_prior_strength,
)
from fantasy_intel.analytics.trades import TradeAnalyzer
from fantasy_intel.config import FPOE_PRIOR_STRENGTH


def _games(pid, name, season, weeks, xfp, fpoe, pos="WR"):
    return [
        {"player_id": pid, "player_display_name": name, "position": pos, "recent_team": "KC",
         "season": season, "week": w, "season_type": "REG", "expected_fp_ppr": xfp,
         "fp_diff": fpoe, "fantasy_points_ppr": xfp + fpoe}
        for w in weeks
    ]


def _with_rolling(df):
    df = df.copy()
    df["rolling_xfp_3w"] = df["expected_fp_ppr"]
    df["rolling_fpoe_3w"] = df["fp_diff"]
    return df


def test_baseline_shrinks_toward_position_mean_and_grows_with_sample():
    # Pool of average WRs (FPOE 0) so the position mean is ~0
    rows = []
    for i in range(30):
        rows += _games(f"avg{i}", f"Avg {i}", 2026, range(1, 11), 12.0, 0.0)
    rows += _games("elite", "Elite", 2026, range(1, 11), 15.0, 4.5)  # +30% efficiency
    out = add_fpoe_baseline(_with_rolling(pd.DataFrame(rows)))
    elite = out[out["player_id"] == "elite"].sort_values("week")

    effs = elite["fpoe_eff"].to_numpy()
    assert np.all(np.diff(effs) > 0)          # more evidence -> more trust
    assert 0 < effs[-1] < 0.30                # but still shrunk below the raw +30%
    k = FPOE_PRIOR_STRENGTH["WR"]
    m = (10 * 4.5) / (30 * 10 * 12.0 + 10 * 15.0)  # pooled WR efficiency (prior mean)
    s_f = sum(4.5 * 0.5 ** (i / 16) for i in range(10))
    s_x = sum(15.0 * 0.5 ** (i / 16) for i in range(10))
    assert effs[-1] == pytest.approx((s_f + k * m) / (s_x + k), rel=0.01)
    last = elite.iloc[-1]
    assert last["proj_ppg"] == pytest.approx(15.0 * (1 + last["fpoe_eff"]), abs=0.01)


def test_history_seeds_baseline_and_absorbs_skill_from_excess():
    pool = []
    for i in range(30):
        pool += _games(f"avg{i}", f"Avg {i}", 2026, range(1, 7), 12.0, 0.0)
    # Two players with an identical 2026 line: +5 FPOE on 15 xFP every week
    current = pd.DataFrame(
        pool
        + _games("vet", "Proven Vet", 2026, range(1, 7), 15.0, 5.0)
        + _games("new", "Unknown", 2026, range(1, 7), 15.0, 5.0)
    )
    # Only the veteran has two prior seasons of the same efficiency
    history = pd.DataFrame(
        _games("vet", "Proven Vet", 2024, range(1, 18), 15.0, 5.0)
        + _games("vet", "Proven Vet", 2025, range(1, 18), 15.0, 5.0)
    )
    out = add_fpoe_baseline(_with_rolling(current), history=history)
    vet = out[out["player_id"] == "vet"].iloc[-1]
    new = out[out["player_id"] == "new"].iloc[-1]

    assert set(out["season"]) == {2026}                      # history rows are not emitted
    assert vet["fpoe_eff_weight"] > new["fpoe_eff_weight"]
    assert vet["fpoe_eff"] > new["fpoe_eff"]
    assert vet["proj_ppg"] > new["proj_ppg"]
    # Same box score, but much of the veteran's output is explained by his baseline.
    # With a 16-game halflife the effective sample caps near ~23 games, so even a
    # 3-year veteran's efficiency gets ~55% credit (consistent with YoY r 0.3-0.5).
    assert vet["fpoe_excess_3w"] < 0.75 * new["fpoe_excess_3w"]
    assert 0.45 < vet["fpoe_eff_weight"] < 0.65


def test_sell_high_skips_efficient_veteran_but_flags_unknown():
    base = dict(position="WR", recent_team="KC", season_type="REG", targets=10, carries=0,
                receiving_air_yards=100, offense_pct=0.85, receptions=7)
    x = float(estimate_expected_fantasy_points(
        pd.DataFrame([{**base, "player_id": "z", "season": 2026, "week": 1, "fantasy_points_ppr": 0.0}]),
        with_baseline=False,
    )["expected_fp_ppr"].iloc[0])

    rows = []
    for i in range(30):
        rows += [{**base, "player_id": f"a{i}", "player_display_name": f"Avg {i}", "season": 2026,
                  "week": w, "fantasy_points_ppr": x} for w in range(1, 7)]
    # Veteran: steady +30% efficiency all season; Unknown: average, then +30% for 3 weeks
    rows += [{**base, "player_id": "vet", "player_display_name": "Proven Vet", "season": 2026,
              "week": w, "fantasy_points_ppr": 1.3 * x} for w in range(1, 7)]
    rows += [{**base, "player_id": "new", "player_display_name": "Unknown", "season": 2026,
              "week": w, "fantasy_points_ppr": x if w <= 3 else 1.3 * x} for w in range(1, 7)]
    weekly = pd.DataFrame(rows)
    history = pd.DataFrame([
        {**base, "player_id": "vet", "player_display_name": "Proven Vet", "season": s, "week": w,
         "fantasy_points_ppr": 1.3 * x}
        for s in (2024, 2025) for w in range(1, 18)
    ])

    analyzer = TradeAnalyzer(weekly, fpoe_history=history)
    latest = analyzer.df[analyzer.df["week"] == 6].set_index("player_id")
    assert latest.loc["vet", "fpoe_excess_3w"] < 4.0 <= latest.loc["new", "fpoe_excess_3w"]

    names = analyzer.get_sell_high_candidates(season=2026)["player_display_name"].tolist()
    assert "Unknown" in names
    assert "Proven Vet" not in names


def test_fit_prior_strength_recovers_reliability():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(150):
        true_eff = rng.normal(0, 0.15)
        for w in range(1, 41):
            x = 12.0
            rows.append({"player_id": f"p{p}", "position": "WR", "season": 2024 + (w - 1) // 20,
                         "week": (w - 1) % 20 + 1, "expected_fp_ppr": x,
                         "fp_diff": x * true_eff + rng.normal(0, 6.0)})
    fit = fit_fpoe_prior_strength(pd.DataFrame(rows))["WR"]
    # signal var 0.15^2 per unit xFP; noise var 36 per game of 12 xFP -> k = 36/12 / 0.0225 ~ 133
    assert 0.3 < fit["r"] < 0.95
    assert 60 < fit["k"] < 300
