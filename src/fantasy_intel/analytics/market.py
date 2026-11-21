"""Trade-market assessment on FantasyCalc values: fairness, acceptance tier, sweeteners.

Lineup math (ros_value) says whether a trade helps each team; the market says
whether the partner will *perceive* it as fair, which is what drives acceptance.
FantasyCalc values are fit from real trades so that traded packages sum equal,
so sides are compared by plain sums (the stud/consolidation premium is already
inside each player's value).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from fantasy_intel.config import (
    MARKET_FAIR_BAND,
    MARKET_FALLER_DISCOUNT,
    MARKET_FALLER_THRESHOLD,
    MARKET_LIKELY_MIN,
    MARKET_LINEUP_OFFSET_PPW,
    MARKET_MIN_TOTAL_VALUE,
    MARKET_PLAUSIBLE_LINEUP_MIN,
    MARKET_PLAUSIBLE_MIN,
    MARKET_SWEETENER_MAX_MULTIPLE,
    MARKET_SWEETENER_OPTIONS,
    ROS_VERDICT_MARGIN_PPW,
)

TIER_ORDER = {"LIKELY": 0, "PLAUSIBLE": 1, "UNLIKELY": 2, "UNKNOWN": 3}


def market_value(values: Mapping[str, Mapping[str, Any]], pid: str) -> float:
    rec = values.get(str(pid))
    return float(rec["value"]) if rec else 0.0


def market_decline(values: Mapping[str, Mapping[str, Any]], pid: str) -> float:
    """Fractional 30-day value drop (0.4 = lost 40% of its value a month ago); 0 if rising."""
    rec = values.get(str(pid)) or {}
    value, trend = float(rec.get("value") or 0.0), rec.get("trend_30d")
    if not trend or trend >= 0:
        return 0.0
    prior = value - float(trend)
    return float(-trend / prior) if prior > 0 else 0.0


def is_faller(values: Mapping[str, Mapping[str, Any]], pid: str) -> bool:
    return market_decline(values, pid) >= MARKET_FALLER_THRESHOLD


def perceived_value(values: Mapping[str, Mapping[str, Any]], pid: str) -> float:
    """Value as a league-mate likely perceives it: steep 30-day fallers are discounted."""
    v = market_value(values, pid)
    d = market_decline(values, pid)
    return v * (1.0 - MARKET_FALLER_DISCOUNT * d) if d >= MARKET_FALLER_THRESHOLD else v


def acceptance_tier(partner_market_pct: Optional[float], partner_lineup_ppw: float) -> str:
    """LIKELY / PLAUSIBLE / UNLIKELY from the partner's market change and lineup change."""
    if partner_market_pct is None:
        return "UNKNOWN"
    m, lineup = partner_market_pct, partner_lineup_ppw
    if m >= MARKET_LIKELY_MIN and lineup >= 0:
        return "LIKELY"
    if m >= MARKET_PLAUSIBLE_MIN or (m >= MARKET_PLAUSIBLE_LINEUP_MIN and lineup >= MARKET_LINEUP_OFFSET_PPW):
        return "PLAUSIBLE"
    return "UNLIKELY"


def assess_trade(
    values: Mapping[str, Mapping[str, Any]],
    you_send: Sequence[str],
    you_get: Sequence[str],
    partner_lineup_ppw: float,
) -> Dict[str, Any]:
    """Market view of a trade from the user's side.

    gap = (get - send) / (get + send): > 0 means you receive more market value.
    The partner's market change is -gap. Uses perceived values (steep 30-day
    fallers discounted, see perceived_value) since perception drives acceptance.
    """
    send = sum(perceived_value(values, p) for p in you_send)
    get = sum(perceived_value(values, p) for p in you_get)
    total = send + get
    fallers = [
        {"player_id": p, "decline_30d": round(market_decline(values, p), 3),
         "listed": round(market_value(values, p)), "perceived": round(perceived_value(values, p))}
        for p in list(you_send) + list(you_get) if is_faller(values, p)
    ]
    if total < MARKET_MIN_TOTAL_VALUE:
        return {
            "send_value": round(send), "get_value": round(get), "gap_pct": None,
            "fair": None, "tier": "UNKNOWN", "value_to_add": 0.0, "fallers": [],
            "summary": "No meaningful market value on either side (players outside FantasyCalc's top ~200).",
        }
    gap = (get - send) / total
    partner_m = -gap
    fair = abs(gap) <= MARKET_FAIR_BAND
    if fair:
        summary = f"Fair by market ({gap:+.1%} to you)"
    elif gap > 0:
        summary = f"You win the market by {gap:.1%}: partner gives more perceived value"
    else:
        summary = f"You overpay by {-gap:.1%} in market terms"
    # Value the user would add so the partner's market change reaches MARKET_LIKELY_MIN
    need = (get - send + MARKET_LIKELY_MIN * total) / (1 - MARKET_LIKELY_MIN)
    return {
        "send_value": round(send),
        "get_value": round(get),
        "fallers": fallers,
        "gap_pct": round(gap, 4),
        "fair": fair,
        "tier": acceptance_tier(partner_m, partner_lineup_ppw),
        "value_to_add": round(max(0.0, need)),
        "summary": summary,
    }


def find_sweeteners(
    values: Mapping[str, Mapping[str, Any]],
    user_roster_ids: Sequence[str],
    you_send: Sequence[str],
    you_get: Sequence[str],
    value_to_add: float,
    evaluate: Callable[[List[str]], Dict[str, Any]],
    names: Optional[Mapping[str, str]] = None,
    min_user_gain_ppw: float = ROS_VERDICT_MARGIN_PPW,
    max_options: int = MARKET_SWEETENER_OPTIONS,
    max_evaluations: int = 4,
    exclude: Optional[Mapping[str, str]] = None,
    skipped: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Players to add to your side so the partner perceives the deal as fair,
    while it still helps your rest-of-season lineup.

    Candidates: your players not already in the deal with perceived value in
    [value_to_add, value_to_add * MARKET_SWEETENER_MAX_MULTIPLE], cheapest
    first. Don't-sell-low guard: players in `exclude` (e.g. running cold vs
    their efficiency baseline) and steep 30-day market fallers are never
    offered; they are appended to `skipped` with the reason. Each is re-evaluated with `evaluate(new_send)` (an evaluate_trade
    result); kept only if your gain stays >= min_user_gain_ppw and the
    partner's lineup does not get worse.
    """
    if value_to_add <= 0:
        return []
    names = names or {}
    exclude = dict(exclude or {})
    in_deal = set(you_send) | set(you_get)
    in_window = [
        p for p in user_roster_ids
        if p not in in_deal
        and value_to_add <= perceived_value(values, p) <= value_to_add * MARKET_SWEETENER_MAX_MULTIPLE
    ]
    candidates = []
    for p in in_window:
        reason = exclude.get(p)
        if reason is None and is_faller(values, p):
            reason = f"market value down {market_decline(values, p):.0%} in 30 days: selling low"
        if reason is not None:
            if skipped is not None:
                skipped.append({"player_id": p, "name": names.get(p, p), "reason": reason})
            continue
        candidates.append(p)
    candidates.sort(key=lambda p: perceived_value(values, p))
    options: List[Dict[str, Any]] = []
    for pid in candidates[:max_evaluations]:
        new_send = list(you_send) + [pid]
        res = evaluate(new_send)
        you_ppw = res["you"]["delta_per_week"]
        partner_ppw = res["partner"]["delta_per_week"]
        if you_ppw < min_user_gain_ppw or partner_ppw < 0:
            continue
        mk = assess_trade(values, new_send, you_get, partner_ppw)
        options.append({
            "add_player_id": pid,
            "add_name": names.get(pid, pid),
            "add_value": round(perceived_value(values, pid)),
            "your_ppw": you_ppw,
            "partner_ppw": partner_ppw,
            "market": mk,
        })
        if len(options) >= max_options:
            break
    return options
