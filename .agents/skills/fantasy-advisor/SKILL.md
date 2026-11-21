---
name: fantasy-advisor
description: >-
  Fantasy football (redraft, Sleeper) advisor backed by a local analytics engine. Use when the
  user asks whether to accept, offer, or counter a trade ("should I trade X for Y", "is this
  fair", "what should I counter with"), wants trade ideas for their league, waiver/FAAB pickups,
  start/sit or matchup advice, buy-low / sell-high targets, injury fallout (next man up), or
  breaking news impact, or who to pick up / drop. Evaluates trades and waiver moves by
  rest-of-season lineup impact and championship odds (both teams for trades) plus FantasyCalc
  market value.
---

# Fantasy Advisor

Engine: `/Users/lukebenfer/PycharmProjects/fantasy-intel-agent`. Run every command as:

```bash
FI=/Users/lukebenfer/PycharmProjects/fantasy-intel-agent/.venv/bin/fantasy-intel
$FI <command> [options]
```

The user's Sleeper username and default league are saved (`$FI connect` changes them). The user
has several leagues: when they name one (e.g. "my work league"), run `$FI leagues` and pass its id as
`--league-id`. Never assume the default league is the one they mean.

Never pass `--season` unless the user asks about a past season; the default is the current season.

Prefer `--json` (available on `trade-eval`, `waiver-moves`, `trades`, `waivers`): stdout is pure
JSON, progress and warnings go to stderr. Read the human tables only for commands without it.

## Workflows

### "Should I make / accept this trade?" (most common)

1. Resolve the league id (above).
2. `$FI trade-eval --league-id <id> --give "A,B" --get "C" --json`
   - `--give` = players leaving the user's roster; `--get` = players arriving (all from one team).
   - For an offer the user *received*, `--give` is what they would send.
   - Last names work. If a name fails or is ambiguous, retry with the last name or the exact
     Sleeper spelling from the error (e.g. "Kenny Gainwell", not "Kenneth").
3. Evaluate every version the user mentions (their offer, the counter) and compare side by side.
4. Check `you_send` / `you_get` for injury status, byes, and `weeks_without_projection`; if a
   key player is injured or newsy, run `$FI news --query "<player>"`.
5. Answer with the format below. Lead with title odds, then lineup and market.

### "What should I counter with?" / "make this work"

- Read `market.sweeteners` (players to add that make the partner likely to accept while the
  trade still helps the user) and `market.sweeteners_skipped` (players deliberately *not*
  offered: running cold or market value falling; selling them would be selling low).
- Try the user's own ideas with `trade-eval`; compare title-odds deltas, not just points.
- Check the lineup tables (`you.lineup_before/after`, `partner.lineup_*`): if a version guts a
  position group or the partner would just cut what they receive, say so.

### "Find me trades" / "who should I trade with"

`$FI trades --league-id <id> --json` returns `buy_low`, `sell_high`, and
`trade_proposals.proposals` (each with `your_title_delta`, `their_title_delta`,
`your_lineup_gain`, `market_tier`, `market_delta`, `sweeteners`, `pitch`) plus
`trade_proposals.current_odds`. Prefer LIKELY/PLAUSIBLE proposals. Before recommending one, run
it through `trade-eval --json` to get both lineups and sweeteners. "0 proposals" is a
legitimate answer (no realistic trade raises the user's title odds); then run `waiver-moves`.

### Waivers: "who should I pick up / drop?"

1. `$FI waiver-moves --league-id <id> --json`: the best free drop→add swaps by
   rest-of-season value, in priority order. Each move has `gain_ppw`, `cumulative_ppw`,
   `odds_after` (title/playoffs/bye), and `add`/`drop` cards (team, injury, byes, ROS pts/week,
   market value). `before.odds` / `after.odds` show the total effect; `alternatives` are
   runner-up claims if a target gets taken; `waiver_type` / `faab_remaining` give the budget.
2. If a `drop` has `has_trade_value: true`, tell the user to shop that player (e.g. as a trade
   sweetener) before cutting him.
3. For speculative adds (breakouts before they show up in projections), add
   `$FI waivers --league-id <id> --json` (opportunity index from snap/target/high-value-touch
   surges, FAAB bid suggestions). `waiver-moves` values players on current projections;
   `waivers` finds rising usage. Mention both when they disagree.
4. Check news on any add with an injury tag or a sudden projection jump before recommending a
   big FAAB bid.

Free waiver moves often matter more than any trade (in testing, three pickups doubled a team's
title odds). Mention them whenever you answer a trade question too (`you.waiver_moves`).

### Start/sit, matchup, injuries, news

- `$FI matchup --league-id <id>`: floor/median/ceiling, favorite/underdog posture, optimal lineup.
- `$FI injuries --positions RB,WR,TE` for sidelined starters and next-man-up beneficiaries.
- `$FI news --query "<player or team>"` (add `--synthesize` for AI injury/depth summaries).
- `$FI roster`, `$FI drops`, `$FI trending --hours 24` as needed.

## Reading trade-eval output

| Field | Meaning |
|---|---|
| `verdict`, `verdict_basis` | Judged by change in the user's **title odds** (±0.5 pts = neutral) when standings are available, else by lineup pts/week |
| `odds.you/partner.before/after/delta` | Playoff, first-round bye, and title probabilities from 4,000 simulated seasons (real schedule and records) |
| `title_leverage` | Title-odds value of one point in each week: shows whether regular season or playoff weeks matter more *for this user's standings* |
| `you.delta_per_week`, `delta_regular`, `delta_playoffs` | Expected starting-lineup points (rest of season), measured after both teams' free waiver moves |
| `you.waiver_moves` | Free pickups the user should make regardless; the trade gets no credit for them |
| `you.drops` / `you.adds` | Roster moves the trade forces (over the roster limit → cut; open spot → pickup) |
| `market.tier` | LIKELY / PLAUSIBLE / UNLIKELY the partner accepts (FantasyCalc perceived value + their lineup change) |
| `market.gap_pct` | > 0: user receives more market value; < 0: user overpays |
| `market.fallers` | Players whose market value fell ≥25% in 30 days (counted at a discount) |

How to weigh them:
- **Title odds decide whether the trade is good for the user.** Lineup points explain why.
  A trade can lose points per week and still raise title odds (better playoff weeks), or the
  reverse.
- **Market tier decides whether the partner will say yes.** A great trade rated UNLIKELY needs
  a sweetener or a different target; a LIKELY trade where `gap_pct` is very negative means the
  user is overpaying and could ask for more.
- **Running hot/cold tags.** "Running hot" = recent scoring well above the player's own
  efficiency baseline (~90% of that excess regresses): acquiring him is buying high. "Running
  cold" with steady usage = buy-low; don't sell him as a throw-in.

## Answer format

1. **Bottom line** in one sentence (accept / decline / counter with X).
2. **Table** per trade version: your title-odds change, playoff-odds change, pts/week (and
   playoffs), partner's change, market gap, likelihood.
3. **Why**: the 2-3 drivers (lineup holes created or filled, byes, playoff weeks, injuries,
   hot/cold players, roster cuts).
4. **Do regardless**: free waiver moves with their value.
5. **Caveats** that matter for this decision only.

Use the user's player and manager names. Say "title odds", "points per week", not internal
field names.

## Check before answering

- **Freshness**: the notes say which weeks are covered ("Weeks 11-17"). If the first week is
  behind the current NFL week, or the user mentions news the engine can't know (an injury from
  today), say so and adjust.
- **Trade deadline**: if output warns the deadline is this week, mention it first.
- **Noise**: differences under ~0.3 pts/week or ~0.3 title-odds points are within simulation
  noise. Don't rank trades on gaps that small.
- **Partner names**: Sleeper team names can differ from usernames; match the user's wording.

## Known limits (state when relevant)

- Projections are Sleeper's weekly projections (they beat this engine's own model out of
  sample); players Sleeper doesn't project fall back to the model.
- Team strength covers QB/RB/WR/TE; kickers and defenses are assumed equal.
- Likelihood tiers are untrained judgment calls; the user's knowledge of the league-mate wins.
- Only 1-for-1, 2-for-1, and 1-for-2 swaps are searched by `trades`; `trade-eval` accepts any
  sizes.

Model details, thresholds, and validation numbers: [references/models.md](references/models.md).
Read it when the user asks how something is calculated or challenges a number.
