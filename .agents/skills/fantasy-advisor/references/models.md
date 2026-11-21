# Fantasy-intel models: how each number is produced

Read when the user asks "how is this calculated", challenges a result, or wants to tune it.
All constants live in `src/fantasy_intel/config.py`.

## Data

- **Weekly stats**: Sleeper per-week stats (team at game time, real snap share
  `off_snp / tm_off_snp`, red-zone counts). nflverse weekly stats 404 for 2025+.
- **Play-by-play**: nflverse pbp for air yards on every target, goal-line splits, completion
  probability / expected YAC, defensive EPA. Weeks nflverse has not published yet get
  goal-line splits imputed from Sleeper inside-20 counts (ratios calibrated from matched rows).
- **Caches**: Sleeper stats/projections 6h; live pbp 24h; schedules (Vegas lines) 12h;
  FantasyCalc 6h. Half-played weeks (e.g. only TNF) are excluded until complete.

## xFP and the efficiency baseline (buy-low / sell-high)

- **xFP**: receiving from per-play pbp (validated corr 0.82 vs actual); rushing from yard-line
  buckets (bias -0.1 pts/game); QB passing from per-play pbp (runs ~10% low: known issue).
- **FPOE** = actual points (league scoring) - xFP.
- **Efficiency baseline**: each player's FPOE per expected point, shrunk toward his position
  mean with prior strength k (QB 340, RB 230, WR 250, TE 450 xFP points; fit by split-half
  reliability, r ≈ 0.35-0.39, TE 0.21) and a 16-game halflife.
- **proj_ppg** = rolling xFP x (1 + efficiency).
- **Excess** = recent FPOE - what this player normally earns. Out of sample, the baseline
  predicts future FPOE at r = 0.35; excess beyond it only r = 0.09 (~89% regresses).
  Sell-high: excess >= +4/game with >= 15 recent PPG. Buy-low: excess <= -3 with xFP >= 12.

## Rest-of-season (ROS) lineup value

- **Projections**: Sleeper weekly projections per remaining week (out of sample on 2026 wks
  5-10: MAE 5.02 vs the model's 5.47; blending hurt). Byes from the schedule (nflverse team
  codes mapped to Sleeper, e.g. LA → LAR).
- **Availability**: P(a regular plays next game) QB .915, RB .960, WR .911, TE .885 (measured);
  next week by tag: Questionable .80, Doubtful .25, Out/IR 0. IR/Out players Sleeper doesn't
  project count 0 until projected again.
- **Simulation**: 300 availability scenarios, best lineup re-solved each week (dedicated slots,
  then flex most-restrictive first), common random numbers so before/after differences are
  the trade only.
- **Roster limits**: over the limit → cut the least valuable player; open spot → best free
  agent; IR-tagged players to open IR slots.
- **Waiver-optimized baseline**: before valuing a trade, both teams make their best free
  drop→add swaps (up to 3, ≥ 0.1 pts/week each; traded players can't be dropped). Without
  this, any trade that opens a roster spot gets credit for a free pickup.

## waiver-moves

- Pool: top 10 unrostered players per position by ROS projection (Sleeper does not expose
  per-league waiver status).
- Greedy swaps (drop one of the 6 lowest-ROS active players, add a free agent) while a swap
  gains ≥ 0.1 pts/week; up to `--max-moves` (default 3). Odds re-simulated after each step.
- `alternatives`: best single swap for each remaining free agent from the final roster.
- `has_trade_value`: suggested drop worth ≥ 500 on FantasyCalc.

## Standings / title odds

- 4,000 simulated seasons: remaining schedule from Sleeper matchups, current records (ties =
  half), points-for tiebreak, optional median game, standard fixed bracket with byes for top
  seeds when playoff teams isn't a power of two.
- Team score ~ Normal(projected lineup mean, σ), σ = pooled within-team SD of this league's
  last 8 weeks. K/DEF excluded (assumed equal). Third-party teams use current rosters.
- Verdict: title-odds change, ±0.5 percentage points = neutral.
- `title_leverage[w]`: title-odds change per projected point in week w.

## Market (FantasyCalc)

- Redraft values for the league's QB count / team count / PPR, keyed by Sleeper id (~190
  players; everyone else ≈ 0).
- FantasyCalc fits values so traded packages sum equal, so sides are compared by plain sums
  (no KTC-style stud adjustment; that would double count).
- **Perceived value**: players down ≥ 25% in 30 days count at value x (1 - 0.5 x decline).
- **Tiers** (partner's market change m, their lineup change L pts/week):
  LIKELY m ≥ -2.5% and L ≥ 0; PLAUSIBLE m ≥ -5%, or m ≥ -10% with L ≥ 1.0; else UNLIKELY.
  Untrained: tune from what actually gets accepted.
- **Sweeteners**: cheapest user player whose perceived value closes the gap to LIKELY, re-run
  through the full evaluation; kept only if the user still gains ≥ 0.5 pts/week. Never
  offers running-cold players or fallers.
- **Trade finder**: screens 1-for-1 / 2-for-1 / 1-for-2 swaps against every roster (drops
  swaps where the partner loses > 10% market value), weights the user's weekly gains by title
  leverage, re-scores the top 25 fully, keeps trades that raise the user's title odds ≥ 0.5
  points without lowering the partner's lineup.
