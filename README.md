# 🏈 Fantasy Football Intelligence Agent (`fantasy-intel`)

> An AI and data-driven Fantasy Football intelligence engine that detects under-the-radar waiver wire gems, identifies buy-low / sell-high trade opportunities using expected fantasy points ($xFP$), and cross-references breaking injury news in real-time.

---

## 🌟 Key Features

1. **Under-The-Radar Waiver Sleeper Detector**:
   - Moves beyond basic box scores by analyzing **leading opportunity metrics**: snap share trajectory ($\Delta \text{snaps}$), Weighted Opportunity Rating (WOPR), target share, and touch share.
   - **Position-specific opportunity index (0–100)** so WR/TEs score via WOPR + unrealized air yards (RACR $< 0.75$) while RBs score via high-value touches (HVT) — one formula no longer locks either position out.
   - **Dynamic FAAB bid sizing**: bids scaled by ΔVORP over your worst bench player, your league's empirical winning-bid history (Sleeper transactions), remaining budget, and standings urgency — with priority-claim recommendations for blind-bid leagues.

2. **Buy-Low & Sell-High Trade Engine**:
   - Calculates **situational Expected Fantasy Points (xFP)** and measures performance variance ($\text{FPOE} = \text{Points} - xFP$) with rolling 3-week exponentially weighted trends.
   - Highlights unlucky players due for positive regression (Buy-Low: high rolling xFP, deeply negative rolling FPOE) and low-volume touchdown luck merchants (Sell-High: sustained $\text{FPOE} > +4.0$).
   - **🤝 Bilateral League Trade Proposals** (`--username` / `--league-id`): scans every opponent roster and ranks 1-for-1 / 2-for-1 swaps using **dynamic league VORP**, the $\alpha = 0.72$ package consolidation penalty, a Pareto-benefit filter (both lineups must improve), and a logistic acceptance-probability score with a ready-to-send pitch.

3. **Live News & Injury Context Engine**:
   - Ingests real-time RSS feeds from ESPN, ProFootballTalk, and beat reporters.
   - Automatically tags news with injury severity and role expansion alerts.

4. **Sleeper API League & Matchup Integration**:
   - Live query of nationwide trending adds and drops across all fantasy leagues.
   - Filter waiver recommendations specifically against your private Sleeper league's rostered players.
   - **Matchup & Start/Sit Optimizer**: context-adjusted Floor/Median/Ceiling projections per starter and bench player — scaled by Vegas implied team totals, wind (≥15 mph outdoor downgrade for QBs/deep-threat WRs, RB boost), and opponent defensive EPA funnels — plus favorite/underdog posture detection ($\Delta M$) and an integer-linear-programming optimal lineup with mathematically justified start/sit swaps.

5. **Antigravity Custom Skill**:
   - Includes `.agents/skills/fantasy-advisor/SKILL.md` so the agent can pair program and provide fantasy advice natively inside Antigravity!

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/your-username/fantasy-intel-agent.git
cd fantasy-intel-agent

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

### 2. CLI Usage

#### Connect Your Sleeper Account (Zero API Key Needed!)
Link your Sleeper username once so all commands automatically sync with your team and leagues:
```bash
fantasy-intel connect --username <YOUR_SLEEPER_USERNAME>
```

#### View Your Leagues & Standings
```bash
fantasy-intel leagues
```

#### View Your Active Team Roster & Injury Status
```bash
fantasy-intel roster
```

#### Discover Waiver Wire Sleepers
Find high-opportunity players flying under the radar (automatically filters out players owned in your league):
```bash
fantasy-intel waivers --season 2024 --limit 10
```
Filter by position or against your Sleeper league:
```bash
fantasy-intel waivers --positions WR,RB --league-id <YOUR_SLEEPER_LEAGUE_ID>
```

#### Evaluate Trades (Buy-Low / Sell-High + Bilateral Proposals)
```bash
fantasy-intel trades --season 2024
```
Add your Sleeper username and league ID to also rank 🤝 mutually beneficial 1-for-1 / 2-for-1 trade proposals against every roster in your league (dynamic VORP, package decay, Pareto filter, and acceptance probability):
```bash
fantasy-intel trades --username <YOUR_SLEEPER_USERNAME> --league-id <YOUR_SLEEPER_LEAGUE_ID>
```

#### Evaluate a Specific Trade (rest-of-season + title odds + market)
```bash
fantasy-intel trade-eval --league-id <LEAGUE_ID> --give "Deebo Samuel" --get "Breece Hall" [--json]
```
*Simulates both teams' lineups for every remaining week (Sleeper projections, byes, injury tags, random availability, roster limits), measures the trade against each team's waiver-optimized roster, runs 4,000 season simulations on your real schedule and standings to report playoff / bye / title odds before and after, and checks FantasyCalc market value to estimate whether the partner accepts (with sweetener suggestions that never sell low).*

`fantasy-intel trades` uses the same engine to search 1-for-1 / 2-for-1 / 1-for-2 swaps with every roster and ranks them by your change in title odds.

#### Best Waiver Moves (rest-of-season value + title odds)
```bash
fantasy-intel waiver-moves --league-id <LEAGUE_ID> [--max-moves 3] [--json]
```
*Greedy drop → add swaps ranked by rest-of-season lineup value, with your playoff / title odds after each move, runner-up claims, and a warning when a suggested drop still has trade value.*

`--json` is available on `trade-eval`, `waiver-moves`, `trades`, and `waivers` (stdout is pure JSON; progress goes to stderr).

#### Live Trending Adds & Drops (Sleeper)
```bash
fantasy-intel trending --hours 24
```

#### Sidelined Starters & Next-Man-Up Beneficiaries
Scan all 32 NFL depth charts for sidelined starters and their direct backup beneficiaries:
```bash
fantasy-intel injuries --positions RB,WR,TE
```

#### AI-Synthesized Breaking News & Injury Wire
Extract injury severity, missed timelines, and promoted depth-chart beneficiaries:
```bash
fantasy-intel news --synthesize --limit 10
```
Or query raw beat reporter wire:
```bash
fantasy-intel news --query "Barkley"
```

#### Situational Matchup & Starting Lineup Advice
Connect directly to your Sleeper account:
```bash
fantasy-intel matchup
```
*Evaluates your starters against your opponent, alerts you to injured starters, projects Floor/Median/Ceiling for every starter and bench player (Vegas ITT, wind, and defensive funnel adjusted), flags your favorite/underdog posture (ΔM), and solves the mathematically optimal start/sit swaps via ILP.*

#### 📱 Discord Bot (Mobile Fantasy Intelligence)
Run your personal Discord bot to access all functionality from your phone:
```bash
fantasy-intel bot --token <YOUR_DISCORD_BOT_TOKEN>
```
*Available mobile Slash Commands:*
- `/roster`: Full roster breakdown with health status
- `/matchup`: Head-to-head matchup + recommended bench swaps
- `/waivers`: Free agent sleeper targets unowned in your league
- `/injuries`: Next-man-up NFL depth chart promotions
- `/leagues`: View all active Sleeper leagues and standings
- `/ask <question>`: Gemini AI analysis with your live team context injected
- *Direct Message / Mention*: Mention `@Bot <question>` for conversational advice on your phone!

---

## 📐 Methodology: Opportunity Over Box Scores

Most fantasy players look in the rearview mirror:
$$\text{Last Week's Fantasy Points} \neq \text{Next Week's Fantasy Points}$$

This engine focuses on structural opportunity metrics:
* **Weighted Opportunity Rating (WOPR)**:
  $$\text{WOPR} = 1.5 \times \text{Target Share} + 0.7 \times \text{Air Yards Share}$$
* **Snap Share Momentum**:
  $$\Delta \text{Snap} = \text{Snap}_{\text{recent}} - \text{Snap}_{\text{prev\_avg}}$$
* **Situational Expected Fantasy Points (xFP)** — touch-level valuation stratified by field position (goal-line proximity and end-zone targets), not a flat per-touch average:
  $$\text{xFP} = \underbrace{0.45 \cdot c_{\text{mid}} + 1.45 \cdot c_{10\text{-}6} + 2.65 \cdot c_{\text{inside 5}}}_{\text{rushing, by yard line}} + \underbrace{(0.65w_{\text{rec}} + 0.52) \cdot t_{\text{reg}} + 0.075 \cdot \text{air\_yds} + (0.42w_{\text{rec}} + 2.15) \cdot t_{\text{EZ}}}_{\text{receiving, catch-rate weighted}}$$
* **Fantasy Points Over Expected (FPOE) & rolling trends**:
  $$\text{FPOE} = \text{Actual Points} - \text{xFP}, \qquad \text{rolling 3-week EWM of xFP and FPOE}$$
  High rolling xFP ($\ge 12.0$ PPG) with deeply negative rolling FPOE ($\le -3.0$) = **buy-low**; sustained $\text{FPOE} > +4.0$ on low xFP = **sell-high** (such players decline ~3.7 PPG over their next 3 games).
* **Efficiency diagnostics**: RACR ($\text{rec yds} / \text{air yds}$, highly mean-reverting), unrealized air yards, and High-Value Touches ($\text{HVT} = \text{targets} + \text{carries inside the 10}$).

### 🎯 Waiver Wire & FAAB Sizing
* **Position-specific Opportunity Index (0–100)** — WR/TE: snap share (25) + WOPR (40) + snap surge (20) + unrealized air yards (15); RB: snap share (25) + HVT (35) + touches (25) + snap surge (15).
* **Dynamic FAAB bids**: $\text{bid} = \min(B_{\text{rem}},\ \text{Bid}_{\text{clearing}} \times \frac{\Delta \text{VORP}}{\text{VORP}_{\max}} \times (\frac{B_{\text{rem}}}{B_{\text{init}}})^{\gamma} \times \phi_{\text{urgency}})$ using your league's empirical winning bids from Sleeper transaction history (priority-claim recommendations for blind-bid leagues).

### 🤝 Bilateral Trade Valuation
* **Dynamic VORP**: replacement rank $K_{\text{pos}} = N \times (S_{\text{pos}} + F_{\text{pos}}) + 1$ from your league's exact roster slots; $\text{VORP}_i = \max(0, P_i - P_{\text{replacement}})$.
* **Package consolidation decay**: $\text{Value}_{\text{pkg}} = V_{(1)} + 0.72 \cdot V_{(2)} + \dots$ — two 10-point players are not one 20-point player.
* **Pareto filter + acceptance score**: both managers' optimal lineups must improve; proposals are ranked by a logistic acceptance probability blending market-value parity, their lineup gain, and positional need.

### ⚔️ Matchup & Start/Sit Optimization
* **Context multipliers**: Vegas implied team total $\left(\text{ITT}/22.5\right)^{0.6}$; outdoor wind $\ge 15$ mph → $0.88\times$ QBs/deep-threat WRs (aDOT > 12) and $1.05\times$ RBs; opponent top-8 defensive EPA funnel → $1.08\times$.
* **Floor / Median / Ceiling**: $P_{15}$ / $P_{50}$ / $P_{85}$ from a moment-matched log-normal over rolling xFP and trailing weekly variance.
* **Posture & ILP**: favored by $\ge 10$ median pts → floor-protect ($0.65 P_{15} + 0.35 P_{50}$); underdog → chase ceiling ($0.65 P_{85} + 0.35 P_{50}$); the optimal lineup across your league's exact slots is solved with integer linear programming.

---

## 🧪 Testing

Run unit tests via `pytest`:
```bash
pytest tests
```

---

## 📄 License
MIT License.
