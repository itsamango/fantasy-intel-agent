# 🏈 Fantasy Football Intelligence Agent (`fantasy-intel`)

> An AI and data-driven Fantasy Football intelligence engine that detects under-the-radar waiver wire gems, identifies buy-low / sell-high trade opportunities using expected fantasy points ($xFP$), and cross-references breaking injury news in real-time.

---

## 🌟 Key Features

1. **Under-The-Radar Waiver Sleeper Detector**:
   - Moves beyond basic box scores by analyzing **leading opportunity metrics**: snap share trajectory ($\Delta \text{snaps}$), Weighted Opportunity Rating (WOPR), target share, and touch share.
   - Catches breakouts *before* they have the multi-touchdown game that costs 40% of your FAAB.

2. **Buy-Low & Sell-High Trade Engine**:
   - Calculates **Expected Fantasy Points (xFP)** and measures performance variance ($\text{Points} - xFP$).
   - Highlights unlucky players due for positive regression (Buy-Low) and low-volume touchdown luck merchants (Sell-High).

3. **Live News & Injury Context Engine**:
   - Ingests real-time RSS feeds from ESPN, ProFootballTalk, and beat reporters.
   - Automatically tags news with injury severity and role expansion alerts.

4. **Sleeper API League & Matchup Integration**:
   - Live query of nationwide trending adds and drops across all fantasy leagues.
   - Filter waiver recommendations specifically against your private Sleeper league's rostered players.
   - **Situational Matchup Inspector**: Inspect your starting lineup vs your opponent, detect active injuries, and receive tactical start/sit swap suggestions.

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

#### Discover Waiver Wire Sleepers
Find high-opportunity players flying under the radar:
```bash
fantasy-intel waivers --season 2024 --limit 10
```
Filter by position or against your Sleeper league:
```bash
fantasy-intel waivers --positions WR,RB --league-id <YOUR_SLEEPER_LEAGUE_ID>
```

#### Evaluate Trades (Buy-Low / Sell-High)
```bash
fantasy-intel trades --season 2024
```

#### Live Trending Adds & Drops (Sleeper)
```bash
fantasy-intel trending --hours 24
```

#### Breaking News & Beat Reporter Wire
```bash
fantasy-intel news --query "Barkley"
```

#### Situational Matchup & Starting Lineup Advice
Connect directly to your Sleeper account:
```bash
fantasy-intel matchup --username <YOUR_SLEEPER_USERNAME>
```
*Evaluates your starters against your opponent, alerts you to injured starters, and suggests optimal bench substitutions.*

---

## 📐 Methodology: Opportunity Over Box Scores

Most fantasy players look in the rearview mirror:
$$\text{Last Week's Fantasy Points} \neq \text{Next Week's Fantasy Points}$$

This engine focuses on structural opportunity metrics:
* **Weighted Opportunity Rating (WOPR)**:
  $$\text{WOPR} = 1.5 \times \text{Target Share} + 0.7 \times \text{Air Yards Share}$$
* **Snap Share Momentum**:
  $$\Delta \text{Snap} = \text{Snap}_{\text{recent}} - \text{Snap}_{\text{prev\_avg}}$$
* **Expected Fantasy Points (xFP)**:
  $$\text{xFP} \approx (\text{Targets} \times 1.3) + (\text{Air Yards} \times 0.08) + (\text{Carries} \times 0.68)$$

---

## 🧪 Testing

Run unit tests via `pytest`:
```bash
pytest tests
```

---

## 📄 License
MIT License.
