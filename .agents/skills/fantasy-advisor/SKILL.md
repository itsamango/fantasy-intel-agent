---
name: fantasy-advisor
description: >-
  Expert Fantasy Football Intelligence Agent. Analyzes multi-year historical NFL data,
  snap counts, target share trends, expected fantasy points (xFP), depth charts, and live news/injury
  feeds (ESPN, Sleeper) to provide under-the-radar waiver wire sleepers and buy-low/sell-high trades.
---

# Fantasy Football Intelligence Skill

Use this skill when the user asks for:
- Waiver wire pickups, sleepers, or breakout candidates
- Trade advice (buy-low / sell-high candidates)
- Injury replacement recommendations and depth chart surges (next-man-up beneficiaries)
- AI-synthesized breaking news and injury severity wire
- Live trending adds and drops across fantasy football platforms (Sleeper)
- Player opportunity analysis (snap share %, WOPR, target share trends, expected points)

---

## Tool Execution

The underlying Python intelligence engine is located at:
`/Users/lukebenfer/PycharmProjects/fantasy-intel-agent`

To query the engine via the CLI, run with the virtualenv:

```bash
/Users/lukebenfer/PycharmProjects/fantasy-intel-agent/.venv/bin/fantasy-intel <command> [options]
```

### Available CLI Commands:

1. **Connect Sleeper Account**:
   ```bash
   /Users/lukebenfer/PycharmProjects/fantasy-intel-agent/.venv/bin/fantasy-intel connect --username <sleeper_username>
   ```

2. **View Your Leagues & Standings**:
   ```bash
   /Users/lukebenfer/PycharmProjects/fantasy-intel-agent/.venv/bin/fantasy-intel leagues
   ```

3. **View Your Active Team Roster & Starters**:
   ```bash
   /Users/lukebenfer/PycharmProjects/fantasy-intel-agent/.venv/bin/fantasy-intel roster
   ```

4. **Under-the-radar Waiver Sleepers**:
   ```bash
   /Users/lukebenfer/PycharmProjects/fantasy-intel-agent/.venv/bin/fantasy-intel waivers --season 2024 --limit 10
   ```
   *Options*:
   - `--week <num>`: Specific week
   - `--positions <pos>`: Filter positions (e.g. `RB`, `WR`, `TE`)
   - `--league-id <id>`: Filter out players already rostered in the user's Sleeper league

2. **Buy-Low / Sell-High Trades**:
   ```bash
   /Users/lukebenfer/PycharmProjects/fantasy-intel-agent/.venv/bin/fantasy-intel trades --season 2024
   ```

3. **Sidelined Starters & Next-Man-Up Beneficiaries**:
   ```bash
   /Users/lukebenfer/PycharmProjects/fantasy-intel-agent/.venv/bin/fantasy-intel injuries --positions RB,WR,TE
   ```
   *Options*:
   - `--team <team>`: Specific team (e.g. `DAL`, `SF`)
   - `--limit <num>`: Max items to display

4. **AI-Synthesized News & Depth Chart Impact Wire**:
   ```bash
   /Users/lukebenfer/PycharmProjects/fantasy-intel-agent/.venv/bin/fantasy-intel news --synthesize --limit 10
   ```
   *Options*:
   - `--query <player_or_team>`: Filter to specific player/team (e.g. `--query "McCaffrey"`)

5. **Live Trending Adds & Drops (Sleeper)**:
   ```bash
   /Users/lukebenfer/PycharmProjects/fantasy-intel-agent/.venv/bin/fantasy-intel trending --hours 24
   ```

6. **Current Sleeper Team, Opponent & Situational Advice**:
   ```bash
   /Users/lukebenfer/PycharmProjects/fantasy-intel-agent/.venv/bin/fantasy-intel matchup
   ```
   *Options*:
   - `--league-id <id>`: Specific Sleeper league
   - `--week <num>`: Specific week
   *Returns*: Starting lineup comparison, bench, critical injury alerts for active starters, swap recommendations, and tactical notes against the opponent.

7. **Discord Bot (Mobile Fantasy Intelligence)**:
   ```bash
   /Users/lukebenfer/PycharmProjects/fantasy-intel-agent/.venv/bin/fantasy-intel bot --token <token>
   ```

---

## Strategy & Reasoning Principles

When synthesizing advice for the user:
- **Prioritize Opportunity over Box Score**: Standard players chase last week's points. Recommend players whose *snap share*, *route participation*, and *Weighted Opportunity Rating (WOPR)* are surging before they explode in fantasy scoring.
- **Identify Regression**: 
  - If a player has high targets/touches but zero touchdowns, highlight them as an immediate **Buy-Low**.
  - If a player scored 2 TDs on only 4 touches and 20% snap share, highlight them as a prime **Sell-High**.
- **Contextualize with Depth Charts**: Cross-reference breaking injuries with team depth charts. When a starter is ruled Out/IR, identify the exact next-man-up backup who inherits high-leverage touches and prioritize them for immediate waiver acquisition.
