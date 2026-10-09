---
name: fantasy-advisor
description: >-
  Expert Fantasy Football Intelligence Agent. Analyzes multi-year historical NFL data,
  snap counts, target share trends, expected fantasy points (xFP), and live news/injury
  feeds (ESPN, Sleeper) to provide under-the-radar waiver wire sleepers and buy-low/sell-high trades.
---

# Fantasy Football Intelligence Skill

Use this skill when the user asks for:
- Waiver wire pickups, sleepers, or breakout candidates
- Trade advice (buy-low / sell-high candidates)
- Injury replacement recommendations and depth chart surges
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

1. **Under-the-radar Waiver Sleepers**:
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

3. **Live Trending Adds & Drops (Sleeper)**:
   ```bash
   /Users/lukebenfer/PycharmProjects/fantasy-intel-agent/.venv/bin/fantasy-intel trending --hours 24
   ```

4. **Live Breaking News & Injury Wire (ESPN / Beat Feeds)**:
   ```bash
   /Users/lukebenfer/PycharmProjects/fantasy-intel-agent/.venv/bin/fantasy-intel news --query "Barkley"
   ```

---

## Strategy & Reasoning Principles

When synthesizing advice for the user:
- **Prioritize Opportunity over Box Score**: Standard players chase last week's points. Recommend players whose *snap share*, *route participation*, and *Weighted Opportunity Rating (WOPR)* are surging before they explode in fantasy scoring.
- **Identify Regression**: 
  - If a player has high targets/touches but zero touchdowns, highlight them as an immediate **Buy-Low**.
  - If a player scored 2 TDs on only 4 touches and 20% snap share, highlight them as a prime **Sell-High**.
- **Contextualize with News**: Cross-reference trending adds with breaking injuries (e.g., if a lead back is ruled out on ESPN, immediately identify the direct backup benefiting from vacant touches).
