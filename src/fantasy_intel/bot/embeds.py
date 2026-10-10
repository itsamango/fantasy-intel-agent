"""Discord embed formatters for fantasy-intel."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import discord
import pandas as pd


def get_status_emoji(status: Optional[str]) -> str:
    """Returns visual emoji for injury status."""
    if not status:
        return "🟢"
    st = status.lower()
    if st in ["out", "ir", "pup"]:
        return "🚨"
    if st in ["doubtful", "questionable"]:
        return "⚠️"
    return "🟡"


def create_roster_embed(roster: Dict[str, Any], username: str) -> discord.Embed:
    """Formats full roster dashboard embed."""
    league_name = roster.get("league_name", "Fantasy League")
    record = roster.get("record", "N/A")
    pf = roster.get("points_for", 0.0)
    pa = roster.get("points_against", 0.0)
    waiver = roster.get("waiver_position", "N/A")

    embed = discord.Embed(
        title=f"🏈 {username}'s Roster Dashboard",
        description=f"**{league_name}** • Record: **{record}** | PF: **{pf}** | PA: **{pa}** | Waiver: **#{waiver}**",
        color=discord.Color.blue(),
    )

    # Starters
    starters_lines = []
    for p in roster.get("starters", []):
        emoji = get_status_emoji(p.get("injury_status"))
        status_text = f" ({p.get('injury_status')})" if p.get("injury_status") else ""
        team = p.get("team") or "FA"
        starters_lines.append(f"{emoji} **{p.get('position')}**: {p.get('name')} *({team})*{status_text}")

    embed.add_field(
        name="⭐ Starting Lineup",
        value="\n".join(starters_lines) if starters_lines else "None",
        inline=False,
    )

    # Bench & Reserves
    bench_lines = []
    for p in roster.get("bench", []):
        emoji = get_status_emoji(p.get("injury_status"))
        status_text = f" ({p.get('injury_status')})" if p.get("injury_status") else ""
        team = p.get("team") or "FA"
        bench_lines.append(f"{emoji} **{p.get('position')}**: {p.get('name')} *({team})*{status_text}")

    for p in roster.get("reserve", []):
        status_text = f" ({p.get('injury_status') or 'IR'})"
        team = p.get("team") or "FA"
        bench_lines.append(f"🚨 **IR**: {p.get('name')} *({team})*{status_text}")

    for p in roster.get("taxi", []):
        team = p.get("team") or "FA"
        bench_lines.append(f"🚕 **Taxi**: {p.get('name')} *({team})*")

    if bench_lines:
        embed.add_field(
            name="🪑 Bench & Reserves",
            value="\n".join(bench_lines),
            inline=False,
        )

    embed.set_footer(text="Fantasy Intel Agent • /roster")
    return embed


def create_matchup_embed(matchup_result: Dict[str, Any]) -> discord.Embed:
    """Formats weekly head-to-head matchup comparison and tactical advice."""
    overview = matchup_result.get("matchup_overview", {})
    advice = matchup_result.get("tactical_advice", {})

    my_team = overview.get("my_team", {})
    opp_team = overview.get("opponent_team")
    my_name = my_team.get("user_name", "My Team")
    opp_name = opp_team.get("user_name", "BYE") if opp_team else "BYE"
    week = overview.get("week", "Current")

    embed = discord.Embed(
        title=f"⚔️ Week {week} Matchup: {my_name} vs {opp_name}",
        description=f"Points Scored So Far: **{my_team.get('points', 0.0)}** vs **{opp_team.get('points', 0.0) if opp_team else 0.0}**",
        color=discord.Color.green(),
    )

    # Starters head to head
    my_starters = my_team.get("starters", [])
    opp_starters = opp_team.get("starters", []) if opp_team else []
    max_len = max(len(my_starters), len(opp_starters))

    lineup_rows = []
    for i in range(max_len):
        mp = my_starters[i] if i < len(my_starters) else {}
        op = opp_starters[i] if i < len(opp_starters) else {}

        my_str = f"{mp.get('name', '-')} ({mp.get('position', '')})" if mp else "-"
        if mp.get("injury_status"):
            my_str += f" 🚨{mp.get('injury_status')}"

        opp_str = f"{op.get('name', '-')} ({op.get('position', '')})" if op else "-"
        if op.get("injury_status"):
            opp_str += f" 🚨{op.get('injury_status')}"

        lineup_rows.append(f"`{i+1:02d}.` {my_str} 🆚 {opp_str}")

    # Chunk lineup rows if too long
    chunk_size = 10
    chunks = [lineup_rows[i:i + chunk_size] for i in range(0, len(lineup_rows), chunk_size)]
    for idx, c in enumerate(chunks):
        title = "📋 Starters Comparison" if idx == 0 else "📋 Starters (Cont.)"
        embed.add_field(name=title, value="\n".join(c), inline=False)

    # Alerts & Recommendations
    alerts = advice.get("injury_alerts", [])
    if alerts:
        alert_lines = [f"🚨 **{a['player']}**: {a['action']}" for a in alerts]
        embed.add_field(name="⚠️ Starting Lineup Warnings", value="\n".join(alert_lines), inline=False)

    suggestions = advice.get("start_sit_suggestions", [])
    if suggestions:
        sug_lines = [f"✅ **Swap**: Bench **{s['sit']}** ➔ Start **{s['start']}** ({s['reason']})" for s in suggestions]
        embed.add_field(name="💡 Recommended Substitutions", value="\n".join(sug_lines), inline=False)

    notes = advice.get("strategy_notes", [])
    if notes:
        embed.add_field(name="🎯 Tactical Strategy", value="\n".join(f"• {n}" for n in notes), inline=False)

    embed.set_footer(text="Fantasy Intel Agent • /matchup")
    return embed


def create_waivers_embed(sleepers: pd.DataFrame, league_name: Optional[str] = None) -> discord.Embed:
    """Formats under-the-radar waiver pickups into a Discord Embed."""
    desc = f"Top high-opportunity targets unowned in **{league_name or 'your league'}**"
    embed = discord.Embed(
        title="🎯 Under-The-Radar Waiver Sleepers",
        description=desc,
        color=discord.Color.gold(),
    )

    if sleepers.empty:
        embed.description = "No sleeper candidates found matching criteria."
        return embed

    for _, row in sleepers.iterrows():
        pname = row.get("player_display_name")
        pos = row.get("position")
        team = row.get("recent_team")
        snaps = f"{int(row.get('offense_pct', 0) * 100)}%"
        delta = row.get("snap_delta", 0)
        delta_str = f"+{int(delta * 100)}%" if delta > 0 else f"{int(delta * 100)}%"
        opp_idx = row.get("opportunity_index", 0.0)
        catalyst = row.get("catalyst", "Usage spike")
        fp_act = row.get("fantasy_points_ppr", 0.0)
        fp_exp = row.get("expected_fp_ppr", 0.0)

        field_title = f"{pname} ({team} {pos}) • Opp Index: {opp_idx}"
        field_val = (
            f"**Snaps**: {snaps} ({delta_str}) | **Actual / xFP**: {fp_act:.1f} / {fp_exp:.1f}\n"
            f"*{catalyst}*"
        )
        embed.add_field(name=field_title, value=field_val, inline=False)

    embed.set_footer(text="Fantasy Intel Agent • /waivers")
    return embed


def create_injuries_embed(injury_reports: List[Dict[str, Any]], limit: int = 10) -> discord.Embed:
    """Formats sidelined starters and promoted next-man-up beneficiaries."""
    embed = discord.Embed(
        title="🚑 Sidelined Starters & Next-Man-Up Beneficiaries",
        description=f"Showing top {min(len(injury_reports), limit)} depth chart promotions across the NFL.",
        color=discord.Color.red(),
    )

    if not injury_reports:
        embed.description = "No currently sidelined starting players detected."
        return embed

    for r in injury_reports[:limit]:
        starter = r["injured_starter"]
        beneficiary = r["promoted_beneficiary"]

        status = f"{starter.get('injury_status')} ({starter.get('injury_body_part') or 'Injury'})"
        b_name = beneficiary.get("name", "Committee / Rotation") if beneficiary else "Committee"
        b_role = f"{beneficiary.get('position')} (Depth {beneficiary.get('depth_order')})" if beneficiary else "N/A"

        embed.add_field(
            name=f"{starter['name']} ({r['team']} {r['position']}) — {status}",
            value=f"➡️ **Promoted Beneficiary**: **{b_name}** ({b_role})\n*Priority waiver target / immediate stash*",
            inline=False,
        )

    embed.set_footer(text="Fantasy Intel Agent • /injuries")
    return embed


def create_leagues_embed(username: str, leagues: List[Dict[str, Any]], active_id: Optional[str]) -> discord.Embed:
    """Formats user's leagues list."""
    embed = discord.Embed(
        title=f"🏆 Sleeper Leagues for {username}",
        description="Your active fantasy leagues and team performance:",
        color=discord.Color.purple(),
    )

    for lg in leagues:
        is_def = "⭐ **Active Default**" if str(lg.get("league_id")) == str(active_id) else ""
        val = (
            f"**ID**: `{lg.get('league_id')}`\n"
            f"**Teams**: {lg.get('total_teams')} | **Record**: **{lg.get('record')}** | **PF**: **{lg.get('points_for')}**\n"
            f"{is_def}"
        ).strip()
        embed.add_field(name=lg.get("name", "League"), value=val, inline=False)

    embed.set_footer(text="Fantasy Intel Agent • /leagues")
    return embed


def create_ai_advice_embed(query: str, advice: str, context_league: Optional[str] = None) -> discord.Embed:
    """Formats Gemini AI tactical advice."""
    embed = discord.Embed(
        title="🧠 AI Fantasy Advisor",
        description=advice[:4000],
        color=discord.Color.teal(),
    )
    footer_text = f"Context: {context_league}" if context_league else "Powered by Gemini & Fantasy Intel"
    embed.set_footer(text=footer_text)
    return embed
