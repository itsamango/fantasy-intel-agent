"""Discord Bot client for Fantasy Football Intelligence Agent."""

from __future__ import annotations
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
import requests

from fantasy_intel.agent.advisor import FantasyAdvisor
from fantasy_intel.config import load_user_config, save_user_config
from fantasy_intel.bot.embeds import (
    create_roster_embed,
    create_matchup_embed,
    create_waivers_embed,
    create_injuries_embed,
    create_leagues_embed,
    create_ai_advice_embed,
)

logger = logging.getLogger(__name__)

# Load .env if present
load_dotenv()


def call_gemini_chat(prompt: str, context: str) -> str:
    """Queries Gemini API for conversational fantasy advice with context."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        return (
            "⚠️ **Gemini API Key not set** (set `GEMINI_API_KEY` in environment or .env).\n\n"
            f"**Based on current roster context:**\n{context[:1500]}"
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
    full_prompt = (
        "You are an expert, highly strategic NFL Fantasy Football Assistant for Discord.\n"
        "Here is the user's current live fantasy team, matchup, and injury context:\n"
        f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"User Question: {prompt}\n\n"
        "Provide direct, concise, high-impact tactical advice. Highlight names and bold recommendations."
    )

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800},
    }

    try:
        resp = requests.post(url, json=payload, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        logger.warning(f"Gemini API request failed: {e}")

    return f"Unable to reach Gemini. Here is your team summary:\n{context[:1500]}"


def create_bot(advisor: Optional[FantasyAdvisor] = None) -> commands.Bot:
    """Initializes and returns the Discord Bot instance."""
    advisor = advisor or FantasyAdvisor()
    intents = discord.Intents.default()
    intents.message_content = True

    bot = commands.Bot(command_prefix="!", intents=intents)

    def get_effective_context(league_id: Optional[str] = None) -> tuple[str, str, int]:
        cfg = load_user_config()
        username = cfg.get("sleeper_username") or "lockedoutagain"
        lid = league_id or cfg.get("default_league_id") or "1389361071038402560"
        season = int(cfg.get("season") or 2026)
        return username, lid, season

    @bot.event
    async def on_ready():
        logger.info(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
        try:
            synced = await bot.tree.sync()
            logger.info(f"Synced {len(synced)} slash commands globally.")
        except Exception as e:
            logger.error(f"Failed to sync slash commands: {e}")

        # Set rich presence
        await bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="NFL Matchups | /roster | /matchup",
            )
        )
        print(f"🤖 Fantasy Intel Bot is online as {bot.user.name}! Ready on Discord.")

    # 1. /roster command
    @bot.tree.command(name="roster", description="Inspect your full team roster, starters, and injuries")
    @app_commands.describe(league_id="Optional Sleeper league ID")
    async def slash_roster(interaction: discord.Interaction, league_id: Optional[str] = None):
        await interaction.response.defer()
        username, lid, _ = get_effective_context(league_id)

        try:
            roster = advisor.get_roster_overview(lid, username)
            if "error" in roster:
                await interaction.followup.send(f"❌ {roster['error']}")
                return

            embed = create_roster_embed(roster, username)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            await interaction.followup.send(f"❌ Error fetching roster: {e}")

    # 2. /matchup command
    @bot.tree.command(name="matchup", description="View head-to-head weekly matchup and tactical start/sit advice")
    @app_commands.describe(league_id="Optional Sleeper league ID", week="Optional NFL Week number")
    async def slash_matchup(interaction: discord.Interaction, league_id: Optional[str] = None, week: Optional[int] = None):
        await interaction.response.defer()
        username, lid, _ = get_effective_context(league_id)

        try:
            result = advisor.get_situational_matchup_advice(username_or_id=username, league_id=lid, week=week)
            if "error" in result:
                await interaction.followup.send(f"❌ {result['error']}")
                return

            embed = create_matchup_embed(result)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            await interaction.followup.send(f"❌ Error evaluating matchup: {e}")

    # 3. /waivers command
    @bot.tree.command(name="waivers", description="Discover under-the-radar waiver sleepers available in your league")
    @app_commands.describe(positions="Filter positions (e.g. WR,RB,TE)", limit="Max candidates (default: 5)")
    async def slash_waivers(interaction: discord.Interaction, positions: Optional[str] = None, limit: int = 5):
        await interaction.response.defer()
        _, lid, season = get_effective_context()

        try:
            pos_list = [p.strip().upper() for p in positions.split(",")] if positions else None
            stats_season = season if season <= 2024 else 2024
            sleepers = advisor.get_waiver_sleepers(
                season=stats_season,
                positions=pos_list,
                league_id=lid,
                limit=limit,
            )
            league_info = advisor.sleeper.get_league_info(lid)
            lname = league_info.get("name") if league_info else "your league"
            embed = create_waivers_embed(sleepers, league_name=lname)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            await interaction.followup.send(f"❌ Error scanning waivers: {e}")

    # 4. /injuries command
    @bot.tree.command(name="injuries", description="View sidelined NFL starters and promoted next-man-up beneficiaries")
    @app_commands.describe(positions="Filter positions (e.g. RB,WR,TE)", limit="Max items (default: 8)")
    async def slash_injuries(interaction: discord.Interaction, positions: Optional[str] = None, limit: int = 8):
        await interaction.response.defer()

        try:
            reports = advisor.get_sidelined_starters_report()
            if positions:
                pos_list = [p.strip().upper() for p in positions.split(",")]
                reports = [r for r in reports if r["position"] in pos_list]

            embed = create_injuries_embed(reports, limit=limit)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            await interaction.followup.send(f"❌ Error fetching injuries: {e}")

    # 5. /leagues command
    @bot.tree.command(name="leagues", description="View all your active Sleeper leagues and standings")
    async def slash_leagues(interaction: discord.Interaction):
        await interaction.response.defer()
        username, active_lid, season = get_effective_context()

        try:
            leagues = advisor.get_user_leagues_overview(username, season=season)
            embed = create_leagues_embed(username, leagues, active_lid)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            await interaction.followup.send(f"❌ Error fetching leagues: {e}")

    # 6. /switch_league command
    @bot.tree.command(name="switch_league", description="Switch your active default Sleeper league")
    @app_commands.describe(league_id="Sleeper League ID to set as default")
    async def slash_switch(interaction: discord.Interaction, league_id: str):
        save_user_config({"default_league_id": league_id})
        await interaction.response.send_message(f"✅ Active default league updated to: `{league_id}`")

    # 7. /ask command (Gemini AI Fantasy Advisor)
    @bot.tree.command(name="ask", description="Ask Gemini AI any question about your fantasy lineup, trades, or start/sits")
    @app_commands.describe(question="Your fantasy football question")
    async def slash_ask(interaction: discord.Interaction, question: str):
        await interaction.response.defer()
        username, lid, _ = get_effective_context()

        # Build context
        roster_info = advisor.get_roster_overview(lid, username)
        context_str = f"User: {username}\nLeague: {roster_info.get('league_name')}\nRecord: {roster_info.get('record')}\n"
        starters = [f"{p['position']} {p['name']} ({p.get('injury_status') or 'Healthy'})" for p in roster_info.get("starters", [])]
        bench = [f"{p['position']} {p['name']} ({p.get('injury_status') or 'Healthy'})" for p in roster_info.get("bench", [])]
        context_str += f"Starters: {', '.join(starters)}\nBench: {', '.join(bench)}\n"

        advice = call_gemini_chat(question, context_str)
        embed = create_ai_advice_embed(question, advice, context_league=roster_info.get("league_name"))
        await interaction.followup.send(embed=embed)

    # Message listener for direct mentions and DMs
    @bot.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return

        # Respond if bot is mentioned or in DM
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mention = bot.user in message.mentions if bot.user else False

        if is_dm or is_mention:
            clean_query = message.content.replace(f"<@{bot.user.id}>", "").strip() if bot.user else message.content
            if not clean_query:
                clean_query = "Who should I start this week?"

            async with message.channel.typing():
                username, lid, _ = get_effective_context()
                roster_info = advisor.get_roster_overview(lid, username)
                context_str = f"User: {username}\nLeague: {roster_info.get('league_name')}\nRecord: {roster_info.get('record')}\n"
                starters = [f"{p['position']} {p['name']} ({p.get('injury_status') or 'Healthy'})" for p in roster_info.get("starters", [])]
                bench = [f"{p['position']} {p['name']} ({p.get('injury_status') or 'Healthy'})" for p in roster_info.get("bench", [])]
                context_str += f"Starters: {', '.join(starters)}\nBench: {', '.join(bench)}\n"

                advice = call_gemini_chat(clean_query, context_str)
                embed = create_ai_advice_embed(clean_query, advice, context_league=roster_info.get("league_name"))
                await message.reply(embed=embed)

        await bot.process_commands(message)

    return bot


def run_bot(token: Optional[str] = None):
    """Starts the Discord bot using token from argument, env, or saved config."""
    cfg = load_user_config()
    bot_token = token or os.environ.get("DISCORD_BOT_TOKEN") or cfg.get("discord_bot_token")

    if not bot_token:
        print("\n❌ [bold red]DISCORD BOT TOKEN NOT FOUND[/bold red]")
        print("\nTo start your Discord Bot:")
        print("1. Go to https://discord.com/developers/applications")
        print("2. Click 'New Application' ➔ Name it 'Fantasy Intel' ➔ Go to 'Bot' tab.")
        print("3. Reset/Copy Token, and enable 'Message Content Intent'.")
        print("4. Run either:")
        print("   fantasy-intel bot --token <YOUR_DISCORD_BOT_TOKEN>")
        print("   or set: export DISCORD_BOT_TOKEN=<YOUR_DISCORD_BOT_TOKEN>\n")
        return

    bot = create_bot()
    bot.run(bot_token)
