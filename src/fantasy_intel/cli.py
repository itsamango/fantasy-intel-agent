"""Command-line interface for fantasy-intel using Rich console output."""

from __future__ import annotations
import argparse
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from fantasy_intel.agent.advisor import FantasyAdvisor
from fantasy_intel.config import DEFAULT_SEASON, load_user_config, save_user_config
from fantasy_intel.news.scraper import NewsCollector

console = Console()


def _fmt_int(val) -> str:
    try:
        return str(int(val))
    except (TypeError, ValueError):
        return "0"


def render_waivers(advisor: FantasyAdvisor, args):
    """Render waiver sleepers table."""
    config = load_user_config()
    league_id = args.league_id or config.get("default_league_id")
    with console.status("[bold green]Analyzing snap trajectories and opportunity metrics..."):
        positions = [p.strip().upper() for p in args.positions.split(",")] if args.positions else None
        sleepers = advisor.get_waiver_sleepers(
            season=args.season,
            week=args.week,
            positions=positions,
            league_id=league_id,
            limit=args.limit,
        )

    if sleepers.empty:
        console.print("[yellow]No sleeper candidates found matching criteria.[/yellow]")
        return

    has_faab = "recommended_faab_bid" in sleepers.columns and sleepers["recommended_faab_bid"].notna().any()
    has_priority = "priority_claim_recommended" in sleepers.columns and not has_faab

    table = Table(
        title=f"🎯 Under-The-Radar Waiver Sleepers ({args.season} Week {args.week or 'Latest'})",
        box=box.ROUNDED,
        header_style="bold cyan",
    )

    table.add_column("Player", style="bold white")
    table.add_column("Pos", style="magenta")
    table.add_column("Team", style="blue")
    table.add_column("Snap %", justify="right")
    table.add_column("Snap Δ", justify="right")
    table.add_column("WOPR / Trg", justify="right")
    table.add_column("HVT", justify="right")
    table.add_column("RACR", justify="right")
    table.add_column("Opp Index", justify="right", style="bold green")
    if has_faab:
        table.add_column("Rec FAAB Bid", justify="right", style="bold green")
    elif has_priority:
        table.add_column("Priority Claim?", justify="center", style="bold green")
    table.add_column("Availability", justify="center")
    table.add_column("Category", style="bold")
    table.add_column("Health / Role", style="yellow")
    table.add_column("Strategic Analysis", style="dim")

    for _, row in sleepers.iterrows():
        snap_pct = f"{int(row.get('offense_pct', 0) * 100)}%"
        delta_val = row.get("snap_delta", 0)
        snap_delta = f"+{int(delta_val * 100)}%" if delta_val > 0 else f"{int(delta_val * 100)}%"
        wopr_val = f"{row.get('wopr', 0):.2f}" if row.get("wopr", 0) > 0 else f"{int(row.get('targets', 0))} trg"
        hvt_str = _fmt_int(row.get("hvt", 0))
        racr_val = row.get("racr")
        racr_str = f"{racr_val:.2f}" if isinstance(racr_val, (int, float)) and racr_val == racr_val else "—"

        if has_faab:
            bid_val = row.get("recommended_faab_bid")
            if bid_val:
                pct_val = row.get("faab_pct_of_remaining") or 0.0
                bid_cell = f"[bold green]${_fmt_int(bid_val)}[/bold green] ({pct_val * 100:.0f}%)"
            else:
                bid_cell = "[dim]$0[/dim]"
        elif has_priority:
            bid_cell = "[bold green]YES[/bold green]" if row.get("priority_claim_recommended") else "[dim]no[/dim]"
        else:
            bid_cell = None

        inj = row.get("injury_status", "Healthy")
        role = row.get("depth_role", "")
        ir_tag = " [bold magenta](IR Stash)[/bold magenta]" if row.get("ir_eligible") else ""
        health_role = f"{inj} [{role}]{ir_tag}"

        avail = row.get("availability", "FREE AGENT")
        avail_str = "[bold green]FREE AGENT[/bold green]" if avail == "FREE AGENT" else "[bold yellow]WAIVER CLAIM[/bold yellow]"
        
        cat = row.get("add_category", "STANDARD_ADD")
        if cat == "HIGH_PRIORITY_ADD":
            cat_style = "[bold green]PRIORITY ADD[/bold green]"
        elif cat == "CONTINGENCY_ADD":
            cat_style = "[bold yellow]CONTINGENCY[/bold yellow]"
        elif cat == "IR_STASH_ONLY":
            cat_style = "[bold magenta]IR STASH[/bold magenta]"
        else:
            cat_style = "[cyan]STANDARD[/cyan]"

        note = str(row.get("strategic_note") or row.get("catalyst"))

        cells = [
            str(row.get("player_display_name")),
            str(row.get("position")),
            str(row.get("recent_team")),
            snap_pct,
            snap_delta,
            wopr_val,
            hvt_str,
            racr_str,
            str(row.get("opportunity_index")),
        ]
        if bid_cell is not None:
            cells.append(bid_cell)
        cells.extend([avail_str, cat_style, health_role, note])
        table.add_row(*cells)

    console.print(table)


def render_trades(advisor: FantasyAdvisor, args):
    """Render buy-low and sell-high regression candidates plus bilateral trade proposals."""
    config = load_user_config()
    username = args.username or config.get("sleeper_username")
    league_id = args.league_id or config.get("default_league_id")

    with console.status("[bold green]Calculating Expected Fantasy Points (xFP) variance..."):
        trades = advisor.get_trade_recommendations(
            season=args.season,
            week=args.week,
            league_id=league_id,
            username=username,
        )

    # Buy Low Table
    buy_low = trades["buy_low"]
    buy_table = Table(
        title=f"📉 Top BUY-LOW Candidates (High Volume, Suppressed Output)",
        box=box.ROUNDED,
        header_style="bold green",
    )
    buy_table.add_column("Player", style="bold white")
    buy_table.add_column("Pos", style="magenta")
    buy_table.add_column("Team", style="blue")
    buy_table.add_column("Snap %", justify="right")
    buy_table.add_column("Touches", justify="right")
    buy_table.add_column("Actual PPR", justify="right", style="red")
    buy_table.add_column("Expected xFP", justify="right", style="green")
    buy_table.add_column("Deficit", justify="right", style="bold red")
    buy_table.add_column("Analysis", style="dim")

    for _, r in buy_low.iterrows():
        buy_table.add_row(
            str(r.get("player_display_name")),
            str(r.get("position")),
            str(r.get("recent_team")),
            f"{int(r.get('offense_pct', 0) * 100)}%",
            str(int(r.get("touches", 0))),
            f"{r.get('fantasy_points_ppr', 0):.1f}",
            f"{r.get('expected_fp_ppr', 0):.1f}",
            f"{r.get('fp_diff', 0):.1f}",
            str(r.get("rationale")),
        )

    console.print(buy_table)
    console.print()

    # Sell High Table
    sell_high = trades["sell_high"]
    sell_table = Table(
        title=f"📈 Top SELL-HIGH Candidates (Spiked Output on Fragile Volume)",
        box=box.ROUNDED,
        header_style="bold red",
    )
    sell_table.add_column("Player", style="bold white")
    sell_table.add_column("Pos", style="magenta")
    sell_table.add_column("Team", style="blue")
    sell_table.add_column("Snap %", justify="right")
    sell_table.add_column("Touches", justify="right")
    sell_table.add_column("Actual PPR", justify="right", style="green")
    sell_table.add_column("Expected xFP", justify="right", style="red")
    sell_table.add_column("Surplus", justify="right", style="bold green")
    sell_table.add_column("Analysis", style="dim")

    for _, r in sell_high.iterrows():
        sell_table.add_row(
            str(r.get("player_display_name")),
            str(r.get("position")),
            str(r.get("recent_team")),
            f"{int(r.get('offense_pct', 0) * 100)}%",
            str(int(r.get("touches", 0))),
            f"{r.get('fantasy_points_ppr', 0):.1f}",
            f"{r.get('expected_fp_ppr', 0):.1f}",
            f"+{r.get('fp_diff', 0):.1f}",
            str(r.get("rationale")),
        )

    console.print(sell_table)

    proposals_bundle = trades.get("trade_proposals")
    if proposals_bundle is None:
        if league_id and not username:
            console.print(
                "[yellow]Bilateral trade proposals need a Sleeper username: "
                "pass --username or run [bold cyan]fantasy-intel connect --username <user>[/bold cyan].[/yellow]"
            )
        return

    cards = proposals_bundle.get("proposals") or []
    if not cards:
        console.print("[yellow]No mutually beneficial bilateral trade proposals found for this league.[/yellow]")
        return

    def player_cell(card):
        text = f"{card['name']} ({card['position']})"
        if card.get("tag"):
            text += f", {card['tag']}"
        return text

    trade_table = Table(
        title="🤝 Personalized Bilateral Trade Proposals",
        box=box.ROUNDED,
        header_style="bold green",
    )
    trade_table.add_column("Send", style="bold white")
    trade_table.add_column("Receive", style="bold cyan")
    trade_table.add_column("Target", style="magenta")
    trade_table.add_column("Your Gain", justify="right", style="bold green")
    trade_table.add_column("Their Gain", justify="right", style="green")
    trade_table.add_column("P(accept)", justify="right", style="bold yellow")
    trade_table.add_column("Pitch", style="dim")

    for c in cards:
        trade_table.add_row(
            " + ".join(player_cell(p) for p in c["send"]),
            " + ".join(player_cell(p) for p in c["receive"]),
            str(c["target_manager"]),
            f"+{c['your_lineup_gain']:.1f}",
            f"+{c['their_lineup_gain']:.1f}",
            f"{c['p_accept'] * 100:.0f}%",
            str(c["pitch"]),
        )

    console.print(trade_table)


def render_trending(advisor: FantasyAdvisor, args):
    """Render national trending adds and drops."""
    with console.status("[bold green]Fetching live Sleeper activity..."):
        trending = advisor.get_trending_movers(lookback_hours=args.hours)

    adds = trending["trending_adds"]
    drops = trending["trending_drops"]

    table = Table(title=f"🔥 Sleeper Trending Moves (Past {args.hours} Hours)", box=box.ROUNDED)
    table.add_column("Rank", justify="center")
    table.add_column("Trending Adds (Player)", style="green")
    table.add_column("Position", style="cyan")
    table.add_column("Team", style="blue")
    table.add_column("Trending Drops (Player)", style="red")

    max_rows = max(len(adds), len(drops))
    for i in range(max_rows):
        add_item = adds[i] if i < len(adds) else {}
        drop_item = drops[i] if i < len(drops) else {}
        table.add_row(
            str(i + 1),
            str(add_item.get("full_name") or add_item.get("player_id", "")),
            str(add_item.get("position", "")),
            str(add_item.get("team", "")),
            str(drop_item.get("full_name") or drop_item.get("player_id", "")),
        )

    console.print(table)


def render_news(advisor: FantasyAdvisor, args):
    """Render live news and AI-analyzed injury/depth-chart impact."""
    if args.synthesize or args.ai:
        with console.status("[bold green]Synthesizing news with AI and mapping depth chart promotions..."):
            items = advisor.get_injury_wire(
                limit=args.limit,
                query=args.query,
                use_ai=True,
            )

        if not items:
            console.print("[yellow]No relevant news found to synthesize.[/yellow]")
            return

        console.print(Panel(
            f"Analyzed {len(items)} breaking updates with depth chart cross-referencing.",
            title="🧠 AI NFL Injury & Depth Chart Impact Wire",
            border_style="cyan",
        ))

        for item in items:
            urgency = item.get("waiver_urgency", "MODERATE")
            border_color = "red" if "CRITICAL" in urgency else ("yellow" if "HIGH" in urgency else "blue")

            lines = [
                f"[bold white]{item.get('player_name')} ({item.get('team')} {item.get('position')})[/bold white] "
                f"— Status: [bold red]{item.get('status')}[/bold red] ({item.get('injury_details')})",
                f"[cyan]• Expected Timeline:[/cyan] {item.get('expected_missed_time')}",
            ]

            if item.get("promoted_beneficiary"):
                lines.append(f"[bold green]• Promoted Beneficiary (Next Man Up):[/bold green] [bold underline]{item.get('promoted_beneficiary')}[/bold underline]")

            lines.append(f"[magenta]• Opportunity Shift:[/magenta] {item.get('opportunity_impact')}")
            lines.append(f"[bold yellow]• Urgency Rating:[/bold yellow] [{border_color}]{urgency}[/{border_color}]")
            lines.append(f"[bold green]• Actionable Advice:[/bold green] {item.get('fantasy_action')}")

            console.print(Panel("\n".join(lines), title=f"🚨 {item.get('title')[:60]}", border_style=border_color))
        return

    # Standard tabular view
    with console.status("[bold green]Parsing ESPN and NFL beat reporter feeds..."):
        articles = advisor.news_collector.fetch_latest_news()

    if args.query:
        q = args.query.lower()
        articles = [a for a in articles if q in a["title"].lower() or q in a["summary"].lower()]

    table = Table(title=f"📰 NFL Breaking News & Injury Wire ({len(articles)} updates)", box=box.ROUNDED)
    table.add_column("Source", style="cyan")
    table.add_column("Headline", style="bold white")
    table.add_column("Summary", style="dim")

    for a in articles[:args.limit]:
        table.add_row(
            a["source"],
            a["title"],
            a["summary"][:120] + ("..." if len(a["summary"]) > 120 else ""),
        )

    console.print(table)


def render_injuries(advisor: FantasyAdvisor, args):
    """Render all sidelined starting players alongside their next-man-up beneficiaries."""
    with console.status("[bold green]Scanning all 32 NFL depth charts for sidelined starters..."):
        reports = advisor.get_sidelined_starters_report()

    if args.positions:
        target_positions = [p.strip().upper() for p in args.positions.split(",")]
        reports = [r for r in reports if r["position"] in target_positions]

    if args.team:
        t = args.team.strip().upper()
        reports = [r for r in reports if r["team"] == t]

    if not reports:
        console.print("[yellow]No sidelined starters found matching criteria.[/yellow]")
        return

    table = Table(
        title=f"🚑 NFL Sidelined Starters & Next-Man-Up Beneficiaries ({len(reports)} detected)",
        box=box.ROUNDED,
        header_style="bold red",
    )
    table.add_column("Team", style="blue")
    table.add_column("Pos", style="cyan")
    table.add_column("Sidelined Starter", style="bold white")
    table.add_column("Status / Injury", style="red")
    table.add_column("Promoted Beneficiary", style="bold green")
    table.add_column("Beneficiary Pos / Depth", style="magenta")
    table.add_column("Action / Waiver Advice", style="dim")

    for r in reports[:args.limit]:
        starter = r["injured_starter"]
        beneficiary = r["promoted_beneficiary"]

        status_str = f"{starter.get('injury_status')} ({starter.get('injury_body_part') or 'Injury'})"
        b_name = beneficiary.get("name", "Committee / Unknown") if beneficiary else "Committee"
        b_pos = f"{beneficiary.get('position', '')} (Depth {beneficiary.get('depth_order', '')})" if beneficiary else "N/A"

        advice = f"Priority add: {b_name} steps into starting role" if beneficiary else "Monitor rotation"

        table.add_row(
            r["team"],
            r["position"],
            starter["name"],
            status_str,
            b_name,
            b_pos,
            advice,
        )

    console.print(table)



def render_connect(advisor: FantasyAdvisor, args):
    """Connect Sleeper account and save username/league to local config."""
    username = args.username.strip()
    with console.status(f"[bold green]Validating Sleeper username '{username}'..."):
        user_info = advisor.sleeper.get_user(username)

    if not user_info:
        console.print(f"[bold red]Error:[/bold red] Sleeper username '{username}' was not found. Please double check the spelling.")
        return

    user_id = user_info.get("user_id")
    display_name = user_info.get("display_name", username)

    # Fetch user's leagues for the current/selected season
    with console.status("[bold green]Fetching your Sleeper leagues..."):
        leagues = advisor.sleeper.get_user_leagues_summary(user_id, season=args.season)

    default_lid = args.league_id
    if not default_lid and leagues:
        default_lid = leagues[0]["league_id"]

    saved = save_user_config({
        "sleeper_username": display_name,
        "default_league_id": default_lid,
        "season": args.season,
    })

    console.print(Panel(
        f"✅ [bold green]Successfully connected to Sleeper![/bold green]\n\n"
        f"• User: [bold white]{display_name}[/bold white] (ID: {user_id})\n"
        f"• Default League: [cyan]{default_lid or 'None'}[/cyan]\n"
        f"• Season: [cyan]{args.season}[/cyan]\n"
        f"• Saved to: [dim]~/.fantasy_intel/config.json[/dim]",
        title="🔗 Sleeper Integration Connected",
        border_style="green",
    ))

    if leagues:
        table = Table(title=f"📋 Your Active Sleeper Leagues ({len(leagues)})", box=box.ROUNDED)
        table.add_column("League Name", style="bold white")
        table.add_column("League ID", style="cyan")
        table.add_column("Teams", justify="center")
        table.add_column("Record", justify="center", style="bold green")
        table.add_column("Points For", justify="right", style="magenta")
        table.add_column("Default", justify="center")

        for lg in leagues:
            is_def = "⭐ Active" if str(lg["league_id"]) == str(default_lid) else ""
            table.add_row(
                lg["name"],
                str(lg["league_id"]),
                str(lg["total_teams"]),
                lg["record"],
                str(lg["points_for"]),
                is_def,
            )
        console.print(table)


def render_leagues(advisor: FantasyAdvisor, args):
    """List all leagues for the connected or specified Sleeper user."""
    config = load_user_config()
    username = args.username or config.get("sleeper_username")
    if not username:
        console.print("[bold red]No Sleeper username provided or saved.[/bold red]")
        console.print("Run [bold cyan]fantasy-intel connect --username <your_username>[/bold cyan] first.")
        return

    with console.status(f"[bold green]Fetching leagues for '{username}'..."):
        leagues = advisor.get_user_leagues_overview(username, season=args.season)

    if not leagues:
        console.print(f"[yellow]No leagues found for '{username}' in season {args.season}.[/yellow]")
        return

    default_lid = config.get("default_league_id")
    table = Table(title=f"🏆 Sleeper Leagues for '{username}' ({args.season})", box=box.ROUNDED)
    table.add_column("League Name", style="bold white")
    table.add_column("League ID", style="cyan")
    table.add_column("Teams", justify="center")
    table.add_column("Record", justify="center", style="bold green")
    table.add_column("Points For", justify="right", style="magenta")
    table.add_column("Default", justify="center")

    for lg in leagues:
        is_def = "⭐ Active" if str(lg["league_id"]) == str(default_lid) else ""
        table.add_row(
            lg["name"],
            str(lg["league_id"]),
            str(lg["total_teams"]),
            lg["record"],
            str(lg["points_for"]),
            is_def,
        )

    console.print(table)


def render_roster(advisor: FantasyAdvisor, args):
    """Inspect detailed roster breakdown, starters, bench, and player injury status."""
    config = load_user_config()
    username = args.username or config.get("sleeper_username")
    league_id = args.league_id or config.get("default_league_id")

    if not username:
        console.print("[bold red]No Sleeper username provided or saved.[/bold red]")
        console.print("Run [bold cyan]fantasy-intel connect --username <your_username>[/bold cyan] first.")
        return

    if not league_id:
        # Try fetching user leagues to pick the first one
        leagues = advisor.get_user_leagues_overview(username, season=args.season)
        if leagues:
            league_id = leagues[0]["league_id"]
        else:
            console.print("[bold red]No Sleeper league ID found.[/bold red]")
            return

    with console.status(f"[bold green]Fetching roster for '{username}' in league {league_id}..."):
        roster = advisor.get_roster_overview(league_id, username)

    if "error" in roster:
        console.print(f"[bold red]Error:[/bold red] {roster['error']}")
        return

    console.print(Panel(
        f"[bold white]{roster['league_name']}[/bold white] — Team Record: [bold green]{roster['record']}[/bold green] "
        f"| PF: [bold magenta]{roster['points_for']}[/bold magenta] | PA: [bold red]{roster['points_against']}[/bold red] "
        f"| Waiver Priority: #{roster['waiver_position']}",
        title=f"🏈 {username}'s Roster Dashboard",
        border_style="cyan",
    ))

    # Starters Table
    s_table = Table(title="⭐ Starting Lineup", box=box.ROUNDED, header_style="bold green")
    s_table.add_column("Player", style="bold white")
    s_table.add_column("Pos", style="cyan")
    s_table.add_column("Team", style="blue")
    s_table.add_column("Depth", justify="center")
    s_table.add_column("Status / Injury", style="red")

    for p in roster.get("starters", []):
        status = f"[bold red]{p.get('injury_status')}[/bold red] ({p.get('injury_body_part') or 'Injury'})" if p.get("injury_status") else "[green]Active[/green]"
        s_table.add_row(
            p.get("name", ""),
            p.get("position", ""),
            p.get("team", ""),
            str(p.get("depth_order") or "-"),
            status,
        )
    console.print(s_table)

    # Bench Table
    b_table = Table(title="🪑 Bench & Reserves", box=box.ROUNDED, header_style="bold yellow")
    b_table.add_column("Player", style="bold white")
    b_table.add_column("Pos", style="cyan")
    b_table.add_column("Team", style="blue")
    b_table.add_column("Role / Slot", justify="center")
    b_table.add_column("Status / Injury", style="red")

    for p in roster.get("bench", []):
        status = f"[bold red]{p.get('injury_status')}[/bold red]" if p.get("injury_status") else "[green]Active[/green]"
        b_table.add_row(p.get("name", ""), p.get("position", ""), p.get("team", ""), "Bench", status)

    for p in roster.get("reserve", []):
        status = f"[bold red]{p.get('injury_status') or 'IR'}[/bold red]"
        b_table.add_row(p.get("name", ""), p.get("position", ""), p.get("team", ""), "[red]IR[/red]", status)

    for p in roster.get("taxi", []):
        status = f"[yellow]{p.get('injury_status')}[/yellow]" if p.get("injury_status") else "[green]Active[/green]"
        b_table.add_row(p.get("name", ""), p.get("position", ""), p.get("team", ""), "[magenta]Taxi[/magenta]", status)

    console.print(b_table)


def render_matchup(advisor: FantasyAdvisor, args):
    """Render head-to-head matchup, starting lineups, injury alerts, and tactical advice."""
    config = load_user_config()
    username = args.username or config.get("sleeper_username")
    league_id = args.league_id or config.get("default_league_id")

    if not username:
        console.print("[bold red]No Sleeper username provided or saved.[/bold red]")
        console.print("Run [bold cyan]fantasy-intel connect --username <your_username>[/bold cyan] or provide --username.")
        return

    with console.status(f"[bold green]Fetching Sleeper matchup for '{username}'..."):
        result = advisor.get_situational_matchup_advice(
            username_or_id=username,
            league_id=league_id,
            week=args.week,
        )

    if "error" in result:
        console.print(f"[bold red]Error:[/bold red] {result['error']}")
        return

    overview = result["matchup_overview"]
    advice = result["tactical_advice"]
    my_team = overview["my_team"]
    opp_team = overview.get("opponent_team")

    console.print(Panel(
        f"[bold white]{my_team['user_name']}[/bold white] vs [bold white]{opp_team['user_name'] if opp_team else 'BYE'}[/bold white] "
        f"— [cyan]NFL Week {overview['week']}[/cyan] (League ID: {overview['league_id']})",
        title="⚔️ Head-to-Head Sleeper Matchup",
        border_style="cyan",
    ))

    # Starters Table
    table = Table(title="📋 Starting Lineups Comparison", box=box.ROUNDED)
    table.add_column("My Starters", style="bold white")
    table.add_column("Pos", style="cyan")
    table.add_column("Status", style="red")
    table.add_column("Opponent Starters", style="bold white")
    table.add_column("Pos", style="magenta")
    table.add_column("Status", style="red")

    my_starters = my_team.get("starters", [])
    opp_starters = opp_team.get("starters", []) if opp_team else []
    max_len = max(len(my_starters), len(opp_starters))

    for i in range(max_len):
        my_p = my_starters[i] if i < len(my_starters) else {}
        opp_p = opp_starters[i] if i < len(opp_starters) else {}

        my_status = f"[bold red]{my_p.get('injury_status')}[/bold red]" if my_p.get("injury_status") in ["Out", "IR"] else (
            f"[yellow]{my_p.get('injury_status')}[/yellow]" if my_p.get("injury_status") else "[green]Active[/green]"
        )
        opp_status = f"[bold red]{opp_p.get('injury_status')}[/bold red]" if opp_p.get("injury_status") in ["Out", "IR"] else (
            f"[yellow]{opp_p.get('injury_status')}[/yellow]" if opp_p.get("injury_status") else "[green]Active[/green]"
        )

        table.add_row(
            f"{my_p.get('name', '')} ({my_p.get('team', '')})",
            str(my_p.get("position", "")),
            my_status,
            f"{opp_p.get('name', '')} ({opp_p.get('team', '')})",
            str(opp_p.get("position", "")),
            opp_status,
        )

    console.print(table)

    # Injury & Lineup Alerts
    alerts = advice.get("injury_alerts", [])
    if alerts:
        alert_lines = []
        for a in alerts:
            color = "red" if a["severity"] == "CRITICAL" else "yellow"
            alert_lines.append(f"[{color}]• [{a['severity']}] {a['action']}[/{color}]")
        console.print(Panel("\n".join(alert_lines), title="🚨 Starting Lineup Alerts", border_style="red"))

    # Context-Adjusted Projections & Matchup Posture
    projections = advice.get("player_projections") or []
    user_projs = [p for p in projections if p.get("side") == "user"]
    has_projections = any(float(p.get("median") or 0) > 0 for p in user_projs)

    if has_projections:
        posture = advice.get("posture")
        delta_median = advice.get("delta_median") or 0.0
        weights = advice.get("objective_weights") or {}
        optimizer = advice.get("optimizer") or {}

        posture_style = {
            "FAVORITE": ("bold green", "FAVORITE (protect the floor)"),
            "UNDERDOG": ("bold red", "UNDERDOG (chase the ceiling)"),
            "NEUTRAL": ("bold yellow", "NEUTRAL (balanced)"),
        }.get(posture, ("bold yellow", str(posture or "UNKNOWN")))

        posture_lines = [
            f"Matchup Posture: [{posture_style[0]}]{posture_style[1]}[/{posture_style[0]}] — "
            f"ΔM [cyan]{delta_median:+.1f}[/cyan] "
            f"(you [green]{optimizer.get('user_median_total', 0):.1f}[/green] vs opp "
            f"[red]{optimizer.get('opponent_median_total', 0):.1f}[/red] projected median)",
            f"Objective: {weights.get('floor', 0):.2f}×Floor + {weights.get('median', 0):.2f}×Median + "
            f"{weights.get('ceiling', 0):.2f}×Ceiling",
        ]
        if optimizer.get("method"):
            lineup = ", ".join(optimizer.get("optimal_lineup") or [])
            posture_lines.append(f"Optimizer: {optimizer['method'].upper()} over league slots — optimal: {lineup}")
        console.print(Panel("\n".join(posture_lines), title="⚖️ Matchup Posture & Optimizer", border_style="cyan"))

        starter_names = {p.get("name") for p in my_starters}
        proj_table = Table(
            title="📊 Context-Adjusted Projections (Vegas ITT · Wind · Defensive Funnel)",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        proj_table.add_column("Player", style="bold white")
        proj_table.add_column("Pos", style="magenta")
        proj_table.add_column("Slot", justify="center")
        proj_table.add_column("Floor", justify="right", style="green")
        proj_table.add_column("Median", justify="right", style="bold cyan")
        proj_table.add_column("Ceiling", justify="right", style="yellow")
        proj_table.add_column("Mult", justify="right")
        proj_table.add_column("Vegas / Weather Note", style="dim")

        for p in user_projs:
            slot = "[green]START[/green]" if p.get("name") in starter_names else "[dim]BENCH[/dim]"
            status = p.get("status")
            name = str(p.get("name") or "")
            if status:
                name += f" [red]({status})[/red]"

            notes = " · ".join(str(n) for n in (p.get("notes") or []))
            if "wind" in notes:
                notes = f"[yellow]{notes} ⚠[/yellow]"
            elif notes:
                notes = f"[dim]{notes}[/dim]"

            proj_table.add_row(
                name,
                str(p.get("position") or ""),
                slot,
                f"{float(p.get('floor') or 0):.1f}",
                f"{float(p.get('median') or 0):.1f}",
                f"{float(p.get('ceiling') or 0):.1f}",
                f"x{float(p.get('multiplier') or 1.0):.2f}",
                notes,
            )

        console.print(proj_table)

    # Tactical Substitutions & Strategy
    suggestions = advice.get("start_sit_suggestions", [])
    notes = advice.get("strategy_notes", [])
    if suggestions or notes:
        rec_lines = []
        for s in suggestions:
            gain = s.get("projected_gain")
            gain_str = f" ({gain:+.1f} adj pts)" if isinstance(gain, (int, float)) else ""
            reason = s.get("reason") or ""
            rec_lines.append(
                f"[bold green]• SWAP SUGGESTION:[/bold green] Bench [red]{s['sit']}[/red] ➔ "
                f"Start [green]{s['start']}[/green]{gain_str} — {reason}"
            )
        for n in notes:
            rec_lines.append(f"[cyan]• TACTICAL NOTE:[/cyan] {n}")
        console.print(Panel("\n".join(rec_lines), title="💡 Optimized Start/Sit Swaps & Strategy", border_style="green"))


def render_drops(advisor: FantasyAdvisor, args):
    """Render comprehensive drop candidates and roster safety evaluations."""
    config = load_user_config()
    league_id = args.league_id or config.get("default_league_id")
    username = args.username or config.get("sleeper_username")

    if not league_id:
        console.print("[red]No league ID provided. Run `fantasy-intel connect` or pass `--league-id`.[/red]")
        return

    with console.status("[bold green]Analyzing roster, depth charts, upstream injuries, and bye weeks..."):
        res = advisor.get_drop_recommendations(league_id=league_id, username_or_id=username, week=args.week)

    if "error" in res:
        console.print(f"[red]{res['error']}[/red]")
        return

    league_name = res.get("league_name", "League")
    week = res.get("week", 1)
    open_ir = res.get("open_ir_slots", 0)
    alerts = res.get("alerts", [])
    evals = res.get("roster_evaluations", [])

    console.print(Panel(
        f"[bold white]{league_name}[/bold white] — Week {week} Roster Audit\n"
        f"[cyan]Open IR Slots:[/cyan] [bold green]{open_ir}[/bold green] | "
        f"[cyan]Teams on Bye:[/cyan] {', '.join(res.get('bye_teams', [])) or 'None'}",
        title="🛡️ Drop Decision & Roster Protection Engine",
        border_style="cyan",
    ))

    if alerts:
        alert_lines = [f"• {a}" for a in alerts]
        console.print(Panel("\n".join(alert_lines), title="🚨 Situational & Injury Alerts", border_style="yellow"))

    table = Table(
        title="📋 Roster Cut Safety & Priority Rankings",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Rec", style="bold")
    table.add_column("Cut Score", justify="right")
    table.add_column("Player", style="bold white")
    table.add_column("Pos", style="cyan")
    table.add_column("Team", style="blue")
    table.add_column("PPG", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Role & Tactical Analysis", style="dim")

    for p in evals:
        score = p["cut_safety_score"]
        rec = p["recommendation"]
        if score >= 35:
            rec_style = "[bold red]DROP CANDIDATE[/bold red]"
        elif score >= 25:
            rec_style = "[bold yellow]CONTINGENCY[/bold yellow]"
        elif rec == "MOVE_TO_IR":
            rec_style = "[bold magenta]MOVE TO IR[/bold magenta]"
        elif rec == "HOLD (BYE)":
            rec_style = "[bold blue]HOLD (BYE)[/bold blue]"
        else:
            rec_style = "[bold green]HOLD/START[/bold green]"

        status_badge = "STARTER" if p["is_starter"] else ("IR" if p["is_on_ir"] else ("BYE" if p["on_bye"] else "BENCH"))

        table.add_row(
            rec_style,
            str(score),
            p["player_name"],
            p["position"],
            p["team"],
            f"{p['ppg']:.1f}",
            status_badge,
            p["tag"] or p["situation"],
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Fantasy Football Intelligence Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Waivers command
    waivers_p = subparsers.add_parser("waivers", help="Find under-the-radar waiver wire sleepers")
    waivers_p.add_argument("--season", type=int, default=DEFAULT_SEASON, help="NFL Season (default: 2026)")
    waivers_p.add_argument("--week", type=int, default=None, help="NFL Week number (default: latest)")
    waivers_p.add_argument("--positions", type=str, default=None, help="Comma-separated positions: WR,RB,TE")
    waivers_p.add_argument("--league-id", type=str, default=None, help="Sleeper league ID to filter out owned players")
    waivers_p.add_argument("--limit", type=int, default=10, help="Max candidates to display")

    # Trades command
    trades_p = subparsers.add_parser("trades", help="Find buy-low and sell-high regression trade targets")
    trades_p.add_argument("--season", type=int, default=DEFAULT_SEASON, help="NFL Season (default: 2026)")
    trades_p.add_argument("--week", type=int, default=None, help="NFL Week number")
    trades_p.add_argument("--username", type=str, default=None, help="Your Sleeper username or user_id (defaults to saved)")
    trades_p.add_argument("--league-id", type=str, default=None, help="Sleeper league ID for bilateral trade proposals (defaults to saved)")

    # Trending command
    trending_p = subparsers.add_parser("trending", help="View Sleeper nationwide trending adds and drops")
    trending_p.add_argument("--hours", type=int, default=24, help="Lookback hours (default: 24)")

    # News command
    news_p = subparsers.add_parser("news", help="Fetch breaking NFL and ESPN news with AI depth-chart impact analysis")
    news_p.add_argument("--query", type=str, default=None, help="Filter news by player or team name")
    news_p.add_argument("--limit", type=int, default=10, help="Number of headlines or items to display")
    news_p.add_argument("--synthesize", "--ai", action="store_true", dest="synthesize", help="Synthesize news with AI and map depth chart promotions")

    # Injuries / Next-Man-Up command
    injuries_p = subparsers.add_parser("injuries", help="View sidelined starters and promoted next-man-up beneficiaries")
    injuries_p.add_argument("--positions", type=str, default=None, help="Comma-separated positions: RB,WR,TE,QB")
    injuries_p.add_argument("--team", type=str, default=None, help="Filter by NFL team abbreviation (e.g. DAL, PHI, KC)")
    injuries_p.add_argument("--limit", type=int, default=20, help="Max entries to display")

    # Connect / Config command
    connect_p = subparsers.add_parser("connect", aliases=["config"], help="Connect your Sleeper account and save defaults")
    connect_p.add_argument("--username", type=str, required=True, help="Your Sleeper username")
    connect_p.add_argument("--league-id", type=str, default=None, help="Default Sleeper league ID")
    connect_p.add_argument("--season", type=int, default=DEFAULT_SEASON, help="NFL Season (default: 2026)")

    # Leagues command
    leagues_p = subparsers.add_parser("leagues", help="List all your Sleeper leagues, records, and points")
    leagues_p.add_argument("--username", type=str, default=None, help="Optional Sleeper username (defaults to saved)")
    leagues_p.add_argument("--season", type=int, default=DEFAULT_SEASON, help="NFL Season (default: 2026)")

    # Roster command
    roster_p = subparsers.add_parser("roster", help="View your full roster, starters vs bench, and player injury status")
    roster_p.add_argument("--username", type=str, default=None, help="Optional Sleeper username (defaults to saved)")
    roster_p.add_argument("--league-id", type=str, default=None, help="Optional Sleeper league ID (defaults to active)")
    roster_p.add_argument("--season", type=int, default=DEFAULT_SEASON, help="NFL Season (default: 2026)")

    # Matchup command
    matchup_p = subparsers.add_parser("matchup", help="Inspect your current Sleeper team, opponent, and get tactical advice")
    matchup_p.add_argument("--username", type=str, default=None, help="Your Sleeper username or user_id (defaults to saved)")
    matchup_p.add_argument("--league-id", type=str, default=None, help="Optional Sleeper league ID")
    matchup_p.add_argument("--week", type=int, default=None, help="Optional NFL Week number")

    # Drops / Roster safety command
    drops_p = subparsers.add_parser("drops", help="Audit your roster to identify cut candidates and critical holds")
    drops_p.add_argument("--username", type=str, default=None, help="Optional Sleeper username (defaults to saved)")
    drops_p.add_argument("--league-id", type=str, default=None, help="Optional Sleeper league ID")
    drops_p.add_argument("--week", type=int, default=None, help="Optional NFL Week number")

    # Bot command
    bot_p = subparsers.add_parser("bot", help="Run the Discord bot for mobile fantasy intelligence")
    bot_p.add_argument("--token", type=str, default=None, help="Discord Bot Token (or set DISCORD_BOT_TOKEN)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "bot":
        from fantasy_intel.bot import run_bot
        run_bot(token=args.token)
        return

    advisor = FantasyAdvisor()

    if args.command in ["connect", "config"]:
        render_connect(advisor, args)
    elif args.command == "leagues":
        render_leagues(advisor, args)
    elif args.command == "roster":
        render_roster(advisor, args)
    elif args.command == "drops":
        render_drops(advisor, args)
    elif args.command == "waivers":
        render_waivers(advisor, args)
    elif args.command == "trades":
        render_trades(advisor, args)
    elif args.command == "trending":
        render_trending(advisor, args)
    elif args.command == "news":
        render_news(advisor, args)
    elif args.command == "injuries":
        render_injuries(advisor, args)
    elif args.command == "matchup":
        render_matchup(advisor, args)


if __name__ == "__main__":
    main()
