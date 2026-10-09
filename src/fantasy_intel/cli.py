"""Command-line interface for fantasy-intel using Rich console output."""

from __future__ import annotations
import argparse
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from fantasy_intel.agent.advisor import FantasyAdvisor
from fantasy_intel.config import DEFAULT_SEASON
from fantasy_intel.news.scraper import NewsCollector

console = Console()


def render_waivers(advisor: FantasyAdvisor, args):
    """Render waiver sleepers table."""
    with console.status("[bold green]Analyzing snap trajectories and opportunity metrics..."):
        positions = [p.strip().upper() for p in args.positions.split(",")] if args.positions else None
        sleepers = advisor.get_waiver_sleepers(
            season=args.season,
            week=args.week,
            positions=positions,
            league_id=args.league_id,
            limit=args.limit,
        )

    if sleepers.empty:
        console.print("[yellow]No sleeper candidates found matching criteria.[/yellow]")
        return

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
    table.add_column("Actual / xFP", justify="right")
    table.add_column("Opp Index", justify="right", style="bold green")
    table.add_column("Catalyst & Context", style="dim")

    for _, row in sleepers.iterrows():
        snap_pct = f"{int(row.get('offense_pct', 0) * 100)}%"
        delta_val = row.get("snap_delta", 0)
        snap_delta = f"+{int(delta_val * 100)}%" if delta_val > 0 else f"{int(delta_val * 100)}%"
        wopr_val = f"{row.get('wopr', 0):.2f}" if row.get("wopr", 0) > 0 else f"{int(row.get('targets', 0))} trg"
        fp_str = f"{row.get('fantasy_points_ppr', 0):.1f} / {row.get('expected_fp_ppr', 0):.1f}"

        table.add_row(
            str(row.get("player_display_name")),
            str(row.get("position")),
            str(row.get("recent_team")),
            snap_pct,
            snap_delta,
            wopr_val,
            fp_str,
            str(row.get("opportunity_index")),
            str(row.get("catalyst")),
        )

    console.print(table)


def render_trades(advisor: FantasyAdvisor, args):
    """Render buy-low and sell-high regression candidates."""
    with console.status("[bold green]Calculating Expected Fantasy Points (xFP) variance..."):
        trades = advisor.get_trade_recommendations(
            season=args.season,
            week=args.week,
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


def render_news(args):
    """Render live news from ESPN and beat reporters."""
    with console.status("[bold green]Parsing ESPN and NFL beat reporter feeds..."):
        collector = NewsCollector()
        articles = collector.fetch_latest_news()

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


def main():
    parser = argparse.ArgumentParser(description="Fantasy Football Intelligence Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Waivers command
    waivers_p = subparsers.add_parser("waivers", help="Find under-the-radar waiver wire sleepers")
    waivers_p.add_argument("--season", type=int, default=DEFAULT_SEASON, help="NFL Season (default: 2024)")
    waivers_p.add_argument("--week", type=int, default=None, help="NFL Week number (default: latest)")
    waivers_p.add_argument("--positions", type=str, default=None, help="Comma-separated positions: WR,RB,TE")
    waivers_p.add_argument("--league-id", type=str, default=None, help="Sleeper league ID to filter out owned players")
    waivers_p.add_argument("--limit", type=int, default=10, help="Max candidates to display")

    # Trades command
    trades_p = subparsers.add_parser("trades", help="Find buy-low and sell-high regression trade targets")
    trades_p.add_argument("--season", type=int, default=DEFAULT_SEASON, help="NFL Season (default: 2024)")
    trades_p.add_argument("--week", type=int, default=None, help="NFL Week number")

    # Trending command
    trending_p = subparsers.add_parser("trending", help="View Sleeper nationwide trending adds and drops")
    trending_p.add_argument("--hours", type=int, default=24, help="Lookback hours (default: 24)")

    # News command
    news_p = subparsers.add_parser("news", help="Fetch breaking NFL and ESPN news")
    news_p.add_argument("--query", type=str, default=None, help="Filter news by player or team name")
    news_p.add_argument("--limit", type=int, default=15, help="Number of headlines to display")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    advisor = FantasyAdvisor()

    if args.command == "waivers":
        render_waivers(advisor, args)
    elif args.command == "trades":
        render_trades(advisor, args)
    elif args.command == "trending":
        render_trending(advisor, args)
    elif args.command == "news":
        render_news(args)


if __name__ == "__main__":
    main()
