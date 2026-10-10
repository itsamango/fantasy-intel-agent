"""Fantasy Intelligence Advisor orchestrating data, news, and league APIs."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd

from fantasy_intel.data.loader import DataLoader
from fantasy_intel.data.sleeper import SleeperClient
from fantasy_intel.analytics.waiver_wire import WaiverWireEngine
from fantasy_intel.analytics.trades import TradeAnalyzer
from fantasy_intel.analytics.depth_chart import DepthChartManager
from fantasy_intel.analytics.matchups import MatchupAdvisor
from fantasy_intel.news.scraper import NewsCollector
from fantasy_intel.news.parser import NewsContextParser


class FantasyAdvisor:
    """The central agent brain for fantasy recommendations."""

    def __init__(self, data_loader: Optional[DataLoader] = None, sleeper_client: Optional[SleeperClient] = None):
        self.loader = data_loader or DataLoader()
        self.sleeper = sleeper_client or SleeperClient()
        self.depth_manager = DepthChartManager(sleeper_client=self.sleeper)
        self.news_collector = NewsCollector(sleeper_client=self.sleeper)
        self.news_parser = NewsContextParser()
        self.matchup_advisor = MatchupAdvisor()

        # Cached analytics engines
        self._waiver_engine: Optional[WaiverWireEngine] = None
        self._trade_analyzer: Optional[TradeAnalyzer] = None
        self._loaded_seasons: List[int] = []

    def _ensure_engines(self, seasons: List[int]):
        if self._waiver_engine is None or self._loaded_seasons != seasons:
            season = seasons[0] if seasons else 2026
            if season >= 2025:
                dataset = self.loader.get_sleeper_season_data(season=season)
                if dataset.empty:
                    from fantasy_intel.config import DEFAULT_SEASON
                    dataset = self.loader.get_weekly_with_snaps(seasons=[DEFAULT_SEASON])
            else:
                dataset = self.loader.get_weekly_with_snaps(seasons=seasons)

            self._waiver_engine = WaiverWireEngine(dataset)
            self._trade_analyzer = TradeAnalyzer(dataset)
            self._loaded_seasons = seasons

    def get_waiver_sleepers(
        self,
        season: int = 2024,
        week: Optional[int] = None,
        positions: Optional[List[str]] = None,
        league_id: Optional[str] = None,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Finds under-the-radar waiver sleepers, optionally filtering out owned players in a Sleeper league."""
        self._ensure_engines([season])
        candidates = self._waiver_engine.scan_sleepers(
            week=week,
            season=season,
            positions=positions,
            limit=limit * 2 if league_id else limit,
        )

        if candidates.empty:
            return pd.DataFrame()

        # If Sleeper league provided, filter out already rostered players
        if league_id:
            rostered_ids = self.sleeper.get_rostered_player_ids(league_id)
            players_db = self.sleeper.get_all_players()
            # Map player names to sleeper IDs
            rostered_names = set()
            for pid in rostered_ids:
                pinfo = players_db.get(pid, {})
                fname = pinfo.get("full_name", "").lower()
                if fname:
                    rostered_names.add(fname)

            # Filter candidates
            candidates["is_rostered"] = candidates["player_display_name"].apply(
                lambda name: name.lower() in rostered_names
            )
            available = candidates[~candidates["is_rostered"]].head(limit)
            return available.reset_index(drop=True)

        return candidates.head(limit).reset_index(drop=True)

    def get_trade_recommendations(
        self,
        season: int = 2024,
        week: Optional[int] = None,
        positions: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Returns buy-low and sell-high regression trade candidates."""
        self._ensure_engines([season])
        return {
            "buy_low": self._trade_analyzer.get_buy_low_candidates(season=season, week=week, positions=positions),
            "sell_high": self._trade_analyzer.get_sell_high_candidates(season=season, week=week, positions=positions),
        }

    def get_trending_movers(self, lookback_hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """Returns national trending adds and drops across fantasy football."""
        adds = self.sleeper.get_trending_adds(lookback_hours=lookback_hours, limit=10)
        drops = self.sleeper.get_trending_drops(lookback_hours=lookback_hours, limit=10)
        return {"trending_adds": adds, "trending_drops": drops}

    def get_injury_wire(
        self,
        limit: int = 12,
        positions: Optional[List[str]] = None,
        query: Optional[str] = None,
        use_ai: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes real-time injury reports from RSS feeds and Sleeper notes,
        evaluates severity with LLM/AI, and identifies promoted next-man-up beneficiaries.
        """
        # Fetch RSS articles
        rss_articles = self.news_collector.fetch_latest_news(limit_per_feed=10)
        # Fetch Sleeper beat notes
        sleeper_notes = self.news_collector.fetch_sleeper_injury_notes(positions=positions)

        combined_items = rss_articles + sleeper_notes

        if query:
            q = query.lower()
            combined_items = [
                item for item in combined_items
                if q in item.get("title", "").lower() or q in item.get("summary", "").lower()
            ]

        if use_ai:
            return self.news_parser.analyze_with_ai(combined_items, limit=limit)

        return [self.news_parser.analyze_article(a) for a in combined_items[:limit]]

    def get_sidelined_starters_report(self) -> List[Dict[str, Any]]:
        """
        Returns all starter-tier players (depth_order == 1 or top WR) currently sidelined,
        paired with their promoted backup beneficiary.
        """
        return self.depth_manager.get_all_sidelined_starters()

    def get_player_intel(self, player_name: str, season: int = 2024) -> Dict[str, Any]:
        """Deep dive dossier on a specific player: stats trajectory, depth chart, and relevant news."""
        self._ensure_engines([season])
        df = self._waiver_engine.df

        # Filter player stats
        matches = df[df["player_display_name"].str.contains(player_name, case=False, na=False)]
        news = self.news_collector.fetch_latest_news(limit_per_feed=25)
        player_news = self.news_parser.filter_by_player(news, player_name)

        # Depth chart & Next man up
        depth_info = self.depth_manager.find_player(player_name)
        beneficiary = self.depth_manager.get_next_man_up(player_name) if depth_info else None

        # AI-analyzed news
        ai_analyzed_news = self.news_parser.analyze_with_ai(player_news, limit=5) if player_news else []

        return {
            "player_name": player_name,
            "depth_chart": depth_info,
            "next_man_up": beneficiary,
            "weekly_stats": matches.sort_values(by="week", ascending=False).to_dict(orient="records"),
            "news_updates": player_news,
            "ai_analyzed_news": ai_analyzed_news,
        }

    def get_situational_matchup_advice(
        self,
        username_or_id: str,
        league_id: Optional[str] = None,
        week: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fetches the user's active matchup and delivers tailored tactical advice:
        - Injury alerts for active starters
        - Recommended bench/starter swaps
        - Opponent strengths/vulnerabilities
        """
        # Resolve username to user_id
        user_info = self.sleeper.get_user(username_or_id)
        user_id = user_info.get("user_id") if user_info else username_or_id

        # If league_id not provided, find user's leagues
        if not league_id:
            leagues = self.sleeper.get_user_leagues(user_id)
            if not leagues:
                return {"error": f"No active leagues found for user '{username_or_id}'."}
            league_id = leagues[0].get("league_id")

        overview = self.sleeper.get_user_matchup_overview(league_id=league_id, user_id=user_id, week=week)
        if "error" in overview:
            return overview

        eval_result = self.matchup_advisor.evaluate_lineup(overview)
        return {
            "matchup_overview": overview,
            "tactical_advice": eval_result,
        }

    def get_user_leagues_overview(self, username_or_id: str, season: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch summary of user's active leagues and performance."""
        return self.sleeper.get_user_leagues_summary(username_or_id, season=season)

    def get_roster_overview(self, league_id: str, username_or_id: str) -> Dict[str, Any]:
        """Fetch full details on the user's active roster in a league."""
        return self.sleeper.get_user_roster_details(league_id, username_or_id)

    def get_drop_recommendations(
        self,
        league_id: str,
        username_or_id: Optional[str] = None,
        week: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the user's roster to determine safe drop candidates and critical holds.
        Accounts for:
        - NFL Bye weeks (prevents cutting benched studs)
        - Upstream depth chart injuries (prevents cutting elevated starters like Will Shipley)
        - Contingency upside (holds players with questionable teammates ahead like Dohnte Meyers)
        - Open IR slots (prompts stashing rather than cutting)
        """
        from fantasy_intel.config import load_user_config

        if not username_or_id:
            cfg = load_user_config()
            username_or_id = cfg.get("sleeper_username")

        if not username_or_id:
            return {"error": "No Sleeper username provided or configured."}

        user_info = self.sleeper.get_user(username_or_id)
        user_id = user_info.get("user_id") if user_info else username_or_id

        state = self.sleeper.get_nfl_state()
        season = int(state.get("season", 2026))
        active_week = week or int(state.get("week", 1))

        rosters = self.sleeper.get_league_rosters(league_id)
        user_roster = next((r for r in rosters if r.get("owner_id") == user_id), None)
        if not user_roster:
            return {"error": f"Could not find roster for user '{username_or_id}' in league {league_id}."}

        players_db = self.sleeper.get_all_players()
        stats = self.sleeper.get_season_stats(season=season)
        bye_teams = self.sleeper.get_bye_teams(season=season, week=active_week)
        league_info = self.sleeper.get_league_info(league_id)

        reserve_slots = int(league_info.get("settings", {}).get("reserve_slots", 0))
        reserve = set(user_roster.get("reserve") or [])
        starters = set(user_roster.get("starters") or [])
        all_pids = user_roster.get("players") or []
        open_ir = max(0, reserve_slots - len(reserve))

        evaluated_players = []
        alerts = []

        for pid in all_pids:
            p = players_db.get(str(pid), {})
            st = stats.get(str(pid), {})
            name = p.get("full_name") or f"Player {pid}"
            pos = p.get("position", "??")
            team = p.get("team", "FA")
            is_starter = pid in starters
            is_on_ir = pid in reserve
            on_bye = team in bye_teams

            if pos in ["K", "DEF"]:
                elevation = {
                    "tier": "SPECIAL_TEAMS",
                    "is_elevated": False,
                    "ir_eligible": False,
                    "summary": f"{team} {pos}",
                }
            else:
                elevation = self.depth_manager.get_player_elevation_context(pid)

            tier = elevation.get("tier")
            is_elevated = elevation.get("is_elevated", False)
            ir_eligible = elevation.get("ir_eligible", False)
            gp = float(st.get("gp") or 0)
            pts = float(st.get("pts_ppr") or 0)
            ppg = round(pts / gp, 1) if gp > 0 else 0.0
            snaps = float(st.get("off_snp", 0.0) or 0.0)
            tm_snaps = float(st.get("tm_off_snp", 0.0) or 0.0)
            snap_pct = round(snaps / tm_snaps, 2) if tm_snaps > 0 else (round(snaps / (gp * 65.0), 2) if gp > 0 else 0.0)

            cut_score = 50 if not is_starter else 15
            tag = ""
            recommendation = "HOLD"

            if ir_eligible and not is_on_ir:
                cut_score = 10
                tag = f"IR ELIGIBLE: Ruled {p.get('injury_status')} ({p.get('injury_body_part') or 'Injured'}). Move directly to open IR slot instead of cutting!"
                recommendation = "MOVE_TO_IR"
                alerts.append(f"🚑 **{name}** can be moved to an open IR slot without dropping anyone.")
            elif on_bye and (ppg >= 14.0 or (pos == "QB" and ppg >= 16.0)):
                cut_score = 0
                tag = "PROTECTED: Franchise starter on Bye Week - DO NOT DROP."
                recommendation = "HOLD (BYE)"
                alerts.append(f"🛡️ **{name}** is on Bye Week (Week {active_week}) — benched for schedule, not skill.")
            elif (tier == "ELEVATED_STARTER" or is_elevated) and is_starter:
                cut_score = 5
                tag = f"CRITICAL HOLD: Elevated starter this week ({elevation.get('summary')})"
                recommendation = "START/HOLD"
                alerts.append(f"⚡ **{name}** is elevated to starter this week ({elevation.get('summary')}).")
            elif (tier == "ELEVATED_STARTER" or is_elevated) and not is_starter:
                # Elevated on paper, but benched. Detect if genuine workhorse vs timeshare committee back
                if snap_pct < 0.45 and ppg < 8.0:
                    cut_score = 40
                    tag = f"Timeshare committee back ({int(snap_pct*100)}% snaps, {ppg:.1f} PPG in timeshare). High cut viability for upside."
                    recommendation = "DROP_CANDIDATE"
                else:
                    cut_score = 15
                    tag = f"High-volume elevated backup ({elevation.get('summary')})"
                    recommendation = "HOLD_HANDCUFF"
            elif tier == "POTENTIAL_SURGE":
                cut_score = 25
                tag = f"CONTINGENCY HOLD: Starters ahead are Questionable ({elevation.get('summary')})"
                recommendation = "HOLD_CONTINGENCY"
            elif tier == "ENTRENCHED_STARTER":
                if is_starter:
                    cut_score = 5
                    tag = f"Starting regular for {team}"
                    recommendation = "HOLD"
                else:
                    if ppg < 7.0 and snap_pct < 0.50:
                        cut_score = 35
                        tag = f"Low-producing bench starter ({int(snap_pct*100)}% snaps, {ppg:.1f} PPG)"
                        recommendation = "DROP_CANDIDATE"
                    else:
                        cut_score = 15
                        tag = f"Core starter on bench ({ppg:.1f} PPG)"
                        recommendation = "HOLD"
            elif tier == "BENCH_BACKUP":
                if ppg < 7.0 and snap_pct < 0.50:
                    cut_score = 35
                    tag = f"Low-volume backup ({elevation.get('summary')})"
                    recommendation = "DROP_CANDIDATE"
                else:
                    cut_score = 25
                    tag = f"Backup ({elevation.get('summary')})"
                    recommendation = "HOLD"

            evaluated_players.append({
                "player_id": str(pid),
                "player_name": name,
                "position": pos,
                "team": team,
                "ppg": ppg,
                "is_starter": is_starter,
                "is_on_ir": is_on_ir,
                "on_bye": on_bye,
                "recommendation": recommendation,
                "cut_safety_score": cut_score,
                "situation": elevation.get("summary", ""),
                "tag": tag,
            })

        evaluated_players.sort(key=lambda x: x["cut_safety_score"], reverse=True)

        return {
            "league_name": league_info.get("name", "Sleeper League"),
            "week": active_week,
            "season": season,
            "bye_teams": sorted(list(bye_teams)),
            "open_ir_slots": open_ir,
            "total_ir_slots": reserve_slots,
            "alerts": alerts,
            "roster_evaluations": evaluated_players,
        }


