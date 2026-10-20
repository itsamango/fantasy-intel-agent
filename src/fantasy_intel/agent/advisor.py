"""Fantasy Intelligence Advisor orchestrating data, news, and league APIs."""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from fantasy_intel.data.loader import DataLoader
from fantasy_intel.data.sleeper import SleeperClient
from fantasy_intel.data.weather import get_game_wind_forecast
from fantasy_intel.analytics.metrics import _normalize_ids, add_projection_distributions, attach_pbp_features
from fantasy_intel.analytics.waiver_wire import WaiverWireEngine, recommend_faab_bids
from fantasy_intel.analytics.trades import TradeAnalyzer, generate_league_trade_proposals
from fantasy_intel.analytics.depth_chart import DepthChartManager
from fantasy_intel.analytics.matchups import MatchupAdvisor, compute_defensive_epa_rates
from fantasy_intel.config import DEFAULT_SEASON, SCORING_PPR, load_user_config
from fantasy_intel.news.scraper import NewsCollector
from fantasy_intel.news.parser import NewsContextParser

logger = logging.getLogger(__name__)


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
        self._loaded_scoring: Optional[str] = None

    def _ensure_engines(self, seasons: List[int], scoring: str = SCORING_PPR):
        if (
            self._waiver_engine is None
            or self._loaded_seasons != seasons
            or self._loaded_scoring != scoring
        ):
            season = seasons[0] if seasons else 2026
            if season >= 2025:
                attached_seasons = [season]
                dataset = self._nflverse_weekly_if_available(season)
                if dataset is None:
                    dataset = self.loader.get_sleeper_season_data(season=season)
                if dataset.empty:
                    dataset = self.loader.get_weekly_with_snaps(seasons=[DEFAULT_SEASON])
                    attached_seasons = [DEFAULT_SEASON]
            else:
                dataset = self.loader.get_weekly_with_snaps(seasons=seasons)
                attached_seasons = list(seasons)

            # Enrich with PBP opportunity features so trajectories carry
            # hvt / unrealized_air_yards / situational xFP (rolling_xfp_3w)
            dataset = attach_pbp_features(dataset, self.loader, seasons=attached_seasons)

            self._waiver_engine = WaiverWireEngine(dataset, scoring=scoring)
            self._trade_analyzer = TradeAnalyzer(dataset)
            self._loaded_seasons = seasons
            self._loaded_scoring = scoring

    def _nflverse_weekly_if_available(self, season: int) -> Optional[pd.DataFrame]:
        """nflverse weekly frame for the season when published, else None.

        Preferred over the Sleeper-derived frame for current seasons because
        weekly granularity gives real rolling xFP / snap deltas plus native
        racr / target_share / air_yards_share; the Sleeper path stays the
        fallback when nflverse has not published that season yet.
        """
        try:
            weekly = self.loader.get_weekly_with_snaps(seasons=[season])
        except Exception as e:
            logger.warning(f"nflverse weekly data unavailable for season {season}: {e}")
            return None
        if weekly.empty or "season" not in weekly.columns:
            return None
        if not (weekly["season"] == season).any():
            return None
        return weekly

    def _resolve_sleeper_ids(self, player_ids: pd.Series) -> pd.Series:
        """Maps a weekly frame's player ids (gsis or sleeper keyed) to Sleeper ids."""
        ids_df = self.loader.get_player_ids()
        normalized = _normalize_ids(player_ids)
        if (
            ids_df.empty
            or "sleeper_id" not in ids_df.columns
            or "gsis_id" not in ids_df.columns
        ):
            return normalized
        gsis_to_sleeper = dict(zip(ids_df["gsis_id"], ids_df["sleeper_id"]))
        frame_ids = set(normalized.dropna())
        sleeper_overlap = len(frame_ids & set(ids_df["sleeper_id"].dropna()))
        gsis_overlap = len(frame_ids & set(gsis_to_sleeper))
        if sleeper_overlap >= gsis_overlap:
            return normalized
        return normalized.map(gsis_to_sleeper)

    def _latest_sleeper_xfp(self) -> Dict[str, float]:
        """Latest-week rolling 3-week xFP per player, keyed by Sleeper id."""
        df = self._waiver_engine.df
        if (
            df is None
            or df.empty
            or "rolling_xfp_3w" not in df.columns
            or "week" not in df.columns
        ):
            return {}
        key = "player_id" if "player_id" in df.columns else "player_display_name"
        sort_cols = [c for c in ("season", "week") if c in df.columns]
        latest = df.sort_values(sort_cols, kind="stable").groupby(key, sort=False).tail(1)
        sleeper_ids = self._resolve_sleeper_ids(latest[key])
        return {
            str(sid): float(xfp)
            for sid, xfp in zip(sleeper_ids, latest["rolling_xfp_3w"])
            if sid is not None and pd.notna(xfp)
        }

    def _build_user_roster(
        self,
        user_roster_record: Dict[str, Any],
        players_db: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Builds the user-roster frame consumed by recommend_faab_bids."""
        starter_ids = {str(p) for p in (user_roster_record.get("starters") or []) if p}
        xfp_by_sleeper_id = self._latest_sleeper_xfp()

        rows = []
        for pid in (user_roster_record.get("players") or []):
            pid = str(pid)
            p = players_db.get(pid, {})
            rows.append({
                "player_id": pid,
                "player_display_name": p.get("full_name") or f"Player {pid}",
                "position": p.get("position"),
                "rolling_xfp_3w": xfp_by_sleeper_id.get(pid),
                "is_starter": pid in starter_ids,
                "injury_status": p.get("injury_status"),
                "owner_id": user_roster_record.get("owner_id"),
                "roster_id": user_roster_record.get("roster_id"),
                "waiver_position": (user_roster_record.get("settings") or {}).get("waiver_position"),
            })
        return pd.DataFrame(rows)

    def _attach_faab_recommendations(
        self,
        candidates: pd.DataFrame,
        league_context: Dict[str, Any],
        username_or_id: Optional[str],
        rosters: List[Dict[str, Any]],
        players_db: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Adds FAAB bid / priority-claim recommendations based on the user's roster context."""
        if not username_or_id:
            username_or_id = load_user_config().get("sleeper_username")
        if not username_or_id:
            return candidates

        user_info = self.sleeper.get_user(username_or_id)
        user_id = user_info.get("user_id") if user_info else username_or_id
        user_roster_record = next(
            (r for r in rosters or [] if str(r.get("owner_id")) == str(user_id)), None
        )
        if not user_roster_record:
            return candidates

        user_roster = self._build_user_roster(user_roster_record, players_db)

        faab_history = None
        if league_context.get("waiver_type") != 0:
            faab_history = self.sleeper.get_league_faab_history(league_id=league_context.get("league_id"))

        return recommend_faab_bids(candidates, user_roster, league_context, faab_history)

    def get_waiver_sleepers(
        self,
        season: int = 2024,
        week: Optional[int] = None,
        positions: Optional[List[str]] = None,
        league_id: Optional[str] = None,
        limit: int = 10,
        username_or_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Finds under-the-radar waiver sleepers, optionally filtering out owned players in a Sleeper league."""
        if not league_id:
            league_id = load_user_config().get("default_league_id")

        scoring = SCORING_PPR
        league_context: Dict[str, Any] = {}
        if league_id:
            league_context = self.sleeper.get_league_context(league_id) or {}
            league_scoring = league_context.get("scoring_type")
            if league_scoring:
                scoring = league_scoring

        self._ensure_engines([season], scoring=scoring)
        candidates = self._waiver_engine.scan_sleepers(
            week=week,
            season=season,
            positions=positions,
            limit=limit * 2 if league_id else limit,
        )

        if candidates.empty:
            return pd.DataFrame()

        if not league_id:
            candidates["availability"] = "FREE AGENT"
            return candidates.head(limit).reset_index(drop=True)

        state = self.sleeper.get_nfl_state()
        active_week = week or int(state.get("week", 1))
        rosters = self.sleeper.get_league_rosters(league_id)
        rostered_ids = {str(p) for r in rosters or [] for p in (r.get("players") or [])}
        dropped_ids = self.sleeper.get_recent_dropped_player_ids(league_id, active_week)
        players_db = self.sleeper.get_all_players()

        # Name fallback for players the ID crosswalk cannot map
        rostered_names = set()
        dropped_names = set()
        for pid in rostered_ids:
            fname = players_db.get(pid, {}).get("full_name", "").lower()
            if fname:
                rostered_names.add(fname)
        for pid in dropped_ids:
            fname = players_db.get(pid, {}).get("full_name", "").lower()
            if fname:
                dropped_names.add(fname)

        if "player_id" in candidates.columns:
            candidate_sleeper_ids = self._resolve_sleeper_ids(candidates["player_id"])
            candidates["sleeper_id"] = candidate_sleeper_ids
            id_rostered = candidate_sleeper_ids.isin(rostered_ids)
            id_dropped = candidate_sleeper_ids.isin(dropped_ids)
        else:
            id_rostered = pd.Series(False, index=candidates.index)
            id_dropped = pd.Series(False, index=candidates.index)

        name_rostered = candidates["player_display_name"].str.lower().isin(rostered_names)
        name_dropped = candidates["player_display_name"].str.lower().isin(dropped_names)

        # Filter candidates via sleeper_id (name fallback covers unmapped players)
        candidates["is_rostered"] = id_rostered | name_rostered
        candidates["is_waiver_hold"] = id_dropped | name_dropped
        candidates["availability"] = candidates["is_waiver_hold"].apply(
            lambda hold: "WAIVER CLAIM" if hold else "FREE AGENT"
        )
        available = candidates[~candidates["is_rostered"]].copy()

        if not available.empty and league_context:
            available = self._attach_faab_recommendations(
                available, league_context, username_or_id, rosters, players_db
            )

        return available.head(limit).reset_index(drop=True)

    def get_trade_recommendations(
        self,
        season: int = 2024,
        week: Optional[int] = None,
        positions: Optional[List[str]] = None,
        league_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Returns buy-low and sell-high regression trade candidates.

        When a Sleeper league and username are resolvable (params or saved
        user config), also runs the bilateral trade engine against every
        league roster and returns the ranked proposals as "trade_proposals".
        """
        if not league_id:
            league_id = load_user_config().get("default_league_id")

        scoring = SCORING_PPR
        league_context: Dict[str, Any] = {}
        if league_id:
            league_context = self.sleeper.get_league_context(league_id) or {}
            if league_context.get("scoring_type"):
                scoring = league_context["scoring_type"]

        self._ensure_engines([season], scoring=scoring)
        result: Dict[str, Any] = {
            "buy_low": self._trade_analyzer.get_buy_low_candidates(season=season, week=week, positions=positions),
            "sell_high": self._trade_analyzer.get_sell_high_candidates(season=season, week=week, positions=positions),
        }
        if not username:
            username = load_user_config().get("sleeper_username")
        if username:
            proposals = self._bilateral_trade_proposals(
                league_id=league_id,
                league_context=league_context,
                username=username,
            )
            if proposals is not None:
                result["trade_proposals"] = proposals
        return result

    def _bilateral_trade_proposals(
        self,
        league_id: Optional[str],
        league_context: Dict[str, Any],
        username: str,
    ) -> Optional[Dict[str, Any]]:
        """Runs the bilateral trade engine when the user's Sleeper roster is resolvable."""
        try:
            user_info = self.sleeper.get_user(username)
            user_id = user_info.get("user_id") if user_info else username

            if not league_id:
                leagues = self.sleeper.get_user_leagues(user_id)
                if not leagues:
                    return None
                league_id = leagues[0].get("league_id")
            if not league_context:
                league_context = self.sleeper.get_league_context(league_id) or {}
                if not league_context:
                    return None

            rosters = self.sleeper.get_league_rosters(league_id)
            if not rosters:
                return None
            user_roster_index = next(
                (i for i, r in enumerate(rosters) if str(r.get("owner_id")) == str(user_id)),
                None,
            )
            if user_roster_index is None:
                return None

            users = self.sleeper.get_league_users(league_id)
            display_names = {
                u.get("user_id"): (u.get("metadata") or {}).get("team_name") or u.get("display_name")
                for u in users
            }
            manager_names = [
                display_names.get(r.get("owner_id")) or f"Roster {r.get('roster_id')}"
                for r in rosters
            ]

            metrics_df = self._latest_metrics_frame(self._waiver_engine.df)
            if metrics_df is None or metrics_df.empty:
                return None

            return generate_league_trade_proposals(
                metrics_df=metrics_df,
                league_context=league_context,
                rosters=[[str(p) for p in (r.get("players") or [])] for r in rosters],
                user_roster_index=user_roster_index,
                roster_ids=[r.get("roster_id") for r in rosters],
                manager_names=manager_names,
                players_db=self.sleeper.get_all_players(),
            )
        except Exception as e:
            logger.warning(f"Bilateral trade evaluation failed: {e}")
            return None

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
        - Context-adjusted floor/median/ceiling projections (Vegas ITT / wind / defensive funnels)
        - Matchup posture and mathematically optimized start/sit swaps
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

        advisor = self._build_matchup_advisor(overview, league_id)
        eval_result = advisor.evaluate_lineup(overview)
        return {
            "matchup_overview": overview,
            "tactical_advice": eval_result,
        }

    def _build_matchup_advisor(self, overview: Dict[str, Any], league_id: Optional[str]) -> MatchupAdvisor:
        """Builds a projection-driven MatchupAdvisor; degrades to the bare advisor on any data failure."""
        week = overview.get("week")

        league_context: Dict[str, Any] = {}
        if league_id:
            try:
                league_context = self.sleeper.get_league_context(league_id) or {}
            except Exception:
                league_context = {}

        scoring = league_context.get("scoring_type") or SCORING_PPR
        season = int(league_context.get("season") or DEFAULT_SEASON)

        roster_positions = [
            slot
            for slot, count in (league_context.get("roster_position_counts") or {}).items()
            for _ in range(int(count))
        ]
        if not roster_positions:
            roster_positions = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]
        overview["roster_positions"] = roster_positions

        metrics_df: Optional[pd.DataFrame] = None
        try:
            self._ensure_engines([season], scoring=scoring)
            dataset = self._waiver_engine.df
            if dataset is not None and not dataset.empty:
                metrics_df = self._latest_metrics_frame(dataset)
        except Exception:
            metrics_df = None

        schedules_df: Optional[pd.DataFrame] = None
        try:
            schedules = self.loader.get_schedules([season])
            if not schedules.empty and week is not None and "week" in schedules.columns:
                schedules = schedules[schedules["week"] == week]
            if not schedules.empty:
                schedules_df = schedules
        except Exception:
            schedules_df = None

        def_epa_df: Optional[pd.DataFrame] = None
        try:
            features = self.loader.get_pbp_opportunity_features([season])
            defense = features.get("defense") if isinstance(features, dict) else None
            if defense is not None and not defense.empty:
                def_epa_df = compute_defensive_epa_rates(defense, season=season, week=week)
        except Exception:
            def_epa_df = None

        if metrics_df is None and schedules_df is None and def_epa_df is None:
            return self.matchup_advisor

        return MatchupAdvisor(
            metrics_df=metrics_df,
            schedules_df=schedules_df,
            def_epa_df=def_epa_df if def_epa_df is not None and not def_epa_df.empty else None,
            weather_fn=get_game_wind_forecast,
        )

    def _latest_metrics_frame(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Latest per-player NFL-wide metrics rows with distribution projections and Sleeper ids."""
        df = add_projection_distributions(dataset)
        if df.empty:
            return df
        key = "player_id" if "player_id" in df.columns else "player_display_name"
        sort_cols = [c for c in ("season", "week") if c in df.columns]
        if sort_cols:
            latest = df.sort_values(sort_cols, kind="stable").groupby(key, sort=False).tail(1).copy()
        else:
            latest = df.copy()
        if "player_id" in latest.columns:
            latest["sleeper_id"] = self._resolve_sleeper_ids(latest["player_id"])
        return latest


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


