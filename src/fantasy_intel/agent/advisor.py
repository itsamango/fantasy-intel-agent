"""Fantasy Intelligence Advisor orchestrating data, news, and league APIs."""

from __future__ import annotations
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd

from fantasy_intel.data.loader import DataLoader
from fantasy_intel.data.sleeper import SleeperClient
from fantasy_intel.data.weather import get_game_wind_forecast
from fantasy_intel.analytics.metrics import _normalize_ids, add_projection_distributions, attach_pbp_features
from fantasy_intel.analytics.waiver_wire import WaiverWireEngine, recommend_faab_bids
from fantasy_intel.analytics.trades import (
    BUY_LOW_MAX_ROLLING_FPOE,
    SELL_HIGH_MIN_ROLLING_FPOE,
    TradeAnalyzer,
    generate_league_trade_proposals,
)
from fantasy_intel.analytics.depth_chart import DepthChartManager
from fantasy_intel.analytics.matchups import MatchupAdvisor, compute_defensive_epa_rates
from fantasy_intel.analytics.market import TIER_ORDER, assess_trade, find_sweeteners, is_faller, market_decline
from fantasy_intel.data.fantasycalc import FantasyCalcClient, market_params_from_league
from fantasy_intel.analytics.standings import (
    SeasonSimulator,
    SeasonState,
    estimate_weekly_sd,
    playoff_round_weeks,
)
from fantasy_intel.analytics.ros_value import (
    LeagueShape,
    plan_waiver_moves,
    title_leverage,
    find_trades,
    RosterState,
    RosValuator,
    build_outlooks,
    evaluate_trade,
    teams_kicked_off,
    teams_playing_from_schedule,
    top_free_agents,
)
from fantasy_intel.config import (
    DEFAULT_SEASON,
    MARKET_PLAUSIBLE_LINEUP_MIN,
    MARKET_TRADEABLE_MIN_VALUE,
    ROS_FA_CANDIDATES_PER_POS,
    ROS_WAIVER_MAX_SWAPS,
    FPOE_HISTORY_SEASONS,
    ROS_SIMS,
    SCORING_PPR,
    load_user_config,
)
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
        self.data_through_week: Optional[int] = None
        self._fpoe_history_cache: Dict[int, Optional[pd.DataFrame]] = {}
        self.market = FantasyCalcClient()
        self._market_cache: Dict[Any, Dict[str, Dict[str, Any]]] = {}

    def _ensure_engines(self, seasons: List[int], scoring: str = SCORING_PPR):
        if (
            self._waiver_engine is None
            or self._loaded_seasons != seasons
            or self._loaded_scoring != scoring
        ):
            season = seasons[0] if seasons else DEFAULT_SEASON
            attached_seasons = [season] if season >= 2025 else list(seasons)
            if season >= 2025:
                # Source order for current seasons:
                # 1. Sleeper per-week stats (live, real weekly rows, Sleeper ids)
                # 2. nflverse weekly (if ever published again for this season)
                # 3. Sleeper season aggregate (one row per player; last resort)
                dataset = self._sleeper_weekly_if_available(season)
                if dataset is None:
                    dataset = self._nflverse_weekly_if_available(season)
                if dataset is None:
                    logger.warning(
                        f"No weekly data for {season}; falling back to Sleeper season totals "
                        "(rolling/trend metrics degrade to season averages)."
                    )
                    dataset = self.loader.get_sleeper_season_data(season=season)
            else:
                dataset = self.loader.get_weekly_with_snaps(seasons=seasons)
            self.data_through_week = (
                dataset.attrs.get("data_through_week")
                if dataset is not None
                else None
            )

            # Enrich with PBP opportunity features so trajectories carry
            # hvt / unrealized_air_yards / situational xFP (rolling_xfp_3w)
            dataset = attach_pbp_features(dataset, self.loader, seasons=attached_seasons)

            # Prior seasons seed each player's FPOE efficiency baseline
            history = self._fpoe_history(season) if season >= 2025 else None

            self._waiver_engine = WaiverWireEngine(dataset, scoring=scoring, fpoe_history=history)
            self._trade_analyzer = TradeAnalyzer(dataset, scoring=scoring, fpoe_history=history)
            self._loaded_seasons = seasons
            self._loaded_scoring = scoring

    def _fpoe_history(self, season: int) -> Optional[pd.DataFrame]:
        """Prior seasons' Sleeper weekly frames (pbp-enriched) for the FPOE baseline, cached per season."""
        if season in self._fpoe_history_cache:
            return self._fpoe_history_cache[season]
        frames = []
        for prior in range(season - FPOE_HISTORY_SEASONS, season):
            weekly = self._sleeper_weekly_if_available(prior)
            if weekly is None:
                logger.info(f"No {prior} weekly history for the FPOE baseline.")
                continue
            try:
                frames.append(attach_pbp_features(weekly, self.loader, seasons=[prior]))
            except Exception as e:
                logger.warning(f"PBP enrichment failed for {prior} history ({e}); using box-score xFP.")
                frames.append(weekly)
        history = pd.concat(frames, ignore_index=True) if frames else None
        self._fpoe_history_cache[season] = history
        return history

    def _sleeper_weekly_if_available(self, season: int) -> Optional[pd.DataFrame]:
        """Sleeper per-week stat frame for the season, else None."""
        fetch = getattr(self.loader, "get_sleeper_weekly_data", None)
        if fetch is None:
            return None
        try:
            weekly = fetch(season=season)
        except Exception as e:
            logger.warning(f"Sleeper weekly stats unavailable for season {season}: {e}")
            return None
        if weekly is None or weekly.empty or not (weekly["season"] == season).any():
            return None
        return weekly

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

    def _latest_sleeper_xfp(self, col: str = "rolling_xfp_3w") -> Dict[str, float]:
        """Latest-week value of `col` (rolling 3-week xFP by default) per player, keyed by Sleeper id."""
        df = self._waiver_engine.df
        if (
            df is None
            or df.empty
            or col not in df.columns
            or "week" not in df.columns
        ):
            return {}
        key = "player_id" if "player_id" in df.columns else "player_display_name"
        sort_cols = [c for c in ("season", "week") if c in df.columns]
        latest = df.sort_values(sort_cols, kind="stable").groupby(key, sort=False).tail(1)
        sleeper_ids = self._resolve_sleeper_ids(latest[key])
        return {
            str(sid): float(xfp)
            for sid, xfp in zip(sleeper_ids, latest[col])
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
        proj_by_sleeper_id = self._latest_sleeper_xfp("proj_ppg")

        rows = []
        for pid in (user_roster_record.get("players") or []):
            pid = str(pid)
            p = players_db.get(pid, {})
            rows.append({
                "player_id": pid,
                "player_display_name": p.get("full_name") or f"Player {pid}",
                "position": p.get("position"),
                "rolling_xfp_3w": xfp_by_sleeper_id.get(pid),
                "proj_ppg": proj_by_sleeper_id.get(pid),
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
        season: int = DEFAULT_SEASON,
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
        season: int = DEFAULT_SEASON,
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
            proposals = self._ros_trade_proposals(league_id=league_id, username=username)
            if proposals is None:
                # Fallback: legacy one-week lineup engine
                proposals = self._bilateral_trade_proposals(
                    league_id=league_id,
                    league_context=league_context,
                    username=username,
                )
            if proposals is not None:
                result["trade_proposals"] = proposals
        return result

    def _ros_trade_proposals(
        self, league_id: Optional[str], username: str, limit: int = 10, n_sims: int = ROS_SIMS
    ) -> Optional[Dict[str, Any]]:
        """Bilateral proposals scored on rest-of-season lineup impact for both teams."""
        try:
            ctx = self.build_ros_context(league_id=league_id, username=username, n_sims=n_sims)
        except Exception as e:
            logger.warning(f"ROS trade context unavailable ({e}); using legacy trade engine.")
            return None
        market_values = self._market_values(ctx["league_context"])
        found = find_trades(
            ctx["valuator"], ctx["rosters"], ctx["user_index"], ctx["free_agents"],
            manager_names=ctx["manager_names"], roster_ids=ctx["roster_ids"], limit=limit,
            market_values=market_values or None,
            min_partner_market_pct=MARKET_PLAUSIBLE_LINEUP_MIN,
            season=ctx.get("season"),
        )
        excess = self._latest_sleeper_xfp("fpoe_excess_3w") if self._waiver_engine is not None else {}
        market_values = market_values or {}

        def card(c: Dict[str, Any], receiving: bool = False) -> Dict[str, Any]:
            pid = str(c.get("player_id"))
            e = excess.get(pid)
            tags = []
            if e is not None and e >= SELL_HIGH_MIN_ROLLING_FPOE:
                tags.append(f"running hot {e:+.1f}/gm")
            elif e is not None and e <= BUY_LOW_MAX_ROLLING_FPOE:
                tags.append(f"running cold {e:+.1f}/gm")
            if receiving and is_faller(market_values, pid):
                tags.append(f"market {-market_decline(market_values, pid):+.0%} in 30d: buy-low window")
            return {**c, "tag": "; ".join(tags) or None}

        proposals = []
        for r in found["proposals"]:
            give_ids = [c["player_id"] for c in r["you_send"]]
            get_ids = [c["player_id"] for c in r["you_get"]]
            r = self._attach_market(ctx, r, give_ids, get_ids, r["target_index"])
            you, them = r["you"], r["partner"]
            pitch = []
            if r.get("odds"):
                o = r["odds"]["you"]
                pitch.append(
                    f"title odds {o['before']['title']:.1%} → {o['after']['title']:.1%}, "
                    f"playoffs {o['before']['playoffs']:.0%} → {o['after']['playoffs']:.0%}"
                )
            pitch.append(
                f"ROS: you {you['delta_per_week']:+.2f}/wk ({you['delta_playoffs']:+.1f} in playoffs), "
                f"them {them['delta_per_week']:+.2f}/wk"
            )
            if them["drops"]:
                pitch.append(f"they cut {', '.join(them['drops'])}")
            if you["drops"]:
                pitch.append(f"you cut {', '.join(you['drops'])}")
            if you["adds"]:
                pitch.append(f"you add {', '.join(you['adds'])} off waivers")
            market = r.get("market") or {}
            if market.get("summary"):
                pitch.append(market["summary"])
            for sw in market.get("sweeteners") or []:
                pitch.append(
                    f"to make it {sw['market']['tier'].lower()}: add {sw['add_name']} "
                    f"(you still {sw['your_ppw']:+.2f}/wk)"
                )
            proposals.append({
                "target_manager": r["target_manager"],
                "target_roster_id": r["target_roster_id"],
                "send": [card(c) for c in r["you_send"]],
                "receive": [card(c, receiving=True) for c in r["you_get"]],
                "your_lineup_gain": you["delta_per_week"],
                "their_lineup_gain": them["delta_per_week"],
                "your_playoff_gain": you["delta_playoffs"],
                "your_title_delta": (r.get("odds") or {}).get("you", {}).get("delta", {}).get("title"),
                "their_title_delta": (r.get("odds") or {}).get("partner", {}).get("delta", {}).get("title"),
                "market_delta": market.get("gap_pct"),
                "market_tier": market.get("tier", "UNKNOWN"),
                "market_send_value": market.get("send_value"),
                "market_get_value": market.get("get_value"),
                "sweeteners": market.get("sweeteners") or [],
                "need_filled": None,
                "p_accept": None,
                "pitch": "; ".join(pitch),
                "ros": r,
            })
        proposals.sort(key=lambda p: (
            TIER_ORDER.get(p["market_tier"], 9),
            -(p["your_title_delta"] if p["your_title_delta"] is not None else p["your_lineup_gain"]),
        ))
        notes = list(ctx["notes"]) + [
            f"Screened {found['swaps_screened']} swaps (1-for-1, 2-for-1, 1-for-2); "
            f"{found.get('market_rejected', 0)} dropped as market-lopsided against the partner "
            f"(> {abs(MARKET_PLAUSIBLE_LINEUP_MIN):.0%} of value); "
            f"{found['passed_screen']} passed the fast screen; top {found['rescored']} re-scored with "
            "simulated availability and roster limits.",
            "Gains are expected starting-lineup points per remaining week; partner must not lose lineup value.",
            "Likelihood = FantasyCalc market fairness for the partner + their lineup change "
            "(LIKELY: they lose <= 2.5% market value and their lineup holds; untrained tiers - tune from experience).",
        ]
        deadline = ctx.get("trade_deadline")
        if deadline and int(ctx["current_week"]) >= int(deadline):
            notes.insert(0, f"League trade deadline: week {deadline} (current week {ctx['current_week']}).")
        return {
            "proposals": proposals,
            "current_odds": ctx.get("current_odds"),
            "engine": "rest_of_season",
            "league": {"name": ctx["league_name"], "scoring_type": ctx["scoring"]},
            "notes": notes,
        }

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

    # ------------------------------------------------------------------ #
    # Rest-of-season trade evaluation
    # ------------------------------------------------------------------ #
    def build_ros_context(
        self,
        league_id: Optional[str] = None,
        username: Optional[str] = None,
        n_sims: int = ROS_SIMS,
        use_model_fallback: bool = True,
        fa_per_position: int = ROS_FA_CANDIDATES_PER_POS,
    ) -> Dict[str, Any]:
        """Gathers everything the ROS valuator needs for one Sleeper league.

        Returns {"valuator", "rosters" (RosterState per team), "roster_ids",
        "manager_names", "user_index", "free_agents", "players_db", "scoring",
        "shape", "notes"}. Raises ValueError with a readable reason when the
        league / user / remaining schedule cannot be resolved.
        """
        cfg = load_user_config()
        league_id = league_id or cfg.get("default_league_id")
        username = username or cfg.get("sleeper_username")
        if not league_id or not username:
            raise ValueError("Need a Sleeper league id and username (run `fantasy-intel connect`).")

        info = self.sleeper.get_league_info(league_id) or {}
        if not info:
            raise ValueError(f"Could not load Sleeper league {league_id}.")
        league_context = self.sleeper.get_league_context(league_id) or {}
        scoring = league_context.get("scoring_type") or SCORING_PPR
        season = int(info.get("season") or DEFAULT_SEASON)

        state = self.sleeper.get_nfl_state() or {}
        start_week = int(state.get("week") or 1)
        if str(state.get("season") or season) != str(season):
            raise ValueError(f"League season {season} is not the current NFL season.")
        shape = LeagueShape.from_sleeper(info.get("roster_positions") or [], info.get("settings") or {}, start_week)
        if not shape.weeks:
            raise ValueError("No fantasy weeks remain this season.")

        user = self.sleeper.get_user(username) or {}
        user_id = str(user.get("user_id") or username)
        raw_rosters = self.sleeper.get_league_rosters(league_id) or []
        user_index = next((i for i, r in enumerate(raw_rosters) if str(r.get("owner_id")) == user_id), None)
        if user_index is None:
            raise ValueError(f"{username} has no roster in league {league_id}.")
        users = self.sleeper.get_league_users(league_id) or []
        display = {
            u.get("user_id"): (u.get("metadata") or {}).get("team_name") or u.get("display_name")
            for u in users
        }

        players_db = self.sleeper.get_all_players() or {}
        rosters: List[RosterState] = []
        rostered: Set[str] = set()
        for r in raw_rosters:
            reserve = [str(p) for p in (r.get("reserve") or [])]
            active = [str(p) for p in (r.get("players") or []) if str(p) not in set(reserve)]
            rosters.append(RosterState(active=active, reserve=reserve))
            rostered.update(active + reserve)

        notes: List[str] = []
        scoring_settings = info.get("scoring_settings") or {}
        projections = {w: self.sleeper.get_week_projections(season=season, week=w) or {} for w in shape.weeks}
        try:
            schedule = self.loader.get_schedules([season])
        except Exception as e:
            schedule = None
            notes.append(f"Schedule unavailable ({e}); byes not modeled.")
        teams_playing = teams_playing_from_schedule(schedule, shape.weeks)

        # Current week: games already kicked off are banked, not re-projected
        kicked = teams_kicked_off(schedule, start_week)
        if schedule is None or getattr(schedule, "empty", True):
            notes.append(f"Schedule unavailable: week {start_week} games already played may be counted again.")
        locked_slots: Dict[str, str] = {}
        banked = np.zeros(len(raw_rosters))
        if kicked:
            idx = {r.get("roster_id"): i for i, r in enumerate(raw_rosters)}
            start_slots = [
                str(s).upper() for s in (info.get("roster_positions") or [])
                if str(s).upper() not in ("BN", "IR", "TAXI")
            ]
            team_of = {pid: (p or {}).get("team") for pid, p in players_db.items()}
            for m in self.sleeper.get_matchups(league_id, start_week) or []:
                i = idx.get(m.get("roster_id"))
                pts = m.get("players_points") or {}
                for slot, pid in zip(start_slots, m.get("starters") or []):
                    pid = str(pid)
                    if pid and pid != "0" and team_of.get(pid) in kicked:
                        locked_slots[pid] = slot
                        if i is not None:
                            banked[i] += float(pts.get(pid) or 0.0)
            notes.append(
                f"Week {start_week} in progress: {len(kicked) // 2} game(s) kicked off; those players' "
                "points are taken from live scores, not projections."
            )

        fallback: Dict[str, float] = {}
        if use_model_fallback:
            try:
                self._ensure_engines([season], scoring=scoring)
                fallback = self._latest_sleeper_xfp("proj_ppg")
            except Exception as e:
                notes.append(f"Model projections unavailable for unprojected players ({e}).")

        skill_ids = {
            pid for pid, p in players_db.items()
            if p.get("position") in ("QB", "RB", "WR", "TE") and (pid in rostered or p.get("team"))
        }
        projected = {pid for wk in projections.values() for pid in wk}
        candidate_ids = (skill_ids & (rostered | projected | set(fallback)))
        outlooks = build_outlooks(
            candidate_ids, players_db, projections, teams_playing, shape.weeks, scoring,
            fallback_ppg=fallback, scoring_settings=scoring_settings, kicked_off=kicked,
        )
        free_agents = top_free_agents(outlooks, rostered, per_position=fa_per_position)
        notes.append(
            f"Weeks {shape.weeks[0]}-{shape.weeks[-1]} (playoffs {sorted(shape.playoff_weeks)}); "
            f"projections: Sleeper weekly scored with this league's settings, model proj_ppg fallback; "
            f"{n_sims} availability simulations."
        )
        valuator = RosValuator(shape, outlooks, n_sims=n_sims, locked_slots=locked_slots)
        season_ctx = None
        current_odds = None
        playoff_start = int((info.get("settings") or {}).get("playoff_week_start") or 15)
        if start_week > playoff_start:
            notes.append(
                f"Playoffs under way (week {start_week}): the season simulator would replay decided "
                "rounds, so title odds are off and verdicts use lineup points."
            )
        else:
            try:
                season_ctx = self._build_season(
                    league_id, info, raw_rosters, rosters, valuator, user_index, banked_points=banked
                )
                current_odds = season_ctx["sim"].run(season_ctx["base_means"]).for_team(user_index)
                notes.append(
                    f"Standings-aware: {season_ctx['sim'].n_sims} season simulations (schedule, records, "
                    f"weekly SD {season_ctx['sim'].state.weekly_sd:.1f} from this league's scores); "
                    "verdicts use change in title odds."
                )
            except Exception as e:
                logger.warning(f"Season simulation unavailable: {e}")
                notes.append(f"Standings simulation unavailable ({e}); verdicts use lineup points.")
        return {
            "valuator": valuator,
            "season": season_ctx,
            "current_odds": current_odds,
            "rosters": rosters,
            "roster_ids": [r.get("roster_id") for r in raw_rosters],
            "manager_names": [
                display.get(r.get("owner_id")) or f"Roster {r.get('roster_id')}" for r in raw_rosters
            ],
            "user_index": user_index,
            "free_agents": free_agents,
            "players_db": players_db,
            "scoring": scoring,
            "shape": shape,
            "league_name": info.get("name"),
            "league_context": league_context,
            "trade_deadline": (info.get("settings") or {}).get("trade_deadline"),
            "current_week": start_week,
            "notes": notes,
        }

    def _build_season(
        self,
        league_id: str,
        info: Mapping[str, Any],
        raw_rosters: Sequence[Mapping[str, Any]],
        rosters: Sequence[RosterState],
        valuator: RosValuator,
        user_index: int,
        banked_points: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """SeasonSimulator inputs from Sleeper: records, remaining schedule, weekly SD, team strength.

        banked_points (per team): current-week points already scored by starters
        whose game kicked off; added to the current week's projected mean."""
        settings = info.get("settings") or {}
        playoff_start = int(settings.get("playoff_week_start") or 15)
        weeks = valuator.shape.weeks
        regular = [w for w in weeks if w < playoff_start]
        rounds = [[w for w in r if w in weeks] or r for r in playoff_round_weeks(settings)]
        idx = {r.get("roster_id"): i for i, r in enumerate(raw_rosters)}

        schedule: Dict[int, List[Tuple[int, int]]] = {}
        for w in regular:
            groups: Dict[Any, List[int]] = {}
            for m in self.sleeper.get_matchups(league_id, w) or []:
                if m.get("matchup_id") is not None and m.get("roster_id") in idx:
                    groups.setdefault(m["matchup_id"], []).append(idx[m["roster_id"]])
            schedule[w] = [tuple(g) for g in groups.values() if len(g) == 2]

        first = weeks[0] if weeks else playoff_start
        scores: Dict[Any, List[float]] = {}
        for w in range(max(1, first - 8), first):
            for m in self.sleeper.get_matchups(league_id, w) or []:
                if m.get("points"):
                    scores.setdefault(m.get("roster_id"), []).append(float(m["points"]))

        def num(st: Mapping[str, Any], k: str) -> float:
            return float(st.get(k) or 0)

        wins = np.array([num(r.get("settings") or {}, "wins") + 0.5 * num(r.get("settings") or {}, "ties") for r in raw_rosters])
        pf = np.array([num(r.get("settings") or {}, "fpts") + num(r.get("settings") or {}, "fpts_decimal") / 100 for r in raw_rosters])
        state = SeasonState(
            team_keys=[r.get("roster_id") for r in raw_rosters],
            wins=wins,
            points_for=pf,
            schedule=schedule,
            regular_weeks=regular,
            playoff_rounds=rounds,
            playoff_teams=int(settings.get("playoff_teams") or 6),
            weekly_sd=estimate_weekly_sd(scores),
            median_game=int(settings.get("league_average_match") or 0) == 1,
        )
        sim = SeasonSimulator(state)
        base = np.zeros((len(rosters), len(state.weeks)))
        for i, r in enumerate(rosters):
            per_week = valuator.team_value(r.all_ids(), "sim").per_week
            fill = float(np.mean(list(per_week.values()))) if per_week else 0.0
            base[i] = [per_week.get(w, fill) for w in state.weeks]
        w0 = weeks[0] if weeks else None
        if banked_points is not None and w0 in state.weeks:
            base[:, state.weeks.index(w0)] += banked_points
        return {
            "sim": sim, "base_means": base, "user_index": user_index,
            "banked_points": banked_points, "current_week": w0,
        }

    def _market_values(self, league_context: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        """FantasyCalc redraft values for this league's settings ({} when unavailable)."""
        params = market_params_from_league(league_context)
        key = (params["numQbs"], params["numTeams"], params["ppr"])
        if key not in self._market_cache:
            self._market_cache[key] = self.market.get_values(
                num_qbs=params["numQbs"], num_teams=params["numTeams"], ppr=params["ppr"]
            )
        return self._market_cache[key]

    def _attach_market(
        self,
        ctx: Dict[str, Any],
        result: Dict[str, Any],
        give_ids: Sequence[str],
        get_ids: Sequence[str],
        partner_index: int,
        find_sweeteners_if_needed: bool = True,
    ) -> Dict[str, Any]:
        """Adds market fairness, acceptance tier, per-player market values, and sweeteners."""
        values = self._market_values(ctx["league_context"])
        if not values:
            result["market"] = {"tier": "UNKNOWN", "summary": "FantasyCalc values unavailable.", "sweeteners": []}
            return result
        params = market_params_from_league(ctx["league_context"])
        market = assess_trade(values, give_ids, get_ids, result["partner"]["delta_per_week"])
        market["source"] = (
            f"FantasyCalc redraft ({params['numQbs']}QB, {params['numTeams']} teams, {params['ppr']:g} PPR)"
        )
        market["sweeteners"] = []
        market["sweeteners_skipped"] = []
        if find_sweeteners_if_needed and market["tier"] in ("PLAUSIBLE", "UNLIKELY") and market["value_to_add"] > 0:
            ui = ctx["user_index"]
            rosters = ctx["rosters"]
            names = {pid: o.name for pid, o in ctx["valuator"].outlooks.items()}
            excess = self._latest_sleeper_xfp("fpoe_excess_3w") if self._waiver_engine is not None else {}
            cold = {
                pid: f"running cold ({e:+.1f}/gm vs his own baseline): a buy-low, don't sell"
                for pid, e in excess.items()
                if e is not None and e <= BUY_LOW_MAX_ROLLING_FPOE
            }
            skipped = market["sweeteners_skipped"]
            market["sweeteners"] = find_sweeteners(
                values,
                rosters[ui].all_ids(),
                give_ids,
                get_ids,
                market["value_to_add"],
                evaluate=lambda send: evaluate_trade(
                    ctx["valuator"], rosters[ui], rosters[partner_index], send, list(get_ids),
                    ctx["free_agents"], mode="sim",
                ),
                names=names,
                exclude=cold,
                skipped=skipped,
            )
        for card in result.get("you_send", []) + result.get("you_get", []):
            mv = values.get(str(card.get("player_id"))) or {}
            card["market_value"] = round(mv["value"]) if mv else 0
            card["market_rank"] = mv.get("overall_rank")
            card["market_trend_30d"] = mv.get("trend_30d")
        names_all = {pid: o.name for pid, o in ctx["valuator"].outlooks.items()}
        for f in market.get("fallers") or []:
            f["name"] = names_all.get(f["player_id"], f["player_id"])
        result["market"] = market
        return result

    @staticmethod
    def _resolve_player_names(
        names: Sequence[str], candidate_ids: Iterable[str], players_db: Mapping[str, Mapping[str, Any]]
    ) -> List[str]:
        """Maps typed names to player ids among candidate_ids (exact, then unique partial match)."""
        def norm(text: Any) -> str:
            return "".join(ch for ch in str(text or "").lower() if ch.isalnum())

        pool = {pid: norm(players_db.get(pid, {}).get("full_name")) for pid in candidate_ids}
        out: List[str] = []
        for raw in names:
            target = norm(raw)
            if not target:
                continue
            exact = [pid for pid, n in pool.items() if n == target]
            matches = exact or [pid for pid, n in pool.items() if target in n]
            if len(matches) == 1:
                out.append(matches[0])
            elif not matches:
                raise ValueError(f"'{raw}' is not on the expected roster(s).")
            else:
                options = ", ".join(sorted(players_db[m].get("full_name", m) for m in matches))
                raise ValueError(f"'{raw}' is ambiguous: {options}.")
        return out

    def get_waiver_moves(
        self,
        league_id: Optional[str] = None,
        username: Optional[str] = None,
        max_moves: int = ROS_WAIVER_MAX_SWAPS,
        n_sims: int = ROS_SIMS,
        fa_per_position: int = 10,
    ) -> Dict[str, Any]:
        """The user's best free waiver moves by rest-of-season value and title odds."""
        ctx = self.build_ros_context(
            league_id=league_id, username=username, n_sims=n_sims, fa_per_position=fa_per_position
        )
        ui = ctx["user_index"]
        result = plan_waiver_moves(
            ctx["valuator"], ctx["rosters"][ui], ctx["free_agents"], season=ctx.get("season"), max_swaps=max_moves
        )
        lc = ctx.get("league_context") or {}
        my_roster_id = ctx["roster_ids"][ui]
        rec = next((r for r in lc.get("rosters") or [] if r.get("roster_id") == my_roster_id), {})
        values = self._market_values(lc) if lc else {}
        for item in [m for m in result["moves"]] + result["alternatives"]:
            for side in ("add", "drop"):
                c = item.get(side)
                if c:
                    mv = values.get(c["player_id"]) or {}
                    c["market_value"] = round(mv["value"]) if mv else 0
                    c["market_trend_30d"] = mv.get("trend_30d")
                    if side == "drop":
                        c["has_trade_value"] = c["market_value"] >= MARKET_TRADEABLE_MIN_VALUE
        result.update({
            "league_name": ctx["league_name"],
            "scoring": ctx["scoring"],
            "current_week": ctx["current_week"],
            "waiver_type": "rolling priority" if lc.get("waiver_type") == 0 else "FAAB",
            "faab_remaining": rec.get("faab_remaining") if isinstance(rec, dict) else None,
            "faab_budget": lc.get("faab_budget"),
            "notes": ctx["notes"] + [
                f"Free-agent pool: top {fa_per_position} unrostered players per position by ROS projection. "
                "Sleeper does not expose per-league waiver status, so some may need a claim.",
            ],
        })
        return result

    def evaluate_trade(
        self,
        give: Sequence[str],
        get: Sequence[str],
        league_id: Optional[str] = None,
        username: Optional[str] = None,
        n_sims: int = ROS_SIMS,
    ) -> Dict[str, Any]:
        """Rest-of-season evaluation of a specific trade (names or Sleeper ids).

        `give` must be on the user's roster; `get` must all be on one other
        roster (the trade partner, found automatically).
        """
        ctx = self.build_ros_context(league_id=league_id, username=username, n_sims=n_sims)
        rosters, players_db, ui = ctx["rosters"], ctx["players_db"], ctx["user_index"]
        give_ids = self._resolve_player_names(give, rosters[ui].all_ids(), players_db)
        others = {pid: i for i, r in enumerate(rosters) if i != ui for pid in r.all_ids()}
        get_ids = self._resolve_player_names(get, others.keys(), players_db)
        partners = {others[pid] for pid in get_ids}
        if len(partners) != 1:
            raise ValueError("All players you get must come from a single opposing roster.")
        pi = partners.pop()
        season = {**ctx["season"], "partner_index": pi} if ctx.get("season") else None
        result = evaluate_trade(
            ctx["valuator"], rosters[ui], rosters[pi], give_ids, get_ids, ctx["free_agents"], mode="sim",
            season=season,
        )
        result = self._attach_market(ctx, result, give_ids, get_ids, pi)
        if season:
            result["title_leverage"] = title_leverage(season, ui)
        result.update({
            "partner_name": ctx["manager_names"][pi],
            "league_name": ctx["league_name"],
            "scoring": ctx["scoring"],
            "current_week": ctx["current_week"],
            "trade_deadline": ctx["trade_deadline"],
            "current_odds": ctx.get("current_odds"),
            "notes": ctx["notes"],
        })
        return result

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

    def get_player_intel(self, player_name: str, season: int = DEFAULT_SEASON) -> Dict[str, Any]:
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


