"""Depth chart tracker and next-man-up beneficiary engine."""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from fantasy_intel.data.sleeper import SleeperClient
from fantasy_intel.data.loader import clean_name

logger = logging.getLogger(__name__)

OFFENSIVE_POSITIONS = {"QB", "RB", "WR", "TE"}

# Injury statuses that rule a player out of the upcoming game
OUT_STATUSES = ["Out", "IR", "PUP", "Doubtful"]

# Sleeper marks practice-squad players via the `status` field; a PS player
# being out never vacates a role on the active roster
PRACTICE_SQUAD_STATUSES = {"practice squad"}

# Placeholder depth order for players no source lists a slot for
UNLISTED_DEPTH_ORDER = 99


def _injury_desc(p: Dict[str, Any]) -> str:
    status = p.get("injury_status")
    part = p.get("injury_body_part") or "Injured"
    return f"{p['name']} ({status} - {part})"


def _is_practice_squad(player: Dict[str, Any]) -> bool:
    """True when Sleeper lists the player with practice-squad status."""
    status = str(player.get("status") or "").strip().lower()
    return status in PRACTICE_SQUAD_STATUSES


class DepthChartManager:
    """Tracks team depth charts and determines next-man-up beneficiaries when injuries occur.

    Sleeper's live players payload is the source of truth for team,
    injury_status, and depth-chart slot/order. nflverse depth charts and
    injury reports are a fallback enrichment only: they fill gaps for
    players Sleeper lists no depth order or injury status for, and only
    when the nflverse frame is from the requested season and agrees with
    the player's live Sleeper team (rows for a former team are discarded
    as stale).
    """

    def __init__(
        self,
        sleeper_client: Optional[SleeperClient] = None,
        season: Optional[int] = None,
        loader: Optional[Any] = None,
    ):
        self.sleeper = sleeper_client or SleeperClient()
        self._season = season
        self._loader = loader
        self._players: Dict[str, Dict[str, Any]] = {}
        self._depth_by_team: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._nflverse_enriched = False

    def _ensure_depth_chart(self):
        if self._players:
            return

        self._players = self.sleeper.get_all_players()
        self._depth_by_team = {}

        for pid, p in self._players.items():
            team = p.get("team")
            pos = p.get("position")
            order = p.get("depth_chart_order")

            if not team or pos not in OFFENSIVE_POSITIONS:
                continue

            # Normalized position
            team_dict = self._depth_by_team.setdefault(team, {})
            pos_list = team_dict.setdefault(pos, [])
            pos_list.append({
                "player_id": str(pid),
                "name": p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or f"Player {pid}",
                "norm_name": clean_name(p.get("full_name", "")),
                "team": team,
                "position": pos,
                "depth_order": int(order) if order is not None else UNLISTED_DEPTH_ORDER,
                "depth_is_real": order is not None,
                "depth_position": p.get("depth_chart_position") or pos,
                "injury_status": p.get("injury_status"),
                "injury_body_part": p.get("injury_body_part"),
                "injury_notes": p.get("injury_notes"),
                "status": p.get("status", "Active"),
            })

        # Sort each team/position list by depth order
        for team, positions in self._depth_by_team.items():
            for pos, players in positions.items():
                players.sort(key=lambda x: x["depth_order"])

    def get_team_depth_chart(self, team: str, position: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get ordered depth chart for a team, optionally filtered by position."""
        self._ensure_depth_chart()
        team_data = self._depth_by_team.get(team.upper(), {})
        if position:
            return team_data.get(position.upper(), [])

        combined = []
        for pos_players in team_data.values():
            combined.extend(pos_players)
        return sorted(combined, key=lambda x: (x["position"], x["depth_order"]))

    def find_player(self, player_identifier: str) -> Optional[Dict[str, Any]]:
        """Find player metadata by ID, full name, or normalized name."""
        self._ensure_depth_chart()
        norm_target = clean_name(player_identifier)
        for team_data in self._depth_by_team.values():
            for pos_players in team_data.values():
                for p in pos_players:
                    if p["player_id"] == str(player_identifier):
                        return p
                    if p["norm_name"] == norm_target or norm_target in p["norm_name"]:
                        return p
        return None

    def _maybe_enrich_with_nflverse(self) -> None:
        """One-shot nflverse fallback enrichment (depth order / injury gaps).

        Only fills gaps Sleeper left open (missing depth order or missing
        injury status); it never overrides Sleeper-provided values. Rows
        must come from the requested season (the loader's caches are
        season-checked) and must agree with the player's live Sleeper team;
        anything else is stale and discarded.
        """
        if self._season is None or self._nflverse_enriched:
            return
        self._nflverse_enriched = True  # never retry within this manager

        if self._loader is None:
            from fantasy_intel.data.loader import DataLoader
            self._loader = DataLoader()

        try:
            depth_df = self._loader.get_depth_charts(seasons=[self._season])
        except Exception as e:
            logger.warning(f"nflverse depth chart enrichment unavailable for {self._season}: {e}")
            depth_df = pd.DataFrame()

        if not depth_df.empty:
            self._fill_unlisted_depth_orders(depth_df)

        try:
            injury_df = self._loader.get_injuries(seasons=[self._season])
        except Exception as e:
            logger.warning(f"nflverse injury enrichment unavailable for {self._season}: {e}")
            injury_df = pd.DataFrame()

        if not injury_df.empty:
            self._fill_missing_injury_statuses(injury_df)

    def _fill_unlisted_depth_orders(self, depth_df: pd.DataFrame) -> None:
        """Fills depth order only for players Sleeper lists no slot for.

        nflverse rows must match the player's live Sleeper team; a row that
        puts a player on a different team is discarded as stale, leaving the
        player unlisted rather than corrupting the live chart.
        """
        frame = depth_df
        if "week" in frame.columns:
            weeks = pd.to_numeric(frame["week"], errors="coerce")
            latest = weeks.max()
            if pd.notna(latest):
                frame = frame[weeks == latest]
        if "formation" in frame.columns:
            frame = frame[frame["formation"] == "Offense"]
        if {"depth_position", "position"}.issubset(frame.columns):
            # Skip special-teams slots (PR/KR/KOR/FB) that do not rank the
            # base offensive position
            frame = frame[frame["depth_position"] == frame["position"]]
        if frame.empty:
            return

        by_key: Dict[tuple, int] = {}
        for _, row in frame.iterrows():
            name = clean_name(row.get("full_name") or "")
            pos = row.get("position")
            team = row.get("club_code")
            if not name or not team or pos not in OFFENSIVE_POSITIONS:
                continue
            try:
                order = int(row.get("depth_team"))
            except (TypeError, ValueError):
                continue
            key = (name, pos, team)
            if key not in by_key or order < by_key[key]:
                by_key[key] = order

        filled = False
        for team_data in self._depth_by_team.values():
            for pos_list in team_data.values():
                for p in pos_list:
                    if p["depth_is_real"]:
                        continue  # Sleeper value is authoritative
                    order = by_key.get((p["norm_name"], p["position"], p["team"]))
                    if order is not None and 0 < order < UNLISTED_DEPTH_ORDER:
                        p["depth_order"] = order
                        p["depth_is_real"] = True
                        filled = True

        if filled:
            for team_data in self._depth_by_team.values():
                for pos, players in team_data.items():
                    players.sort(key=lambda x: x["depth_order"])

    def _fill_missing_injury_statuses(self, injury_df: pd.DataFrame) -> None:
        """Fills injury status only where Sleeper reports none.

        Uses only rows from the live NFL week's official report (or the
        frame's latest week when the live week is unknown) so an old week's
        report can never be mixed into the current roster. Sleeper-provided
        injury values are never overridden.
        """
        frame = injury_df
        if "week" in frame.columns:
            weeks = pd.to_numeric(frame["week"], errors="coerce")
            live_week = self._live_nfl_week()
            target_week = live_week if live_week is not None else weeks.max()
            if pd.notna(target_week):
                frame = frame[weeks == target_week]
        if frame.empty:
            return

        by_key: Dict[tuple, Dict[str, Any]] = {}
        for _, row in frame.iterrows():
            name = clean_name(row.get("full_name") or "")
            pos = row.get("position")
            team = row.get("team")
            status = row.get("report_status")
            if not name or not team or pos not in OFFENSIVE_POSITIONS:
                continue
            if not isinstance(status, str) or not status.strip():
                continue
            by_key[(name, pos, team)] = {
                "injury_status": status.strip(),
                "injury_body_part": row.get("report_primary_injury"),
            }

        for team_data in self._depth_by_team.values():
            for pos_list in team_data.values():
                for p in pos_list:
                    if p.get("injury_status"):
                        continue  # Sleeper value is authoritative
                    info = by_key.get((p["norm_name"], p["position"], p["team"]))
                    if info:
                        p["injury_status"] = info["injury_status"]
                        p["injury_body_part"] = info["injury_body_part"]

    def _live_nfl_week(self) -> Optional[int]:
        """Current NFL week from Sleeper's state endpoint, if reachable."""
        try:
            state = self.sleeper.get_nfl_state()
            week = int(state.get("display_week") or state.get("week") or 0)
            return week or None
        except Exception:
            return None

    @staticmethod
    def _role_label(player: Dict[str, Any]) -> str:
        """Human-readable depth slot, e.g. 'WR2 (SWR)', 'RB1', 'WR depth'."""
        pos = player["position"]
        order = player["depth_order"]
        slot = player.get("depth_position")
        base = f"{pos}{order}" if 1 <= order <= 4 else f"{pos} depth"
        if slot and slot != pos:
            return f"{base} ({slot})"
        return base

    def get_player_elevation_context(self, player_identifier: str) -> Dict[str, Any]:
        """
        Calculates the situational elevation of a player based on upstream teammate injuries.
        Identifies whether a player is an entrenched starter, an elevated starter (due to
        injuries ahead of them on the depth chart), a potential surge beneficiary, or a
        standard backup.

        Sleeper's live depth data is the source of truth: an "elevated
        starter" narrative only fires when an active-roster teammate ranked
        AHEAD of the player at his position is ruled out — an established
        starter is never described as elevated because a backup behind him
        got hurt, practice-squad players never count as sidelined starters,
        and players without reliable depth data get a muted role note
        instead of an invented vacancy claim.
        """
        self._ensure_depth_chart()
        player = self.find_player(player_identifier)
        if not player:
            return {
                "player_name": player_identifier,
                "tier": "UNKNOWN",
                "is_elevated": False,
                "ir_eligible": False,
                "summary": "Player not found on depth chart",
                "sidelined_ahead": [],
            }

        if _is_practice_squad(player):
            # Practice-squad player: not on the active roster, so no
            # elevation/vacancy narrative (he must be activated to play)
            return {
                "player_id": player["player_id"],
                "player_name": player["name"],
                "team": player["team"],
                "position": player["position"],
                "depth_order": player["depth_order"],
                "tier": "PRACTICE_SQUAD",
                "is_elevated": False,
                "is_temporary_starter": False,
                "ir_eligible": False,
                "own_injury": player.get("injury_status"),
                "own_body_part": player.get("injury_body_part"),
                "healthy_role": f"{player['team']} {player['position']} (practice squad)",
                "sidelined_ahead": [],
                "summary": f"{player['position']} for {player['team']} — practice-squad status; requires activation",
            }

        if self._season is not None:
            # nflverse fallback enrichment runs at most once, and only
            # after a real player was located (never on the not-found path)
            self._maybe_enrich_with_nflverse()
            player = self.find_player(player_identifier) or player

        pos = player["position"]
        team = player["team"]
        depth_order = player["depth_order"]
        depth_is_real = player.get("depth_is_real", depth_order != UNLISTED_DEPTH_ORDER)
        own_injury = player.get("injury_status")
        own_part = player.get("injury_body_part")
        ir_eligible = own_injury in ["Out", "IR", "PUP", "Sus"]

        pos_list = self._depth_by_team.get(team, {}).get(pos, [])
        # Only active-roster teammates ranked AHEAD can vacate a role for
        # this player; practice-squad players never vacate a role, and
        # injuries to backups ranked behind are irrelevant to his role.
        ahead = [
            p for p in pos_list
            if p["depth_order"] < depth_order and not _is_practice_squad(p)
        ]
        out_ahead = [p for p in ahead if p.get("injury_status") in OUT_STATUSES]
        q_ahead = [p for p in ahead if p.get("injury_status") == "Questionable"]

        role_label = self._role_label(player)

        if not depth_is_real:
            # No live source lists a depth slot for this player: drop all
            # elevation/vacancy claims rather than invent one
            return {
                "player_id": player["player_id"],
                "player_name": player["name"],
                "team": team,
                "position": pos,
                "depth_order": depth_order,
                "tier": "DEPTH_UNLISTED",
                "is_elevated": False,
                "is_temporary_starter": False,
                "ir_eligible": ir_eligible,
                "own_injury": own_injury,
                "own_body_part": own_part,
                "healthy_role": f"{team} {pos} (depth unlisted)",
                "sidelined_ahead": [],
                "summary": f"{pos} for {team}; no reliable depth-chart data from live sources",
            }

        is_starter_tier = depth_order == 1 or (pos == "WR" and depth_order <= 2)
        is_temporary = False

        if is_starter_tier:
            if out_ahead:
                # A top-slot player with the starter ahead of him ruled out
                tier = "ELEVATED_STARTER"
                is_elevated = True
                is_temporary = True
                desc = ", ".join(_injury_desc(p) for p in out_ahead)
                verb = "is" if len(out_ahead) == 1 else "are"
                summary = f"Moving into the top {pos} role for {team} while {desc} {verb} sidelined"
            else:
                # Established starter atop the chart: report the role,
                # never claim an injury behind him elevated him
                tier = "ENTRENCHED_STARTER"
                is_elevated = False
                summary = f"{role_label} for {team} — full-time starter role"
        elif ahead and len(out_ahead) == len(ahead):
            tier = "ELEVATED_STARTER"
            is_elevated = True
            is_temporary = True
            desc = ", ".join(_injury_desc(p) for p in out_ahead)
            verb = "is" if len(out_ahead) == 1 else "are"
            summary = f"Moving into the top {pos} role for {team} while {desc} {verb} sidelined"
        elif out_ahead or q_ahead:
            tier = "POTENTIAL_SURGE"
            is_elevated = True
            all_inj = out_ahead + q_ahead
            desc = ", ".join(_injury_desc(p) for p in all_inj)
            summary = f"Target/touch boost if injured teammates sit ({desc})"
        else:
            tier = "BENCH_BACKUP"
            is_elevated = False
            starter_names = ", ".join([p["name"] for p in ahead])
            summary = f"Backup behind healthy starter(s): {starter_names}"

        return {
            "player_id": player["player_id"],
            "player_name": player["name"],
            "team": team,
            "position": pos,
            "depth_order": depth_order,
            "tier": tier,
            "is_elevated": is_elevated,
            "is_temporary_starter": is_temporary,
            "ir_eligible": ir_eligible,
            "own_injury": own_injury,
            "own_body_part": own_part,
            "healthy_role": f"{team} {role_label}",
            "sidelined_ahead": out_ahead + q_ahead,
            "summary": summary,
        }

    def get_next_man_up(self, player_name: str) -> Optional[Dict[str, Any]]:
        """
        Given a player (typically an injured starter), identify the direct
        beneficiary who moves up in the depth chart.
        """
        self._ensure_depth_chart()
        player = self.find_player(player_name)
        if not player:
            return None

        team = player["team"]
        pos = player["position"]
        depth_order = player["depth_order"]

        pos_list = self._depth_by_team.get(team, {}).get(pos, [])
        # Find healthy players ranked below this player
        for candidate in pos_list:
            if candidate["depth_order"] > depth_order:
                # Exclude players also ruled out or on IR
                if candidate.get("injury_status") in OUT_STATUSES:
                    continue
                # Practice-squad players are not reliable next-man-up beneficiaries
                if _is_practice_squad(candidate):
                    continue
                return candidate

        return None

    def get_all_sidelined_starters(self) -> List[Dict[str, Any]]:
        """
        Scan all teams for starters (depth_order == 1 or 2 in WR) sidelined with injury,
        and pair them with their promoted beneficiaries.
        """
        self._ensure_depth_chart()
        sidelined_reports = []

        for team, pos_dict in self._depth_by_team.items():
            for pos, plist in pos_dict.items():
                for p in plist:
                    if _is_practice_squad(p):
                        continue  # a PS player is never a sidelined starter
                    is_starter_tier = (p["depth_order"] == 1) or (pos == "WR" and p["depth_order"] <= 2)
                    is_injured = p.get("injury_status") in ["Out", "IR", "PUP", "Doubtful", "Questionable"]

                    if is_starter_tier and is_injured:
                        beneficiary = self.get_next_man_up(p["name"])
                        sidelined_reports.append({
                            "injured_starter": p,
                            "promoted_beneficiary": beneficiary,
                            "team": team,
                            "position": pos,
                            "severity": "CRITICAL" if p.get("injury_status") in ["Out", "IR", "PUP"] else "MONITOR",
                        })

        return sorted(
            sidelined_reports,
            key=lambda x: (0 if x["severity"] == "CRITICAL" else 1, x["position"]),
        )
