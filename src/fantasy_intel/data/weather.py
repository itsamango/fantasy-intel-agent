"""Static NFL stadium locations and game-time wind forecast lookup via Open-Meteo."""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# team abbreviation -> (latitude, longitude, dome or retractable roof)
# Keyed by nflverse team abbreviation as it appears in schedules home_team/away_team.
STADIUM_LOCATIONS: dict[str, Tuple[float, float, bool]] = {
    "ARI": (33.527, -112.262, True),   # State Farm Stadium (retractable)
    "ATL": (33.755, -84.401, True),    # Mercedes-Benz Stadium (retractable)
    "BAL": (39.278, -76.622, False),   # M&T Bank Stadium
    "BUF": (42.773, -78.787, False),   # Highmark Stadium
    "CAR": (35.228, -80.851, False),   # Bank of America Stadium
    "CHI": (41.862, -87.617, False),   # Soldier Field
    "CIN": (39.096, -84.516, False),   # Paycor Stadium
    "CLE": (41.506, -81.699, False),   # Huntington Bank Field
    "DAL": (32.748, -97.094, True),    # AT&T Stadium (retractable)
    "DEN": (39.744, -105.020, False),  # Empower Field at Mile High
    "DET": (42.340, -83.046, True),    # Ford Field
    "GB": (44.501, -88.062, False),    # Lambeau Field
    "HOU": (29.685, -95.411, True),    # NRG Stadium (retractable)
    "IND": (39.760, -86.164, True),    # Lucas Oil Stadium (retractable)
    "JAX": (30.324, -81.638, False),   # EverBank Stadium
    "KC": (39.049, -94.484, False),    # GEHA Field at Arrowhead Stadium
    "LAC": (33.953, -118.339, True),   # SoFi Stadium
    "LAR": (33.953, -118.339, True),   # SoFi Stadium
    "LV": (36.091, -115.184, True),    # Allegiant Stadium
    "MIA": (25.958, -80.239, False),   # Hard Rock Stadium
    "MIN": (44.974, -93.258, True),    # U.S. Bank Stadium
    "NE": (42.091, -71.264, False),    # Gillette Stadium
    "NO": (29.951, -90.081, True),     # Caesars Superdome
    "NYG": (40.813, -74.074, False),   # MetLife Stadium
    "NYJ": (40.813, -74.074, False),   # MetLife Stadium
    "PHI": (39.900, -75.168, False),   # Lincoln Financial Field
    "PIT": (40.447, -80.016, False),   # Acrisure Stadium
    "SEA": (47.595, -122.332, False),  # Lumen Field
    "SF": (37.403, -121.970, False),   # Levi's Stadium
    "TB": (27.976, -82.504, False),    # Raymond James Stadium
    "TEN": (36.166, -86.771, False),   # Nissan Stadium
    "WAS": (38.866, -76.866, False),   # Commanders Field
}

DOME_TEAMS = frozenset(
    team for team, (_, _, is_dome) in STADIUM_LOCATIONS.items() if is_dome
)


def get_game_wind_forecast(team: str, game_datetime: datetime) -> Optional[float]:
    """Returns the forecast sustained wind (mph) at game time for a team's home stadium.

    game_datetime must be stadium-local (nflverse schedules gameday/gametime are
    stadium-local). Returns None for dome/retractable-roof stadiums, unknown teams,
    games beyond the 16-day forecast horizon, or any lookup failure.
    """
    if not team or game_datetime is None:
        return None

    entry = STADIUM_LOCATIONS.get(str(team).upper())
    if entry is None or entry[2]:
        return None

    latitude, longitude, _ = entry
    try:
        resp = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "windspeed_10m",
                "wind_speed_unit": "mph",
                "timezone": "auto",
                "forecast_days": 16,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Wind forecast lookup failed for {team}: {e}")
        return None

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    speeds = hourly.get("windspeed_10m") or []
    if not times or not speeds:
        return None

    # nearest forecast hour to kickoff; reject matches more than 6 hours away
    best_delta = None
    best_speed = None
    for time_str, speed in zip(times, speeds):
        if speed is None:
            continue
        try:
            hour = datetime.fromisoformat(time_str)
        except ValueError:
            continue
        delta = abs((hour - game_datetime).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_speed = float(speed)

    if best_delta is None or best_delta > 6 * 3600:
        return None
    return best_speed
