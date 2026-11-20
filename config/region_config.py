"""Configurable region, map, timezone, and real-time scheduler settings for AquaGrid.

Default region: Dubai, UAE (Asia/Dubai).
All values can be overridden via environment variables / .env (see config/settings.py).
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from config.settings import settings


def get_region_tz():
    """Return ZoneInfo for configured region timezone (with safe UTC+4 fallback for Asia/Dubai)."""
    try:
        return ZoneInfo(settings.REGION_TIMEZONE)
    except Exception:
        return timezone(timedelta(hours=4))


def now_local() -> datetime:
    """Return current real local wall-clock time in the configured region timezone."""
    return datetime.now(get_region_tz())


# Region & Map Configuration
REGION_CONFIG: dict = {
    "region_name": settings.REGION_NAME,
    "timezone": settings.REGION_TIMEZONE,
    "map_center": [settings.MAP_CENTER_LAT, settings.MAP_CENTER_LON],
    "map_zoom": settings.MAP_DEFAULT_ZOOM,
    "osm_tiles": "OpenStreetMap",
    "osm_attribution": "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors",
    "live_interval_seconds": settings.LIVE_INTERVAL_SECONDS,
    "demo_accelerated_interval_seconds": settings.DEMO_ACCELERATED_INTERVAL_SECONDS,
    "ui_refresh_seconds": settings.UI_REFRESH_SECONDS,
    "default_fault_probability": settings.DEFAULT_FAULT_PROBABILITY,
    # Energy intensity of RO desalination (~3.5 kWh per m3 produced) for hourly leak cost estimation
    "desal_kwh_per_m3": 3.5,
    "usd_to_aed": 3.6725,
}

# Distinct high-contrast map colors for Sectors 1-6
SECTOR_COLORS: dict[str, str] = {
    "Sector 1": "#38BDF8",  # Sky Blue (Deira / Al Rigga)
    "Sector 2": "#A78BFA",  # Violet (Bur Dubai / Karama)
    "Sector 3": "#2DD4BF",  # Teal (Downtown / Business Bay)
    "Sector 4": "#F59E0B",  # Amber (Jumeirah / Al Wasl — Sector 4 Main)
    "Sector 5": "#FB7185",  # Rose (Al Barsha / Umm Suqeim)
    "Sector 6": "#34D399",  # Emerald (Dubai Marina / JLT)
}

# Severity marker colors per specification
LEAK_SEVERITY_COLORS: dict[str, str] = {
    "Low": "#FACC15",       # Yellow
    "Medium": "#F97316",    # Orange
    "High": "#EF4444",      # Red
    "Critical": "#991B1B",  # Dark Red
    "Resolved": "#6B7280",  # Grey
}
