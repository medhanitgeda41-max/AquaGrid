"""Unified pure-Python real-time reservoir depletion & projection model.

Used everywhere in AquaGrid — live simulation, scenarios, schedule validator,
chat answers, status bar, and 3D digital twin — so every number always agrees:

    level_ML(t + dt) = clamp(0, capacity_ML, level_ML(t) + (production_MLD - demand_MLD) * dt_hours / 24)

Capacity defaults to 500.0 ML. Effective production is the current DESAL-01 setpoint
capped by active algal-bloom / raw-water quality derating.
"""
from typing import Mapping, Optional, Sequence

from config.constants import (
    DEFAULT_SCADA_HOUR,
    DESAL_MAX_CAPACITY_MLD,
    RESERVOIR_FLOOR_PERCENT,
    RESERVOIR_TOTAL_CAPACITY_ML,
)
from guards.safety_floors import evaluate_quality_and_bloom_derating


def step_reservoir_volume_ml(
    current_volume_ml: float,
    production_mld: float,
    demand_mld: float,
    dt_hours: float = 1.0,
    capacity_ml: float = RESERVOIR_TOTAL_CAPACITY_ML,
) -> float:
    """
    Advance reservoir volume (ML) by `dt_hours` using the exact shared balance equation:
        level_ML(t + dt) = level_ML(t) + (production_MLD - demand_MLD) * dt_hours / 24
    Clamped strictly to [0.0, capacity_ml].
    """
    cap = max(1e-6, float(capacity_ml))
    delta_ml = (float(production_mld) - float(demand_mld)) * (float(dt_hours) / 24.0)
    return max(0.0, min(cap, float(current_volume_ml) + delta_ml))


def compute_effective_production_mld(
    setpoint_mld: float,
    bloom_severity: str = "NONE",
    turbidity_ntu: float = 1.2,
    chlorine_ppm: float = 1.8,
    max_capacity_mld: float = DESAL_MAX_CAPACITY_MLD,
) -> float:
    """Return effective DESAL-01 production (MLD), clamped to [0, max_capacity_mld] and capped by algae derating."""
    raw = min(max(float(setpoint_mld), 0.0), float(max_capacity_mld))
    derating = evaluate_quality_and_bloom_derating(
        turbidity_ntu=turbidity_ntu,
        chlorine_ppm=chlorine_ppm,
        bloom_severity=bloom_severity,
    )
    if str(derating.get("bloom_severity", "NONE")).upper() != "NONE":
        return min(raw, float(derating["max_safe_output_mld"]))
    return raw


def format_hours_minutes(hours: float) -> str:
    """Format positive fractional hours as 'X h Y m' (or 'X h' / 'Y m' cleanly)."""
    total_mins = max(0, int(round(float(hours) * 60.0)))
    h, m = divmod(total_mins, 60)
    if h == 0:
        return f"{m} m"
    if m == 0:
        return f"{h} h"
    return f"{h} h {m} m"


def _fmt_floor_pct(floor_percent: float) -> str:
    val = round(float(floor_percent), 1)
    return f"{int(val)}%" if abs(val - round(val)) < 1e-6 else f"{val:.1f}%"


def evaluate_reservoir_status(
    current_volume_ml: float,
    production_mld: float,
    demand_mld: float,
    floor_percent: float = RESERVOIR_FLOOR_PERCENT * 100.0,
    capacity_ml: float = RESERVOIR_TOTAL_CAPACITY_ML,
    bloom_severity: str = "NONE",
    turbidity_ntu: float = 1.2,
    chlorine_ppm: float = 1.8,
    apply_bloom_cap: bool = True,
) -> dict:
    """
    Compute instantaneous reservoir rate (%/h), trend arrow, time to safety floor,
    and warning state (NORMAL / WARNING within 5% / CRITICAL at or below floor).
    """
    cap = max(1e-6, float(capacity_ml))
    vol = max(0.0, min(cap, float(current_volume_ml)))
    level_pct = (vol / cap) * 100.0
    floor_pct = float(floor_percent)
    floor_ml = cap * (floor_pct / 100.0)
    floor_tag = _fmt_floor_pct(floor_pct)

    eff_prod = (
        compute_effective_production_mld(
            setpoint_mld=production_mld,
            bloom_severity=bloom_severity,
            turbidity_ntu=turbidity_ntu,
            chlorine_ppm=chlorine_ppm,
        )
        if apply_bloom_cap
        else max(0.0, float(production_mld))
    )
    dem = max(0.0, float(demand_mld))

    rate_ml_per_hour = (eff_prod - dem) / 24.0
    rate_pct_per_hour = (rate_ml_per_hour / cap) * 100.0

    # Trend arrow & formatted rate string
    if rate_pct_per_hour < -0.005:
        trend_arrow = "↓"
        rate_str = f"↓ {abs(rate_pct_per_hour):.2f} %/h" if abs(rate_pct_per_hour) < 0.1 else f"↓ {abs(rate_pct_per_hour):.1f} %/h"
    elif rate_pct_per_hour > 0.005:
        trend_arrow = "↑"
        rate_str = f"↑ {abs(rate_pct_per_hour):.2f} %/h" if abs(rate_pct_per_hour) < 0.1 else f"↑ {abs(rate_pct_per_hour):.1f} %/h"
    else:
        trend_arrow = "→"
        rate_str = "→ 0.0 %/h"

    # Time to safety floor
    at_or_below_floor = level_pct <= floor_pct + 1e-6
    within_warning_band = (not at_or_below_floor) and (level_pct <= floor_pct + 5.0 + 1e-6)
    if at_or_below_floor:
        hours_to_floor: Optional[float] = 0.0
        floor_eta_str = f"at/below {floor_tag} floor"
        warning_state = "CRITICAL"
    elif rate_ml_per_hour < -1e-6:
        hours_to_floor = (vol - floor_ml) / abs(rate_ml_per_hour)
        floor_eta_str = f"{floor_tag} floor in {format_hours_minutes(hours_to_floor)}"
        warning_state = "WARNING" if within_warning_band else "NORMAL"
    elif rate_ml_per_hour > 1e-6:
        hours_to_floor = None
        floor_eta_str = "rising"
        warning_state = "WARNING" if within_warning_band else "NORMAL"
    else:
        hours_to_floor = None
        floor_eta_str = "stable"
        warning_state = "WARNING" if within_warning_band else "NORMAL"

    return {
        "volume_ml": round(vol, 2),
        "capacity_ml": round(cap, 1),
        "level_percent": round(level_pct, 2),
        "floor_percent": round(floor_pct, 2),
        "floor_volume_ml": round(floor_ml, 2),
        "effective_production_mld": round(eff_prod, 2),
        "demand_mld": round(dem, 2),
        "rate_ml_per_hour": round(rate_ml_per_hour, 4),
        "rate_pct_per_hour": round(rate_pct_per_hour, 3),
        "trend_arrow": trend_arrow,
        "rate_str": rate_str,
        "hours_to_floor": round(hours_to_floor, 3) if hours_to_floor is not None else None,
        "floor_eta_str": floor_eta_str,
        "warning_state": warning_state,
        "within_warning_band": within_warning_band,
        "at_or_below_floor": at_or_below_floor,
    }


def project_reservoir_series(
    current_volume_ml: float,
    current_hour: int = DEFAULT_SCADA_HOUR,
    production_mld: float = 100.0,
    demand_by_hour: Optional[Mapping[int, float]] = None,
    floor_percent: float = RESERVOIR_FLOOR_PERCENT * 100.0,
    capacity_ml: float = RESERVOIR_TOTAL_CAPACITY_ML,
    history_hours: int = 6,
    forecast_hours: int = 12,
    history_points: Optional[Sequence[dict]] = None,
    bloom_severity: str = "NONE",
    turbidity_ntu: float = 1.2,
    chlorine_ppm: float = 1.8,
) -> list[dict]:
    """
    Build a combined (last `history_hours` + next `forecast_hours`) reservoir level series
    using the unified reservoir balance model, with the active safety floor attached to every point.
    """
    cap = max(1e-6, float(capacity_ml))
    eff_prod = compute_effective_production_mld(
        setpoint_mld=production_mld,
        bloom_severity=bloom_severity,
        turbidity_ntu=turbidity_ntu,
        chlorine_ppm=chlorine_ppm,
    )
    dem_map = {int(k): float(v) for k, v in (demand_by_hour or {}).items()}
    avg_dem = sum(dem_map.values()) / max(1, len(dem_map)) if dem_map else 85.0

    def _dem(h: int) -> float:
        return dem_map.get(int(h) % 24, avg_dem)

    rows: list[dict] = []

    # 1. Last 6h history (from recorded history_points if available, else back-projected cleanly)
    if history_points and len(history_points) >= 2:
        trimmed = list(history_points)[-history_hours:]
        for idx, pt in enumerate(trimmed):
            off = idx - len(trimmed)
            rows.append(
                {
                    "hour_offset": off,
                    "hour_label": str(pt.get("hour_label", f"{(current_hour + off) % 24:02d}:00")),
                    "level_percent": round(float(pt["level_percent"]), 2),
                    "volume_ml": round(float(pt.get("volume_ml", float(pt["level_percent"]) * cap / 100.0)), 2),
                    "floor_percent": round(float(floor_percent), 2),
                    "segment": "History (last 6h)",
                }
            )
    else:
        # Back-project backwards hour-by-hour so history joins seamlessly with current volume at offset 0
        back_vols = [float(current_volume_ml)]
        vol_b = float(current_volume_ml)
        for step_back in range(1, history_hours + 1):
            prev_h = (current_hour - step_back) % 24
            # Inverse of step: vol_prev = vol_curr - (prod - dem)/24
            vol_b = max(0.0, min(cap, vol_b - (eff_prod - _dem(prev_h)) / 24.0))
            back_vols.append(vol_b)
        for step_back in range(history_hours, 0, -1):
            h_lbl = f"{(current_hour - step_back) % 24:02d}:00"
            v = back_vols[step_back]
            rows.append(
                {
                    "hour_offset": -step_back,
                    "hour_label": h_lbl,
                    "level_percent": round((v / cap) * 100.0, 2),
                    "volume_ml": round(v, 2),
                    "floor_percent": round(float(floor_percent), 2),
                    "segment": "History (last 6h)",
                }
            )

    # Current point (offset 0) joins History and Projection smoothly
    curr_vol = max(0.0, min(cap, float(current_volume_ml)))
    curr_pct = round((curr_vol / cap) * 100.0, 2)
    curr_lbl = f"{current_hour % 24:02d}:00 (Now)"
    rows.append(
        {
            "hour_offset": 0,
            "hour_label": curr_lbl,
            "level_percent": curr_pct,
            "volume_ml": round(curr_vol, 2),
            "floor_percent": round(float(floor_percent), 2),
            "segment": "History (last 6h)",
        }
    )
    rows.append(
        {
            "hour_offset": 0,
            "hour_label": curr_lbl,
            "level_percent": curr_pct,
            "volume_ml": round(curr_vol, 2),
            "floor_percent": round(float(floor_percent), 2),
            "segment": "Projection (next 12h)",
        }
    )

    # 2. Next 12h forward projection using the EXACT shared step_reservoir_volume_ml
    vol_f = curr_vol
    for off in range(1, forecast_hours + 1):
        h_step = (current_hour + off - 1) % 24
        next_h = (current_hour + off) % 24
        day_sfx = " (+1d)" if (current_hour + off) >= 24 else ""
        vol_f = step_reservoir_volume_ml(
            current_volume_ml=vol_f,
            production_mld=eff_prod,
            demand_mld=_dem(h_step),
            dt_hours=1.0,
            capacity_ml=cap,
        )
        rows.append(
            {
                "hour_offset": off,
                "hour_label": f"{next_h:02d}:00{day_sfx}",
                "level_percent": round((vol_f / cap) * 100.0, 2),
                "volume_ml": round(vol_f, 2),
                "floor_percent": round(float(floor_percent), 2),
                "segment": "Projection (next 12h)",
            }
        )

    return rows
