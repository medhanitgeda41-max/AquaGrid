"""Pure Python safety-floor enforcement engine.

Enforces non-negotiable physical constraints:
1. Reservoir storage must remain >= 35% across the forecast horizon.
2. Raw water quality limits (turbidity, chlorine).
3. Algal-bloom capacity derating matrix.

These safety floors deterministically override tariff cost optimization.
"""
from datetime import datetime, timezone
from typing import Optional, Sequence
from config.constants import (
    RESERVOIR_FLOOR_PERCENT,
    RESERVOIR_TOTAL_CAPACITY_ML,
    RESERVOIR_FLOOR_ML,
    TURBIDITY_MAX_NTU,
    CHLORINE_MIN_PPM,
    CHLORINE_MAX_PPM,
    DESAL_MAX_CAPACITY_MLD,
)
from config.region_config import get_region_tz


def _parse_utc_timestamp(ts: object) -> Optional[datetime]:
    """Parse SQLite / ISO timestamp string or datetime into a UTC-aware datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    raw = str(ts).strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def compute_effective_reservoir_floor(
    commitments: Sequence[dict],
    total_capacity_ml: float = RESERVOIR_TOTAL_CAPACITY_ML,
    base_floor_percent: float = RESERVOIR_FLOOR_PERCENT,
    at_time: Optional[datetime] = None,
) -> dict:
    """
    Pure-Python dynamic reservoir safety floor calculation:
        floor_ML = (base_floor_percent * total_capacity_ml) + sum(active emergency commitments in ML)

    A commitment counts ONLY when start_time <= at_time <= end_time.
    Example (500 ML tank + 2,000 m³ Civil Defence fire request = 2.0 ML):
        175.0 ML (35.0%) + 2.0 ML (0.4%) = 177.0 ML (35.4%).
    """
    now_utc = _parse_utc_timestamp(at_time) or datetime.now(timezone.utc)
    base_ml = float(total_capacity_ml) * float(base_floor_percent)
    base_pct = round(float(base_floor_percent) * 100.0, 2)

    active: list[dict] = []
    total_emergency_m3 = 0.0
    for c in commitments or []:
        start_dt = _parse_utc_timestamp(c.get("start_time"))
        end_dt = _parse_utc_timestamp(c.get("end_time"))
        if start_dt is not None and end_dt is not None:
            if not (start_dt <= now_utc <= end_dt):
                continue
        vol_m3 = max(0.0, float(c.get("volume_m3", 0.0)))
        total_emergency_m3 += vol_m3
        active.append(dict(c))

    emergency_ml = total_emergency_m3 / 1000.0  # 1,000 m³ = 1.0 ML
    emergency_pct = (emergency_ml / float(total_capacity_ml)) * 100.0 if total_capacity_ml > 0 else 0.0
    effective_ml = base_ml + emergency_ml
    effective_ratio = effective_ml / float(total_capacity_ml) if total_capacity_ml > 0 else base_floor_percent
    effective_pct = effective_ratio * 100.0

    if active:
        dubai_tz = get_region_tz()
        parts = []
        for c in active:
            end_dt = _parse_utc_timestamp(c.get("end_time"))
            until_str = end_dt.astimezone(dubai_tz).strftime("%H:%M") if end_dt else str(c.get("end_time", ""))
            parts.append(f"{c.get('source', 'Emergency')}, {c.get('sector', '')}, until {until_str}")
        banner = (
            f"Reservoir floor: {base_pct:.1f}% + {emergency_pct:.1f}% emergency = "
            f"{effective_pct:.1f}% ({'; '.join(parts)})"
        )
    else:
        banner = f"Reservoir floor: {base_pct:.1f}%"

    return {
        "base_floor_percent": round(base_pct, 2),
        "base_floor_ml": round(base_ml, 2),
        "emergency_volume_m3": round(total_emergency_m3, 1),
        "emergency_volume_ml": round(emergency_ml, 3),
        "emergency_percent": round(emergency_pct, 2),
        "effective_floor_ml": round(effective_ml, 2),
        "effective_floor_percent": round(effective_pct, 2),
        "effective_floor_ratio": round(effective_ratio, 6),
        "active_commitments": active,
        "has_active_emergency": len(active) > 0,
        "banner": banner,
    }


def simulate_reservoir_trajectory(
    current_volume_ml: float,
    plant_inflow_mld: float,
    demand_forecast_mld: Sequence[float],
    total_capacity_ml: float = RESERVOIR_TOTAL_CAPACITY_ML,
    min_floor_percent: float = RESERVOIR_FLOOR_PERCENT,
) -> dict:
    """
    Deterministically simulate reservoir storage volume hour-by-hour.

    Args:
        current_volume_ml: Starting volume in Megalitres (ML).
        plant_inflow_mld: Proposed production rate from desalination plant in MLD.
        demand_forecast_mld: Sequence of hourly town demand forecasts in MLD.
        total_capacity_ml: Total capacity of the reservoir (default: 500 ML).
        min_floor_percent: Minimum safety floor percentage (default: 0.35 / 35%).

    Returns:
        dict containing:
            'is_safe': bool,
            'min_level_percent': float,
            'min_volume_ml': float,
            'breach_hour': int | None,
            'floor_volume_ml': float,
            'hourly_levels': list[dict],
            'reason': str
    """
    min_floor_volume_ml = total_capacity_ml * min_floor_percent
    volume = current_volume_ml
    hourly_levels = []
    min_vol_reached = volume
    breach_hour = None

    for hour_idx, demand_mld in enumerate(demand_forecast_mld):
        # Convert daily rates (MLD) into 1-hour volume delta (ML)
        hourly_inflow_ml = plant_inflow_mld / 24.0
        hourly_demand_ml = demand_mld / 24.0
        net_delta_ml = hourly_inflow_ml - hourly_demand_ml

        volume = max(0.0, volume + net_delta_ml)
        level_percent = (volume / total_capacity_ml) * 100.0

        if volume < min_vol_reached:
            min_vol_reached = volume

        is_hour_safe = volume >= min_floor_volume_ml
        if not is_hour_safe and breach_hour is None:
            breach_hour = hour_idx + 1

        hourly_levels.append({
            "hour_offset": hour_idx + 1,
            "volume_ml": round(volume, 2),
            "level_percent": round(level_percent, 2),
            "is_safe": is_hour_safe,
        })

    min_level_percent = (min_vol_reached / total_capacity_ml) * 100.0
    is_safe = min_vol_reached >= min_floor_volume_ml

    if is_safe:
        reason = f"Reservoir trajectory remains above {min_floor_percent*100:.1f}% floor (minimum {min_level_percent:.1f}% reached)."
    else:
        reason = (
            f"SAFETY FLOOR BREACH: Reservoir projected to drop to {min_level_percent:.1f}% "
            f"({min_vol_reached:.1f} ML), which is below the mandatory {min_floor_percent*100:.1f}% floor "
            f"({min_floor_volume_ml:.1f} ML) at hour offset +{breach_hour}."
        )

    return {
        "is_safe": is_safe,
        "min_level_percent": round(min_level_percent, 2),
        "min_volume_ml": round(min_vol_reached, 2),
        "floor_volume_ml": round(min_floor_volume_ml, 2),
        "breach_hour": breach_hour,
        "hourly_levels": hourly_levels,
        "reason": reason,
    }


def evaluate_quality_and_bloom_derating(
    turbidity_ntu: float,
    chlorine_ppm: float,
    bloom_severity: str = "NONE",
) -> dict:
    """
    Evaluate raw water quality and calculate maximum safe desalination plant capacity.

    Algal Bloom Derating Rules (Intake 2 affected; Intake 1 remains operational):
    - CRITICAL: Intake 2 isolated — plant capacity capped at 50% (max 60.0 MLD).
    - MODERATE: Plant capacity derated by 50% (max 60.0 MLD).
    - LOW: Plant capacity derated by 20% (max 96.0 MLD).
    - NONE: Full capacity (120.0 MLD) available.

    Potable Standards:
    - Turbidity <= 4.0 NTU
    - Chlorine between 0.5 and 4.0 ppm
    """
    severity = (bloom_severity or "NONE").upper()
    derating_factors = {
        "NONE": 1.0,
        "LOW": 0.80,
        "MODERATE": 0.50,
        "CRITICAL": 0.50,
    }
    derating_factor = derating_factors.get(severity, 0.50)
    max_safe_output_mld = DESAL_MAX_CAPACITY_MLD * derating_factor

    # Water quality threshold violations
    quality_violations = []
    if turbidity_ntu > TURBIDITY_MAX_NTU:
        quality_violations.append(f"Turbidity {turbidity_ntu:.2f} NTU exceeds limit of {TURBIDITY_MAX_NTU} NTU")
    if chlorine_ppm < CHLORINE_MIN_PPM:
        quality_violations.append(f"Chlorine residual {chlorine_ppm:.2f} ppm is below minimum {CHLORINE_MIN_PPM} ppm")
    if chlorine_ppm > CHLORINE_MAX_PPM:
        quality_violations.append(f"Chlorine residual {chlorine_ppm:.2f} ppm exceeds maximum {CHLORINE_MAX_PPM} ppm")

    is_quality_safe = len(quality_violations) == 0

    return {
        "is_quality_safe": is_quality_safe,
        "bloom_severity": severity,
        "derating_factor": derating_factor,
        "max_safe_output_mld": max_safe_output_mld,
        "violations": quality_violations,
    }
