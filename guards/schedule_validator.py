"""Pure Python deterministic cumulative validator for LLM-proposed desalination schedules.

Parses time-blocked production schedules (e.g., "00:00-06:00 at 110 MLD, 14:00-18:00 at 60 MLD")
and simulates the strategic reservoir CUMULATIVELY hour-by-hour starting at current_hour
(default 14:00) through the next 24 hours (14:00 -> 24:00, then 00:00 -> 14:00 next day).

Checks every block and hour against three non-negotiable physical & safety constraints:
1. Physical desalination plant capacity (0 <= output_mld <= DESAL_MAX_CAPACITY_MLD = 120.0 MLD).
2. Water quality & algal-bloom capacity derating limit.
3. Mandatory 35% strategic reservoir storage floor carried forward hour-by-hour.

Any violating block or floor breach flags the schedule as 'REJECTED_BY_CODE' with the
exact first breach hour and projected percentage.
"""
import re
from typing import Mapping, Sequence, Union
from config.constants import (
    DEFAULT_SCADA_HOUR,
    DESAL_BASELINE_OUTPUT_MLD,
    DESAL_MAX_CAPACITY_MLD,
    RESERVOIR_FLOOR_PERCENT,
    RESERVOIR_TOTAL_CAPACITY_ML,
)
from guards.safety_floors import evaluate_quality_and_bloom_derating

# Matches blocks like:
#   "00:00-06:00 at 110 MLD"
#   "14:00-18:00 throttle proposal to 15 MLD"
#   "14:00-18:00 (peak $0.48/kWh): 60 MLD"
_SCHEDULE_BLOCK_RE = re.compile(
    r"(?P<start>\d{1,2}):\d{2}\s*(?:-|–|—|to)\s*(?P<end>\d{1,2}):\d{2}"
    r"[^,;\n]*?"
    r"(?P<mld>-?\d+(?:\.\d+)?)\s*MLD",
    re.IGNORECASE,
)


def parse_schedule_blocks(schedule: Union[str, Sequence[dict]]) -> list[dict]:
    """Extract normalized schedule blocks from an LLM schedule string or list of dicts."""
    if isinstance(schedule, (list, tuple)):
        parsed = []
        for item in schedule:
            if not isinstance(item, dict):
                continue
            start_h = int(item.get("start_hour", 0))
            end_h = int(item.get("end_hour", 24))
            mld = float(item.get("output_mld", item.get("proposed_mld", 0.0)))
            parsed.append({
                "window": f"{start_h:02d}:00-{end_h:02d}:00",
                "start_hour": start_h,
                "end_hour": end_h,
                "proposed_mld": mld,
            })
        return parsed

    if not isinstance(schedule, str) or not schedule.strip():
        return []

    blocks = []
    for match in _SCHEDULE_BLOCK_RE.finditer(schedule):
        start_h = int(match.group("start"))
        end_h = int(match.group("end"))
        mld = float(match.group("mld"))
        blocks.append({
            "window": f"{start_h:02d}:00-{end_h:02d}:00",
            "start_hour": start_h,
            "end_hour": end_h,
            "proposed_mld": mld,
        })
    return blocks


def _hours_in_block(start_h: int, end_h: int) -> list[int]:
    """Return list of hour indices [0..23] covered by [start_h, end_h)."""
    s = start_h % 24
    if start_h == 0 and end_h == 24:
        return list(range(24))
    if end_h == 24:
        return list(range(s, 24))
    e = end_h % 24
    if s < e:
        return list(range(s, e))
    if s > e:
        return list(range(s, 24)) + list(range(0, e))
    return [s]


def _resolve_hourly_demand(
    hour: int,
    demand_by_hour: Mapping[int, float] | None,
    fallback_demand_mld: Sequence[float] | None,
) -> float:
    """Return hourly demand (MLD) for the given clock hour."""
    if demand_by_hour:
        if hour in demand_by_hour:
            return float(demand_by_hour[hour])
        return sum(float(v) for v in demand_by_hour.values()) / max(len(demand_by_hour), 1)
    if fallback_demand_mld:
        seq = [float(x) for x in fallback_demand_mld]
        return seq[hour % len(seq)]
    return 85.0


def validate_production_schedule(
    proposed_schedule: Union[str, Sequence[dict]],
    current_volume_ml: float,
    demand_by_hour: Mapping[int, float] | None = None,
    fallback_demand_mld: Sequence[float] | None = None,
    bloom_severity: str = "NONE",
    turbidity_ntu: float = 1.2,
    chlorine_ppm: float = 1.8,
    max_capacity_mld: float = DESAL_MAX_CAPACITY_MLD,
    total_capacity_ml: float = RESERVOIR_TOTAL_CAPACITY_ML,
    min_floor_percent: float = RESERVOIR_FLOOR_PERCENT,
    current_hour: int = DEFAULT_SCADA_HOUR,
    current_plant_output_mld: float = DESAL_BASELINE_OUTPUT_MLD,
) -> dict:
    """
    Validate an LLM-proposed 24h production schedule using CUMULATIVE hour-by-hour
    reservoir simulation starting at current_hour (default 14:00 -> 24:00 -> 00:00 -> 14:00).

    Every block's min_projected_percent is the lowest carried-forward level within that
    block (including its starting level, so it can never exceed the level at block entry).
    """
    derating = evaluate_quality_and_bloom_derating(
        turbidity_ntu=turbidity_ntu,
        chlorine_ppm=chlorine_ppm,
        bloom_severity=bloom_severity,
    )
    bloom_max_mld = float(derating["max_safe_output_mld"])
    normalized_bloom = derating["bloom_severity"]
    floor_pct_display = min_floor_percent * 100.0
    floor_volume_ml = total_capacity_ml * min_floor_percent

    parsed_blocks = parse_schedule_blocks(proposed_schedule)
    if not parsed_blocks:
        reason = "No parseable hour-range and MLD blocks found in proposed_schedule."
        return {
            "status": "REJECTED_BY_CODE",
            "is_valid": False,
            "current_hour": current_hour,
            "max_capacity_mld": max_capacity_mld,
            "bloom_severity": normalized_bloom,
            "bloom_max_safe_mld": bloom_max_mld,
            "min_overall_percent": round((current_volume_ml / total_capacity_ml) * 100.0, 2),
            "first_breach_hour": None,
            "first_breach_percent": None,
            "cumulative_trajectory": [],
            "blocks": [],
            "gap_blocks": [],
            "chronological_blocks": [],
            "rejected_blocks": [
                {
                    "window": "UNPARSEABLE",
                    "proposed_mld": None,
                    "status": "REJECTED_BY_CODE",
                    "reasons": [reason],
                    "min_projected_percent": None,
                }
            ],
            "summary": f"REJECTED_BY_CODE: {reason}",
        }

    # Map each clock hour [0..23] strictly to its explicitly covered block index.
    # Hours not covered by any block remain uncovered (gaps) and use current_plant_output_mld capped by bloom limit.
    hour_to_block_idx: dict[int, int] = {}
    for idx, blk in enumerate(parsed_blocks):
        hrs = _hours_in_block(blk["start_hour"], blk["end_hour"])
        for h in hrs:
            hour_to_block_idx[h] = idx

    # Per-block accumulators for carried-forward simulation
    block_stats: list[dict] = [
        {
            "first_seen_offset": None,
            "start_pct": None,
            "min_pct": None,
            "end_pct": None,
            "breach_hours": [],
            "breach_min_pct": None,
            "caused_or_deepened_breach": False,
        }
        for _ in parsed_blocks
    ]

    # Cumulative hour-by-hour simulation: 24 hours from current_hour (e.g., 14 -> 23, 0 -> 13)
    volume_ml = float(current_volume_ml)
    start_pct_overall = round((volume_ml / total_capacity_ml) * 100.0, 2)
    min_overall_pct = start_pct_overall
    first_breach_hour: str | None = None
    first_breach_percent: float | None = None
    first_breach_offset: int | None = None

    raw_current_setpoint = min(max(float(current_plant_output_mld), 0.0), max_capacity_mld)
    gap_inflow_mld = min(
        raw_current_setpoint,
        bloom_max_mld if normalized_bloom != "NONE" else max_capacity_mld,
    )
    gap_blocks: list[dict] = []
    active_gap: dict | None = None

    cumulative_trajectory: list[dict] = [
        {
            "hour_offset": 0,
            "clock_hour": current_hour,
            "hour_label": f"{current_hour:02d}:00",
            "volume_ml": round(volume_ml, 2),
            "level_percent": start_pct_overall,
            "inflow_mld": None,
            "demand_mld": None,
            "block_window": "START",
            "is_safe": volume_ml >= floor_volume_ml,
        }
    ]

    def _finalize_gap(gap: dict | None) -> None:
        if gap is None:
            return
        start_h = gap["start_hour"]
        end_h = (gap["last_hour"] + 1) % 24
        end_display = 24 if end_h == 0 and gap["last_hour"] == 23 else end_h
        window_str = f"{start_h:02d}:00-{end_display:02d}:00"
        reasons: list[str] = []
        if gap["caused_or_deepened_breach"] and gap["breach_hours"]:
            breach_hr = gap["breach_hours"][0]
            breach_pct = gap["breach_min_pct"]
            breach_vol = round((breach_pct / 100.0) * total_capacity_ml, 1)
            reasons.append(
                f"SAFETY FLOOR BREACH: Cumulative reservoir level drops to {breach_pct:.1f}% "
                f"({breach_vol:.1f} ML) below the mandatory {floor_pct_display:.1f}% floor "
                f"({floor_volume_ml:.1f} ML) during gap, first breached at {breach_hr}."
            )
        cap_detail = (
            f"current setpoint {raw_current_setpoint:.1f} MLD, bloom-capped to {gap_inflow_mld:.1f} MLD"
            if gap_inflow_mld < raw_current_setpoint
            else f"current setpoint {gap_inflow_mld:.1f} MLD"
        )
        gap_blocks.append({
            "window": window_str,
            "start_hour": start_h,
            "end_hour": end_display,
            "proposed_mld": round(gap_inflow_mld, 2),
            "is_gap": True,
            "gap_note": f"gap ({cap_detail})",
            "status": "REJECTED_BY_CODE" if reasons else "GAP",
            "reasons": reasons,
            "start_projected_percent": gap["start_pct"],
            "min_projected_percent": gap["min_pct"],
            "end_projected_percent": gap["end_pct"],
            "first_seen_offset": gap["first_seen_offset"],
            "effective_capped_mld": round(gap_inflow_mld, 2),
        })

    for offset in range(24):
        clock_h = (current_hour + offset) % 24
        next_clock_h = (current_hour + offset + 1) % 24
        day_suffix = " (+1d)" if (current_hour + offset + 1) >= 24 else ""
        label_end = f"{next_clock_h:02d}:00{day_suffix}"

        b_idx = hour_to_block_idx.get(clock_h)
        entry_pct = round((volume_ml / total_capacity_ml) * 100.0, 2)

        if b_idx is not None:
            if active_gap is not None:
                _finalize_gap(active_gap)
                active_gap = None
            raw_mld = float(parsed_blocks[b_idx]["proposed_mld"])
            # Physical inflow cannot exceed plant max capacity or be negative
            sim_inflow_mld = min(max(raw_mld, 0.0), max_capacity_mld)
            block_window = parsed_blocks[b_idx]["window"]
            st = block_stats[b_idx]
            if st["first_seen_offset"] is None:
                st["first_seen_offset"] = offset
                st["start_pct"] = entry_pct
                st["min_pct"] = entry_pct
            elif entry_pct < st["min_pct"]:
                st["min_pct"] = entry_pct
        else:
            sim_inflow_mld = gap_inflow_mld
            block_window = "GAP"
            if active_gap is None:
                active_gap = {
                    "first_seen_offset": offset,
                    "start_hour": clock_h,
                    "last_hour": clock_h,
                    "start_pct": entry_pct,
                    "min_pct": entry_pct,
                    "end_pct": entry_pct,
                    "breach_hours": [],
                    "breach_min_pct": None,
                    "caused_or_deepened_breach": False,
                }
            else:
                active_gap["last_hour"] = clock_h
                if entry_pct < active_gap["min_pct"]:
                    active_gap["min_pct"] = entry_pct

        demand_mld = _resolve_hourly_demand(clock_h, demand_by_hour, fallback_demand_mld)
        net_delta_ml = (sim_inflow_mld - demand_mld) / 24.0
        prev_volume_ml = volume_ml
        volume_ml = max(0.0, min(total_capacity_ml, volume_ml + net_delta_ml))
        level_pct = round((volume_ml / total_capacity_ml) * 100.0, 2)

        if level_pct < min_overall_pct:
            min_overall_pct = level_pct

        is_hour_safe = volume_ml >= floor_volume_ml
        if not is_hour_safe and first_breach_hour is None:
            first_breach_hour = label_end
            first_breach_percent = level_pct
            first_breach_offset = offset + 1

        if b_idx is not None:
            st = block_stats[b_idx]
            if level_pct < st["min_pct"]:
                st["min_pct"] = level_pct
            st["end_pct"] = level_pct
            if not is_hour_safe:
                st["breach_hours"].append(label_end)
                if st["breach_min_pct"] is None or level_pct < st["breach_min_pct"]:
                    st["breach_min_pct"] = level_pct
                # Blame this block if it crossed below floor OR actively drained (output < demand) while below floor
                if prev_volume_ml >= floor_volume_ml or sim_inflow_mld < demand_mld:
                    st["caused_or_deepened_breach"] = True
        elif active_gap is not None:
            if level_pct < active_gap["min_pct"]:
                active_gap["min_pct"] = level_pct
            active_gap["end_pct"] = level_pct
            if not is_hour_safe:
                active_gap["breach_hours"].append(label_end)
                if active_gap["breach_min_pct"] is None or level_pct < active_gap["breach_min_pct"]:
                    active_gap["breach_min_pct"] = level_pct
                if prev_volume_ml >= floor_volume_ml or sim_inflow_mld < demand_mld:
                    active_gap["caused_or_deepened_breach"] = True

        cumulative_trajectory.append({
            "hour_offset": offset + 1,
            "clock_hour": next_clock_h,
            "hour_label": label_end,
            "volume_ml": round(volume_ml, 2),
            "level_percent": level_pct,
            "inflow_mld": round(sim_inflow_mld, 2),
            "demand_mld": round(demand_mld, 2),
            "block_window": block_window,
            "is_safe": is_hour_safe,
        })

    if active_gap is not None:
        _finalize_gap(active_gap)

    validated_blocks: list[dict] = []
    rejected_blocks: list[dict] = []

    for idx, block in enumerate(parsed_blocks):
        start_h = block["start_hour"]
        end_h = block["end_hour"]
        mld = float(block["proposed_mld"])
        st = block_stats[idx]
        reasons: list[str] = []

        # 1. Physical plant capacity check
        if mld < 0.0:
            reasons.append(f"Proposed output {mld:.1f} MLD cannot be negative.")
        elif mld > max_capacity_mld:
            reasons.append(
                f"Proposed output {mld:.1f} MLD exceeds DESAL-01 max capacity of {max_capacity_mld:.1f} MLD."
            )

        # 2. Algal bloom derating check
        if normalized_bloom != "NONE" and mld > bloom_max_mld:
            reasons.append(
                f"Proposed output {mld:.1f} MLD exceeds {normalized_bloom} algal bloom derating limit "
                f"of {bloom_max_mld:.1f} MLD (derating factor {derating['derating_factor']:.0%})."
            )

        # 3. Cumulative 35% reservoir floor check
        if st["caused_or_deepened_breach"] and st["breach_hours"]:
            breach_hr = st["breach_hours"][0]
            breach_pct = st["breach_min_pct"]
            breach_vol = round((breach_pct / 100.0) * total_capacity_ml, 1)
            reasons.append(
                f"SAFETY FLOOR BREACH: Cumulative reservoir level drops to {breach_pct:.1f}% "
                f"({breach_vol:.1f} ML) below the mandatory {floor_pct_display:.1f}% floor "
                f"({floor_volume_ml:.1f} ML), first breached at {breach_hr}."
            )

        block_status = "REJECTED_BY_CODE" if reasons else "VALID"
        effective_inflow = min(
            max(mld, 0.0),
            max_capacity_mld,
            bloom_max_mld if normalized_bloom != "NONE" else max_capacity_mld,
        )
        entry = {
            "window": block["window"],
            "start_hour": start_h,
            "end_hour": end_h,
            "proposed_mld": mld,
            "is_gap": False,
            "status": block_status,
            "reasons": reasons,
            "start_projected_percent": st["start_pct"],
            "min_projected_percent": st["min_pct"],
            "end_projected_percent": st["end_pct"],
            "first_seen_offset": st["first_seen_offset"] if st["first_seen_offset"] is not None else 99,
            "effective_capped_mld": effective_inflow,
        }
        validated_blocks.append(entry)
        if block_status == "REJECTED_BY_CODE":
            rejected_blocks.append(entry)

    chronological_blocks = sorted(validated_blocks + gap_blocks, key=lambda b: b["first_seen_offset"])

    # Overall schedule is REJECTED_BY_CODE if ANY block failed OR ANY hour breached the 35% floor
    breach_occurred = first_breach_hour is not None
    is_valid = (len(rejected_blocks) == 0) and (not breach_occurred)
    overall_status = "VALID" if is_valid else "REJECTED_BY_CODE"

    if is_valid:
        summary = (
            f"VALID: All {len(validated_blocks)} schedule block(s) passed capacity, bloom derating, "
            f"and 24h cumulative 35.0% reservoir floor checks (min projected: {min_overall_pct:.2f}%)."
        )
    else:
        breach_note = (
            f" First 35.0% floor breach at {first_breach_hour} ({first_breach_percent:.2f}%)."
            if breach_occurred
            else ""
        )
        summary = (
            f"REJECTED_BY_CODE: {len(rejected_blocks)} of {len(validated_blocks)} schedule block(s) "
            f"violated physical or safety constraints.{breach_note}"
        )

    return {
        "status": overall_status,
        "is_valid": is_valid,
        "current_hour": current_hour,
        "max_capacity_mld": max_capacity_mld,
        "bloom_severity": normalized_bloom,
        "bloom_max_safe_mld": bloom_max_mld,
        "min_overall_percent": min_overall_pct,
        "first_breach_hour": first_breach_hour,
        "first_breach_percent": first_breach_percent,
        "first_breach_offset": first_breach_offset,
        "cumulative_trajectory": cumulative_trajectory,
        "blocks": validated_blocks,
        "gap_blocks": gap_blocks,
        "chronological_blocks": chronological_blocks,
        "rejected_blocks": rejected_blocks,
        "summary": summary,
    }
