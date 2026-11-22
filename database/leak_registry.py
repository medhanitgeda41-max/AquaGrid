"""Single Source of Truth for AquaGrid Leaks (Registry + Telemetry Sync + Consistency Check).

Every component in AquaGrid — AI read tools, status bar, 3D digital twin, 2D OpenStreetMap,
leak table, scenarios, live simulator, and HITL approval cards — reads active leaks ONLY
through `get_active_leaks()` from the single `leaks` SQLite table.

Rules:
- Canonical statuses: SUSPECTED, CONFIRMED, UNDER_REPAIR, RESOLVED, FALSE_ALARM.
- Active statuses: SUSPECTED, CONFIRMED, UNDER_REPAIR.
- Never two leak IDs for the same junction (e.g. J-14 always maps to LEAK-001).
"""
from datetime import datetime, timezone
import json
import logging
from typing import Optional

from database.connection import get_db_connection, transaction_scope
from database.dubai_network import (
    DUBAI_PIPE_SEGMENTS,
    JUNCTION_COORDINATES,
    JUNCTION_SECTOR_MAP,
    point_on_polyline,
)

# Authoritative valve mapping per segment (mirrors digital_twin.hydraulics.SEGMENT_VALVE_MAP with zero circular import)
SEGMENT_VALVE_MAP: dict[str, str] = {
    "SEG-S1-01": "V-01", "SEG-S1-02": "V-01", "SEG-S1-03": "V-02", "SEG-S1-04": "V-02", "SEG-S1-05": "V-03",
    "SEG-S2-01": "V-04", "SEG-S2-02": "V-04", "SEG-S2-03": "V-05", "SEG-S2-04": "V-05", "SEG-S2-05": "V-06",
    "SEG-S3-01": "V-07", "SEG-S3-02": "V-08", "SEG-S3-03": "V-08", "SEG-S3-04": "V-09", "SEG-S3-05": "V-09",
    "SEG-S4-01": "V-12", "SEG-S4-02": "V-10", "SEG-S4-03": "V-11", "SEG-S4-04": "V-11", "SEG-S4-05": "V-12",
    "SEG-S5-01": "V-13", "SEG-S5-02": "V-14", "SEG-S5-03": "V-15", "SEG-S5-04": "V-16",
    "SEG-S6-01": "V-17", "SEG-S6-02": "V-18", "SEG-S6-03": "V-19", "SEG-S6-04": "V-20",
}

logger = logging.getLogger("aquagrid.leaks")

CANONICAL_LEAK_STATUSES: tuple[str, ...] = (
    "SUSPECTED",
    "CONFIRMED",
    "UNDER_REPAIR",
    "RESOLVED",
    "FALSE_ALARM",
)

ACTIVE_LEAK_STATUSES: tuple[str, ...] = (
    "SUSPECTED",
    "CONFIRMED",
    "UNDER_REPAIR",
)

# Canonical 1-to-1 mapping for known seeded/primary junctions -> deterministic leak_id
CANONICAL_JUNCTION_LEAK_IDS: dict[str, str] = {
    "J-14": "LEAK-001",
    "J-11": "LEAK-002",
    "J-17": "LEAK-003",
    "J-01": "LEAK-004",
}

# Map pipe_segment_id -> primary junction_id (derived from related_sensor_ids, e.g. 'P-14,F-14' -> 'J-14')
SEGMENT_JUNCTION_MAP: dict[str, str] = {}
JUNCTION_SEGMENT_MAP: dict[str, str] = {}
for _seg in DUBAI_PIPE_SEGMENTS:
    _rel = [x.strip() for x in str(_seg.get("related_sensor_ids", "")).split(",") if x.strip()]
    _jid = _rel[0].replace("P-", "J-") if _rel and _rel[0].startswith("P-") else "J-14"
    SEGMENT_JUNCTION_MAP[_seg["segment_id"]] = _jid
    if _jid not in JUNCTION_SEGMENT_MAP:
        JUNCTION_SEGMENT_MAP[_jid] = _seg["segment_id"]


def normalize_leak_status(raw_status: Optional[str]) -> str:
    """Normalize legacy or canonical leak status strings into one of the 5 canonical status codes."""
    s = str(raw_status or "").strip().upper()
    if s in ("CONFIRMED", "CONFIRMED_PIPE_LEAK"):
        return "CONFIRMED"
    if s in ("SUSPECTED",):
        return "SUSPECTED"
    if s in ("UNDER_REPAIR", "UNDER REPAIR"):
        return "UNDER_REPAIR"
    if s in ("RESOLVED",):
        return "RESOLVED"
    if "FALSE" in s or "SENSOR" in s or s == "FALSE_ALARM":
        return "FALSE_ALARM"
    return "SUSPECTED"


def is_active_leak_status(raw_status: Optional[str]) -> bool:
    """Return True iff the leak status is active (SUSPECTED, CONFIRMED, or UNDER_REPAIR)."""
    return normalize_leak_status(raw_status) in ACTIVE_LEAK_STATUSES


def get_junction_metadata(junction_id: str) -> dict:
    """Resolve a junction ID (e.g. 'J-14') to its pipe segment, sector, isolating valve, and coordinates."""
    jid = str(junction_id or "").strip().upper()
    seg_id = JUNCTION_SEGMENT_MAP.get(jid)
    seg = next((s for s in DUBAI_PIPE_SEGMENTS if s["segment_id"] == seg_id), None)
    if seg is None:
        sec_hint = JUNCTION_SECTOR_MAP.get(jid, "Sector 4")
        seg = next((s for s in DUBAI_PIPE_SEGMENTS if s["sector"] == sec_hint), DUBAI_PIPE_SEGMENTS[0])
        seg_id = seg["segment_id"]

    valve_id = SEGMENT_VALVE_MAP.get(seg_id, seg.get("isolation_valve_id", "V-12"))
    if jid in JUNCTION_COORDINATES:
        lat, lon = JUNCTION_COORDINATES[jid]
    else:
        lat, lon = point_on_polyline(seg["geometry"], fraction=0.5)

    return {
        "junction_id": jid,
        "canonical_leak_id": CANONICAL_JUNCTION_LEAK_IDS.get(jid),
        "pipe_segment_id": seg_id,
        "sector": seg["sector"],
        "isolating_valve_id": valve_id,
        "location_name": f"{seg['street_area']} ({seg['name']})",
        "latitude": float(lat),
        "longitude": float(lon),
        "related_sensor_ids": seg.get("related_sensor_ids", ""),
    }


def _row_to_leak_dict(row, valve_statuses: dict[str, str], seg_map: dict[str, dict]) -> dict:
    """Convert a SQLite `leaks` row into a normalized, canonical leak dictionary."""
    d = dict(row)
    seg_id = str(d.get("pipe_segment_id") or "")
    seg = seg_map.get(seg_id, {})
    jid = str(d.get("junction_id") or "").strip() or SEGMENT_JUNCTION_MAP.get(seg_id, "")
    if not jid:
        rel = [x.strip() for x in str(d.get("related_sensor_ids") or "").split(",") if x.strip()]
        jid = rel[0].replace("P-", "J-") if rel and rel[0].startswith("P-") else "J-14"
    valve_id = str(d.get("isolating_valve_id") or "").strip() or SEGMENT_VALVE_MAP.get(seg_id, seg.get("isolation_valve_id", "V-12"))
    v_status = str(valve_statuses.get(valve_id, "OPEN")).upper()
    raw_status = str(d.get("status") or "SUSPECTED")
    norm_status = normalize_leak_status(raw_status)
    loss_m3h = round(float(d.get("estimated_loss_m3_per_hour") or 0.0), 1)
    updated_at = str(d.get("updated_at") or d.get("detected_at") or "")
    detected_by = str(d.get("detected_by") or "SCADA_MASS_BALANCE")
    is_isolated = (v_status == "CLOSED") or (norm_status in ("RESOLVED", "FALSE_ALARM"))
    effective_loss = 0.0 if is_isolated else loss_m3h
    loc = str(d.get("location_name") or seg.get("street_area") or d.get("sector") or "")

    summary_line = (
        f"{d['leak_id']} at {jid}, {d['sector']}, {norm_status}, "
        f"{effective_loss:.1f} m³/hr (valve {valve_id}: {v_status})"
    )

    return {
        "leak_id": d["leak_id"],
        "junction_id": jid,
        "pipe_segment_id": seg_id,
        "sector": d["sector"],
        "location": loc,
        "location_name": loc,
        "latitude": float(d["latitude"]),
        "longitude": float(d["longitude"]),
        "severity": str(d.get("severity") or "Medium"),
        "status": norm_status,
        "raw_status": raw_status,
        "is_active": norm_status in ACTIVE_LEAK_STATUSES,
        "is_isolated": is_isolated,
        "detected_at": str(d.get("detected_at") or ""),
        "updated_at": updated_at,
        "loss_rate_m3h": effective_loss,
        "raw_loss_rate_m3h": loss_m3h,
        "estimated_loss_m3_per_hour": effective_loss,
        "isolating_valve_id": valve_id,
        "valve_status": v_status,
        "detected_by": detected_by,
        "confidence": round(float(d.get("confidence") or 0.85), 2),
        "related_sensor_ids": str(d.get("related_sensor_ids") or ""),
        "ai_summary": str(d.get("ai_summary") or ""),
        "ai_suggestions_json": str(d.get("ai_suggestions_json") or "{}"),
        "diagnostic": summary_line,
    }


def get_all_leaks(
    sector: Optional[str] = None,
    include_history: bool = True,
    db_path: Optional[str] = None,
) -> list[dict]:
    """
    Read leaks from the single `leaks` table with normalized statuses, junction_id,
    isolating_valve_id, and valve isolation state.
    If `include_history=False`, returns only active leaks (SUSPECTED, CONFIRMED, UNDER_REPAIR).
    """
    conn = get_db_connection(db_path)
    try:
        if sector:
            rows = conn.execute(
                "SELECT * FROM leaks WHERE LOWER(sector) = LOWER(?) ORDER BY leak_id ASC;",
                (sector.strip(),),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM leaks ORDER BY leak_id ASC;").fetchall()
        v_rows = conn.execute("SELECT valve_id, status FROM valves;").fetchall()
    finally:
        conn.close()

    valve_statuses = {r["valve_id"]: str(r["status"]).upper() for r in v_rows}
    seg_map = {s["segment_id"]: s for s in DUBAI_PIPE_SEGMENTS}
    result = [_row_to_leak_dict(r, valve_statuses, seg_map) for r in rows]
    if not include_history:
        result = [lk for lk in result if lk["is_active"]]
    return result


def get_active_leaks(
    sector: Optional[str] = None,
    db_path: Optional[str] = None,
) -> list[dict]:
    """
    THE SINGLE SOURCE OF TRUTH for active leaks (SUSPECTED, CONFIRMED, UNDER_REPAIR).
    Used by AI read tools, status bar, 3D digital twin, 2D map, leak table, scenarios, and approvals.
    """
    return get_all_leaks(sector=sector, include_history=False, db_path=db_path)


def get_leak_by_id_or_junction(
    identifier: str,
    db_path: Optional[str] = None,
) -> Optional[dict]:
    """Look up a leak record by either leak_id (e.g. 'LEAK-001') or junction_id (e.g. 'J-14')."""
    ident = str(identifier or "").strip().upper()
    if not ident:
        return None
    all_leaks = get_all_leaks(include_history=True, db_path=db_path)
    for lk in all_leaks:
        if lk["leak_id"].upper() == ident or lk["junction_id"].upper() == ident:
            return lk
    return None


def upsert_leak_for_junction(
    junction_id: str,
    sector: Optional[str] = None,
    status: str = "CONFIRMED",
    pressure_delta_psi: Optional[float] = None,
    flow_delta_lps: Optional[float] = None,
    severity: Optional[str] = None,
    loss_rate_m3h: Optional[float] = None,
    detected_by: str = "SCADA_MASS_BALANCE",
    summary: Optional[str] = None,
    pipe_segment_id: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: Optional[str] = None,
    confidence: Optional[float] = None,
    db_path: Optional[str] = None,
    conn=None,
) -> dict:
    """
    Create or update the SINGLE leak record for `junction_id` in the `leaks` table.
    Guarantees there are never two leak IDs for the same junction (e.g. J-14 -> LEAK-001).
    """
    jid = str(junction_id or "").strip().upper()
    meta = get_junction_metadata(jid)
    eff_sector = sector or meta["sector"]
    eff_seg_id = pipe_segment_id or meta["pipe_segment_id"]
    eff_valve = SEGMENT_VALVE_MAP.get(eff_seg_id, meta["isolating_valve_id"])
    eff_lat = float(latitude) if latitude is not None else meta["latitude"]
    eff_lon = float(longitude) if longitude is not None else meta["longitude"]
    eff_loc = location_name or meta["location_name"]
    norm_status = normalize_leak_status(status)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Estimate loss rate and severity from hydraulic deltas if not explicitly provided
    if loss_rate_m3h is None:
        if flow_delta_lps is not None and abs(float(flow_delta_lps)) > 1.0:
            # 1 L/s = 3.6 m3/hr; scale realistic burst fraction
            loss_rate_m3h = round(max(12.0, min(140.0, abs(float(flow_delta_lps)) * 0.85)), 1)
        else:
            loss_rate_m3h = 64.5 if jid == "J-14" else 38.0
    if severity is None:
        if float(loss_rate_m3h) >= 80.0:
            severity = "Critical"
        elif float(loss_rate_m3h) >= 45.0:
            severity = "High"
        elif float(loss_rate_m3h) >= 20.0:
            severity = "Medium"
        else:
            severity = "Low"
    eff_conf = float(confidence) if confidence is not None else (0.93 if norm_status == "CONFIRMED" else 0.78)
    eff_summary = summary or (
        f"Hydraulic mass balance confirmed leak at {jid} ({eff_sector}): "
        f"Pressure {pressure_delta_psi:+.1f} psi, Flow {flow_delta_lps:+.1f} L/s."
        if (pressure_delta_psi is not None and flow_delta_lps is not None)
        else f"Leak record at {jid} ({eff_sector}) — {norm_status}, {loss_rate_m3h:.1f} m³/hr."
    )

    def _run(tx) -> str:
        # 1. Check for existing record by junction_id OR canonical leak_id for this junction
        canonical_id = meta["canonical_leak_id"]
        existing = tx.execute(
            "SELECT * FROM leaks WHERE UPPER(junction_id) = ? OR (? IS NOT NULL AND UPPER(leak_id) = ?) ORDER BY leak_id ASC LIMIT 1;",
            (jid, canonical_id, canonical_id),
        ).fetchone()
        if existing is None:
            # Also check legacy rows seeded without junction_id that match pipe_segment_id
            existing = tx.execute(
                "SELECT * FROM leaks WHERE pipe_segment_id = ? AND (junction_id = '' OR junction_id IS NULL) LIMIT 1;",
                (eff_seg_id,),
            ).fetchone()

        if existing is not None:
            lid = existing["leak_id"]
            existing_norm = normalize_leak_status(existing["status"])
            # Preserve UNDER_REPAIR if a field crew is actively repairing it and anomaly persists
            final_status = "UNDER_REPAIR" if (existing_norm == "UNDER_REPAIR" and norm_status == "CONFIRMED") else norm_status
            tx.execute(
                """
                UPDATE leaks
                SET junction_id = ?,
                    sector = ?,
                    severity = ?,
                    estimated_loss_m3_per_hour = ?,
                    confidence = ?,
                    status = ?,
                    updated_at = ?,
                    isolating_valve_id = ?,
                    detected_by = ?,
                    ai_summary = ?
                WHERE leak_id = ?;
                """,
                (
                    jid,
                    eff_sector,
                    severity,
                    round(float(loss_rate_m3h), 1),
                    round(eff_conf, 2),
                    final_status,
                    now_utc,
                    eff_valve,
                    detected_by,
                    eff_summary,
                    lid,
                ),
            )
            return lid

        # Allocate deterministic canonical ID (e.g. J-14 -> LEAK-001) or next unused LEAK-NNN
        if canonical_id:
            lid = canonical_id
        else:
            all_ids = [r["leak_id"] for r in tx.execute("SELECT leak_id FROM leaks;").fetchall()]
            max_num = 0
            for x in all_ids:
                if str(x).upper().startswith("LEAK-"):
                    try:
                        max_num = max(max_num, int(str(x).split("-")[1]))
                    except (IndexError, ValueError):
                        pass
            lid = f"LEAK-{max_num + 1:03d}"

        tx.execute(
            """
            INSERT INTO leaks (
                leak_id, junction_id, pipe_segment_id, sector, latitude, longitude, location_name,
                detected_at, updated_at, severity, estimated_loss_m3_per_hour, confidence, status,
                isolating_valve_id, detected_by, related_sensor_ids, ai_summary, ai_suggestions_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}');
            """,
            (
                lid,
                jid,
                eff_seg_id,
                eff_sector,
                eff_lat,
                eff_lon,
                eff_loc,
                now_utc,
                now_utc,
                severity,
                round(float(loss_rate_m3h), 1),
                round(eff_conf, 2),
                norm_status,
                eff_valve,
                detected_by,
                meta["related_sensor_ids"],
                eff_summary,
            ),
        )
        return lid

    if conn is not None:
        lid = _run(conn)
    else:
        with transaction_scope(db_path) as tx:
            lid = _run(tx)

    return get_leak_by_id_or_junction(lid, db_path=db_path) or {"leak_id": lid, "junction_id": jid, "status": norm_status}


def resolve_telemetry_leak_for_junction(
    junction_id: str,
    reason: str = "Telemetry mass balance returned to nominal.",
    db_path: Optional[str] = None,
) -> bool:
    """
    Mark a telemetry-detected leak (`detected_by == 'SCADA_MASS_BALANCE'`) on `junction_id`
    as RESOLVED once telemetry returns to normal (never resolves manual/field-dispatched tickets
    unless telemetry originally created/updated it).
    """
    jid = str(junction_id or "").strip().upper()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with transaction_scope(db_path) as tx:
        row = tx.execute(
            "SELECT leak_id, status, detected_by FROM leaks WHERE UPPER(junction_id) = ? LIMIT 1;",
            (jid,),
        ).fetchone()
        if not row:
            return False
        if str(row["detected_by"] or "") != "SCADA_MASS_BALANCE":
            return False
        if normalize_leak_status(row["status"]) not in ("CONFIRMED", "SUSPECTED"):
            return False
        tx.execute(
            "UPDATE leaks SET status = 'RESOLVED', updated_at = ?, ai_summary = ? WHERE leak_id = ?;",
            (now_utc, reason, row["leak_id"]),
        )
        return True


def verify_leak_count_consistency(
    counts: dict[str, int],
    context: str = "runtime",
) -> bool:
    """Log an ERROR if active-leak counts across status bar, 3D twin, 2D map, and AI tools disagree."""
    unique_vals = set(int(v) for v in counts.values())
    if len(unique_vals) > 1:
        logger.error("LEAK_COUNT_MISMATCH (%s): %s", context, json.dumps(counts, sort_keys=True))
        return False
    return True
