"""Tool 1 (Read-Only): Network Telemetry Ingestion Service.

Queries live hydraulic network telemetry, cross-checks each sensor using
deterministic Python rules, quarantines drifting/flatlined/impossible sensors,
and returns verified telemetry to the agent.
"""
from typing import Optional
from config.constants import DEFAULT_SCADA_HOUR
from database.connection import get_db_connection, transaction_scope
from guards.injection_filter import inspect_untrusted_text
from guards.sensor_checks import (
    check_flatline,
    check_drift,
    check_bounds,
    check_mass_balance,
)


def read_network_telemetry(
    sector_id: Optional[str] = None,
    current_hour: int = DEFAULT_SCADA_HOUR,
    db_path: Optional[str] = None,
    log_audit: bool = False,
) -> dict:
    """
    Read verified water distribution network telemetry.

    Cross-validates all sensor readings against deterministic Python checks:
    - Quarantines flatlined sensors (e.g. P-17)
    - Quarantines drifting sensors (e.g. T-INTAKE-2)
    - Quarantines physically impossible readings (e.g. negative pressure)
    - Localizes hydraulic mass-balance anomalies (leak signatures)

    Args:
        sector_id: Optional filter for a specific sector (e.g. 'Sector 4').
        db_path: Optional database path override.

    Returns:
        Structured dict containing verified telemetry, quarantined sensors, and leak diagnostics.
    """
    conn = get_db_connection(db_path)
    try:
        latest_hour = current_hour

        # Query all sensors in scope
        sensor_query = "SELECT * FROM sensors"
        params: list = []
        if sector_id:
            sensor_query += " WHERE sector = ?"
            params.append(sector_id)
        sensors = conn.execute(sensor_query, params).fetchall()

        verified_telemetry = []
        quarantined_sensors = []
        leak_alerts = []

        # Gather baseline readings for drift comparisons
        intake_turbidity_vals: list[float] = []
        for s in sensors:
            if s["sensor_type"] == "turbidity" and s["sensor_id"] == "T-INTAKE-1":
                r = conn.execute(
                    "SELECT value FROM telemetry WHERE sensor_id = ? AND hour = ?;",
                    (s["sensor_id"], latest_hour),
                ).fetchone()
                if r:
                    intake_turbidity_vals.append(r["value"])

        for sensor in sensors:
            s_id = sensor["sensor_id"]
            s_type = sensor["sensor_type"]
            j_id = sensor["junction_id"]
            sec = sensor["sector"]
            unit = sensor["unit"]
            nom_min = sensor["nominal_min"]
            nom_max = sensor["nominal_max"]

            # Fetch recent history (up to last 4 hours) for flatline evaluation
            history_rows = conn.execute(
                """
                SELECT hour, value, status, timestamp
                FROM telemetry
                WHERE sensor_id = ? AND hour <= ?
                ORDER BY hour DESC
                LIMIT 4;
                """,
                (s_id, latest_hour),
            ).fetchall()

            if not history_rows:
                continue

            current_row = history_rows[0]
            current_val = float(current_row["value"])
            current_ts = current_row["timestamp"]
            history_vals = [float(r["value"]) for r in history_rows]

            quarantine_reasons = []

            # 1. Bounds check (physically impossible readings)
            if not check_bounds(current_val, min_val=0.0, max_val=200.0):
                quarantine_reasons.append(f"Physically impossible reading: {current_val} {unit}")

            # 2. Flatline check across recent history
            if len(history_vals) >= 3 and check_flatline(history_vals, threshold=0.01):
                quarantine_reasons.append("Zero variance detected across consecutive timestamps (FLATLINE)")

            # 3. Drift check for redundant sensors
            if s_id == "T-INTAKE-2" and intake_turbidity_vals:
                if check_drift(current_val, intake_turbidity_vals, max_deviation=2.0):
                    quarantine_reasons.append(
                        f"Sensor deviation ({current_val:.2f} NTU) deviates significantly from reference Intake-1 ({intake_turbidity_vals[0]:.2f} NTU)"
                    )

            if quarantine_reasons:
                quarantined_sensors.append({
                    "sensor_id": s_id,
                    "sensor_type": s_type,
                    "junction_id": j_id,
                    "sector": sec,
                    "last_reading": current_val,
                    "unit": unit,
                    "timestamp": current_ts,
                    "status": "QUARANTINED",
                    "reasons": quarantine_reasons,
                })
            else:
                verified_telemetry.append({
                    "sensor_id": s_id,
                    "sensor_type": s_type,
                    "junction_id": j_id,
                    "sector": sec,
                    "value": current_val,
                    "unit": unit,
                    "timestamp": current_ts,
                    "nominal_range": f"{nom_min} - {nom_max}",
                    "status": "NORMAL",
                })

        # 4. Sector mass-balance checks for genuine pipe leaks
        # Compare current pressure and flow vs baseline (hour 10) for each junction
        junction_rows = conn.execute(
            """
            SELECT DISTINCT junction_id, sector
            FROM sensors
            WHERE sensor_type IN ('pressure', 'flow')
            """ + (" AND sector = ?" if sector_id else "") + ";",
            params,
        ).fetchall()

        for j in junction_rows:
            j_id = j["junction_id"]
            sec = j["sector"]

            # Pressure readings (latest vs baseline)
            p_latest_row = conn.execute(
                "SELECT value FROM telemetry WHERE junction_id = ? AND sensor_id LIKE 'P-%' AND hour = ?;",
                (j_id, latest_hour),
            ).fetchone()
            p_base_row = conn.execute(
                "SELECT value FROM telemetry WHERE junction_id = ? AND sensor_id LIKE 'P-%' AND hour = 10;",
                (j_id,),
            ).fetchone()

            # Flow readings (latest vs baseline)
            f_latest_row = conn.execute(
                "SELECT value FROM telemetry WHERE junction_id = ? AND sensor_id LIKE 'F-%' AND hour = ?;",
                (j_id, latest_hour),
            ).fetchone()
            f_base_row = conn.execute(
                "SELECT value FROM telemetry WHERE junction_id = ? AND sensor_id LIKE 'F-%' AND hour = 10;",
                (j_id,),
            ).fetchone()

            if p_latest_row and p_base_row and f_latest_row and f_base_row:
                p_delta = float(p_latest_row["value"]) - float(p_base_row["value"])
                f_delta = float(f_latest_row["value"]) - float(f_base_row["value"])

                diagnostic = check_mass_balance(p_delta, f_delta)
                if diagnostic == "LEAK_CONFIRMED":
                    leak_alerts.append({
                        "junction_id": j_id,
                        "sector": sec,
                        "pressure_delta_psi": round(p_delta, 2),
                        "flow_delta_lps": round(f_delta, 2),
                        "status": "CONFIRMED_PIPE_LEAK",
                        "diagnostic": (
                            f"Hydraulic mass balance confirmed leak at {j_id} ({sec}): "
                            f"Pressure dropped {abs(p_delta):.1f} psi while Flow spiked +{f_delta:.1f} L/s"
                        ),
                    })

        # 5. Fetch latest per-sector chlorine residual readings (with timestamps)
        if sector_id:
            wq_rows = conn.execute(
                """
                SELECT sector, chlorine_mg_l, timestamp
                FROM sector_water_quality
                WHERE id IN (
                    SELECT MAX(id) FROM sector_water_quality WHERE sector = ? GROUP BY sector
                )
                ORDER BY sector ASC;
                """,
                (sector_id,),
            ).fetchall()
            ticket_rows = conn.execute(
                """
                SELECT id, sector, text, created_at
                FROM field_tickets
                WHERE sector = ?
                ORDER BY id DESC
                LIMIT 20;
                """,
                (sector_id,),
            ).fetchall()
        else:
            wq_rows = conn.execute(
                """
                SELECT sector, chlorine_mg_l, timestamp
                FROM sector_water_quality
                WHERE id IN (SELECT MAX(id) FROM sector_water_quality GROUP BY sector)
                ORDER BY sector ASC;
                """
            ).fetchall()
            ticket_rows = conn.execute(
                """
                SELECT id, sector, text, created_at
                FROM field_tickets
                ORDER BY id DESC
                LIMIT 20;
                """
            ).fetchall()

        sector_water_quality = [
            {
                "sector": r["sector"],
                "chlorine_mg_l": float(r["chlorine_mg_l"]),
                "unit": "mg/L",
                "timestamp": r["timestamp"],
            }
            for r in wq_rows
        ]
        field_tickets = []
        for r in ticket_rows:
            insp = inspect_untrusted_text(r["text"], source_id=f"TICKET-{r['id']}")
            field_tickets.append({
                "id": r["id"],
                "ticket_ref": f"TICKET-{r['id']}",
                "sector": r["sector"],
                "trust_level": "UNTRUSTED_TEXT",
                "text": insp["sanitized_text"],
                "quoted_data": insp["quoted_data"],
                "is_injection": insp["is_injection"],
                "command_or_approval_claim_detected": insp["command_or_approval_claim_detected"],
                "flagged_patterns": insp["flagged_patterns"],
                "mentioned_valves": insp["mentioned_valves"],
                "created_at": r["created_at"],
            })
        all_timestamps = [
            item["timestamp"]
            for item in (verified_telemetry + quarantined_sensors)
            if item.get("timestamp")
        ]
        latest_telemetry_ts = max(all_timestamps) if all_timestamps else None

        result = {
            "sector_filter": sector_id or "ALL",
            "evaluation_hour": latest_hour,
            "latest_telemetry_timestamp": latest_telemetry_ts,
            "total_sensors_evaluated": len(sensors),
            "verified_sensor_count": len(verified_telemetry),
            "quarantined_sensor_count": len(quarantined_sensors),
            "verified_telemetry": verified_telemetry,
            "quarantined_sensors": quarantined_sensors,
            "leak_alerts": leak_alerts,
            "sector_water_quality": sector_water_quality,
            "field_tickets": field_tickets,
        }
    finally:
        conn.close()

    if log_audit:
        with transaction_scope(db_path) as audit_conn:
            audit_conn.execute(
                """
                INSERT INTO audit_log (event_type, target_id, details, actor, status)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    "TOOL_CALL",
                    "read_network_telemetry",
                    (
                        f"Evaluated {result['total_sensors_evaluated']} sensors at hour {latest_hour}: "
                        f"{result['verified_sensor_count']} verified, {result['quarantined_sensor_count']} quarantined, "
                        f"{len(leak_alerts)} confirmed leak(s)."
                    ),
                    "CODE_GUARD",
                    "ALLOWED",
                ),
            )
            for q in quarantined_sensors:
                audit_conn.execute(
                    """
                    INSERT INTO audit_log (event_type, target_id, details, actor, status)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (
                        "SENSOR_QUARANTINED",
                        q["sensor_id"],
                        f"Sensor {q['sensor_id']} ({q['junction_id']}, {q['sector']}) quarantined: {'; '.join(q['reasons'])}",
                        "CODE_GUARD",
                        "BLOCKED",
                    ),
                )

    return result
