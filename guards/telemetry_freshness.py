"""Pure-Python telemetry freshness & TELEMETRY_LOSS guard (Phase 7 — U-5.5).

Every telemetry reading is checked against TELEMETRY_MAX_AGE_SECONDS (default 180s = 3 min):
- Normal 2-minute batches (120s) and 5s Demo accelerated batches are always FRESH.
- Readings older than TELEMETRY_MAX_AGE_SECONDS are marked STALE.
- When key telemetry is STALE, system enters TELEMETRY_LOSS:
  no new proposals, tariff optimisation paused, Approve refused (CANCELLED_STALE_TELEMETRY),
  last safe state held until feed returns and readings are re-validated.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from config.constants import DEFAULT_SCADA_HOUR, TELEMETRY_MAX_AGE_SECONDS
from config.region_config import get_region_tz
from database.connection import get_db_connection, transaction_scope
from guards.safety_floors import _parse_utc_timestamp


def evaluate_telemetry_freshness(
    readings: Sequence[dict],
    max_age_seconds: int = TELEMETRY_MAX_AGE_SECONDS,
    now_utc: Optional[datetime] = None,
) -> dict:
    """Evaluate freshness of a sequence of readings containing 'timestamp'."""
    ref_utc = _parse_utc_timestamp(now_utc) or datetime.now(timezone.utc)
    dubai_tz = get_region_tz()

    stale_ids: list[str] = []
    fresh_count = 0
    latest_dt: Optional[datetime] = None

    for item in readings or []:
        ts_dt = _parse_utc_timestamp(item.get("timestamp"))
        if ts_dt is None:
            stale_ids.append(str(item.get("sensor_id", "UNKNOWN")))
            continue
        if latest_dt is None or ts_dt > latest_dt:
            latest_dt = ts_dt
        age_sec = (ref_utc - ts_dt).total_seconds()
        if age_sec > max_age_seconds:
            stale_ids.append(str(item.get("sensor_id", "UNKNOWN")))
        else:
            fresh_count += 1

    total = len(readings or [])
    if latest_dt is not None:
        latest_age_sec = max(0.0, (ref_utc - latest_dt).total_seconds())
        last_data_local = latest_dt.astimezone(dubai_tz).strftime("%H:%M:%S")
        last_data_utc = latest_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        latest_age_sec = float(max_age_seconds + 1)
        last_data_local = "UNKNOWN"
        last_data_utc = None

    # Key telemetry is stale if no readings exist, latest timestamp > max_age_seconds, or >=50% stale
    is_loss = (total == 0) or (latest_age_sec > max_age_seconds) or (len(stale_ids) >= max(1, total // 2))
    age_minutes = max(1, int(round(latest_age_sec / 60.0))) if is_loss else round(latest_age_sec / 60.0, 1)

    if is_loss:
        status = "TELEMETRY_LOSS"
        banner = f"TELEMETRY LOSS — last data {last_data_local} ({age_minutes} min ago). Holding last safe state."
    else:
        status = "FRESH"
        banner = f"Telemetry fresh (last data {last_data_local})"

    return {
        "status": status,
        "is_telemetry_loss": is_loss,
        "max_age_seconds": int(max_age_seconds),
        "total_checked": total,
        "fresh_count": fresh_count,
        "stale_count": len(stale_ids),
        "stale_sensor_ids": stale_ids,
        "latest_timestamp_utc": last_data_utc,
        "last_data_local": last_data_local,
        "age_seconds": round(latest_age_sec, 1),
        "age_minutes": age_minutes,
        "banner": banner,
    }


def check_db_telemetry_freshness(
    db_path: Optional[str] = None,
    current_hour: Optional[int] = None,
    max_age_seconds: int = TELEMETRY_MAX_AGE_SECONDS,
    now_utc: Optional[datetime] = None,
    conn=None,
) -> dict:
    """Query the latest telemetry timestamps from SQLite and evaluate TELEMETRY_LOSS."""
    own_conn = conn is None
    db_conn = get_db_connection(db_path) if own_conn else conn
    try:
        if current_hour is not None:
            rows = db_conn.execute(
                "SELECT sensor_id, timestamp FROM telemetry WHERE hour = ?;",
                (current_hour,),
            ).fetchall()
        else:
            rows = db_conn.execute(
                """
                SELECT sensor_id, MAX(timestamp) AS timestamp
                FROM telemetry
                GROUP BY sensor_id;
                """
            ).fetchall()
    finally:
        if own_conn:
            db_conn.close()

    return evaluate_telemetry_freshness(
        readings=[dict(r) for r in rows],
        max_age_seconds=max_age_seconds,
        now_utc=now_utc,
    )


def fast_forward_telemetry_age(
    minutes: int = 10,
    db_path: Optional[str] = None,
) -> dict:
    """
    Simulate telemetry silence by shifting stored telemetry & reservoir timestamps back by `minutes`.
    Logs TELEMETRY_FEED_STALED in audit_log and returns the resulting freshness status.
    """
    stale_utc = datetime.now(timezone.utc) - timedelta(minutes=int(minutes))
    stale_str = stale_utc.strftime("%Y-%m-%d %H:%M:%S")
    with transaction_scope(db_path) as conn:
        conn.execute("UPDATE telemetry SET timestamp = ?;", (stale_str,))
        conn.execute("UPDATE reservoirs SET updated_at = ?;", (stale_str,))
        conn.execute("UPDATE sector_water_quality SET timestamp = ?;", (stale_str,))
        conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                "TELEMETRY_LOSS_DETECTED",
                "SCADA_FEED",
                f"Telemetry feed silent for {minutes} minutes (last timestamp={stale_str} UTC > {TELEMETRY_MAX_AGE_SECONDS}s limit). Tariff optimisation paused; holding last safe state.",
                "CODE_GUARD",
                "TELEMETRY_LOSS",
            ),
        )
    return check_db_telemetry_freshness(db_path=db_path)


def restore_telemetry_feed(
    db_path: Optional[str] = None,
    current_hour: int = DEFAULT_SCADA_HOUR,
) -> dict:
    """
    Restore fresh telemetry timestamps and re-validate all sensor checks (stuck, drift, bounds)
    before planning is allowed to resume.
    """
    from tools.read_telemetry import read_network_telemetry

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with transaction_scope(db_path) as conn:
        conn.execute("UPDATE telemetry SET timestamp = ?;", (now_str,))
        conn.execute("UPDATE reservoirs SET updated_at = ?;", (now_str,))
        conn.execute("UPDATE sector_water_quality SET timestamp = ?;", (now_str,))

    # Mandatory re-validation of stuck, drift, range, and freshness checks before resuming planning
    reval = read_network_telemetry(current_hour=current_hour, db_path=db_path, log_audit=True)
    with transaction_scope(db_path) as conn:
        conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                "TELEMETRY_RESTORED",
                "SCADA_FEED",
                (
                    f"Telemetry restored — readings re-validated at {now_str} UTC "
                    f"(verified={reval['verified_sensor_count']}, quarantined={reval['quarantined_sensor_count']}, "
                    f"freshness={reval['telemetry_freshness']['status']})."
                ),
                "CODE_GUARD",
                "ALLOWED",
            ),
        )

    return {
        "status": "TELEMETRY_RESTORED",
        "message": "Telemetry restored — readings re-validated",
        "restored_at_utc": now_str,
        "telemetry_freshness": reval["telemetry_freshness"],
        "verified_sensor_count": reval["verified_sensor_count"],
        "quarantined_sensor_count": reval["quarantined_sensor_count"],
    }
