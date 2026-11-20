"""Tests for real-time 2-minute scheduler, Dubai pipe snapping, AI Run Log, and Leak Map HITL flow."""
from datetime import datetime, timedelta
import json
from unittest.mock import patch
import pytest

from agent.live_analyzer import run_live_ai_analysis
from config.region_config import REGION_CONFIG, get_region_tz
from database.connection import get_db_connection
from database.dubai_network import (
    DUBAI_PIPE_SEGMENTS,
    distance_to_polyline_meters,
    snap_point_to_polyline,
)
from database.scenarios import reset_database
from simulator import LiveScadaSimulator, realistic_daily_demand_mld, tariff_for_hour
from tools.write_gateway import (
    execute_human_approval,
    update_leak_status_by_operator,
)


@pytest.fixture(autouse=True)
def clean_db():
    reset_database()
    yield
    reset_database()


def test_snap_point_to_polyline_snaps_offroad_point_onto_pipe():
    """Snapping an off-pipe point projects it strictly onto the pipe segment polyline (~0m error)."""
    al_wasl_poly = DUBAI_PIPE_SEGMENTS[15]["geometry"]  # SEG-S4-01 Al Wasl Rd
    off_lat, off_lon = 25.1995, 55.2500  # offset from road centerline

    snapped_lat, snapped_lon, orig_dist_m = snap_point_to_polyline(off_lat, off_lon, al_wasl_poly)
    assert orig_dist_m > 10.0, "Test point should start off the polyline"

    residual_m = distance_to_polyline_meters(snapped_lat, snapped_lon, al_wasl_poly)
    assert residual_m < 0.25, f"Snapped point must lie ON the polyline (got {residual_m}m)"


def test_all_seeded_and_simulated_leaks_sit_on_pipe_segments():
    """All 28 Dubai pipe segments exist across Sectors 1-6, and every seeded & simulated leak sits on its pipe."""
    # Trigger a simulated leak in addition to the 4 seeded leaks
    sim = LiveScadaSimulator(
        interval_seconds=120.0,
        fault_probability=0.0,
        real_time_clock=True,
        run_ai_after_batch=True,
        seed=42,
    )
    summary = sim.step(force_fault="leak")
    assert summary["new_leak_id"] is not None

    conn = get_db_connection()
    try:
        segs = {r["segment_id"]: dict(r) for r in conn.execute("SELECT * FROM pipe_segments;").fetchall()}
        leaks = [dict(r) for r in conn.execute("SELECT * FROM leaks;").fetchall()]
    finally:
        conn.close()

    assert len(segs) == 28
    sectors = {s["sector"] for s in segs.values()}
    assert sectors == {"Sector 1", "Sector 2", "Sector 3", "Sector 4", "Sector 5", "Sector 6"}
    assert len(leaks) >= 5  # 4 seeded + 1 newly simulated

    for lk in leaks:
        seg = segs[lk["pipe_segment_id"]]
        poly = json.loads(seg["geometry_json"])
        dist_m = distance_to_polyline_meters(float(lk["latitude"]), float(lk["longitude"]), poly)
        assert dist_m < 1.0, (
            f"Leak {lk['leak_id']} at ({lk['latitude']}, {lk['longitude']}) is {dist_m:.2f}m off {lk['pipe_segment_id']}!"
        )


def test_realtime_2min_scheduler_uses_real_clock_smooth_reservoir_and_logs_ai_run():
    """
    Real-time 2-minute scheduler uses Dubai local time (HH:MM:SS), realistic diurnal demand,
    smooth 2-minute reservoir changes (never jumping), stores scada_history, and logs an AI run.
    """
    dubai_tz = get_region_tz()
    t_now = datetime(2026, 11, 20, 15, 30, 0, tzinfo=dubai_tz)  # 15:30 Dubai time (Peak tariff)

    sim = LiveScadaSimulator(
        interval_seconds=120.0,
        fault_probability=0.05,
        real_time_clock=True,
        run_ai_after_batch=True,
        use_llm_for_ai=False,
        seed=42,
    )
    summary = sim.step(force_fault="none", now_override=t_now)

    assert summary["real_time_clock"] is True
    assert summary["clock_display"] == "15:30:00"
    assert summary["hour"] == 15
    assert summary["tariff_rate_per_kwh"] == 0.48
    assert summary["tariff_tier"] == "Peak"
    assert sim.last_update_at == t_now
    assert sim.next_update_at == t_now + timedelta(seconds=120)

    # Reservoir change over 2 minutes (1/30 hr) must be tiny and physically smooth (never jump!)
    assert abs(summary["reservoir_level_percent"] - 48.0) < 0.05

    # Diurnal demand curve check: night (03:00) < morning peak (08:30) and evening peak (19:30)
    d_night = realistic_daily_demand_mld(datetime(2026, 11, 20, 3, 0, tzinfo=dubai_tz))
    d_morning = realistic_daily_demand_mld(datetime(2026, 11, 20, 8, 30, tzinfo=dubai_tz))
    d_evening = realistic_daily_demand_mld(datetime(2026, 11, 20, 19, 30, tzinfo=dubai_tz))
    assert d_night < 60.0 < d_morning
    assert d_night < d_evening

    # Tariff bands check
    assert tariff_for_hour(3) == (0.09, 0, "Off-Peak")
    assert tariff_for_hour(10) == (0.20, 0, "Mid-Peak")
    assert tariff_for_hour(15) == (0.48, 1, "Peak")

    # Verify scada_history and ai_run_log both recorded the 2-minute batch
    conn = get_db_connection()
    try:
        hist_cnt = conn.execute("SELECT COUNT(*) AS cnt FROM scada_history WHERE recorded_at = ?;", (summary["recorded_at_utc"],)).fetchone()["cnt"]
        ai_row = conn.execute("SELECT * FROM ai_run_log WHERE trigger_source = 'SCHEDULED_2MIN' ORDER BY run_id DESC LIMIT 1;").fetchone()
    finally:
        conn.close()

    assert hist_cnt == 28  # 1 history row per Dubai pipe segment
    assert ai_row is not None
    assert ai_row["error_message"] is None
    assert len(json.loads(ai_row["grid_suggestions_json"])) >= 2


def test_ai_run_failure_is_logged_and_code_guards_keep_running():
    """If the LLM fails or times out during a 2-min check, the error is logged and CODE checks still succeed."""
    with patch("agent.llm_factory.llm_client.query", side_effect=TimeoutError("Simulated LLM timeout")):
        res = run_live_ai_analysis(current_hour=14, trigger_source="SCHEDULED_2MIN", use_llm=True)

    assert res["status"] == "ERROR_CODE_FALLBACK"
    assert "TimeoutError" in (res["error_message"] or "")
    assert len(res["findings"]) >= 1
    assert len(res["grid_suggestions"]) >= 2

    conn = get_db_connection()
    try:
        row = conn.execute("SELECT status, error_message FROM ai_run_log WHERE run_id = ?;", (res["run_id"],)).fetchone()
    finally:
        conn.close()
    assert row["status"] == "ERROR_CODE_FALLBACK"
    assert "TimeoutError" in row["error_message"]


def test_leak_map_hitl_approval_and_operator_decisions_logged():
    """AI stages valve isolation ONLY as PENDING (valves unchanged); human approval commits; operator leak decisions log to audit_log."""
    # Run AI analysis -> stages CLOSE_VALVE V-12 for High-severity LEAK-001 on SEG-S4-01
    ai_res = run_live_ai_analysis(current_hour=14, trigger_source="SCHEDULED_2MIN", use_llm=False, stage_hitl_actions=True)
    v12_staged = [a for a in ai_res["staged_actions"] if a["valve_id"] == "V-12"]
    assert len(v12_staged) >= 1
    action_id = v12_staged[0]["gateway"]["action_id"]

    # Valve V-12 MUST still be OPEN before human approval
    conn = get_db_connection()
    try:
        assert conn.execute("SELECT status FROM valves WHERE valve_id = 'V-12';").fetchone()["status"] == "OPEN"
    finally:
        conn.close()

    # Human approves -> commits CLOSED
    exec_res = execute_human_approval(action_id, approver="Operator_Tariq")
    assert exec_res["status"] == "EXECUTED"

    # Human marks LEAK-003 as False Alarm and LEAK-001 as Resolved
    fa = update_leak_status_by_operator("LEAK-003", "False Alarm – Sensor Fault", operator_name="Operator_Tariq")
    res_op = update_leak_status_by_operator("LEAK-001", "Resolved", operator_name="Operator_Tariq")
    assert fa["status"] == "UPDATED"
    assert res_op["status"] == "UPDATED"

    conn = get_db_connection()
    try:
        assert conn.execute("SELECT status FROM valves WHERE valve_id = 'V-12';").fetchone()["status"] == "CLOSED"
        assert conn.execute("SELECT status FROM leaks WHERE leak_id = 'LEAK-001';").fetchone()["status"] == "Resolved"
        assert conn.execute("SELECT status FROM leaks WHERE leak_id = 'LEAK-003';").fetchone()["status"] == "False Alarm – Sensor Fault"
        audit_cnt = conn.execute(
            "SELECT COUNT(*) AS cnt FROM audit_log WHERE event_type = 'LEAK_STATUS_UPDATED' AND details LIKE '%Operator_Tariq%';"
        ).fetchone()["cnt"]
        assert audit_cnt == 2
    finally:
        conn.close()
