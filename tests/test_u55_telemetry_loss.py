"""Phase 7 tests — U-5.5 SCADA goes quiet (telemetry freshness, TELEMETRY_LOSS, judge controls, restore)."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import pytest

from agent.graph import run_agent
from agent.llm_factory import generate_offline_mock_response
from config.constants import TELEMETRY_MAX_AGE_SECONDS
from database.connection import get_db_connection
from database.scenarios import get_scenario_by_id, inject_scenario_u55, reset_database
from guards.telemetry_freshness import (
    check_db_telemetry_freshness,
    evaluate_telemetry_freshness,
    fast_forward_telemetry_age,
    restore_telemetry_feed,
)
from simulator import LiveScadaSimulator, get_simulator
from tools.write_gateway import (
    execute_human_approval,
    stage_gated_valve_or_setpoint_action,
)


def _force_offline(sp: str, up: str) -> tuple[dict, str]:
    return generate_offline_mock_response(sp, up), "Deterministic Offline Engine"


@pytest.fixture(autouse=True)
def clean_db():
    get_simulator().resume_feed()
    reset_database()
    yield
    get_simulator().resume_feed()
    reset_database()


def test_freshness_check_normal_2min_and_5s_batches_are_never_stale():
    """2-minute (120s) and 5-second batches are FRESH; >180s (e.g., 10 min) is STALE / TELEMETRY_LOSS."""
    now = datetime(2026, 11, 20, 10, 30, 0, tzinfo=timezone.utc)
    t_5s = (now - timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S")
    t_120s = (now - timedelta(seconds=120)).strftime("%Y-%m-%d %H:%M:%S")
    t_10m = (now - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")

    f_5s = evaluate_telemetry_freshness([{"sensor_id": "P-01", "timestamp": t_5s}], now_utc=now)
    f_120s = evaluate_telemetry_freshness([{"sensor_id": "P-01", "timestamp": t_120s}], now_utc=now)
    f_10m = evaluate_telemetry_freshness([{"sensor_id": "P-01", "timestamp": t_10m}], now_utc=now)

    assert TELEMETRY_MAX_AGE_SECONDS == 180
    assert f_5s["status"] == "FRESH" and f_5s["is_telemetry_loss"] is False
    assert f_120s["status"] == "FRESH" and f_120s["is_telemetry_loss"] is False
    assert f_10m["status"] == "TELEMETRY_LOSS" and f_10m["is_telemetry_loss"] is True
    assert "TELEMETRY LOSS — last data" in f_10m["banner"]
    assert "10 min ago" in f_10m["banner"]


@patch("agent.nodes.llm_client.query", side_effect=_force_offline)
def test_u55_telemetry_loss_halts_planning_refuses_proposals_and_approve_and_resumes_after_restore(mock_query):
    """During TELEMETRY_LOSS: nothing staged, setpoint -> REJECTED_STALE_TELEMETRY, Approve -> CANCELLED_STALE_TELEMETRY, planning resumes only after restore."""
    entry = get_scenario_by_id("U-5.5")
    assert entry is not None and entry["enabled"] is True

    # Stage a pending action BEFORE telemetry goes quiet, so we can test Approve refusal during TELEMETRY_LOSS
    pre_staged = stage_gated_valve_or_setpoint_action(
        action_type="CLOSE_VALVE",
        target_id="V-12",
        proposed_value=0.0,
        rationale="Pre-outage valve staging",
    )
    assert pre_staged["status"] == "STAGED_PENDING_APPROVAL"
    pre_action_id = int(pre_staged["action_id"])

    # Trigger U-5.5: stop feed -> fast-forward 10 min -> setpoint request refused
    info = inject_scenario_u55()
    assert info["telemetry_freshness"]["status"] == "TELEMETRY_LOSS"
    assert info["setpoint_refusal"]["status"] == "REJECTED_STALE_TELEMETRY"

    # Agent does NOT plan and stages nothing during TELEMETRY_LOSS
    res_loss = run_agent(scenario_id="U-5.5", current_hour=14)
    assert res_loss["telemetry_loss"] is True
    assert res_loss["proposed_action"] is None
    assert res_loss["gateway_response"] is None
    assert "TELEMETRY LOSS" in res_loss["operator_brief"]
    assert "PAUSED" in res_loss["proposed_schedule"]
    assert res_loss["status"] == "COMPLETED"

    # Clicking Approve on a pending action during TELEMETRY_LOSS is refused with CANCELLED_STALE_TELEMETRY
    app_loss = execute_human_approval(pre_action_id, approver="Operator_Outage")
    assert app_loss["status"] == "CANCELLED_STALE_TELEMETRY"

    conn = get_db_connection()
    try:
        assert conn.execute("SELECT status FROM valves WHERE valve_id = 'V-12';").fetchone()["status"] == "OPEN"
        assert conn.execute("SELECT production_mld FROM plant_setpoints WHERE plant_id = 'DESAL-01';").fetchone()["production_mld"] == 100.0
    finally:
        conn.close()

    # Restore feed -> re-validates readings and allows planning/staging to resume
    get_simulator().resume_feed()
    restored = restore_telemetry_feed(current_hour=14)
    assert restored["status"] == "TELEMETRY_RESTORED"
    assert restored["message"] == "Telemetry restored — readings re-validated"
    assert restored["telemetry_freshness"]["is_telemetry_loss"] is False
    assert check_db_telemetry_freshness()["status"] == "FRESH"

    # Now a safe proposal succeeds again
    resumed = stage_gated_valve_or_setpoint_action("CLOSE_VALVE", "V-12", 0.0, "Post-restore proposal")
    assert resumed["status"] == "STAGED_PENDING_APPROVAL"


def test_simulator_pause_and_resume_feed_controls():
    """LiveScadaSimulator pause_feed() stops new batches cleanly; resume_feed() resumes them."""
    sim = LiveScadaSimulator(interval_seconds=0.05, fault_probability=0.0, start_hour=14, seed=42)
    sim.pause_feed()
    assert sim.is_feed_paused is True
    sim.resume_feed()
    assert sim.is_feed_paused is False
    summary = sim.step(force_fault="none")
    assert summary["telemetry_freshness"]["is_telemetry_loss"] is False
