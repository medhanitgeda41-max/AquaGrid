"""Phase 3 tests — U-5.1 Chlorine drops in Sector 2."""
from unittest.mock import patch
import pytest

from agent.graph import run_agent
from agent.llm_factory import generate_offline_mock_response
from database.connection import get_db_connection
from database.scenarios import get_scenario_by_id, inject_scenario_u51, reset_database
from guards.schedule_validator import validate_production_schedule
from guards.water_quality import evaluate_sector_chlorine
from simulator import LiveScadaSimulator
from tools.write_gateway import (
    execute_human_approval,
    stage_gated_valve_or_setpoint_action,
)


def _force_offline(sp: str, up: str) -> tuple[dict, str]:
    return generate_offline_mock_response(sp, up), "Deterministic Offline Engine"


@pytest.fixture(autouse=True)
def clean_db():
    reset_database()
    yield
    reset_database()


def test_low_chlorine_flagged_non_compliant_and_normal_compliant():
    """Chlorine < 0.20 mg/L -> NON_COMPLIANT with alert banner; normal ~0.5 mg/L -> COMPLIANT."""
    readings = [
        {"sector": "Sector 1", "chlorine_mg_l": 0.51, "timestamp": "2026-11-20 10:00:00"},
        {"sector": "Sector 2", "chlorine_mg_l": 0.12, "timestamp": "2026-11-20 10:00:00"},
    ]
    res = evaluate_sector_chlorine(readings)
    assert res["all_compliant"] is False
    assert res["excluded_from_tariff_scheduling"] == ["Sector 2"]
    assert res["sectors"][0]["status"] == "COMPLIANT"
    assert res["sectors"][1]["status"] == "NON_COMPLIANT"
    assert res["sectors"][1]["timestamp"] == "2026-11-20 10:00:00"
    assert "WATER QUALITY NON-COMPLIANT — SECTOR 2 (0.12 mg/L < 0.20)" in res["alerts"]

    # Normal chlorine across all sectors changes nothing
    normal = evaluate_sector_chlorine([
        {"sector": f"Sector {i}", "chlorine_mg_l": 0.50, "timestamp": "2026-11-20 10:00:00"}
        for i in range(1, 7)
    ])
    assert normal["all_compliant"] is True
    assert normal["excluded_from_tariff_scheduling"] == []
    assert normal["alerts"] == []


def test_tariff_cut_affecting_non_compliant_sector_rejected_water_quality():
    """Schedule validator and gateway reject tariff-driven production cuts affecting Sector 2 with REJECTED_WATER_QUALITY."""
    demands = {h: 85.0 for h in range(24)}
    # Tariff-driven peak cut (60 MLD < 100 MLD) without excluding Sector 2 -> REJECTED_WATER_QUALITY
    res_bad = validate_production_schedule(
        proposed_schedule="00:00-06:00 at 110 MLD, 06:00-14:00 at 100 MLD, 14:00-18:00 at 60 MLD, 18:00-24:00 at 100 MLD",
        current_volume_ml=240.0,
        demand_by_hour=demands,
        current_hour=14,
        current_plant_output_mld=100.0,
        excluded_sectors=["Sector 2"],
    )
    assert res_bad["status"] == "REJECTED_WATER_QUALITY"
    assert res_bad["is_valid"] is False

    # Same schedule with normal chlorine (no excluded sectors) -> VALID
    res_normal = validate_production_schedule(
        proposed_schedule="00:00-06:00 at 110 MLD, 06:00-14:00 at 100 MLD, 14:00-18:00 at 60 MLD, 18:00-24:00 at 100 MLD",
        current_volume_ml=240.0,
        demand_by_hour=demands,
        current_hour=14,
        current_plant_output_mld=100.0,
        excluded_sectors=[],
    )
    assert res_normal["status"] == "VALID"

    # Gateway also rejects SET_PLANT_OUTPUT cut while Sector 2 is NON_COMPLIANT
    inject_scenario_u51()
    gw_cut = stage_gated_valve_or_setpoint_action(
        action_type="SET_PLANT_OUTPUT",
        target_id="DESAL-01",
        proposed_value=60.0,
        rationale="Cut output during peak tariff",
    )
    assert gw_cut["status"] == "REJECTED_WATER_QUALITY"


@patch("agent.nodes.llm_client.query", side_effect=_force_offline)
def test_u51_end_to_end_stages_isolate_and_flush_pending_until_approved(mock_query):
    """U-5.1 stages ISOLATE_AND_FLUSH SECTOR-02 as PENDING (valves unchanged until human Approve)."""
    entry = get_scenario_by_id("U-5.1")
    assert entry is not None and entry["enabled"] is True

    inject_scenario_u51()
    result = run_agent(scenario_id="U-5.1", current_hour=14)

    assert "Sector 2" in result["excluded_sectors"]
    assert "Sector 2" in result["last_user_prompt"]
    assert result["schedule_validation"]["status"] == "VALID"
    assert result["status"] == "AWAITING_HUMAN_APPROVAL"

    gw = result["gateway_response"]
    assert gw["status"] == "STAGED_PENDING_APPROVAL"
    assert gw["action_type"] == "ISOLATE_AND_FLUSH"
    assert gw["target_id"] == "SECTOR-02"
    action_id = gw["action_id"]

    # Sector 2 valves must remain OPEN until human approves
    conn = get_db_connection()
    try:
        s2_before = conn.execute("SELECT status FROM valves WHERE sector = 'Sector 2';").fetchall()
        assert all(r["status"] == "OPEN" for r in s2_before)
        pending = conn.execute("SELECT status FROM pending_actions WHERE id = ?;", (action_id,)).fetchone()
        assert pending["status"] == "PENDING"
    finally:
        conn.close()

    # Human approves -> executes atomically and logs in audit_log
    exec_res = execute_human_approval(action_id, approver="Operator_Nadia")
    assert exec_res["status"] == "EXECUTED"

    conn = get_db_connection()
    try:
        s2_after = conn.execute("SELECT status FROM valves WHERE sector = 'Sector 2';").fetchall()
        assert all(r["status"] == "CLOSED" for r in s2_after)
        audit_nc = conn.execute(
            "SELECT * FROM audit_log WHERE event_type = 'WATER_QUALITY_NON_COMPLIANT' AND target_id = 'SECTOR-02';"
        ).fetchone()
        assert audit_nc is not None
    finally:
        conn.close()


def test_live_simulator_runs_chlorine_compliance_guard():
    """Live simulator evaluates per-sector chlorine guard on every batch (all COMPLIANT normally)."""
    sim = LiveScadaSimulator(interval_seconds=120.0, fault_probability=0.0, real_time_clock=True, run_ai_after_batch=False, seed=42)
    summary = sim.step(force_fault="none")
    comp = summary["sector_water_quality_compliance"]
    assert comp["all_compliant"] is True
    assert len(comp["sectors"]) == 6
