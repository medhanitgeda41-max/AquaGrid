"""Phase 4 tests — U-5.2 Fire in Sector 6 (dynamic emergency reservoir floor)."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import pytest

from agent.graph import run_agent
from agent.llm_factory import generate_offline_mock_response
from database.connection import get_db_connection
from database.scenarios import get_scenario_by_id, inject_scenario_u52, reset_database
from guards.safety_floors import compute_effective_reservoir_floor, simulate_reservoir_trajectory
from guards.schedule_validator import validate_production_schedule
from tools.write_gateway import (
    add_emergency_commitment,
    stage_gated_valve_or_setpoint_action,
)


def _force_offline(sp: str, up: str) -> tuple[dict, str]:
    return generate_offline_mock_response(sp, up), "Deterministic Offline Engine"


@pytest.fixture(autouse=True)
def clean_db():
    reset_database()
    yield
    reset_database()


def test_dynamic_floor_35_4_inside_window_and_35_0_after():
    """Floor is 35.4% (177.0 ML) inside Civil Defence 3h window and returns to 35.0% (175.0 ML) after."""
    t0 = datetime(2026, 11, 20, 10, 0, 0, tzinfo=timezone.utc)  # 14:00 Dubai
    commitment = {
        "sector": "Sector 6",
        "volume_m3": 2000.0,
        "start_time": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": (t0 + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Civil Defence",
    }

    inside = compute_effective_reservoir_floor([commitment], at_time=t0 + timedelta(hours=1))
    after = compute_effective_reservoir_floor([commitment], at_time=t0 + timedelta(hours=4))
    before = compute_effective_reservoir_floor([commitment], at_time=t0 - timedelta(minutes=5))

    print(
        f"[U-5.2 floor numbers] inside={inside['effective_floor_percent']}% ({inside['effective_floor_ml']} ML) | "
        f"after={after['effective_floor_percent']}% ({after['effective_floor_ml']} ML) | banner='{inside['banner']}'"
    )

    assert inside["effective_floor_ml"] == 177.0
    assert inside["effective_floor_percent"] == 35.4
    assert inside["emergency_percent"] == 0.4
    assert "Reservoir floor: 35.0% + 0.4% emergency = 35.4% (Civil Defence, Sector 6, until 17:00)" == inside["banner"]

    assert after["effective_floor_ml"] == 175.0
    assert after["effective_floor_percent"] == 35.0
    assert after["has_active_emergency"] is False

    assert before["effective_floor_percent"] == 35.0


def test_plan_with_min_35_2_valid_without_commitment_and_rejected_with_it():
    """A plan with minimum projected 35.2% is VALID at 35.0% floor and REJECTED at 35.4% emergency floor."""
    # From 240 ML (48.0%) over 6h peak (demand=312 MLD): 56.0 MLD -> 176.0 ML = 35.20%
    sim_no_em = simulate_reservoir_trajectory(
        current_volume_ml=240.0,
        plant_inflow_mld=56.0,
        demand_forecast_mld=[312.0] * 6,
        min_floor_percent=0.35,
    )
    sim_with_em = simulate_reservoir_trajectory(
        current_volume_ml=240.0,
        plant_inflow_mld=56.0,
        demand_forecast_mld=[312.0] * 6,
        min_floor_percent=0.354,
    )
    print(
        f"[U-5.2 35.2% plan] without_commitment: min={sim_no_em['min_level_percent']}% safe={sim_no_em['is_safe']} | "
        f"with_commitment (35.4%): min={sim_with_em['min_level_percent']}% safe={sim_with_em['is_safe']}"
    )
    assert sim_no_em["min_level_percent"] == 35.2
    assert sim_no_em["is_safe"] is True
    assert sim_with_em["min_level_percent"] == 35.2
    assert sim_with_em["is_safe"] is False

    # Same check through cumulative schedule_validator
    demand_24h = {h: (312.0 if 14 <= h <= 19 else 60.0) for h in range(24)}
    sched_str = "14:00-20:00 at 56 MLD, 20:00-24:00 at 100 MLD, 00:00-14:00 at 100 MLD"
    val_base = validate_production_schedule(sched_str, current_volume_ml=240.0, demand_by_hour=demand_24h, min_floor_percent=0.35)
    val_em = validate_production_schedule(sched_str, current_volume_ml=240.0, demand_by_hour=demand_24h, min_floor_percent=0.354)
    assert val_base["status"] == "VALID"
    assert val_base["min_overall_percent"] == 35.2
    assert val_em["status"] == "REJECTED_BY_CODE"
    assert val_em["min_overall_percent"] == 35.2


@patch("agent.nodes.llm_client.query", side_effect=_force_offline)
def test_pending_action_rechecked_when_commitment_added_and_u52_replans(mock_query):
    """Adding a Civil Defence commitment cancels violating PENDING actions and U-5.2 replans above 35.4%."""
    entry = get_scenario_by_id("U-5.2")
    assert entry is not None and entry["enabled"] is True

    info = inject_scenario_u52()
    old_plan = info["old_plan"]
    cancelled = info["commitment"]["revalidation"]["cancelled"]

    print(
        f"[U-5.2 recheck] old_plan={old_plan['proposed_mld']} MLD min={old_plan['min_projected_percent']}% "
        f"-> {old_plan['recheck_status']} (floor={info['commitment']['effective_floor']['effective_floor_percent']}%)"
    )

    assert old_plan["initial_status"] == "STAGED_PENDING_APPROVAL"
    assert old_plan["min_projected_percent"] == 35.2
    assert len(cancelled) == 1
    assert cancelled[0]["action_id"] == old_plan["action_id"]
    assert cancelled[0]["status"] == "CANCELLED_BY_SAFETY_RECHECK"

    # Confirm DB updated old PENDING row to CANCELLED_BY_SAFETY_RECHECK
    conn = get_db_connection()
    try:
        old_row = conn.execute("SELECT status FROM pending_actions WHERE id = ?;", (old_plan["action_id"],)).fetchone()
        assert old_row["status"] == "CANCELLED_BY_SAFETY_RECHECK"
    finally:
        conn.close()

    # Agent replans against the 35.4% floor -> stages 65.0 MLD (min 35.65% >= 35.4%)
    result = run_agent(scenario_id="U-5.2", current_hour=14)
    gw = result["gateway_response"]
    print(
        f"[U-5.2 replan] new_plan={gw['proposed_value']} MLD min={gw['min_projected_percent']}% "
        f"floor={gw['floor_threshold_percent']}% status={gw['status']}"
    )
    assert gw["status"] == "STAGED_PENDING_APPROVAL"
    assert gw["proposed_value"] == 65.0
    assert gw["min_projected_percent"] >= 35.4
    assert gw["floor_threshold_percent"] == 35.4
    assert result["status"] == "AWAITING_HUMAN_APPROVAL"
