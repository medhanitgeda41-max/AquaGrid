"""Tests for the single SCENARIO_REGISTRY in database/scenarios.py and Live Simulation independence."""
from unittest.mock import patch
from agent.graph import run_agent
from agent.llm_factory import generate_offline_mock_response
from database.scenarios import (
    SCENARIO_REGISTRY,
    get_scenario_by_id,
    get_scenarios_by_round,
    reset_database,
)
from simulator import LiveScadaSimulator


def _force_offline(sp: str, up: str) -> tuple[dict, str]:
    return generate_offline_mock_response(sp, up), "Deterministic Offline Engine"


def test_scenario_registry_has_round_1_and_round_2_entries():
    """Registry holds 4 enabled Round 1 core scenarios and 5 disabled Round 2 challenge scenarios."""
    r1 = get_scenarios_by_round(1)
    r2 = get_scenarios_by_round(2)

    assert [s["id"] for s in r1] == ["R1-S1", "R1-S2", "R1-S3", "R1-S4"]
    assert [s["id"] for s in r2] == ["U-5.1", "U-5.2", "U-5.3", "U-5.4", "U-5.5"]
    assert len(SCENARIO_REGISTRY) == 9

    for entry in r1:
        assert entry["enabled"] is True
        assert callable(entry["injector"])
        for field in ("id", "round", "title", "brief", "watch_for", "enabled"):
            assert field in entry and entry[field]

    for entry in r2:
        assert isinstance(entry["enabled"], bool)
        if entry["enabled"]:
            assert callable(entry["injector"])
        for field in ("id", "round", "title", "brief", "watch_for"):
            assert field in entry and entry[field]


def test_get_scenario_by_id_lookup():
    """get_scenario_by_id returns exact registry entries and None for unknown IDs."""
    s1 = get_scenario_by_id("R1-S1")
    assert s1 is not None and s1["title"] == "Leak vs. Broken Sensor"
    u51 = get_scenario_by_id("U-5.1")
    assert u51 is not None and u51["title"] == "Chlorine drops in Sector 2"
    assert get_scenario_by_id("UNKNOWN-99") is None


@patch("agent.nodes.llm_client.query", side_effect=_force_offline)
def test_scenarios_run_independently_with_live_simulator_running(mock_query):
    """Running a Round 1 scenario works cleanly while LiveScadaSimulator is active (and leaves it running)."""
    reset_database()
    sim = LiveScadaSimulator(interval_seconds=120.0, fault_probability=0.0, real_time_clock=True, run_ai_after_batch=False)
    sim.start()
    try:
        assert sim.is_running is True
        entry = get_scenario_by_id("R1-S1")
        entry["injector"]()
        result = run_agent(scenario_id=entry["scenario_num"], current_hour=14)
        assert result["status"] == "AWAITING_HUMAN_APPROVAL"
        assert sim.is_running is True
    finally:
        sim.stop()
        reset_database()
