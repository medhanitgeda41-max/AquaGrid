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


def test_simulator_init_creates_step_lock_and_replaces_stale_singleton():
    """LiveScadaSimulator always initializes step_lock, and get_simulator() replaces stale objects lacking step_lock."""
    import threading
    import simulator as sim_mod

    fresh = LiveScadaSimulator(interval_seconds=60.0, fault_probability=0.0, start_hour=14)
    assert hasattr(fresh, "step_lock")
    assert isinstance(fresh.step_lock, type(threading.Lock()))

    class _OldStaleSimulator:
        is_running = False

    prev = sim_mod._GLOBAL_SIMULATOR
    try:
        sim_mod._GLOBAL_SIMULATOR = _OldStaleSimulator()  # type: ignore[assignment]
        replaced = sim_mod.get_simulator()
        assert isinstance(replaced, LiveScadaSimulator)
        assert hasattr(replaced, "step_lock")
        assert isinstance(replaced.step_lock, type(threading.Lock()))
    finally:
        sim_mod._GLOBAL_SIMULATOR = prev


@patch("agent.nodes.llm_client.query", side_effect=_force_offline)
def test_execute_scenario_with_live_off_live_on_none_and_old_simulator(mock_query):
    """_execute_scenario succeeds with Live Sim OFF, Live Sim ON, sim=None, and an old simulator lacking step_lock."""
    import os
    import sys
    import simulator as sim_mod

    app_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    from components import scenario_selector as sel

    fake_session: dict = {}

    class _OldSimulatorNoLock:
        """Simulates a stale hot-reloaded simulator object without step_lock or resume_feed."""

        def __init__(self, running: bool):
            self.is_running = running

        def stop(self):
            self.is_running = False

    prev_global = sim_mod._GLOBAL_SIMULATOR
    try:
        with patch.object(sel.st, "session_state", fake_session):
            # 1. Live Simulation OFF (stopped simulator — no lock required)
            sim_off = LiveScadaSimulator(interval_seconds=120.0, fault_probability=0.0, real_time_clock=True, run_ai_after_batch=False)
            assert sim_off.is_running is False
            with patch.object(sel, "get_simulator", return_value=sim_off):
                sel._execute_scenario("R1-S1")
            assert fake_session["active_scenario_id"] == "R1-S1"
            assert fake_session["last_agent_result"]["status"] == "AWAITING_HUMAN_APPROVAL"

            # 2. Live Simulation ON (running simulator — serialized with step_lock)
            sim_on = LiveScadaSimulator(interval_seconds=120.0, fault_probability=0.0, real_time_clock=True, run_ai_after_batch=False)
            sim_on.start()
            try:
                assert sim_on.is_running is True
                with patch.object(sel, "get_simulator", return_value=sim_on):
                    sel._execute_scenario("U-5.1")
                assert fake_session["active_scenario_id"] == "U-5.1"
                assert fake_session["last_agent_result"]["status"] == "AWAITING_HUMAN_APPROVAL"
                assert sim_on.is_running is True
            finally:
                sim_on.stop()

            # 3. Simulator is None (runs cleanly without needing a lock)
            with patch.object(sel, "get_simulator", return_value=None):
                sel._execute_scenario("R1-S3")
            assert fake_session["active_scenario_id"] == "R1-S3"
            assert fake_session["last_agent_result"]["status"] == "AWAITING_HUMAN_APPROVAL"

            # 4. Old/stale simulator object without step_lock (stopped AND running)
            old_stopped = _OldSimulatorNoLock(running=False)
            with patch.object(sel, "get_simulator", return_value=old_stopped):
                sel._execute_scenario("R1-S1")
            assert fake_session["active_scenario_id"] == "R1-S1"
            assert not hasattr(old_stopped, "step_lock")  # Stopped simulator didn't even need a lock

            old_running = _OldSimulatorNoLock(running=True)
            with patch.object(sel, "get_simulator", return_value=old_running):
                sel._execute_scenario("R1-S2")
            assert fake_session["active_scenario_id"] == "R1-S2"
            assert hasattr(old_running, "step_lock")  # Safely attached lock for running stale object

            # 5. Stale object stored in module singleton is auto-replaced by get_simulator()
            sim_mod._GLOBAL_SIMULATOR = _OldSimulatorNoLock(running=False)  # type: ignore[assignment]
            sel._execute_scenario("U-5.2")
            assert fake_session["active_scenario_id"] == "U-5.2"
            assert isinstance(sim_mod._GLOBAL_SIMULATOR, LiveScadaSimulator)
            assert hasattr(sim_mod._GLOBAL_SIMULATOR, "step_lock")
    finally:
        sim_mod._GLOBAL_SIMULATOR = prev_global
        reset_database()
