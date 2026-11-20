"""Unit and safety-boundary tests for the optional Live SCADA Simulator (simulator.py)."""
from pathlib import Path
import re
import time
import pytest

from database.connection import get_db_connection
from database.scenarios import reset_database
from simulator import LiveScadaSimulator
from tools.read_telemetry import read_network_telemetry


@pytest.fixture(autouse=True)
def clean_db():
    """Ensure every simulator test starts and ends with the fixed seeded baseline."""
    reset_database()
    yield
    reset_database()


def test_simulator_advances_clock_updates_reservoir_and_switches_tariff_tiers():
    """Simulator ticks advance the clock, update reservoir math, write noisy telemetry, and switch tariff tiers."""
    sim = LiveScadaSimulator(interval_seconds=0.1, fault_probability=0.0, start_hour=17, seed=42)

    # Tick 1: 17 -> 18 (still PEAK at $0.48/kWh)
    s1 = sim.step(force_fault="none")
    assert s1["hour"] == 18
    assert s1["tariff_rate_per_kwh"] == 0.48
    assert s1["is_peak_tariff"] is True

    # Tick 2: 18 -> 19 (switches to evening MID-PEAK at $0.20/kWh)
    s2 = sim.step(force_fault="none")
    assert s2["hour"] == 19
    assert s2["tariff_rate_per_kwh"] == 0.20
    assert s2["is_peak_tariff"] is False

    # Advance through midnight to 01:00 (OFF-PEAK at $0.09/kWh)
    s_last = s2
    while s_last["hour"] != 1:
        s_last = sim.step(force_fault="none")
    assert s_last["hour"] == 1
    assert s_last["tariff_rate_per_kwh"] == 0.09
    assert s_last["is_peak_tariff"] is False

    # Verify DB tariffs, reservoir, and telemetry were updated
    conn = get_db_connection()
    try:
        t1 = conn.execute("SELECT rate_per_kwh, is_peak FROM tariffs WHERE hour = 1;").fetchone()
        assert float(t1["rate_per_kwh"]) == 0.09
        assert int(t1["is_peak"]) == 0

        res = conn.execute("SELECT volume_ml, level_percent FROM reservoirs WHERE id = 'RES-01';").fetchone()
        assert float(res["level_percent"]) == s_last["reservoir_level_percent"]

        tel_count = conn.execute("SELECT COUNT(*) AS cnt FROM telemetry WHERE hour = 1;").fetchone()["cnt"]
        assert tel_count == 44  # 20 pressure + 20 flow + 4 intake sensors
    finally:
        conn.close()


def test_simulator_injects_leak_fault_detected_by_code_guards():
    """Forced pipe-leak fault must be caught by pure-Python mass-balance code guards and logged as an alert."""
    sim = LiveScadaSimulator(interval_seconds=0.1, fault_probability=0.0, start_hour=14, seed=42)
    summary = sim.step(force_fault="leak")

    assert summary["injected_fault"] is not None
    assert summary["injected_fault"]["type"] == "PIPE_LEAK"
    leak_target = summary["injected_fault"]["target"]

    # Code guard check must confirm the leak
    telemetry_check = read_network_telemetry(current_hour=summary["hour"])
    leak_jids = [lk["junction_id"] for lk in telemetry_check["leak_alerts"]]
    assert leak_target in leak_jids, f"Expected leak at {leak_target} to be confirmed; got {leak_jids}"
    assert any("LEAK CONFIRMED" in a for a in summary["raised_alerts"])

    # Alert must be recorded in quality_alerts
    conn = get_db_connection()
    try:
        alert = conn.execute(
            "SELECT * FROM quality_alerts WHERE location = ? ORDER BY id DESC LIMIT 1;",
            (leak_target,),
        ).fetchone()
        assert alert is not None
        assert "LIVE SIM FAULT" in alert["raw_text"]
    finally:
        conn.close()


def test_simulator_injects_stuck_sensor_quarantined_by_code_guards():
    """Forced stuck-sensor fault must be quarantined as FLATLINE by pure-Python code guards."""
    sim = LiveScadaSimulator(interval_seconds=0.1, fault_probability=0.0, start_hour=14, seed=7)
    summary = sim.step(force_fault="stuck_sensor")

    assert summary["injected_fault"] is not None
    assert summary["injected_fault"]["type"] == "STUCK_SENSOR"
    stuck_id = summary["injected_fault"]["target"]

    telemetry_check = read_network_telemetry(current_hour=summary["hour"])
    quarantined_ids = [s["sensor_id"] for s in telemetry_check["quarantined_sensors"]]
    assert stuck_id in quarantined_ids, f"Expected stuck sensor {stuck_id} to be quarantined; got {quarantined_ids}"
    assert any("SENSOR QUARANTINED" in a and stuck_id in a for a in summary["raised_alerts"])


def test_simulator_never_writes_to_valves_or_plant_setpoints():
    """
    Hard architectural invariant:
    1. Static check: simulator.py contains zero INSERT/UPDATE/DELETE statements on valves or plant_setpoints.
    2. Runtime check: running 10 ticks with fault_probability=1.0 leaves valves and plant_setpoints unchanged.
    """
    sim_path = Path(__file__).resolve().parent.parent / "simulator.py"
    source = sim_path.read_text(encoding="utf-8")

    forbidden_sql = re.compile(
        r"\b(INSERT\s+INTO|UPDATE|DELETE\s+FROM|REPLACE\s+INTO)\s+(valves|plant_setpoints)\b",
        re.IGNORECASE,
    )
    assert not forbidden_sql.search(source), (
        "simulator.py must NEVER write to valves or plant_setpoints!"
    )

    conn = get_db_connection()
    try:
        valves_before = {
            r["valve_id"]: r["status"]
            for r in conn.execute("SELECT valve_id, status FROM valves ORDER BY valve_id;").fetchall()
        }
        plants_before = {
            r["plant_id"]: float(r["production_mld"])
            for r in conn.execute("SELECT plant_id, production_mld FROM plant_setpoints ORDER BY plant_id;").fetchall()
        }
    finally:
        conn.close()

    sim = LiveScadaSimulator(interval_seconds=0.1, fault_probability=1.0, start_hour=14, seed=99)
    for _ in range(10):
        sim.step()

    conn = get_db_connection()
    try:
        valves_after = {
            r["valve_id"]: r["status"]
            for r in conn.execute("SELECT valve_id, status FROM valves ORDER BY valve_id;").fetchall()
        }
        plants_after = {
            r["plant_id"]: float(r["production_mld"])
            for r in conn.execute("SELECT plant_id, production_mld FROM plant_setpoints ORDER BY plant_id;").fetchall()
        }
    finally:
        conn.close()

    assert valves_after == valves_before, "Simulator mutated valves table!"
    assert plants_after == plants_before, "Simulator mutated plant_setpoints table!"


def test_simulator_background_thread_starts_stops_and_resets_cleanly():
    """Background thread starts, executes ticks, stops cleanly, and DB reset restores baseline."""
    sim = LiveScadaSimulator(interval_seconds=0.05, fault_probability=0.0, start_hour=14, seed=42)
    sim.start()
    assert sim.is_running is True

    time.sleep(0.22)
    sim.stop(timeout=2.0)
    assert sim.is_running is False
    assert sim.tick_count >= 1

    # Reset restores clean baseline
    sim.reset_clock(14)
    reset_database()
    assert sim.current_hour == 14
    assert sim.tick_count == 0

    conn = get_db_connection()
    try:
        res = conn.execute("SELECT level_percent FROM reservoirs WHERE id = 'RES-01';").fetchone()
        assert float(res["level_percent"]) == 48.0
    finally:
        conn.close()
