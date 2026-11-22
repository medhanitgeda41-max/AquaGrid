"""Part 1 tests: Single Source of Truth for Leaks + Self-Updating 3D Digital Twin."""
import logging
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from database.connection import get_db_connection, transaction_scope
from database.leak_registry import (
    get_active_leaks,
    get_all_leaks,
    get_leak_by_id_or_junction,
    verify_leak_count_consistency,
)
from database.scenarios import inject_scenario_1, reset_database
from digital_twin.scene_state import build_3d_scene_state, build_twin_caption
from simulator import LiveScadaSimulator
from tools.read_telemetry import read_network_telemetry
from tools.write_gateway import (
    execute_human_approval,
    stage_gated_valve_or_setpoint_action,
    update_leak_status_by_operator,
)


@pytest.fixture(autouse=True)
def clean_db():
    reset_database()
    yield
    reset_database()


def test_j14_detection_creates_updates_leak_001_no_duplicates_and_resolves_when_normal():
    """Mass-balance detection at J-14 updates LEAK-001 (never duplicates) and marks RESOLVED when J-14 returns to nominal."""
    inject_scenario_1()

    # Run telemetry detection multiple times -> always updates LEAK-001, never creates a second ID for J-14
    tel1 = read_network_telemetry(current_hour=14, log_audit=False)
    tel2 = read_network_telemetry(current_hour=14, log_audit=False)

    conn = get_db_connection()
    try:
        j14_rows = conn.execute("SELECT leak_id, junction_id, status, detected_by FROM leaks WHERE junction_id = 'J-14';").fetchall()
    finally:
        conn.close()

    assert len(j14_rows) == 1, f"Expected exactly 1 leak record for J-14, got {len(j14_rows)}"
    assert j14_rows[0]["leak_id"] == "LEAK-001"
    assert j14_rows[0]["status"] == "CONFIRMED"
    assert j14_rows[0]["detected_by"] == "SCADA_MASS_BALANCE"

    # AI read tool output lists BOTH leak_id and junction_id
    j14_alerts = [a for a in tel2["leak_alerts"] if a["junction_id"] == "J-14"]
    assert len(j14_alerts) == 1
    assert j14_alerts[0]["leak_id"] == "LEAK-001"
    assert "LEAK-001" in j14_alerts[0]["diagnostic"]
    assert "J-14" in j14_alerts[0]["diagnostic"]

    # When J-14 telemetry returns to nominal (matching hour-10 baseline), telemetry-detected leak resolves automatically
    with transaction_scope() as tx:
        p10 = tx.execute("SELECT value FROM telemetry WHERE sensor_id = 'P-14' AND hour = 10;").fetchone()["value"]
        f10 = tx.execute("SELECT value FROM telemetry WHERE sensor_id = 'F-14' AND hour = 10;").fetchone()["value"]
        tx.execute("UPDATE telemetry SET value = ?, status = 'NORMAL' WHERE sensor_id = 'P-14' AND hour = 14;", (p10,))
        tx.execute("UPDATE telemetry SET value = ?, status = 'NORMAL' WHERE sensor_id = 'F-14' AND hour = 14;", (f10,))

    tel_after = read_network_telemetry(current_hour=14, log_audit=False)
    lk_001 = get_leak_by_id_or_junction("LEAK-001")
    assert lk_001 is not None and lk_001["status"] == "RESOLVED"
    assert all(a["leak_id"] != "LEAK-001" for a in tel_after["leak_alerts"])


def test_get_active_leaks_is_single_source_for_ai_status_bar_twin_and_map(caplog):
    """Status bar, 3D twin, 2D map, and AI read tools all report the exact same active leak count from get_active_leaks()."""
    from components.leak_map_view import build_leak_folium_map

    active = get_active_leaks()
    tel = read_network_telemetry(current_hour=14, log_audit=False)
    scene = build_3d_scene_state(current_hour=14, focus="all", show_leak_history=False)

    assert len(active) == len(tel["leak_alerts"])
    assert len(active) == int(scene["active_leak_count"])
    assert len(active) == len(scene["leaks"])
    assert {l["leak_id"] for l in active} == {l["leak_id"] for l in tel["leak_alerts"]}
    assert {l["leak_id"] for l in active} == {l["leak_id"] for l in scene["leaks"]}

    # Consistency verifier passes when equal and logs ERROR when mismatched
    assert verify_leak_count_consistency(
        {"status_bar": len(active), "twin": len(scene["leaks"]), "map": len(active), "ai": len(tel["leak_alerts"])},
        context="unit_test",
    ) is True
    with caplog.at_level(logging.ERROR, logger="aquagrid.leaks"):
        assert verify_leak_count_consistency({"status_bar": 3, "ai": 1}, context="mismatch_test") is False
    assert "LEAK_COUNT_MISMATCH" in caplog.text


def test_resolved_leaks_are_not_drawn_as_active_and_history_toggle_shows_them():
    """RESOLVED and FALSE_ALARM leaks are excluded from active 3D/map/AI views unless show_leak_history=True."""
    res_upd = update_leak_status_by_operator("LEAK-001", "RESOLVED", operator_name="Op_Test")
    assert res_upd["status"] == "UPDATED"

    active_ids = {l["leak_id"] for l in get_active_leaks()}
    assert "LEAK-001" not in active_ids
    assert "LEAK-003" not in active_ids  # FALSE_ALARM

    scene_default = build_3d_scene_state(current_hour=14, show_leak_history=False)
    assert all(l["leak_id"] != "LEAK-001" for l in scene_default["leaks"])
    assert all(l["leak_id"] != "LEAK-003" for l in scene_default["leaks"])

    scene_hist = build_3d_scene_state(current_hour=14, show_leak_history=True)
    hist_by_id = {l["leak_id"]: l for l in scene_hist["leaks"]}
    assert "LEAK-001" in hist_by_id and hist_by_id["LEAK-001"]["is_history"] is True
    assert "LEAK-003" in hist_by_id and hist_by_id["LEAK-003"]["is_history"] is True


def test_tell_me_about_leak_001_returns_live_details_and_prompt_forbids_cached_blame():
    """Asking 'Tell me about LEAK-001' returns its live details from the single registry; prompt forbids blaming stale twin."""
    from agent.chat_assistant import ASK_AQUAGRID_SYSTEM_PROMPT, run_chat_turn

    assert "Never say the twin is stale or cached" in ASK_AQUAGRID_SYSTEM_PROMPT

    with patch("agent.chat_assistant.llm_client.chat_with_tools", return_value=(None, "Deterministic Offline Engine", [])):
        reply = run_chat_turn("Tell me about LEAK-001", history=[], current_hour=14)

    txt = reply["content"]
    assert "LEAK-001" in txt
    assert "J-14" in txt
    assert "Sector 4" in txt
    assert "CONFIRMED" in txt
    assert "64.5" in txt
    assert "V-12" in txt
    assert "cached" not in txt.lower()
    assert reply["twin_view"]["focus"] == "LEAK-001"


def test_twin_scene_version_changes_after_approval_and_after_live_batch():
    """build_3d_scene_state scene_version/scene_hash changes immediately after valve approval and after a live sim batch."""
    s0 = build_3d_scene_state(current_hour=14, focus="all")
    v0 = s0["scene_version"]
    assert v0 == s0["scene_hash"] and len(v0) == 16
    cap = build_twin_caption(s0)
    assert "Last update " in cap

    # 1. Approve CLOSE_VALVE V-12 -> scene_version must change immediately
    gw = stage_gated_valve_or_setpoint_action(
        action_type="CLOSE_VALVE",
        target_id="V-12",
        proposed_value=0.0,
        rationale="Isolate LEAK-001",
    )
    assert gw["status"] == "STAGED_PENDING_APPROVAL"
    ex = execute_human_approval(gw["action_id"], approver="Op_Test")
    assert ex["status"] == "EXECUTED"

    s1 = build_3d_scene_state(current_hour=14, focus="all")
    v1 = s1["scene_version"]
    assert v1 != v0, "Scene version must change after valve approval!"

    # 2. Run one live simulator batch -> scene_version must change again
    sim = LiveScadaSimulator(interval_seconds=0.1, fault_probability=0.0, start_hour=14, seed=42)
    summary = sim.step(force_fault="none")
    s2 = build_3d_scene_state(current_hour=summary["hour"], focus="all")
    v2 = s2["scene_version"]
    assert v2 != v1, "Scene version must change after live simulation batch!"
