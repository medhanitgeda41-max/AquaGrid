"""One-click SCADA scenario control bar (4 demo scenarios + Reset & Re-seed).

Queues clicks via on_click callbacks with a re-entrancy lock so each button click
executes strictly ONCE while displaying a visible st.spinner during run_agent().
"""
import streamlit as st
from config.constants import DEFAULT_SCADA_HOUR
from database.scenarios import (
    inject_scenario_1,
    inject_scenario_2,
    inject_scenario_3,
    inject_scenario_4,
    reset_database,
)
from agent.graph import run_agent
from simulator import get_simulator
from tools.write_gateway import refuse_direct_write_bypass
from components.badges import section_header


def _stop_live_sim_if_running() -> None:
    """Ensure live simulator thread is stopped and toggles reset before deterministic scenarios or DB reset."""
    sim = get_simulator()
    if sim.is_running:
        sim.stop()
    sim.reset_clock(DEFAULT_SCADA_HOUR)
    st.session_state["live_sim_toggle"] = False
    st.session_state["demo_accel_toggle"] = False


def _queue_scenario(scenario_id: int) -> None:
    """Callback: queue a single scenario run if not already running or queued."""
    if st.session_state.get("_scenario_lock") or st.session_state.get("_queued_scenario") is not None:
        return
    _stop_live_sim_if_running()
    st.session_state["_queued_scenario"] = scenario_id


def _execute_scenario(scenario_id: int) -> None:
    """Reset DB, inject scenario, invoke LangGraph agent, and cache outputs."""
    injectors = {
        1: inject_scenario_1,
        2: inject_scenario_2,
        3: inject_scenario_3,
        4: inject_scenario_4,
    }
    reset_database()
    info = injectors[scenario_id]()

    result = run_agent(scenario_id=scenario_id, current_hour=DEFAULT_SCADA_HOUR)

    bypass_result = None
    flagged = [a for a in result.get("sanitized_alerts", []) if a.get("injection_detected")]
    if scenario_id == 4 or flagged:
        bypass_result = refuse_direct_write_bypass(target_valve_id="V-12", attempted_status="OPEN")

    st.session_state["active_scenario"] = scenario_id
    st.session_state["active_scenario_name"] = f"Scenario {scenario_id}: {info['name']}"
    st.session_state["last_agent_result"] = result
    st.session_state["last_bypass_result"] = bypass_result
    st.session_state["last_hitl_feedback"] = None


def _reset_database_callback() -> None:
    """Callback: stop live simulator and reset/re-seed database once."""
    if st.session_state.get("_scenario_lock"):
        return

    st.session_state["_scenario_lock"] = True
    try:
        _stop_live_sim_if_running()
        reset_database()
        st.session_state["_queued_scenario"] = None
        st.session_state["active_scenario"] = None
        st.session_state["active_scenario_name"] = "Clean 24h Baseline (Re-seeded)"
        st.session_state["last_agent_result"] = None
        st.session_state["last_bypass_result"] = None
        st.session_state["selected_leak_id"] = None
        st.session_state["last_hitl_feedback"] = {
            "kind": "info",
            "message": "Database reset to clean 24h baseline.",
        }
    finally:
        st.session_state["_scenario_lock"] = False


def render_scenario_selector() -> None:
    """Render the 5 persistent one-click scenario and reset buttons."""
    section_header("Scenario Control Bar", ["CODE", "LLM"])
    is_locked = bool(st.session_state.get("_scenario_lock", False))

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.button(
            "Scenario 1: Leak vs. Broken Sensor",
            width="stretch",
            key="btn_scenario_1",
            disabled=is_locked,
            on_click=_queue_scenario,
            args=(1,),
        )

    with col2:
        st.button(
            "Scenario 2: Throttle Aborted (Safety Floor)",
            width="stretch",
            key="btn_scenario_2",
            disabled=is_locked,
            on_click=_queue_scenario,
            args=(2,),
        )

    with col3:
        st.button(
            "Scenario 3: Valve Needs Human Approval",
            width="stretch",
            key="btn_scenario_3",
            disabled=is_locked,
            on_click=_queue_scenario,
            args=(3,),
        )

    with col4:
        st.button(
            "Scenario 4: Injection Attack Blocked",
            width="stretch",
            key="btn_scenario_4",
            disabled=is_locked,
            on_click=_queue_scenario,
            args=(4,),
        )

    with col5:
        st.button(
            "Reset and Re-seed Database",
            width="stretch",
            key="btn_reset_db",
            type="secondary",
            disabled=is_locked,
            on_click=_reset_database_callback,
        )

    queued_scenario = st.session_state.pop("_queued_scenario", None)
    if queued_scenario is not None and not st.session_state.get("_scenario_lock"):
        st.session_state["_scenario_lock"] = True
        try:
            with st.spinner("Running AquaGrid agent… LLM thinking, CODE checking"):
                _execute_scenario(queued_scenario)
        finally:
            st.session_state["_scenario_lock"] = False
        st.rerun()
