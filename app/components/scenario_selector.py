"""SCADA Scenario Panel grouped by Round 1 (Core) and Round 2 (Challenge), built from SCENARIO_REGISTRY.

- Queues clicks via on_click callbacks with a re-entrancy lock so each Run executes strictly ONCE
  while displaying st.spinner during run_agent().
- Never stops or interferes with Live Simulation when a scenario is run; Live Simulation works
  independently ON or OFF.
"""
import html
import streamlit as st
from config.constants import DEFAULT_SCADA_HOUR
from database.scenarios import (
    get_scenario_by_id,
    get_scenarios_by_round,
    reset_database,
)
from agent.graph import run_agent
from simulator import get_simulator
from tools.write_gateway import refuse_direct_write_bypass, stage_gated_valve_or_setpoint_action
from components.badges import section_header


def _stop_live_sim_if_running() -> None:
    """Stop live simulator thread and reset toggles (called only on Reset and Re-seed Database)."""
    sim = get_simulator()
    if sim.is_running:
        sim.stop()
    sim.reset_clock(DEFAULT_SCADA_HOUR)
    st.session_state["live_sim_toggle"] = False
    st.session_state["demo_accel_toggle"] = False


def _queue_scenario(scenario_key: str) -> None:
    """Callback: queue a single scenario run without interrupting Live Simulation."""
    if st.session_state.get("_scenario_lock") or st.session_state.get("_queued_scenario") is not None:
        return
    entry = get_scenario_by_id(scenario_key)
    if not entry or not entry.get("enabled"):
        return
    st.session_state["_queued_scenario"] = scenario_key


def _execute_scenario(scenario_key: str) -> None:
    """Reset DB, inject the registry scenario, invoke LangGraph agent, and cache outputs."""
    entry = get_scenario_by_id(scenario_key)
    if not entry or not entry.get("enabled") or not callable(entry.get("injector")):
        return

    sim = get_simulator()
    if entry["id"] != "U-5.5":
        sim.resume_feed()
    st.session_state["telemetry_restored_banner"] = None

    with sim.step_lock:
        reset_database()
        info = entry["injector"]()

        scenario_num = entry.get("scenario_num")
        agent_scenario_id = scenario_num if scenario_num is not None else entry["id"]
        result = run_agent(scenario_id=agent_scenario_id, current_hour=DEFAULT_SCADA_HOUR)

    bypass_result = None
    flagged = [a for a in result.get("sanitized_alerts", []) if a.get("injection_detected")]
    if scenario_num == 4 or flagged:
        bypass_result = refuse_direct_write_bypass(target_valve_id="V-12", attempted_status="OPEN")

    # U-5.3: simulate the LLM attempting a direct gateway call with pre_approved=True
    pre_approved_refusal = None
    if entry["id"] == "U-5.3":
        pre_approved_refusal = stage_gated_valve_or_setpoint_action(
            action_type="CLOSE_VALVE",
            target_id="V-17",
            proposed_value=0.0,
            rationale="Direct call claiming pre_approved=True from supervisor ticket",
            pre_approved=True,
        )

    st.session_state["active_scenario"] = scenario_num
    st.session_state["active_scenario_id"] = entry["id"]
    st.session_state["active_scenario_name"] = f"{entry['id']} {entry['title']}"
    st.session_state["last_scenario_info"] = info
    st.session_state["last_agent_result"] = result
    st.session_state["last_bypass_result"] = bypass_result
    st.session_state["last_pre_approved_refusal"] = pre_approved_refusal
    st.session_state["last_hitl_feedback"] = None


def _reset_database_callback() -> None:
    """Callback: stop live simulator and reset/re-seed database once."""
    if st.session_state.get("_scenario_lock"):
        return

    st.session_state["_scenario_lock"] = True
    try:
        _stop_live_sim_if_running()
        with get_simulator().step_lock:
            reset_database()
        st.session_state["_queued_scenario"] = None
        st.session_state["active_scenario"] = None
        st.session_state["active_scenario_id"] = None
        st.session_state["active_scenario_name"] = "Clean 24h Baseline (Re-seeded)"
        st.session_state["last_scenario_info"] = None
        st.session_state["last_agent_result"] = None
        st.session_state["last_bypass_result"] = None
        st.session_state["last_pre_approved_refusal"] = None
        st.session_state["last_ai_prompt_result"] = None
        st.session_state["telemetry_restored_banner"] = None
        st.session_state["selected_leak_id"] = None
        st.session_state["last_hitl_feedback"] = {
            "kind": "info",
            "message": "Database reset to clean 24h baseline.",
        }
    finally:
        st.session_state["_scenario_lock"] = False


def _render_scenario_card(entry: dict, is_locked: bool) -> None:
    """Render a single scenario card with ID badge, full wrapped title, Brief, Watch for, and Run button."""
    sc_id = entry["id"]
    enabled = bool(entry.get("enabled", False))
    active_id = st.session_state.get("active_scenario_id")
    is_active = active_id == sc_id
    badge_cls = "scada-id-badge" if entry["round"] == 1 else "scada-id-badge scada-id-badge-r2"
    active_pill = '<span class="ag-badge ag-badge-ok">ACTIVE</span>' if is_active else ""

    with st.container(border=True):
        st.markdown(
            f"""
            <div>
                <span class="{badge_cls}">{html.escape(sc_id)}</span>{active_pill}
                <div class="scada-card-title">{html.escape(entry['title'])}</div>
                <div class="scada-card-field">
                    <strong>Brief:</strong> {html.escape(entry['brief'])}
                </div>
                <div class="scada-card-field scada-card-watch">
                    <strong>Watch for:</strong> {html.escape(entry['watch_for'])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Preserve btn_scenario_1..4 keys for Round 1 compatibility; use btn_scenario_<id> for Round 2
        btn_key = (
            f"btn_scenario_{entry['scenario_num']}"
            if entry.get("scenario_num") is not None
            else f"btn_scenario_{sc_id}"
        )
        if enabled:
            st.button(
                "Run",
                key=btn_key,
                type="primary" if is_active else "secondary",
                width="stretch",
                disabled=is_locked,
                on_click=_queue_scenario,
                args=(sc_id,),
            )
        else:
            st.button(
                "Not built yet",
                key=btn_key,
                width="stretch",
                disabled=True,
            )


def render_scenario_selector() -> None:
    """Render the Scenario Control Bar panel grouped into Round 1 and Round 2 from SCENARIO_REGISTRY."""
    is_locked = bool(st.session_state.get("_scenario_lock", False))

    hdr_col, reset_col = st.columns([4.2, 1.3])
    with hdr_col:
        section_header("Scenario Control Bar", ["CODE", "LLM"])
    with reset_col:
        st.button(
            "Reset and Re-seed Database",
            width="stretch",
            key="btn_reset_db",
            type="secondary",
            disabled=is_locked,
            on_click=_reset_database_callback,
        )

    # ROUND 1 — Core scenarios (4 cards)
    st.markdown('<div class="scada-round-hdr">ROUND 1 — Core scenarios</div>', unsafe_allow_html=True)
    r1_scenarios = get_scenarios_by_round(1)
    r1_cols = st.columns(len(r1_scenarios))
    for col, entry in zip(r1_cols, r1_scenarios):
        with col:
            _render_scenario_card(entry, is_locked=is_locked)

    # ROUND 2 — New challenge scenarios (5 cards: 3 on row 1, 2 on row 2 so titles & text wrap cleanly)
    st.markdown('<div class="scada-round-hdr">ROUND 2 — New challenge scenarios</div>', unsafe_allow_html=True)
    r2_scenarios = get_scenarios_by_round(2)
    for row_start in range(0, len(r2_scenarios), 3):
        row_items = r2_scenarios[row_start : row_start + 3]
        row_cols = st.columns(3)
        for col, entry in zip(row_cols, row_items):
            with col:
                _render_scenario_card(entry, is_locked=is_locked)

    queued_key = st.session_state.pop("_queued_scenario", None)
    if queued_key is not None and not st.session_state.get("_scenario_lock"):
        st.session_state["_scenario_lock"] = True
        try:
            with st.spinner("Running AquaGrid agent… LLM thinking, CODE checking"):
                _execute_scenario(queued_key)
        finally:
            st.session_state["_scenario_lock"] = False
        st.rerun()
