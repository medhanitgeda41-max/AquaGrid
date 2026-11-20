"""AquaGrid Streamlit SCADA Control Room — Clean Operator Dashboard & AI Prompt.

Launch from project root:
    python -m streamlit run app/app.py
"""
import os
import sys

# Ensure both app/ (for components.*) and project root (for agent/database/guards/tools/config) are on sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
for _p in (PROJECT_ROOT, APP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
from config.constants import DEFAULT_SCADA_HOUR
from config.region_config import now_local
from database.connection import get_db_connection, init_db
from database.seed_data import seed_database
from simulator import get_simulator
from components.badges import inject_scada_css
from components.header import render_header
from components.live_sim_control import render_live_sim_controls
from components.approval_card import render_approval_card
from components.ai_prompt_panel import render_ai_prompt_panel
from components.incident_brief import render_incident_brief
from components.scenario_selector import render_scenario_selector
from components.leak_map_view import render_leak_map_view
from components.telemetry_view import render_telemetry_view
from components.ai_suggestions_view import render_ai_suggestions_and_run_log
from components.audit_log_view import render_audit_log


def _ensure_database_initialized() -> None:
    """Initialize and seed SQLite database on first launch if empty or missing new tables/valves."""
    init_db()
    conn = get_db_connection()
    try:
        res_count = conn.execute("SELECT COUNT(*) AS cnt FROM reservoirs;").fetchone()["cnt"]
        pipe_count = conn.execute("SELECT COUNT(*) AS cnt FROM pipe_segments;").fetchone()["cnt"]
        valve_count = conn.execute("SELECT COUNT(*) AS cnt FROM valves;").fetchone()["cnt"]
        wq_count = conn.execute("SELECT COUNT(*) AS cnt FROM sector_water_quality;").fetchone()["cnt"]
    finally:
        conn.close()
    if res_count == 0 or pipe_count == 0 or valve_count < 20 or wq_count == 0:
        seed_database()


def main() -> None:
    st.set_page_config(
        page_title="AquaGrid SCADA Control Room",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_scada_css()
    _ensure_database_initialized()

    # 1. Header: 5 Key Operator Values + Active Alerts
    sim = get_simulator()
    if st.session_state.get("live_sim_toggle", False):
        display_hour = sim.current_hour if not sim.real_time_clock else now_local().hour
    else:
        display_hour = DEFAULT_SCADA_HOUR
    snapshot = render_header(display_hour=display_hour)

    # 2. Compact Live Simulation Bar
    render_live_sim_controls()

    # 3. Pending Approvals surfaced immediately whenever operator signoff is needed
    pending_count = snapshot.get("pending_count", 0)
    if pending_count > 0 or st.session_state.get("last_hitl_feedback"):
        render_approval_card(key_prefix="top", show_history=False)

    st.divider()

    # 4. Clean 4-Tab Operator Workspace
    approvals_badge = f" ({pending_count})" if pending_count > 0 else ""
    tab_main, tab_map, tab_telemetry, tab_logs = st.tabs(
        [
            f"Overview & AI{approvals_badge}",
            "Leak Map",
            "Telemetry",
            "Logs & Details",
        ]
    )

    with tab_main:
        # AI Prompt Area (request box + Run + step-by-step trace + clear final result)
        render_ai_prompt_panel(display_hour=display_hour)

        # Latest Scenario Result (concise summary + collapsed Show details)
        if st.session_state.get("last_agent_result"):
            st.divider()
            render_incident_brief()

        st.divider()

        # Round 1 & Round 2 Scenarios + Reset
        render_scenario_selector()

    with tab_map:
        render_leak_map_view(snapshot)

    with tab_telemetry:
        render_telemetry_view(snapshot)

    with tab_logs:
        render_approval_card(key_prefix="center", show_history=True)
        st.divider()
        render_ai_suggestions_and_run_log()
        st.divider()
        render_audit_log()


if __name__ == "__main__":
    main()
