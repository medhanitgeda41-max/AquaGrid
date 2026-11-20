"""AquaGrid Streamlit SCADA Control Room — Phase 4 Entry Point.

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
from components.scenario_selector import render_scenario_selector
from components.live_sim_control import render_live_sim_controls
from components.leak_map_view import render_leak_map_view
from components.ai_suggestions_view import render_ai_suggestions_and_run_log
from components.incident_brief import render_incident_brief
from components.telemetry_view import render_telemetry_view
from components.approval_card import render_approval_card
from components.audit_log_view import render_audit_log


def _ensure_database_initialized() -> None:
    """Initialize and seed SQLite database on first launch if empty or missing new geospatial tables."""
    init_db()
    conn = get_db_connection()
    try:
        res_count = conn.execute("SELECT COUNT(*) AS cnt FROM reservoirs;").fetchone()["cnt"]
        pipe_count = conn.execute("SELECT COUNT(*) AS cnt FROM pipe_segments;").fetchone()["cnt"]
    finally:
        conn.close()
    if res_count == 0 or pipe_count == 0:
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

    # Top Header & Real-Time KPIs (real Dubai hour when Live Sim is ON, or 14:00 for deterministic scenarios)
    sim = get_simulator()
    if st.session_state.get("live_sim_toggle", False):
        display_hour = sim.current_hour if not sim.real_time_clock else now_local().hour
    else:
        display_hour = DEFAULT_SCADA_HOUR
    snapshot = render_header(display_hour=display_hour)

    # Persistent 5-Button Scenario & Reset Control Bar
    render_scenario_selector()

    # Optional Live Simulation Mode (default OFF; real-time 2-min batches + optional accelerated demo mode)
    render_live_sim_controls()

    st.divider()

    # Main 6-Tab SCADA Layout
    pending_count = snapshot.get("pending_count", 0)
    approval_tab_label = f"HITL Approval Center ({pending_count} PENDING)" if pending_count > 0 else "HITL Approval Center"

    tab_map, tab_report, tab_telemetry, tab_ai_log, tab_approval, tab_audit = st.tabs(
        [
            "2D Leak Map (Dubai)",
            "Incident & Guard Report",
            "SCADA Telemetry & Reservoir",
            "AI Suggestions & Run Log",
            approval_tab_label,
            "Audit Log",
        ]
    )

    with tab_map:
        render_leak_map_view(snapshot)

    with tab_report:
        # Surface pending approval card right alongside the scenario report when HITL action is waiting
        if pending_count > 0:
            render_approval_card(key_prefix="inline", show_history=False)
            st.divider()
        render_incident_brief()

    with tab_telemetry:
        render_telemetry_view(snapshot)

    with tab_ai_log:
        render_ai_suggestions_and_run_log()

    with tab_approval:
        render_approval_card(key_prefix="center", show_history=True)

    with tab_audit:
        render_audit_log()


if __name__ == "__main__":
    main()
