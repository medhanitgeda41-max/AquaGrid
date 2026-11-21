"""AquaGrid Chat-First SCADA Control Room.

Launch from project root:
    python -m streamlit run app/app.py
"""
import importlib
import inspect
import os
import sys

# Ensure PROJECT_ROOT is first (so 'app.app' stays a package) and APP_DIR is available for 'components.*'
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)


def _ensure_module_attr(module_name: str, required_attr: str) -> None:
    """Reload a cached module if a Streamlit hot-reload left an older version in sys.modules."""
    mod = sys.modules.get(module_name)
    if mod is not None and not hasattr(mod, required_attr):
        importlib.reload(mod)


def _ensure_callable_param(module_name: str, attr_path: str, param_name: str) -> None:
    """Reload a cached module if a function/method signature is missing a required parameter after hot-reload."""
    mod = sys.modules.get(module_name)
    if mod is None:
        return
    obj = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            importlib.reload(mod)
            return
    try:
        params = inspect.signature(obj).parameters
    except (TypeError, ValueError):
        return
    if param_name not in params:
        importlib.reload(mod)


# Guard against stale sys.modules across live Streamlit hot-reloads
_ensure_module_attr("simulator", "simulator_step_lock")
_ensure_module_attr("simulator", "STEP_LOCK")
_ensure_module_attr("guards.telemetry_freshness", "refresh_baseline_timestamps")
_ensure_module_attr("agent.llm_factory", "provider_supports_streaming")
_ensure_callable_param("agent.llm_factory", "LLMClient.chat_with_tools", "status_callback")
_ensure_module_attr("agent.chat_assistant", "collect_system_alerts")
_ensure_callable_param("agent.chat_assistant", "run_chat_turn", "status_callback")
_ensure_callable_param("agent.chat_assistant", "run_scenario_for_chat", "status_callback")
_ensure_callable_param("agent.chat_assistant", "run_production_planner_for_chat", "status_callback")
_ensure_module_attr("components.badges", "get_sim_step_lock")
_ensure_module_attr("components.chat_widgets", "render_chat_widget")
_ensure_module_attr("components.header", "render_header")
_ensure_module_attr("components.ai_prompt_panel", "queue_scenario_request")
_ensure_module_attr("components.ai_prompt_panel", "render_ask_aquagrid_chat")
_ensure_module_attr("components.sidebar", "render_sidebar")

import streamlit as st
from agent.chat_assistant import collect_system_alerts
from config.constants import DEFAULT_SCADA_HOUR
from config.region_config import now_local
from database.connection import get_db_connection, init_db
from database.seed_data import seed_database
from guards.telemetry_freshness import refresh_baseline_timestamps
from simulator import get_simulator
from components.badges import inject_scada_css
from components.header import render_header
from components.sidebar import render_sidebar
from components.ai_prompt_panel import render_ask_aquagrid_chat


def _ensure_database_initialized() -> None:
    """Initialize and seed SQLite database on first launch, and keep baseline timestamps fresh on startup."""
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
    elif not st.session_state.get("_startup_ts_refreshed"):
        refresh_baseline_timestamps()
    st.session_state["_startup_ts_refreshed"] = True


def main() -> None:
    st.set_page_config(
        page_title="Ask AquaGrid — SCADA Control Room",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_scada_css()
    _ensure_database_initialized()

    # Determine active clock hour (real Dubai hour when Live Sim is ON, or 14:00 for deterministic scenarios)
    sim = get_simulator()
    if st.session_state.get("live_sim_toggle", False) and sim is not None:
        display_hour = sim.current_hour if not getattr(sim, "real_time_clock", True) else now_local().hour
    else:
        display_hour = DEFAULT_SCADA_HOUR

    # 1. Thin always-visible 1-line status bar at top (placeholder allows live refresh after a chat/scenario turn)
    header_placeholder = st.empty()
    with header_placeholder.container():
        snapshot = render_header(display_hour=display_hour)

    # 2. Compact sidebar: Live Simulation, Scenarios (Round 1 & 2), Open Audit Log, Tech Details, Clear Chat
    render_sidebar()

    # 3. Auto-post any new system ALERT messages (leaks, water quality, telemetry loss, floor breach, live checks) into chat
    seen_alerts = st.session_state.setdefault("_seen_alert_keys", set())
    live_summary = getattr(sim, "last_tick_summary", None) if st.session_state.get("live_sim_toggle", False) else None
    new_alert_msgs, updated_seen = collect_system_alerts(
        telemetry_data=snapshot.get("telemetry_data") or {},
        feeds_data=snapshot.get("feeds_data") or {},
        seen_alert_keys=seen_alerts,
        live_tick_summary=live_summary,
    )
    if new_alert_msgs:
        st.session_state["_seen_alert_keys"] = updated_seen
        st.session_state.setdefault("ask_aquagrid_messages", []).extend(new_alert_msgs)

    # 4. Full-width "Ask AquaGrid" multi-turn chat with inline visual widgets & button-only approvals
    turn_ran = render_ask_aquagrid_chat(display_hour=display_hour, snapshot=snapshot)
    if turn_ran:
        # Refresh top status bar in place (preserving st.status('Done') & streamed output)
        with header_placeholder.container():
            render_header(display_hour=display_hour)


if __name__ == "__main__":
    main()
