"""Compact Sidebar: Live Simulation controls, Round 1 & 2 Scenarios, Audit Log, Technical Details & Clear Chat."""
import html
import streamlit as st

from agent.chat_assistant import collect_system_alerts, run_scenario_for_chat
from config.constants import DEFAULT_SCADA_HOUR
from config.region_config import REGION_CONFIG
from database.scenarios import get_scenarios_by_round, reset_database
from guards.telemetry_freshness import fast_forward_telemetry_age, restore_telemetry_feed
from simulator import get_simulator, simulator_step_lock
from tools.read_feeds import read_tariff_and_quality_feeds
from tools.read_telemetry import read_network_telemetry


def _fmt_countdown(seconds: int) -> str:
    mins, secs = divmod(max(0, int(seconds)), 60)
    return f"{mins}:{secs:02d}"


@st.fragment(run_every=int(REGION_CONFIG["ui_refresh_seconds"]))
def _render_sidebar_live_ticker() -> None:
    """Auto-refreshing sidebar status line for Live Simulation ('Last update · Next update in') + push live updates to chat."""
    sim = get_simulator()
    last_dt = getattr(sim, "last_update_at", None)
    last_str = last_dt.strftime("%H:%M:%S") if last_dt else "Starting..."
    countdown = "PAUSED" if getattr(sim, "is_feed_paused", False) else _fmt_countdown(sim.seconds_until_next_update())
    tick = getattr(sim, "tick_count", 0)

    st.markdown(
        f'<div class="ag-mono" style="font-size:0.78rem; color:#CBD5E1; margin-bottom:0.35rem;">'
        f'Batch #{tick} · Last update: <strong>{last_str}</strong> · Next update in <strong>{countdown}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Push any newly generated live batch summary or alert into chat automatically
    summary = getattr(sim, "last_tick_summary", None)
    if summary and summary.get("tick"):
        sim_hour = summary.get("hour", DEFAULT_SCADA_HOUR)
        tel = read_network_telemetry(current_hour=sim_hour, log_audit=False)
        feeds = read_tariff_and_quality_feeds(horizon_hours=6, current_hour=sim_hour, log_audit=False)
        seen = st.session_state.setdefault("_seen_alert_keys", set())
        new_msgs, updated = collect_system_alerts(tel, feeds, seen, live_tick_summary=summary)
        if new_msgs:
            st.session_state["_seen_alert_keys"] = updated
            st.session_state.setdefault("ask_aquagrid_messages", []).extend(new_msgs)
            st.rerun()


def _on_sidebar_reset() -> None:
    """on_click callback: stop live simulator, re-seed DB, and reset widget toggles safely before rerun."""
    sim = get_simulator()
    if sim is not None:
        if getattr(sim, "is_running", False) and hasattr(sim, "stop"):
            sim.stop()
        if hasattr(sim, "reset_clock"):
            sim.reset_clock(DEFAULT_SCADA_HOUR)
    with simulator_step_lock(sim):
        reset_database()
    st.session_state["live_sim_toggle"] = False
    st.session_state["demo_accel_toggle"] = False
    st.session_state["active_scenario"] = None
    st.session_state["active_scenario_id"] = None
    st.session_state["active_scenario_name"] = "Clean Baseline"
    st.session_state["last_agent_result"] = None
    st.session_state["_seen_alert_keys"] = set()
    st.session_state.setdefault("ask_aquagrid_messages", []).append(
        {
            "role": "assistant",
            "kind": "text",
            "content": "Database reset to clean 24h baseline.",
            "provider": "System",
            "tools_called": [],
            "code_checks": [],
            "widgets": [],
            "staged_action": None,
        }
    )


def render_sidebar() -> None:
    """Render compact left sidebar with Live Simulation, Scenarios (R1 & R2), Audit Log, Tech Details, Clear Chat."""
    sim = get_simulator()

    with st.sidebar:
        st.markdown("### Live Simulation")
        live_on = st.toggle("Live Simulation: ON/OFF", value=False, key="live_sim_toggle")
        demo_accel = st.toggle("Demo mode (accelerated)", value=False, key="demo_accel_toggle", disabled=not live_on)

        with st.expander("Feed & Simulator Controls", expanded=False):
            fault_prob = st.slider(
                "Fault Chance / Cycle",
                min_value=0.0,
                max_value=1.0,
                value=float(REGION_CONFIG["default_fault_probability"]),
                step=0.01,
                key="live_sim_fault_prob",
                disabled=not live_on,
            )
            if st.button("Stop telemetry feed", key="btn_stop_feed", width="stretch"):
                if sim is not None and hasattr(sim, "pause_feed"):
                    sim.pause_feed()
                st.rerun()
            if st.button("Fast-forward 10 minutes", key="btn_ff_10min", width="stretch"):
                if sim is not None and hasattr(sim, "pause_feed"):
                    sim.pause_feed()
                fast_forward_telemetry_age(minutes=10)
                st.rerun()
            if st.button("Restore feed", key="btn_restore_feed", width="stretch"):
                if sim is not None and hasattr(sim, "resume_feed"):
                    sim.resume_feed()
                res_hour = sim.current_hour if (live_on and sim is not None) else DEFAULT_SCADA_HOUR
                restored = restore_telemetry_feed(current_hour=res_hour)
                st.session_state.setdefault("ask_aquagrid_messages", []).append(
                    {
                        "role": "assistant",
                        "kind": "text",
                        "content": f"**{restored['message']}** (`TELEMETRY_RESTORED`).",
                        "provider": "CODE_GUARD",
                        "tools_called": [],
                        "code_checks": [{"check": "Telemetry Freshness", "status": "PASS", "code": "TELEMETRY_RESTORED", "detail": restored["message"]}],
                        "widgets": [],
                        "staged_action": None,
                    }
                )
                st.rerun()

        if sim is not None and hasattr(sim, "configure_mode"):
            sim.configure_mode(
                demo_accelerated=bool(demo_accel),
                fault_probability=float(fault_prob),
                run_ai_after_batch=True,
            )

        if live_on:
            if sim is not None and not getattr(sim, "is_running", False):
                sim.start()
            _render_sidebar_live_ticker()
        else:
            if sim is not None and getattr(sim, "is_running", False):
                sim.stop()
            if not getattr(sim, "is_feed_paused", False):
                st.caption("Live simulation paused")

        st.divider()

        # ── Scenarios (Round 1 & Round 2) ─────────────────────────────────────
        st.markdown("### Scenarios")
        st.button(
            "Reset and Re-seed Database",
            key="btn_reset_db",
            width="stretch",
            on_click=_on_sidebar_reset,
        )

        st.markdown('<div class="scada-round-hdr">ROUND 1 — Core scenarios</div>', unsafe_allow_html=True)
        for entry in get_scenarios_by_round(1):
            _render_sidebar_scenario_row(entry)

        st.markdown('<div class="scada-round-hdr">ROUND 2 — New challenge scenarios</div>', unsafe_allow_html=True)
        for entry in get_scenarios_by_round(2):
            _render_sidebar_scenario_row(entry)

        st.divider()

        # ── Audit Log, Technical Details, Clear Chat ──────────────────────────
        if st.button("Open Audit Log", key="btn_open_audit_log", width="stretch"):
            st.session_state.setdefault("ask_aquagrid_messages", []).append(
                {
                    "role": "assistant",
                    "kind": "text",
                    "content": "**Immutable SCADA Audit Log** (latest 40 entries):",
                    "provider": "CODE_GUARD (Audit Log)",
                    "tools_called": [{"tool": "show_table", "args": {"what": "audit"}, "summary": "Displayed live Audit Log"}],
                    "code_checks": [],
                    "widgets": [{"type": "table", "what": "audit"}],
                    "staged_action": None,
                }
            )
            st.rerun()

        st.toggle("Show technical details", value=False, key="show_tech_details")

        if st.button("Clear chat", key="btn_clear_chat", width="stretch"):
            st.session_state["ask_aquagrid_messages"] = []
            st.rerun()


def _render_sidebar_scenario_row(entry: dict) -> None:
    """Render one compact scenario item (ID, title, one-line brief, Run button) in the sidebar."""
    sc_id = entry["id"]
    btn_key = (
        f"btn_scenario_{entry['scenario_num']}"
        if entry.get("scenario_num") is not None
        else f"btn_scenario_{sc_id}"
    )
    is_active = st.session_state.get("active_scenario_id") == sc_id
    brief_short = entry["brief"]

    c_txt, c_btn = st.columns([3.0, 1.1])
    with c_txt:
        st.markdown(
            f'<div style="line-height:1.25; margin-bottom:0.35rem;">'
            f'<span class="scada-id-badge">{html.escape(sc_id)}</span>'
            f'<strong style="font-size:0.82rem; color:#F8FAFC;">{html.escape(entry["title"])}</strong>'
            f'<div style="font-size:0.73rem; color:#94A3B8; margin-top:0.15rem;">{html.escape(brief_short)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c_btn:
        if st.button("Run", key=btn_key, type="primary" if is_active else "secondary", width="stretch"):
            sim = get_simulator()
            if sc_id != "U-5.5" and sim is not None and hasattr(sim, "resume_feed"):
                sim.resume_feed()
            with simulator_step_lock(sim):
                msg = run_scenario_for_chat(sc_id, current_hour=DEFAULT_SCADA_HOUR)
            meta = msg.get("scenario_meta") or {}
            st.session_state["active_scenario"] = meta.get("scenario_num")
            st.session_state["active_scenario_id"] = sc_id
            st.session_state["active_scenario_name"] = meta.get("name", sc_id)
            st.session_state["last_agent_result"] = meta.get("result")
            st.session_state["last_scenario_info"] = meta.get("info")
            st.session_state["last_bypass_result"] = meta.get("bypass_result")
            st.session_state["last_pre_approved_refusal"] = meta.get("pre_approved_refusal")
            st.session_state.setdefault("ask_aquagrid_messages", []).append(
                {"role": "user", "content": f"Run {sc_id}: {entry['title']}"}
            )
            st.session_state["ask_aquagrid_messages"].append(msg)
            st.rerun()
