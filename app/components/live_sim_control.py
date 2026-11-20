"""Compact Live Simulation Controls & Live Check Result (extra controls tucked in Show details)."""
import html
import streamlit as st

from config.constants import DEFAULT_SCADA_HOUR
from config.region_config import REGION_CONFIG, now_local
from guards.telemetry_freshness import (
    check_db_telemetry_freshness,
    fast_forward_telemetry_age,
    restore_telemetry_feed,
)
from simulator import get_simulator, realistic_daily_demand_mld, tariff_for_hour
from tools.read_telemetry import read_network_telemetry


def _format_countdown(seconds: int) -> str:
    mins, secs = divmod(max(0, int(seconds)), 60)
    return f"{mins}:{secs:02d}"


@st.fragment(run_every=int(REGION_CONFIG["ui_refresh_seconds"]))
def _render_live_sim_feed() -> None:
    """Auto-refreshing compact live status strip & latest live check result (no duplicate KPIs)."""
    sim = get_simulator()
    dt_now = now_local()

    last_dt = sim.last_update_at
    last_update_str = last_dt.strftime("%H:%M:%S") if last_dt else "Starting..."
    countdown_str = "PAUSED" if sim.is_feed_paused else _format_countdown(sim.seconds_until_next_update())

    summary = sim.last_tick_summary
    sim_hour = summary["hour"] if summary else dt_now.hour
    tick_num = sim.tick_count
    is_accel = not sim.real_time_clock

    telemetry_check = read_network_telemetry(current_hour=sim_hour, log_audit=False)
    freshness = telemetry_check.get("telemetry_freshness") or check_db_telemetry_freshness(current_hour=sim_hour)
    is_loss = bool(freshness.get("is_telemetry_loss", False))

    demand_mld = float(summary["demand_mld"]) if summary else realistic_daily_demand_mld(dt_now)
    _, _, tier_label = tariff_for_hour(sim_hour)
    ai_err = sim.last_ai_error
    ai_run = (summary or {}).get("ai_run") or {}
    raised = (summary or {}).get("raised_alerts") or []

    if sim.is_feed_paused or is_loss:
        mode_badge = '<span class="ag-badge ag-badge-danger">FEED PAUSED</span>'
        card_cls = "scada-card-alert"
    elif is_accel:
        mode_badge = '<span class="ag-badge ag-badge-warn">DEMO (5s)</span>'
        card_cls = "scada-card-ok"
    else:
        mode_badge = '<span class="ag-badge ag-badge-ok">REAL-TIME (2m)</span>'
        card_cls = "scada-card-ok"

    loss_inline = (
        f' &nbsp;|&nbsp; <span class="ag-badge ag-badge-danger">{html.escape(freshness["banner"])}</span>'
        if is_loss
        else ""
    )

    # Concise 1-line live check result
    if raised:
        check_line = f"Live check: {len(raised)} alert(s) — {raised[0]}"
    elif ai_run.get("run_id"):
        check_line = (
            f"Live check OK · Demand {demand_mld:.0f} MLD ({tier_label}) · "
            f"AI run #{ai_run['run_id']} ({ai_run.get('status', 'OK')})"
        )
    else:
        check_line = f"Live check OK · Demand {demand_mld:.0f} MLD ({tier_label})"

    st.markdown(
        f"""
        <div class="{card_cls}" style="margin-top:0.35rem; padding:0.65rem 1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div>{mode_badge} <strong>Batch #{tick_num}</strong>{loss_inline}</div>
                <div class="ag-mono" style="font-size:0.82rem; color:#CBD5E1;">
                    Last update: <strong>{last_update_str}</strong> &nbsp;|&nbsp;
                    Next update in <strong>{countdown_str}</strong>
                </div>
            </div>
            <div style="margin-top:0.3rem; font-size:0.84rem; color:#CBD5E1;">
                {html.escape(check_line)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ai_err:
        st.error(f"**AI Check Error:** {ai_err}")


def render_live_sim_controls() -> bool:
    """Render compact Live Simulation toggle bar; hide fault slider & feed controls in Show details."""
    sim = get_simulator()

    c_toggle, c_demo, c_spacer = st.columns([1.2, 1.3, 2.5])
    with c_toggle:
        live_on = st.toggle(
            "Live Simulation: ON/OFF",
            value=False,
            key="live_sim_toggle",
        )
    with c_demo:
        demo_accel = st.toggle(
            "Demo mode (accelerated)",
            value=False,
            key="demo_accel_toggle",
            disabled=not live_on,
        )

    with st.expander("Show details — Feed & Simulator Controls", expanded=False):
        fault_prob = st.slider(
            "Fault Chance / 2-min Cycle",
            min_value=0.0,
            max_value=1.0,
            value=float(REGION_CONFIG["default_fault_probability"]),
            step=0.01,
            key="live_sim_fault_prob",
            disabled=not live_on,
        )
        jc1, jc2, jc3 = st.columns(3)
        with jc1:
            if st.button("Stop telemetry feed", key="btn_stop_feed", width="stretch"):
                sim.pause_feed()
                st.session_state["telemetry_restored_banner"] = None
                st.rerun()
        with jc2:
            if st.button("Fast-forward 10 minutes", key="btn_ff_10min", width="stretch"):
                sim.pause_feed()
                fast_forward_telemetry_age(minutes=10)
                st.session_state["telemetry_restored_banner"] = None
                st.rerun()
        with jc3:
            if st.button("Restore feed", key="btn_restore_feed", width="stretch"):
                sim.resume_feed()
                res_hour = sim.current_hour if live_on else DEFAULT_SCADA_HOUR
                restored = restore_telemetry_feed(current_hour=res_hour)
                st.session_state["telemetry_restored_banner"] = restored["message"]
                st.rerun()

    sim.configure_mode(
        demo_accelerated=bool(demo_accel),
        fault_probability=float(fault_prob),
        run_ai_after_batch=True,
    )

    restored_msg = st.session_state.get("telemetry_restored_banner")
    if restored_msg:
        st.success(f"**{restored_msg}**")

    if live_on:
        if not sim.is_running:
            sim.start()
        _render_live_sim_feed()
    else:
        if sim.is_running:
            sim.stop()

    return bool(live_on)
