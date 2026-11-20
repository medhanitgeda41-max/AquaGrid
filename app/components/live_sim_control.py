"""Real-Time (2-Minute) & Optional Accelerated Live Simulation Controls for AquaGrid."""
import streamlit as st

from config.constants import RESERVOIR_FLOOR_PERCENT
from config.region_config import REGION_CONFIG, now_local
from simulator import get_simulator, realistic_daily_demand_mld, tariff_for_hour
from tools.read_feeds import read_tariff_and_quality_feeds
from tools.read_telemetry import read_network_telemetry


def _format_countdown(seconds: int) -> str:
    mins, secs = divmod(max(0, int(seconds)), 60)
    return f"{mins}:{secs:02d}"


@st.fragment(run_every=int(REGION_CONFIG["ui_refresh_seconds"]))
def _render_live_sim_feed() -> None:
    """Auto-refreshing live monitor showing non-duplicated live values and alerts."""
    sim = get_simulator()
    dt_now = now_local()

    last_dt = sim.last_update_at
    last_update_str = last_dt.strftime("%H:%M:%S") if last_dt else "Initializing..."
    countdown_str = _format_countdown(sim.seconds_until_next_update())

    summary = sim.last_tick_summary
    sim_hour = summary["hour"] if summary else dt_now.hour
    tick_num = sim.tick_count
    is_accel = not sim.real_time_clock

    telemetry_check = read_network_telemetry(current_hour=sim_hour, log_audit=False)
    feeds_check = read_tariff_and_quality_feeds(horizon_hours=6, current_hour=sim_hour, log_audit=False)

    res = feeds_check.get("reservoir", {})
    res_pct = float(res.get("current_level_percent", 48.0))
    res_vol = float(res.get("current_volume_ml", 240.0))
    plant_mld = float(feeds_check.get("current_plant_output_mld", 100.0))
    demand_mld = (
        float(summary["demand_mld"])
        if summary
        else realistic_daily_demand_mld(dt_now)
    )
    curr_rate, is_peak, tier_label = tariff_for_hour(sim_hour)

    leaks = telemetry_check.get("leak_alerts", [])
    quarantined = telemetry_check.get("quarantined_sensors", [])
    injected_fault = summary.get("injected_fault") if summary else None
    ai_err = sim.last_ai_error

    mode_badge = (
        '<span class="ag-badge ag-badge-warn">DEMO (5s)</span>'
        if is_accel
        else '<span class="ag-badge ag-badge-ok">REAL-TIME (2m)</span>'
    )

    st.markdown(
        f"""
        <div class="scada-card-ok" style="margin-top:0.5rem; padding:0.65rem 1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div>{mode_badge} <strong>Batch #{tick_num}</strong></div>
                <div class="ag-mono" style="font-size:0.84rem; color:#7CE8FF;">
                    Last update: <strong>{last_update_str}</strong> &nbsp;|&nbsp;
                    Next update in <strong>{countdown_str}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "DESAL-01 vs Demand",
        f"{plant_mld:.0f} MLD",
        delta=f"Demand {demand_mld:.1f} MLD | ${curr_rate:.2f} {tier_label}",
    )
    m2.metric(
        "Active Leaks",
        str(len(leaks)),
        delta="Alert!" if leaks else "Clear",
        delta_color="inverse" if leaks else "off",
    )
    m3.metric(
        "Quarantined Sensors",
        str(len(quarantined)),
        delta="Fault Detected" if quarantined else "All Healthy",
        delta_color="inverse" if quarantined else "off",
    )

    if ai_err:
        st.error(f"**AI Agent Error:** {ai_err}")

    wq_compliance = feeds_check.get("sector_water_quality_compliance", {})
    for wq_alert in wq_compliance.get("alerts", []):
        st.error(wq_alert)

    eff_floor_pct = float(res.get("safety_floor_percent", RESERVOIR_FLOOR_PERCENT * 100.0))
    if res.get("has_active_emergency"):
        st.warning(f"**{res.get('floor_banner', '')}**")

    if res_pct < eff_floor_pct:
        st.error(
            f"**RESERVOIR ALERT:** Strategic Reservoir RES-01 dropped to **{res_pct:.2f}% ({res_vol:.1f} ML)** — "
            f"below the mandatory **{eff_floor_pct:.1f}% safety floor**!"
        )

    if leaks:
        for lk in leaks:
            st.error(f"**LEAK CONFIRMED — {lk['junction_id']} ({lk['sector']}):** {lk['diagnostic']}")

    if quarantined:
        for qs in quarantined:
            st.warning(
                f"**SENSOR QUARANTINED — {qs['sensor_id']} ({qs['junction_id']}, {qs['sector']}):** "
                f"{'; '.join(qs['reasons'])}"
            )

    if injected_fault and not leaks and not quarantined:
        st.info(f"**LIVE EVENT:** {injected_fault['type']} on {injected_fault['target']}: {injected_fault['details']}")


def render_live_sim_controls() -> bool:
    """Render Live Simulation ON/OFF (default OFF), Demo mode (accelerated) toggle (default OFF), and fault chance."""
    sim = get_simulator()

    c_toggle, c_demo, c_prob = st.columns([1.1, 1.2, 1.4])
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

    with c_prob:
        fault_prob = st.slider(
            "Fault Chance / 2-min Cycle",
            min_value=0.0,
            max_value=1.0,
            value=float(REGION_CONFIG["default_fault_probability"]),
            step=0.01,
            key="live_sim_fault_prob",
            disabled=not live_on,
        )

    sim.configure_mode(
        demo_accelerated=bool(demo_accel),
        fault_probability=float(fault_prob),
        run_ai_after_batch=True,
    )

    if live_on:
        if not sim.is_running:
            sim.start()
        _render_live_sim_feed()
    else:
        if sim.is_running:
            sim.stop()

    return bool(live_on)
