"""Clean SCADA header: 5 key operator metrics and active alerts."""
import html
import streamlit as st
from config.constants import DEFAULT_SCADA_HOUR, RESERVOIR_FLOOR_PERCENT
from config.region_config import REGION_CONFIG, now_local
from database.connection import get_db_connection
from tools.read_telemetry import read_network_telemetry
from tools.read_feeds import read_tariff_and_quality_feeds
from components.badges import status_badge


def _compute_grid_health(telemetry_data: dict, feeds_data: dict, pending_count: int) -> tuple[int, str]:
    """Deterministically calculate Grid Health Score (0-100) from live SCADA telemetry."""
    score = 100
    leak_count = len(telemetry_data.get("leak_alerts", []))
    quarantined_count = len(telemetry_data.get("quarantined_sensors", []))
    injection_count = sum(1 for a in feeds_data.get("sanitized_alerts", []) if a.get("injection_detected"))
    wq = feeds_data.get("water_quality", {})
    bloom_sev = str(wq.get("bloom_severity", "NONE")).upper()
    res = feeds_data.get("reservoir", {})
    res_pct = float(res.get("current_level_percent", 48.0))
    eff_floor = float(res.get("safety_floor_percent", RESERVOIR_FLOOR_PERCENT * 100.0))
    non_compliant = len(feeds_data.get("excluded_sectors", []))

    score -= leak_count * 20
    score -= quarantined_count * 6
    score -= injection_count * 12
    score -= non_compliant * 25
    if telemetry_data.get("telemetry_loss"):
        score -= 35
    if bloom_sev == "CRITICAL":
        score -= 15
    elif bloom_sev == "MODERATE":
        score -= 10
    elif bloom_sev == "LOW":
        score -= 5
    if res_pct < eff_floor:
        score -= 30
    if pending_count > 0:
        score -= 4

    score = max(0, min(100, score))
    if score >= 85:
        label = "NOMINAL"
    elif score >= 65:
        label = "ELEVATED WATCH"
    else:
        label = "ALERT ACTIVE"
    return score, label


def render_header(display_hour: int = DEFAULT_SCADA_HOUR) -> dict:
    """Render clean banner, 5 operator KPIs, and active alerts. Returns live snapshot dict."""
    conn = get_db_connection()
    try:
        plant = conn.execute("SELECT * FROM plant_setpoints WHERE plant_id = 'DESAL-01';").fetchone()
        pending_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM pending_actions WHERE status = 'PENDING';"
        ).fetchone()["cnt"]
        v12 = conn.execute("SELECT valve_id, sector, status FROM valves WHERE valve_id = 'V-12';").fetchone()
    finally:
        conn.close()

    telemetry_data = read_network_telemetry(current_hour=display_hour)
    feeds_data = read_tariff_and_quality_feeds(horizon_hours=6, current_hour=display_hour)

    _, health_label = _compute_grid_health(telemetry_data, feeds_data, pending_count)
    active_scenario_name = st.session_state.get("active_scenario_name", "Clean Baseline")

    res = feeds_data.get("reservoir", {})
    res_pct = float(res.get("current_level_percent", 48.0))
    eff_floor_pct = float(res.get("safety_floor_percent", 35.0))
    floor_banner = res.get("floor_banner", f"Reservoir floor: {eff_floor_pct:.1f}%")
    has_emergency = bool(res.get("has_active_emergency", False))
    plant_mld = float(plant["production_mld"]) if plant else 100.0
    plant_status = plant["status"] if plant else "ONLINE"

    tariffs = feeds_data.get("tariffs", [])
    curr_tariff = tariffs[0]["rate_per_kwh"] if tariffs else 0.48
    is_peak = feeds_data.get("is_current_peak_tariff", True)
    v12_status = v12["status"] if v12 else "OPEN"

    leaks = telemetry_data.get("leak_alerts", [])
    quarantined = telemetry_data.get("quarantined_sensors", [])
    leak_count = len(leaks)
    dubai_clock_str = now_local().strftime("%H:%M:%S")

    st.markdown(
        f"""
        <div class="scada-banner">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div>
                    <div class="scada-title">AQUAGRID SCADA CONTROL ROOM</div>
                    <div class="scada-subtitle">
                        {html.escape(REGION_CONFIG['region_name'])} · {dubai_clock_str}
                        &nbsp;|&nbsp; {html.escape(active_scenario_name)}
                    </div>
                </div>
                <div>{status_badge(health_label)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5 Core Operator Values
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Reservoir",
        f"{res_pct:.1f}%",
        delta=f"Floor {eff_floor_pct:.1f}%",
        delta_color="normal" if res_pct >= eff_floor_pct else "inverse",
    )
    c2.metric(
        "Plant Output",
        f"{plant_mld:.0f} MLD",
        delta=plant_status,
    )
    c3.metric(
        "Tariff",
        f"${curr_tariff:.2f}/kWh",
        delta="PEAK" if is_peak else "OFF-PEAK",
        delta_color="inverse" if is_peak else "normal",
    )
    c4.metric(
        "Active Leaks",
        str(leak_count),
        delta="Leak" if leak_count > 0 else "None",
        delta_color="inverse" if leak_count > 0 else "off",
    )
    c5.metric(
        "Pending Approvals",
        str(pending_count),
        delta="Pending" if pending_count > 0 else "Clear",
        delta_color="inverse" if pending_count > 0 else "off",
    )

    # Active Alerts (shown only when triggered; red = danger, orange = warning)
    if telemetry_data.get("telemetry_loss"):
        st.error(f"**{telemetry_data['telemetry_freshness']['banner']}**")

    wq_compliance = feeds_data.get("sector_water_quality_compliance", {})
    for wq_alert in wq_compliance.get("alerts", []):
        st.error(f"**{wq_alert}**")

    if res_pct < eff_floor_pct:
        st.error(f"**RESERVOIR LOW:** RES-01 at {res_pct:.1f}% (below {eff_floor_pct:.1f}% floor).")

    if has_emergency:
        st.warning(f"**{floor_banner}**")

    for lk in leaks:
        st.error(f"**CONFIRMED_PIPE_LEAK — {lk['junction_id']} ({lk['sector']}):** {lk['diagnostic']}")

    if quarantined:
        q_ids = ", ".join(q["sensor_id"] for q in quarantined)
        st.warning(f"**QUARANTINED ({len(quarantined)}):** {q_ids}")

    inj_alerts = [a for a in feeds_data.get("sanitized_alerts", []) if a.get("injection_detected")]
    if inj_alerts:
        st.error(f"**BLOCKED:** {len(inj_alerts)} prompt injection alert(s) quarantined before reaching AI.")

    return {
        "telemetry_data": telemetry_data,
        "feeds_data": feeds_data,
        "pending_count": pending_count,
        "plant_mld": plant_mld,
        "v12_status": v12_status,
    }
