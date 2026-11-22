"""Thin always-visible 1-line SCADA status bar for chat-first AquaGrid."""
import html
import streamlit as st
from config.constants import DEFAULT_SCADA_HOUR, RESERVOIR_FLOOR_PERCENT
from config.region_config import REGION_CONFIG, now_local
from database.connection import get_db_connection
from simulator import get_simulator
from tools.read_telemetry import read_network_telemetry
from tools.read_feeds import read_tariff_and_quality_feeds
from components.badges import status_badge


def _compute_grid_health(telemetry_data: dict, feeds_data: dict, pending_count: int) -> tuple[int, str]:
    """Deterministically calculate system health & top status badge (NOMINAL / ALERT ACTIVE / TELEMETRY LOSS)."""
    if telemetry_data.get("telemetry_loss"):
        return 20, "TELEMETRY LOSS"

    score = 100
    leak_count = len(telemetry_data.get("leak_alerts", []))
    quarantined_count = len(telemetry_data.get("quarantined_sensors", []))
    injection_count = sum(1 for a in feeds_data.get("sanitized_alerts", []) if a.get("injection_detected"))
    wq = feeds_data.get("water_quality", {})
    bloom_sev = str(wq.get("bloom_severity", "NONE")).upper()
    res = feeds_data.get("reservoir", {})
    res_pct = float(res.get("current_level_percent", 48.0))
    eff_floor = float(res.get("safety_floor_percent", RESERVOIR_FLOOR_PERCENT * 100.0))
    warning_state = str(res.get("warning_state", "NORMAL")).upper()
    non_compliant = len(feeds_data.get("excluded_sectors", []))

    score -= leak_count * 25
    score -= quarantined_count * 6
    score -= injection_count * 20
    score -= non_compliant * 30
    if bloom_sev == "CRITICAL":
        score -= 15
    elif bloom_sev == "MODERATE":
        score -= 10
    if res_pct <= eff_floor or warning_state == "CRITICAL":
        score -= 35
    elif warning_state == "WARNING" or res_pct <= eff_floor + 5.0:
        score -= 18
    if pending_count > 0:
        score -= 4

    score = max(0, min(100, score))
    if leak_count > 0 or non_compliant > 0 or injection_count > 0 or res_pct <= eff_floor or warning_state == "CRITICAL" or score < 65:
        label = "ALERT ACTIVE"
    elif warning_state == "WARNING" or res_pct <= eff_floor + 5.0:
        label = "RESERVOIR WARNING"
    elif score >= 85 and not res.get("has_active_emergency"):
        label = "NOMINAL"
    else:
        label = "ELEVATED WATCH"
    return score, label


def render_header(display_hour: int = DEFAULT_SCADA_HOUR) -> dict:
    """Render thin one-line top status bar. Returns live snapshot dict."""
    conn = get_db_connection()
    try:
        plant = conn.execute("SELECT * FROM plant_setpoints WHERE plant_id = 'DESAL-01';").fetchone()
        pending_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM pending_actions WHERE status = 'PENDING';"
        ).fetchone()["cnt"]
        v12 = conn.execute("SELECT valve_id, sector, status FROM valves WHERE valve_id = 'V-12';").fetchone()
    finally:
        conn.close()

    telemetry_data = read_network_telemetry(current_hour=display_hour, log_audit=False)
    feeds_data = read_tariff_and_quality_feeds(horizon_hours=6, current_hour=display_hour, log_audit=False)

    _, health_label = _compute_grid_health(telemetry_data, feeds_data, pending_count)

    res = feeds_data.get("reservoir", {})
    res_pct = float(res.get("current_level_percent", 48.0))
    eff_floor_pct = float(res.get("safety_floor_percent", 35.0))
    rate_str = str(res.get("rate_str") or "→ 0.0 %/h")
    floor_eta_str = str(res.get("floor_eta_str") or "stable")
    res_warn_state = str(res.get("warning_state") or "NORMAL").upper()
    plant_mld = float(plant["production_mld"]) if plant else 100.0
    tariffs = feeds_data.get("tariffs", [])
    curr_tariff = float(tariffs[0]["rate_per_kwh"]) if tariffs else 0.48
    is_peak = bool(feeds_data.get("is_current_peak_tariff", True))
    v12_status = v12["status"] if v12 else "OPEN"
    leak_count = len(telemetry_data.get("leak_alerts", []))

    sim = get_simulator()
    if st.session_state.get("live_sim_toggle", False) and sim is not None and not getattr(sim, "real_time_clock", True):
        clock_str = f"{sim.current_hour:02d}:00:00"
    else:
        clock_str = now_local().strftime("%H:%M:%S")

    if res_warn_state == "CRITICAL" or res_pct <= eff_floor_pct:
        res_color = "#EF4444"
        rate_color = "#FCA5A5"
    elif res_warn_state == "WARNING" or res_pct <= eff_floor_pct + 5.0:
        res_color = "#F59E0B"
        rate_color = "#FCD34D"
    else:
        res_color = "#38BDF8"
        rate_color = "#CBD5E1"

    leak_color = "#EF4444" if leak_count > 0 else "#10B981"
    pend_color = "#F59E0B" if pending_count > 0 else "#10B981"
    peak_tag = "PEAK" if is_peak else "OFF-PEAK"

    if health_label == "TELEMETRY LOSS":
        badge_html = status_badge("TELEMETRY_LOSS").replace("TELEMETRY_LOSS", "TELEMETRY LOSS")
    elif health_label == "RESERVOIR WARNING":
        badge_html = '<span class="ag-badge ag-badge-warn">ALERT · RESERVOIR LOW</span>'
    else:
        badge_html = status_badge(health_label)

    live_paused_html = (
        '<span style="font-size:0.75rem; color:#64748B; margin-right:0.45rem;">Live simulation paused</span>'
        if (not st.session_state.get("live_sim_toggle", False) and not telemetry_data.get("telemetry_loss"))
        else ""
    )

    st.markdown(
        f"""
        <div class="scada-banner" style="padding:0.5rem 0.9rem; margin-bottom:0.55rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.65rem; font-size:0.84rem;">
                <div style="display:flex; align-items:center; flex-wrap:wrap; gap:0.85rem;">
                    <strong style="font-size:0.94rem; letter-spacing:0.03em; color:#F8FAFC;">AQUAGRID</strong>
                    <span class="ag-mono" style="color:#94A3B8;">{html.escape(clock_str)}</span>
                    <span>RES-01 <strong class="ag-mono" style="color:{res_color};">{res_pct:.1f}%</strong>
                        <span class="ag-mono" style="color:{rate_color}; font-size:0.78rem; margin-left:0.2rem;">{html.escape(rate_str)} · {html.escape(floor_eta_str)}</span>
                    </span>
                    <span>DESAL-01 <strong class="ag-mono">{plant_mld:.0f} MLD</strong></span>
                    <span>Tariff <strong class="ag-mono">${curr_tariff:.2f}/kWh</strong> <span style="color:#94A3B8; font-size:0.76rem;">({peak_tag})</span></span>
                    <span>Leaks <strong class="ag-mono" style="color:{leak_color};">{leak_count}</strong></span>
                    <span>HITL <strong class="ag-mono" style="color:{pend_color};">{pending_count}</strong></span>
                </div>
                <div style="display:flex; align-items:center;">{live_paused_html}{badge_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return {
        "telemetry_data": telemetry_data,
        "feeds_data": feeds_data,
        "pending_count": pending_count,
        "plant_mld": plant_mld,
        "v12_status": v12_status,
        "health_label": health_label,
    }
