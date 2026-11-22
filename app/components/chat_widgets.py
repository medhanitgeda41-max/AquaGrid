"""Inline visual widgets rendered inside Ask AquaGrid chat messages (map, charts, tables, schedule checks, approvals, scenario cards)."""
import html
import json
import altair as alt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from config.constants import DEFAULT_SCADA_HOUR
from database.connection import get_db_connection
from digital_twin.hydraulics import simulate_valve_change_impact
from guards.injection_filter import detect_injection
from guards.reservoir_model import project_reservoir_series
from tools.read_feeds import read_tariff_and_quality_feeds
from tools.write_gateway import execute_human_approval, execute_human_rejection
from components.badges import format_dubai_time, status_badge
from components.digital_twin_3d import render_3d_digital_twin, set_twin_view
from components.leak_map_view import build_leak_folium_map
from components.telemetry_view import _build_reservoir_chart


def _format_val(action_type: str, value: float) -> str:
    if action_type in ("CLOSE_VALVE", "OPEN_VALVE"):
        return "OPEN" if float(value) >= 0.5 else "CLOSED"
    if action_type == "ISOLATE_AND_FLUSH":
        return f"{float(value):.2f} mg/L" if float(value) > 0 else "ISOLATE_AND_FLUSH"
    return f"{float(value):.1f} MLD"


def render_inline_map(
    sector: str | None,
    leak_id: str | None,
    widget_key: str,
    snapshot: dict,
    show_boundaries_default: bool = True,
    show_view_toggle: bool = True,
) -> None:
    """Render interactive Leaflet map with sector boundaries, layer toggles, English basemap toggle, and single-sector summary table."""
    from config.region_config import SECTOR_COLOR_NAMES
    from database.dubai_network import JUNCTION_SECTOR_MAP, VALVE_SECTOR_MAP

    conn = get_db_connection()
    try:
        seg_rows = conn.execute("SELECT * FROM pipe_segments ORDER BY sector, segment_id;").fetchall()
        leak_rows = conn.execute("SELECT * FROM leaks ORDER BY detected_at DESC;").fetchall()
        valve_rows = conn.execute("SELECT valve_id, sector, status FROM valves ORDER BY valve_id ASC;").fetchall()
        wq_rows = conn.execute(
            """
            SELECT sector, chlorine_mg_l
            FROM sector_water_quality
            WHERE id IN (SELECT MAX(id) FROM sector_water_quality GROUP BY sector);
            """
        ).fetchall()
        p_rows = conn.execute(
            """
            SELECT pipe_segment_id, pressure_psi
            FROM scada_history
            WHERE id IN (SELECT MAX(id) FROM scada_history GROUP BY pipe_segment_id);
            """
        ).fetchall()
    finally:
        conn.close()

    segments = [dict(r) for r in seg_rows]
    all_leaks = [dict(r) for r in leak_rows]
    valve_statuses = {r["valve_id"]: r["status"] for r in valve_rows}
    wq_by_sector = {r["sector"]: float(r["chlorine_mg_l"]) for r in wq_rows}
    latest_pressures = {r["pipe_segment_id"]: float(r["pressure_psi"]) for r in p_rows if r["pressure_psi"] is not None}
    quarantined_list = (snapshot.get("telemetry_data") or {}).get("quarantined_sensors", [])
    quarantined_ids = {s["sensor_id"] for s in quarantined_list}

    norm_sector = (sector or "").strip()
    if norm_sector.lower() in ("", "all", "all sectors"):
        highlight_sector = None
    elif norm_sector.upper().startswith("LEAK-"):
        leak_id = norm_sector.upper()
        highlight_sector = None
    else:
        highlight_sector = norm_sector

    # "2D Map / 3D View" toggle so operators can inspect either the 2D OSM map or the 3D digital twin
    if show_view_toggle:
        view_mode = st.radio(
            "Map View",
            options=["2D Map", "3D View"],
            index=0,
            horizontal=True,
            key=f"map_view_mode_{widget_key}",
            label_visibility="collapsed",
        )
        if view_mode == "3D View":
            focus_arg = leak_id or highlight_sector or "all"
            render_3d_digital_twin(
                focus=focus_arg,
                widget_key=f"inline3d_{widget_key}",
                height=430,
                current_hour=DEFAULT_SCADA_HOUR,
                show_controls=True,
            )
            return

    # Compact layer toggles (OpenStreetMap is the only basemap)
    t1, t2, t3, t4, t5 = st.columns([1.3, 0.9, 1.0, 0.9, 0.9])
    with t1:
        show_boundaries = st.checkbox("Boundaries", value=bool(show_boundaries_default), key=f"map_b_{widget_key}")
    with t2:
        show_pipes = st.checkbox("Pipes", value=True, key=f"map_p_{widget_key}")
    with t3:
        show_sensors = st.checkbox("Sensors", value=True, key=f"map_s_{widget_key}")
    with t4:
        show_valves = st.checkbox("Valves", value=True, key=f"map_v_{widget_key}")
    with t5:
        show_leaks = st.checkbox("Leaks", value=True, key=f"map_l_{widget_key}")

    matched_leaks = [
        lk for lk in all_leaks
        if (not highlight_sector or lk["sector"].lower() == highlight_sector.lower())
        and (not leak_id or lk["leak_id"].upper() == leak_id.upper())
    ]
    selected_leak = matched_leaks[0] if (leak_id and matched_leaks) else None

    fmap = build_leak_folium_map(
        segments=segments,
        leaks=all_leaks,
        quarantined_ids=quarantined_ids,
        latest_pressures=latest_pressures,
        selected_leak=selected_leak,
        show_pipes=show_pipes,
        pipe_color_mode="Sector",
        show_leaks=show_leaks,
        show_sensors=show_sensors,
        show_boundaries=show_boundaries,
        show_valves=show_valves,
        highlight_sector=highlight_sector,
        valve_statuses=valve_statuses,
        fit_bounds=True,
    )
    st_folium(fmap, height=380, width=None, returned_objects=[], key=f"map_{widget_key}")

    scope_lbl = leak_id or highlight_sector or "All 6 Sectors (Dubai Network)"
    st.caption(f"Map: {scope_lbl} · Basemap: OpenStreetMap (© OpenStreetMap contributors)")

    # Small summary table under the map when focused on a single sector
    if highlight_sector:
        sec_pipes = [s for s in segments if s["sector"].lower() == highlight_sector.lower()]
        sec_valves = [v for v in valve_rows if v["sector"].lower() == highlight_sector.lower()]
        sec_juncs = [j for j, sc in JUNCTION_SECTOR_MAP.items() if sc.lower() == highlight_sector.lower()]
        sec_quar = [q for q in quarantined_list if str(q.get("sector", "")).lower() == highlight_sector.lower()]
        sec_active_leaks = [l for l in matched_leaks if str(l.get("status", "")).lower() != "resolved"]
        cl_val = wq_by_sector.get(highlight_sector)
        if cl_val is None:
            for k, v in wq_by_sector.items():
                if k.lower() == highlight_sector.lower():
                    cl_val = v
                    break
        cl_str = f"{cl_val:.2f} mg/L" if cl_val is not None else "-"
        sec_status = "NON_COMPLIANT" if (cl_val is not None and cl_val < 0.20) else ("ALERT" if sec_active_leaks else "NOMINAL")
        open_v = sum(1 for v in sec_valves if v["status"] == "OPEN")
        closed_v = len(sec_valves) - open_v
        color_nm = SECTOR_COLOR_NAMES.get(highlight_sector, "-")

        sec_df = pd.DataFrame(
            [
                {
                    "Sector": highlight_sector,
                    "Color": color_nm,
                    "Pipes": len(sec_pipes),
                    "Valves": f"{len(sec_valves)} ({open_v} OPEN / {closed_v} CLOSED)",
                    "Sensors": len(sec_juncs) * 2,
                    "Active Leaks": len(sec_active_leaks),
                    "Quarantined": len(sec_quar),
                    "Chlorine": cl_str,
                    "Status": sec_status,
                }
            ]
        )
        st.dataframe(sec_df, width="stretch", hide_index=True)


def _build_reservoir_trend_chart(snapshot: dict) -> alt.LayerChart:
    """Build compact reservoir trend chart (last 6h history + next 12h projection + active safety floor line)."""
    feeds_data = snapshot.get("feeds_data") or read_tariff_and_quality_feeds(horizon_hours=6, current_hour=DEFAULT_SCADA_HOUR, log_audit=False)
    res = feeds_data.get("reservoir") or {}
    wq = feeds_data.get("water_quality") or {}
    curr_hour = int(feeds_data.get("current_hour", DEFAULT_SCADA_HOUR))
    plant_mld = float(feeds_data.get("current_plant_output_mld", snapshot.get("plant_mld", 100.0)))
    floor_pct = float(res.get("safety_floor_percent", 35.0))
    cap_ml = float(res.get("capacity_ml", 500.0))

    # Pull any recent live reservoir snapshots from scada_history for the last 6h segment
    history_pts: list[dict] = []
    conn = get_db_connection()
    try:
        h_rows = conn.execute(
            """
            SELECT recorded_at, reservoir_level_percent, reservoir_volume_ml
            FROM scada_history
            WHERE reservoir_level_percent IS NOT NULL
            ORDER BY id DESC
            LIMIT 6;
            """
        ).fetchall()
    finally:
        conn.close()
    if len(h_rows) >= 2:
        for r in reversed(h_rows):
            history_pts.append(
                {
                    "hour_label": format_dubai_time(r["recorded_at"])[11:16],
                    "level_percent": float(r["reservoir_level_percent"]),
                    "volume_ml": float(r["reservoir_volume_ml"] or (float(r["reservoir_level_percent"]) * cap_ml / 100.0)),
                }
            )

    series = project_reservoir_series(
        current_volume_ml=float(res.get("current_volume_ml", 240.0)),
        current_hour=curr_hour,
        production_mld=plant_mld,
        demand_by_hour=feeds_data.get("hourly_demand_24h") or {},
        floor_percent=floor_pct,
        capacity_ml=cap_ml,
        history_hours=6,
        forecast_hours=12,
        history_points=history_pts if len(history_pts) >= 2 else None,
        bloom_severity=str(wq.get("bloom_severity", "NONE")),
        turbidity_ntu=float(wq.get("intake_turbidity_ntu", 1.2)),
        chlorine_ppm=float(wq.get("intake_chlorine_ppm", 1.8)),
    )
    df = pd.DataFrame(series)
    x_order = list(dict.fromkeys(df["hour_label"].tolist()))
    y_min = max(0.0, min(floor_pct - 5.0, float(df["level_percent"].min()) - 3.0))
    y_max = min(100.0, max(55.0, float(df["level_percent"].max()) + 4.0))

    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.6, point=alt.OverlayMarkDef(size=42))
        .encode(
            x=alt.X("hour_label:N", sort=x_order, title="Last 6h & Next 12h Projection", axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
            y=alt.Y("level_percent:Q", title="RES-01 Level (%)", scale=alt.Scale(domain=[y_min, y_max])),
            color=alt.Color(
                "segment:N",
                scale=alt.Scale(domain=["History (last 6h)", "Projection (next 12h)"], range=["#94A3B8", "#38BDF8"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            strokeDash=alt.StrokeDash(
                "segment:N",
                scale=alt.Scale(domain=["History (last 6h)", "Projection (next 12h)"], range=[[1, 0], [5, 3]]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("hour_label:N", title="Time"),
                alt.Tooltip("segment:N", title="Series"),
                alt.Tooltip("level_percent:Q", title="Level (%)", format=".2f"),
                alt.Tooltip("volume_ml:Q", title="Volume (ML)", format=".1f"),
            ],
        )
    )
    floor_df = pd.DataFrame({"y": [floor_pct], "label": [f"Safety Floor ({floor_pct:.1f}%)"]})
    rule = alt.Chart(floor_df).mark_rule(color="#EF4444", strokeDash=[6, 4], strokeWidth=2.2).encode(y="y:Q")
    lbl = alt.Chart(floor_df).mark_text(align="left", dx=5, dy=-6, color="#FCA5A5", fontWeight="bold", fontSize=10).encode(
        x=alt.value(4), y="y:Q", text="label:N"
    )
    return (line + rule + lbl).properties(height=225)


def render_inline_chart(metric: str, asset: str | None, hours: int, snapshot: dict) -> None:
    """Render live pressure, flow, reservoir, reservoir_trend (6h+12h), or tariff chart inside chat."""
    metric = (metric or "reservoir").lower()

    if metric == "reservoir_trend":
        chart = _build_reservoir_trend_chart(snapshot)
        st.altair_chart(chart, width="stretch")
        return

    if metric == "reservoir":
        feeds_data = snapshot.get("feeds_data") or {}
        plant_mld = float(snapshot.get("plant_mld", 100.0))
        agent_result = st.session_state.get("last_agent_result")
        active_scenario = st.session_state.get("active_scenario")
        chart = _build_reservoir_chart(feeds_data, plant_mld, agent_result, active_scenario)
        st.altair_chart(chart, width="stretch")
        return

    if metric == "tariff":
        conn = get_db_connection()
        try:
            td_rows = conn.execute(
                "SELECT t.hour, t.rate_per_kwh, t.is_peak, d.demand_mld FROM tariffs t JOIN demand_forecast d ON t.hour = d.hour ORDER BY t.hour ASC;"
            ).fetchall()
        finally:
            conn.close()
        td_df = pd.DataFrame(
            [
                {
                    "hour_label": f"{int(r['hour']):02d}:00",
                    "tariff_rate": float(r["rate_per_kwh"]),
                    "tier": "PEAK ($0.48)" if r["is_peak"] else "Off/Mid-Peak",
                    "demand_mld": float(r["demand_mld"]),
                }
                for r in td_rows
            ]
        )
        tariff_chart = (
            alt.Chart(td_df)
            .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
            .encode(
                x=alt.X("hour_label:N", sort=None, title="Hour", axis=alt.Axis(labelAngle=-45, labelFontSize=10)),
                y=alt.Y("tariff_rate:Q", title="Rate ($/kWh)", scale=alt.Scale(domain=[0.0, 0.55])),
                color=alt.Color("tier:N", scale=alt.Scale(domain=["PEAK ($0.48)", "Off/Mid-Peak"], range=["#EF4444", "#64748B"]), legend=alt.Legend(orient="top", title=None)),
                tooltip=["hour_label", "tariff_rate", "demand_mld", "tier"],
            )
            .properties(height=240)
        )
        st.altair_chart(tariff_chart, width="stretch")
        return

    # Pressure or Flow chart for an asset (junction, sensor, or sector)
    target = (asset or "J-14").strip()
    prefix = "P-%" if metric == "pressure" else "F-%"
    y_title = "Pressure (psi)" if metric == "pressure" else "Flow (L/s)"
    color_hex = "#10B981" if metric == "pressure" else "#F59E0B"

    conn = get_db_connection()
    try:
        # First check scada_history (2-min resolution) if matching segment/sensor/sector
        hist_rows = conn.execute(
            """
            SELECT recorded_at, pressure_psi, flow_lps, sensor_id, sector
            FROM scada_history
            WHERE sensor_id = ? OR sector = ? OR pipe_segment_id = ?
            ORDER BY recorded_at DESC
            LIMIT ?;
            """,
            (target.replace("J-", "P-"), target, target, max(12, int(hours) * 30)),
        ).fetchall()

        tel_rows = conn.execute(
            """
            SELECT hour, sensor_id, junction_id, sector, value, unit
            FROM telemetry
            WHERE (junction_id = ? OR sensor_id = ? OR sector = ?) AND sensor_id LIKE ?
            ORDER BY hour ASC;
            """,
            (target, target, target, prefix),
        ).fetchall()
    finally:
        conn.close()

    if tel_rows:
        df = pd.DataFrame(
            [
                {
                    "time_label": f"{int(r['hour']):02d}:00",
                    "value": float(r["value"]),
                    "sensor": r["sensor_id"],
                }
                for r in tel_rows[-max(6, int(hours) * 2):]
            ]
        )
        chart = (
            alt.Chart(df)
            .mark_line(color=color_hex, strokeWidth=2.6, point=alt.OverlayMarkDef(size=55, color=color_hex))
            .encode(
                x=alt.X("time_label:N", sort=None, title=f"{target} — Time"),
                y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=False)),
                tooltip=["time_label", "sensor", "value"],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, width="stretch")
    elif hist_rows:
        ordered = list(reversed(hist_rows))
        col_name = "pressure_psi" if metric == "pressure" else "flow_lps"
        df = pd.DataFrame(
            [
                {
                    "time_label": format_dubai_time(r["recorded_at"])[11:16],
                    "value": float(r[col_name] or 0.0),
                }
                for r in ordered
            ]
        )
        chart = (
            alt.Chart(df)
            .mark_line(color=color_hex, strokeWidth=2.4)
            .encode(
                x=alt.X("time_label:N", sort=None, title=f"{target} — Last {hours}h"),
                y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=False)),
                tooltip=["time_label", "value"],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.info(f"No {metric} readings found for `{target}`.")


def render_inline_table(what: str) -> None:
    """Render live table for leaks, sensors, schedule, audit, or valves inside chat."""
    what = (what or "leaks").lower()
    conn = get_db_connection()
    try:
        if what == "leaks":
            rows = conn.execute(
                "SELECT leak_id, sector, location_name, severity, estimated_loss_m3_per_hour, status, detected_at FROM leaks ORDER BY detected_at DESC LIMIT 20;"
            ).fetchall()
            df = pd.DataFrame(
                [
                    {
                        "Leak": r["leak_id"],
                        "Sector": r["sector"],
                        "Location": r["location_name"],
                        "Severity": r["severity"],
                        "Loss (m3/h)": round(float(r["estimated_loss_m3_per_hour"]), 1),
                        "Status": r["status"],
                        "Detected": format_dubai_time(r["detected_at"]),
                    }
                    for r in rows
                ]
            )
        elif what == "sensors":
            rows = conn.execute(
                """
                SELECT sensor_id, junction_id, sector, value, unit, status, timestamp
                FROM telemetry
                WHERE id IN (SELECT MAX(id) FROM telemetry GROUP BY sensor_id)
                ORDER BY sensor_id ASC;
                """
            ).fetchall()
            df = pd.DataFrame(
                [
                    {
                        "Sensor": r["sensor_id"],
                        "Junction": r["junction_id"],
                        "Sector": r["sector"],
                        "Value": f"{float(r['value']):.2f} {r['unit']}",
                        "Status": r["status"],
                        "Timestamp": format_dubai_time(r["timestamp"]),
                    }
                    for r in rows
                ]
            )
        elif what == "schedule":
            rows = conn.execute(
                "SELECT t.hour, t.rate_per_kwh, t.is_peak, d.demand_mld FROM tariffs t JOIN demand_forecast d ON t.hour = d.hour ORDER BY t.hour ASC;"
            ).fetchall()
            df = pd.DataFrame(
                [
                    {
                        "Hour": f"{int(r['hour']):02d}:00",
                        "Tariff ($/kWh)": f"${float(r['rate_per_kwh']):.2f}",
                        "Tier": "PEAK" if r["is_peak"] else "Off/Mid-Peak",
                        "Demand (MLD)": round(float(r["demand_mld"]), 1),
                    }
                    for r in rows
                ]
            )
        elif what == "valves":
            rows = conn.execute("SELECT valve_id, sector, status, last_updated FROM valves ORDER BY valve_id ASC;").fetchall()
            df = pd.DataFrame(
                [
                    {
                        "Valve": r["valve_id"],
                        "Sector": r["sector"],
                        "Status": r["status"],
                        "Updated": format_dubai_time(r["last_updated"]),
                    }
                    for r in rows
                ]
            )
        else:  # audit
            rows = conn.execute(
                "SELECT id, timestamp, actor, event_type, target_id, status, details FROM audit_log ORDER BY id DESC LIMIT 40;"
            ).fetchall()
            df = pd.DataFrame(
                [
                    {
                        "ID": r["id"],
                        "Time": format_dubai_time(r["timestamp"]),
                        "Actor": r["actor"],
                        "Event": r["event_type"],
                        "Target": r["target_id"] or "-",
                        "Status": r["status"],
                        "Details": r["details"],
                    }
                    for r in rows
                ]
            )
    finally:
        conn.close()

    if df.empty:
        st.info(f"No rows available for `{what}`.")
    else:
        st.dataframe(df, width="stretch", hide_index=True, height=min(360, 42 + len(df) * 35))


def render_inline_schedule_validation(plan: str, validation: dict) -> None:
    """Render hour-by-hour schedule validation result inside chat."""
    if not validation:
        return
    sv_status = validation.get("status", "UNKNOWN")
    card_cls = "scada-card-ok" if sv_status == "VALID" else "scada-card-alert"
    lines_html = [
        f"<div>{status_badge(sv_status)} <strong>{html.escape(str(validation.get('summary', '')))}</strong></div>"
    ]
    if validation.get("first_breach_hour"):
        lines_html.append(
            f'<div style="margin-top:0.25rem; color:#FCA5A5;">'
            f'<strong>First Floor Breach:</strong> '
            f'<span class="ag-mono">{html.escape(str(validation.get("first_breach_hour")))} '
            f'({validation.get("first_breach_percent")}%)</span></div>'
        )
    blocks = validation.get("chronological_blocks") or validation.get("blocks", [])
    for blk in blocks:
        blk_badge = status_badge(blk.get("status", "UNKNOWN"))
        win = html.escape(str(blk.get("window")))
        mld = blk.get("proposed_mld")
        s_pct = blk.get("start_projected_percent")
        m_pct = blk.get("min_projected_percent")
        e_pct = blk.get("end_projected_percent")
        gap_html = f' <span style="color:#FCD34D; font-size:0.8rem;">[{html.escape(str(blk.get("gap_note")))}]</span>' if blk.get("is_gap") else ""
        reasons_html = "".join(
            f'<div style="color:#FCA5A5; margin-left:1rem; font-size:0.82rem;">&rarr; {html.escape(str(r))}</div>'
            for r in blk.get("reasons", [])
        )
        lines_html.append(
            f'<div style="margin-top:0.3rem;">{blk_badge} '
            f'<span class="ag-mono">{win} @ {mld} MLD</span>{gap_html} '
            f'<span style="color:#94A3B8;">({s_pct}% &rarr; min {m_pct}% &rarr; {e_pct}%)</span>'
            f'{reasons_html}</div>'
        )
    st.markdown(f'<div class="{card_cls}">{"".join(lines_html)}</div>', unsafe_allow_html=True)


def render_inline_3d(
    focus: str | None,
    widget_key: str,
    snapshot: dict,
    height: int = 440,
) -> None:
    """Render the interactive 3D Digital Twin inside chat with a '3D View / 2D Map' toggle."""
    focus_val = (focus or "all").strip() or "all"
    view_mode = st.radio(
        "Twin View",
        options=["3D View", "2D Map"],
        index=0,
        horizontal=True,
        key=f"twin_view_mode_{widget_key}",
        label_visibility="collapsed",
    )
    if view_mode == "2D Map":
        sec_arg = focus_val if focus_val.lower().startswith("sector") else "all"
        lid_arg = focus_val.upper() if focus_val.upper().startswith("LEAK-") else None
        render_inline_map(
            sector=sec_arg,
            leak_id=lid_arg,
            widget_key=f"fb2d_{widget_key}",
            snapshot=snapshot,
            show_boundaries_default=True,
            show_view_toggle=False,
        )
        return

    render_3d_digital_twin(
        focus=focus_val,
        widget_key=f"twin3d_{widget_key}",
        height=height,
        current_hour=DEFAULT_SCADA_HOUR,
        show_controls=True,
    )


def build_approval_impact_preview(action_type: str, target_id: str, db_path: str | None = None) -> dict:
    """Compute hydraulic impact preview & 3D focus metadata for a pending HITL approval card (read-only)."""
    a_type = str(action_type or "").upper()
    t_id = str(target_id or "").upper()
    if a_type in ("CLOSE_VALVE", "OPEN_VALVE"):
        new_st = "CLOSED" if a_type == "CLOSE_VALVE" else "OPEN"
        impact = simulate_valve_change_impact(valve_id=t_id, new_status=new_st, db_path=db_path)
        return {
            "focus": t_id,
            "highlight_pipe_ids": impact["affected_pipe_ids"],
            "summary_lines": impact["summary_lines"],
            "affected_pipes": impact["affected_pipes"],
        }
    if a_type == "SET_PLANT_OUTPUT":
        return {
            "focus": "reservoir",
            "highlight_pipe_ids": [],
            "summary_lines": [f"DESAL-01 output setpoint change affects RES-01 level trajectory"],
            "affected_pipes": [],
        }
    if a_type == "ISOLATE_AND_FLUSH" and t_id.startswith("SECTOR-"):
        sec_num = t_id.split("-")[-1].lstrip("0") or "1"
        return {
            "focus": f"Sector {sec_num}",
            "highlight_pipe_ids": [],
            "summary_lines": [f"Isolate & flush Sector {sec_num} to restore chlorine residual >= 0.20 mg/L"],
            "affected_pipes": [],
        }
    return {"focus": "all", "highlight_pipe_ids": [], "summary_lines": [], "affected_pipes": []}


def render_inline_approvals(widget_key: str) -> None:
    """Render Pending Approval cards with compact 3D hydraulic preview and Approve / Reject BUTTONS directly inside chat."""
    conn = get_db_connection()
    try:
        pending_rows = conn.execute("SELECT * FROM pending_actions WHERE status = 'PENDING' ORDER BY id DESC;").fetchall()
    finally:
        conn.close()

    if not pending_rows:
        st.markdown('<div class="scada-card-ok"><strong>No PENDING actions awaiting approval.</strong></div>', unsafe_allow_html=True)
        return

    operator_name = st.session_state.get("operator_signature", "Lead_Operator_SCADA")

    for row in pending_rows:
        action_id = int(row["id"])
        action_type = row["action_type"]
        target_id = row["target_id"]
        curr_fmt = _format_val(action_type, float(row["current_value"]))
        prop_fmt = _format_val(action_type, float(row["proposed_value"]))
        rationale = row["rationale"]
        prov = row["provenance"] if "provenance" in row.keys() and row["provenance"] else "AGENT_ANALYSIS"
        prov_ref = row["provenance_ref"] if "provenance_ref" in row.keys() and row["provenance_ref"] else ""
        prov_txt = f"{prov} ({prov_ref})" if prov_ref else prov
        prov_cls = "ag-badge-warn" if prov == "UNTRUSTED_TEXT" else "ag-badge-neutral"

        preview = build_approval_impact_preview(action_type, target_id)
        flow_lines = preview.get("summary_lines") or []
        flow_html = ""
        if flow_lines:
            flow_joined = " &middot; ".join(html.escape(line) for line in flow_lines[:3])
            flow_html = (
                f'<div style="margin-top:0.25rem; font-size:0.83rem; color:#FDE68A;">'
                f'<strong>Computed Flow Change:</strong> <span class="ag-mono">{flow_joined}</span>'
                f'</div>'
            )

        st.markdown(
            f"""
            <div class="scada-card-pending">
                <div>
                    {status_badge("STAGED_PENDING_APPROVAL")}
                    <span class="ag-badge {prov_cls}">{html.escape(prov_txt)}</span>
                    <strong>Action #{action_id}:</strong>
                    <span class="ag-mono">{html.escape(action_type)} {html.escape(target_id)} ({html.escape(curr_fmt)} &rarr; {html.escape(prop_fmt)})</span>
                </div>
                <div style="margin-top:0.3rem; font-size:0.86rem; color:#CBD5E1;">
                    <strong>Reason:</strong> {html.escape(str(rationale))}
                </div>
                {flow_html}
                <div style="margin-top:0.2rem; font-size:0.8rem; color:#94A3B8;">
                    Safety check: PASS at staging · Clicking Approve re-checks safety floor, water quality &amp; freshness and consumes a single-use token.
                </div>
                <div style="margin-top:0.25rem; font-size:0.79rem; color:#38BDF8;">
                    &rarr; {html.escape(preview['focus'])} &amp; affected downstream pipes shown in the twin view
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, _ = st.columns([1.5, 1.5, 0.3])
        with c1:
            if st.button(f"Approve #{action_id}", key=f"chat_approve_{widget_key}_{action_id}", type="primary", width="stretch"):
                res = execute_human_approval(action_id=action_id, approver=operator_name)
                st_code = res.get("status", "UNKNOWN")
                decision_widgets: list[dict] = []
                if st_code == "EXECUTED":
                    if action_type == "CLOSE_VALVE":
                        evt_type = "valve_close"
                        short_ev = f"{target_id} closed — {flow_lines[0]}" if flow_lines else f"{target_id} closed"
                    elif action_type == "OPEN_VALVE":
                        evt_type = "valve_open"
                        short_ev = f"{target_id} opened — {flow_lines[0]}" if flow_lines else f"{target_id} opened"
                    elif action_type == "SET_PLANT_OUTPUT":
                        evt_type = "plant_change"
                        old_mld = float(row["current_value"])
                        new_mld = float(row["proposed_value"])
                        fresh_feeds = read_tariff_and_quality_feeds(horizon_hours=6, current_hour=DEFAULT_SCADA_HOUR, log_audit=False)
                        f_res = fresh_feeds.get("reservoir") or {}
                        r_str = str(f_res.get("rate_str") or "→ 0.0 %/h")
                        eta_str = str(f_res.get("floor_eta_str") or "stable")
                        short_ev = f"DESAL-01 {old_mld:.0f} -> {new_mld:.0f} MLD — reservoir now {r_str}, {eta_str}"
                        decision_widgets.append({"type": "chart", "metric": "reservoir_trend", "asset": "RES-01", "hours": 18})
                    else:
                        evt_type = "isolate_flush"
                        short_ev = f"{target_id} isolated and flushed"

                    set_twin_view(
                        st.session_state,
                        focus="plant" if action_type == "SET_PLANT_OUTPUT" else preview["focus"],
                        event_type=evt_type,
                        note=short_ev,
                        highlight_pipe_ids=preview["highlight_pipe_ids"],
                    )
                    impact_note = f" — {'; '.join(flow_lines[:2])}" if (flow_lines and action_type != "SET_PLANT_OUTPUT") else ""
                    msg_txt = f"**{short_ev}** (Action #{action_id} `{st_code}` by `{res.get('approver')}`{impact_note})"
                    decision_widgets.append({"type": "twin_3d", "focus": preview["focus"]})
                else:
                    msg_txt = f"**Action #{action_id} `{st_code}`:** {res.get('reason') or res.get('message', '')}"
                st.session_state.setdefault("ask_aquagrid_messages", []).append(
                    {
                        "role": "assistant",
                        "kind": "decision",
                        "content": msg_txt,
                        "provider": f"HUMAN_OPERATOR ({operator_name})",
                        "tools_called": [],
                        "code_checks": [
                            {
                                "check": "JIT Safety Re-check & Single-Use Token",
                                "status": "PASS" if st_code == "EXECUTED" else "BLOCKED",
                                "code": st_code,
                                "detail": res.get("details") or res.get("reason") or "",
                            }
                        ],
                        "widgets": decision_widgets,
                        "staged_action": res,
                    }
                )
                st.rerun()
        with c2:
            if st.button(f"Reject #{action_id}", key=f"chat_reject_{widget_key}_{action_id}", width="stretch"):
                res = execute_human_rejection(
                    action_id=action_id,
                    rejection_reason="Operator declined via chat approval card",
                    approver=operator_name,
                )
                set_twin_view(
                    st.session_state,
                    focus=preview["focus"],
                    event_type="valve_reject",
                    note="Rejected — no change",
                )
                st.session_state.setdefault("ask_aquagrid_messages", []).append(
                    {
                        "role": "assistant",
                        "kind": "decision",
                        "content": f"**Action #{action_id} ({action_type} {target_id}) `REJECTED`** by `{res.get('approver')}`. Rejected — no change.",
                        "provider": f"HUMAN_OPERATOR ({operator_name})",
                        "tools_called": [],
                        "code_checks": [],
                        "widgets": [],
                        "staged_action": res,
                    }
                )
                st.rerun()


def render_inline_scenario_card(payload: dict) -> None:
    """Render concise Scenario Result card inside chat with collapsed Show details."""
    sc_id = payload.get("scenario_id", "")
    sc_title = payload.get("scenario_title", "")
    result = payload.get("agent_result") or {}
    scenario_info = payload.get("scenario_info") or {}
    bypass_result = payload.get("bypass_result")
    pre_approved_refusal = payload.get("pre_approved_refusal")

    gw_resp = result.get("gateway_response") or {}
    wq_comp = result.get("sector_water_quality_compliance") or {}
    res_st = result.get("reservoir_status") or {}
    u55_refusal = scenario_info.get("setpoint_refusal")
    u54_token = scenario_info.get("token")
    flagged_alerts = [a for a in result.get("sanitized_alerts", []) if a.get("injection_detected")]

    if result.get("telemetry_loss"):
        top_status, top_cls = "TELEMETRY_LOSS", "scada-card-alert"
    elif flagged_alerts or bypass_result:
        top_status, top_cls = "BLOCKED", "scada-card-alert"
    elif pre_approved_refusal:
        top_status, top_cls = pre_approved_refusal.get("status", "REJECTED_UNAUTHORIZED_APPROVAL_CLAIM"), "scada-card-pending"
    elif u54_token:
        top_status, top_cls = "REJECTED_APPROVAL_MISMATCH", "scada-card-alert"
    elif gw_resp.get("status") == "STAGED_PENDING_APPROVAL":
        top_status, top_cls = "STAGED_PENDING_APPROVAL", "scada-card-pending"
    elif wq_comp.get("alerts"):
        top_status, top_cls = "NON_COMPLIANT", "scada-card-alert"
    else:
        top_status, top_cls = result.get("status", "COMPLETED"), "scada-card-ok"

    st.markdown(
        f"""
        <div class="{top_cls}">
            <div>
                <span class="scada-id-badge">{html.escape(str(sc_id))}</span>
                <strong>{html.escape(str(sc_title))}</strong>
                &nbsp;{status_badge(top_status)}
            </div>
            <div style="margin-top:0.35rem; font-size:0.89rem; color:#F8FAFC;">
                {html.escape(str(result.get('operator_brief', '')))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    old_plan = scenario_info.get("old_plan")
    if res_st.get("has_active_emergency") and old_plan:
        st.markdown(
            f"""
            <div class="scada-card-pending">
                <div><strong>Old Plan:</strong> <span class="ag-mono">DESAL-01 &rarr; {old_plan.get('proposed_mld')} MLD</span> (min {old_plan.get('min_projected_percent')}%) {status_badge(old_plan.get('recheck_status', 'CANCELLED_BY_SAFETY_RECHECK'))}</div>
                <div style="margin-top:0.25rem;"><strong>New Plan:</strong> <span class="ag-mono">DESAL-01 &rarr; {gw_resp.get('proposed_value', 65.0)} MLD</span> (min {gw_resp.get('min_projected_percent')}%) {status_badge(gw_resp.get('status', 'STAGED_PENDING_APPROVAL'))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if u55_refusal:
        st.markdown(
            f'<div class="scada-card-alert"><strong>Setpoint Request During Silence:</strong> <span class="ag-mono">DESAL-01 &rarr; 60.0 MLD</span> {status_badge(u55_refusal.get("status", "REJECTED_STALE_TELEMETRY"))}</div>',
            unsafe_allow_html=True,
        )

    if pre_approved_refusal:
        st.markdown(
            f'<div class="scada-card-alert"><strong>Pre-Approved Call Refused:</strong> <span class="ag-mono">CLOSE_VALVE V-17 (pre_approved=True)</span> {status_badge(pre_approved_refusal.get("status", "REJECTED_UNAUTHORIZED_APPROVAL_CLAIM"))}</div>',
            unsafe_allow_html=True,
        )

    if u54_token:
        replay_v18 = scenario_info.get("replay_v18") or {}
        replay_v17 = scenario_info.get("replay_v17") or {}
        st.markdown(
            f"""
            <div class="scada-card-alert">
                <div><strong>14:05 Token Executed:</strong> V-17 {status_badge('CLOSED')} · V-18 {status_badge('OPEN')}</div>
                <div style="margin-top:0.25rem;"><strong>14:40 Replay V-18:</strong> {status_badge(replay_v18.get('status', 'REJECTED_APPROVAL_MISMATCH'))}</div>
                <div style="margin-top:0.25rem;"><strong>14:40 Replay V-17:</strong> {status_badge(replay_v17.get('status', 'REJECTED_APPROVAL_ALREADY_USED'))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if bypass_result:
        st.markdown(
            f'<div class="scada-card-alert"><strong>Direct Write Bypass:</strong> <span class="ag-mono">{html.escape(str(bypass_result.get("attempted_action")))}</span> {status_badge(bypass_result.get("status", "REFUSED"))} · V-12 {status_badge(str(bypass_result.get("valve_status_after", "OPEN")))}</div>',
            unsafe_allow_html=True,
        )


def render_inline_plan_table(horizon_hours: int, rows: list[dict]) -> None:
    """Render an N-hour production plan table inside chat."""
    if not rows:
        return
    st.markdown(f"**{horizon_hours}-Hour Production Plan**")
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True, height=min(360, 42 + len(df) * 35))


def render_chat_widget(widget: dict, widget_key: str, snapshot: dict) -> None:
    """Dispatch a single chat widget spec. 3D/2D spatial views render in the persistent top twin, with a short note in chat."""
    w_type = widget.get("type")
    if w_type in ("twin_3d", "map"):
        target_lbl = widget.get("focus") or widget.get("leak_id") or widget.get("sector") or "All 6 Sectors"
        st.markdown(
            f'<div style="font-size:0.82rem; color:#38BDF8; margin:0.2rem 0;">'
            f'&rarr; shown in the 3D view above (<span class="ag-mono">{html.escape(str(target_lbl))}</span>)'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif w_type == "chart":
        render_inline_chart(widget.get("metric", "reservoir"), widget.get("asset"), int(widget.get("hours", 24)), snapshot)
    elif w_type == "table":
        render_inline_table(widget.get("what", "leaks"))
    elif w_type == "plan_table":
        render_inline_plan_table(int(widget.get("horizon_hours", len(widget.get("rows") or []))), widget.get("rows") or [])
    elif w_type == "schedule_validation":
        render_inline_schedule_validation(widget.get("plan", ""), widget.get("validation") or {})
    elif w_type == "approvals":
        render_inline_approvals(widget_key)
    elif w_type == "scenario_result":
        render_inline_scenario_card(widget)
