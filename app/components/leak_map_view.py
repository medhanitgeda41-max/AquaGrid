"""Interactive 2D OpenStreetMap Leak Map (Folium + streamlit-folium) & Synced Leak Details Panel."""
from datetime import datetime, timedelta, timezone
import html
import json
import math
import altair as alt
from branca.element import MacroElement, Template
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from config.region_config import LEAK_SEVERITY_COLORS, REGION_CONFIG, SECTOR_COLOR_NAMES, SECTOR_COLORS
from database.connection import get_db_connection
from database.dubai_network import (
    JUNCTION_COORDINATES,
    JUNCTION_SECTOR_MAP,
    VALVE_COORDINATES,
    VALVE_SECTOR_MAP,
    compute_sector_boundaries,
)
from tools.write_gateway import (
    execute_human_approval,
    execute_human_rejection,
    stage_gated_valve_or_setpoint_action,
    update_leak_status_by_operator,
)
from components.badges import format_dubai_time, section_header, status_badge


def _parse_utc(ts_str: str | None) -> datetime:
    if not ts_str:
        return datetime.now(timezone.utc)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(str(ts_str).strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def _leak_color(severity: str, status: str) -> str:
    """Color leaks by canonical status: CONFIRMED red, SUSPECTED orange, UNDER_REPAIR yellow, RESOLVED/FALSE_ALARM grey."""
    from database.leak_registry import normalize_leak_status

    norm = normalize_leak_status(status)
    if norm == "CONFIRMED":
        return "#EF4444"
    if norm == "SUSPECTED":
        return "#F97316"
    if norm == "UNDER_REPAIR":
        return "#EAB308"
    if norm in ("RESOLVED", "FALSE_ALARM"):
        return "#94A3B8"
    return LEAK_SEVERITY_COLORS.get(severity, "#F97316")


def _pressure_color(pressure_psi: float | None) -> str:
    if pressure_psi is None:
        return "#38BDF8"
    if pressure_psi < 38.0:
        return "#EF4444"
    if pressure_psi < 46.0:
        return "#F59E0B"
    return "#10B981"


def _build_segment_history_charts(segment_id: str) -> tuple[alt.Chart | None, alt.Chart | None]:
    """Build 2-hour Pressure & Flow Altair charts for a pipe segment from scada_history."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT recorded_at, pressure_psi, flow_lps
            FROM scada_history
            WHERE pipe_segment_id = ?
            ORDER BY recorded_at DESC
            LIMIT 60;
            """,
            (segment_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return None, None

    ordered = list(reversed(rows))
    df = pd.DataFrame(
        [
            {
                "time_label": format_dubai_time(r["recorded_at"])[11:16],
                "full_time": format_dubai_time(r["recorded_at"]),
                "pressure_psi": float(r["pressure_psi"] or 55.0),
                "flow_lps": float(r["flow_lps"] or 110.0),
            }
            for r in ordered
        ]
    )

    p_chart = (
        alt.Chart(df)
        .mark_line(color="#38BDF8", strokeWidth=2.4)
        .encode(
            x=alt.X("time_label:N", sort=None, title="Time (Asia/Dubai)", axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
            y=alt.Y("pressure_psi:Q", title="Pressure (psi)", scale=alt.Scale(domain=[15, 70])),
            tooltip=[
                alt.Tooltip("full_time:N", title="Time"),
                alt.Tooltip("pressure_psi:Q", title="Pressure (psi)", format=".1f"),
            ],
        )
        .properties(height=150)
    )

    f_chart = (
        alt.Chart(df)
        .mark_line(color="#F59E0B", strokeWidth=2.4)
        .encode(
            x=alt.X("time_label:N", sort=None, title="Time (Asia/Dubai)", axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
            y=alt.Y("flow_lps:Q", title="Flow (L/s)", scale=alt.Scale(domain=[60, 220])),
            tooltip=[
                alt.Tooltip("full_time:N", title="Time"),
                alt.Tooltip("flow_lps:Q", title="Flow (L/s)", format=".1f"),
            ],
        )
        .properties(height=150)
    )
    return p_chart, f_chart


def _centroid_distance_m(c1: list[float], c2: list[float]) -> float:
    """Approximate planar distance in meters between two [lat, lon] centroids."""
    ref_lat = 0.5 * (float(c1[0]) + float(c2[0]))
    dy = (float(c1[0]) - float(c2[0])) * 111_320.0
    dx = (float(c1[1]) - float(c2[1])) * 111_320.0 * math.cos(math.radians(ref_lat))
    return math.hypot(dx, dy)


def _build_map_legend_html() -> str:
    """Build compact collapsible bottom-right Leaflet legend listing Sectors 1-6 colors + asset symbols."""
    sec_items = "".join(
        f'<div style="display:flex; align-items:center; gap:5px; margin:2px 0;">'
        f'<span style="display:inline-block; width:10px; height:10px; border-radius:2px; '
        f'background:{SECTOR_COLORS[s]}33; border:2px solid {SECTOR_COLORS[s]};"></span>'
        f'<span><strong>{s}</strong> ({SECTOR_COLOR_NAMES[s]})</span></div>'
        for s in ("Sector 1", "Sector 2", "Sector 3", "Sector 4", "Sector 5", "Sector 6")
    )
    return f"""
    {{% macro html(this, kwargs) %}}
    <details style="
        position: fixed;
        bottom: 26px;
        right: 10px;
        z-index: 9999;
        background: rgba(17, 24, 39, 0.90);
        color: #F8FAFC;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 4px 8px;
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 10px;
        line-height: 1.3;
        max-width: 170px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.32);
    ">
        <summary style="cursor:pointer; font-weight:700; color:#E2E8F0; user-select:none;">
            Legend
        </summary>
        <div style="margin-top:4px; padding-top:4px; border-top:1px solid #334155;">
            {sec_items}
            <div style="margin-top:4px; padding-top:4px; border-top:1px solid #334155;">
                <div style="display:flex; align-items:center; gap:5px; margin:2px 0;">
                    <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#0EA5E9; border:1px solid #fff;"></span>
                    <span>Sensor · &#9888; Quarantined</span>
                </div>
                <div style="display:flex; align-items:center; gap:5px; margin:2px 0;">
                    <span style="display:inline-block; width:8px; height:8px; background:#16A34A; border:1px solid #fff;"></span>
                    <span>Valve OPEN · <span style="color:#EF4444;">&#9632;</span> CLOSED</span>
                </div>
                <div style="display:flex; align-items:center; gap:5px; margin:2px 0;">
                    <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#EF4444; border:1px solid #FCA5A5;"></span>
                    <span>Active Leak</span>
                </div>
            </div>
        </div>
    </details>
    {{% endmacro %}}
    """


def build_leak_folium_map(
    segments: list[dict],
    leaks: list[dict],
    quarantined_ids: set[str],
    latest_pressures: dict[str, float],
    selected_leak: dict | None,
    show_pipes: bool = True,
    pipe_color_mode: str = "Sector",
    show_leaks: bool = True,
    show_sensors: bool = True,
    show_boundaries: bool = True,
    show_valves: bool = True,
    highlight_sector: str | None = None,
    valve_statuses: dict[str, str] | None = None,
    basemap: str = "OpenStreetMap",
    fit_bounds: bool = True,
    focus_point_latlon: tuple[float, float] | None = None,
    focus_label: str | None = None,
    preserve_center: tuple[float, float] | None = None,
    preserve_zoom: int | None = None,
) -> folium.Map:
    """Construct interactive OpenStreetMap Leaflet map with buffered (~500m) sector boundaries, pipes, valves, sensors, leaks, DESAL-01/RES-01 & collapsible legend."""
    if preserve_center is not None and preserve_zoom is not None:
        center = [float(preserve_center[0]), float(preserve_center[1])]
        zoom = int(preserve_zoom)
    elif focus_point_latlon is not None:
        center = [float(focus_point_latlon[0]), float(focus_point_latlon[1])]
        zoom = 14
    elif selected_leak:
        center = [float(selected_leak["latitude"]), float(selected_leak["longitude"])]
        zoom = max(int(REGION_CONFIG["map_zoom"]) + 2, 14)
    else:
        center = list(REGION_CONFIG["map_center"])
        zoom = int(REGION_CONFIG["map_zoom"])

    # Always use standard OpenStreetMap tiles (CARTO removed completely)
    tiles = REGION_CONFIG["osm_tiles"]
    attr = REGION_CONFIG["osm_attribution"]

    fmap = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=tiles,
        attr=attr,
        control_scale=True,
    )

    boundaries = compute_sector_boundaries(segments, buffer_meters=500.0)
    hl_norm = highlight_sector.strip().lower() if highlight_sector else None

    # 1. Sector Boundaries Layer (500 m buffered convex hull per sector + clean non-overlapping text labels)
    if show_boundaries and boundaries:
        boundary_group = folium.FeatureGroup(name="Sector Boundaries")
        # Order so highlighted sector gets label placement priority if any centroids are close
        ordered_secs = list(boundaries.keys())
        if hl_norm is not None:
            ordered_secs.sort(key=lambda s: 0 if s.lower() == hl_norm else 1)

        placed_label_centroids: list[list[float]] = []
        min_label_sep_m = 1600.0

        for sec in ordered_secs:
            binfo = boundaries[sec]
            color = SECTOR_COLORS.get(sec, "#2563EB")
            color_name = SECTOR_COLOR_NAMES.get(sec, "blue")
            is_hl = hl_norm is not None and sec.lower() == hl_norm
            is_dim = hl_norm is not None and sec.lower() != hl_norm

            weight = 4.5 if is_hl else (1.5 if is_dim else 3.0)
            opacity = 0.95 if is_hl else (0.30 if is_dim else 0.90)
            fill_opacity = 0.22 if is_hl else (0.04 if is_dim else 0.12)

            folium.Polygon(
                locations=binfo["polygon"],
                color=color,
                weight=weight,
                opacity=opacity,
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity,
                tooltip=f"{sec} ({color_name} boundary)",
            ).add_to(boundary_group)

            # Clean transparent text label at centroid only if it won't overlap another placed label
            centroid = binfo["centroid"]
            overlaps = any(_centroid_distance_m(centroid, prev) < min_label_sep_m for prev in placed_label_centroids)
            if not overlaps:
                placed_label_centroids.append(centroid)
                label_opacity = "0.40" if is_dim else "1.0"
                folium.Marker(
                    location=centroid,
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="opacity:{label_opacity}; white-space:nowrap; '
                            f'background:transparent; border:none; color:{color}; '
                            f'font-family:system-ui,-apple-system,sans-serif; '
                            f'font-size:11px; font-weight:700; letter-spacing:0.2px; '
                            f'text-shadow:-1px -1px 0 rgba(255,255,255,0.92),1px -1px 0 rgba(255,255,255,0.92),'
                            f'-1px 1px 0 rgba(255,255,255,0.92),1px 1px 0 rgba(255,255,255,0.92),0 0 3px #fff; '
                            f'transform:translate(-50%,-50%); pointer-events:none;">'
                            f'{html.escape(sec)}</div>'
                        ),
                        icon_size=(0, 0),
                    ),
                ).add_to(boundary_group)
        boundary_group.add_to(fmap)

    # 2. Pipe Network Layer
    if show_pipes:
        pipe_group = folium.FeatureGroup(name="Pipe Network")
        for seg in segments:
            coords = json.loads(seg["geometry_json"]) if seg.get("geometry_json") else seg.get("geometry", [])
            seg_pressure = latest_pressures.get(seg["segment_id"])
            color = (
                _pressure_color(seg_pressure)
                if pipe_color_mode == "Pressure Status"
                else SECTOR_COLORS.get(seg["sector"], "#2563EB")
            )
            is_selected_seg = selected_leak and selected_leak["pipe_segment_id"] == seg["segment_id"]
            is_dim = hl_norm is not None and str(seg.get("sector", "")).lower() != hl_norm
            tooltip_txt = (
                f"{seg['segment_id']} — {seg['name']} | {seg['sector']} | "
                f"{seg['diameter_mm']}mm {seg['material']} ({seg['install_year']}) | "
                f"Valve: {seg['isolation_valve_id']}"
            )
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=6 if is_selected_seg else (2.5 if is_dim else 4),
                opacity=0.95 if is_selected_seg else (0.30 if is_dim else 0.85),
                tooltip=tooltip_txt,
            ).add_to(pipe_group)
        pipe_group.add_to(fmap)

    # 3. Valves Layer (V-01..V-20)
    if show_valves:
        v_map = valve_statuses or {}
        valve_group = folium.FeatureGroup(name="Valves")
        for v_id, (v_lat, v_lon) in VALVE_COORDINATES.items():
            v_sec = VALVE_SECTOR_MAP.get(v_id, "")
            if hl_norm is not None and v_sec.lower() != hl_norm:
                continue
            v_status = str(v_map.get(v_id, "OPEN")).upper()
            v_color = "#16A34A" if v_status == "OPEN" else "#DC2626"
            folium.RegularPolygonMarker(
                location=[v_lat, v_lon],
                number_of_sides=4,
                rotation=45,
                radius=6,
                color="#FFFFFF",
                weight=1.5,
                fill=True,
                fill_color=v_color,
                fill_opacity=0.95,
                tooltip=f"Valve {v_id} ({v_sec}) — {v_status}",
            ).add_to(valve_group)
        valve_group.add_to(fmap)

    # 4. Sensors Layer
    if show_sensors:
        sensor_group = folium.FeatureGroup(name="Sensors")
        for j_id, (lat, lon) in JUNCTION_COORDINATES.items():
            j_sec = JUNCTION_SECTOR_MAP.get(j_id, "")
            if hl_norm is not None and j_sec.lower() != hl_norm:
                continue
            p_id = j_id.replace("J-", "P-") if j_id.startswith("J-") else ("T-INTAKE-1" if j_id == "INTAKE-01" else "T-INTAKE-2")
            is_quarantined = p_id in quarantined_ids
            if is_quarantined:
                folium.Marker(
                    location=[lat, lon],
                    tooltip=f"QUARANTINED SENSOR: {p_id} ({j_id}, {j_sec})",
                    icon=folium.Icon(color="orange", icon="exclamation-triangle", prefix="fa"),
                ).add_to(sensor_group)
            else:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4.5,
                    color="#FFFFFF",
                    weight=1.2,
                    fill=True,
                    fill_color="#0EA5E9",
                    fill_opacity=0.92,
                    tooltip=f"Sensor {p_id} ({j_id}, {j_sec}) — ACTIVE",
                ).add_to(sensor_group)
        sensor_group.add_to(fmap)

    # 5. Leaks Layer (status-driven colors: CONFIRMED red, SUSPECTED orange, UNDER_REPAIR yellow+wrench, history grey)
    if show_leaks:
        from database.leak_registry import normalize_leak_status

        leak_group = folium.FeatureGroup(name="Leaks")
        for lk in leaks:
            if hl_norm is not None and str(lk.get("sector", "")).lower() != hl_norm:
                continue
            lat = float(lk["latitude"])
            lon = float(lk["longitude"])
            loss = float(lk.get("loss_rate_m3h", lk.get("estimated_loss_m3_per_hour", 0.0)))
            norm_st = normalize_leak_status(lk.get("status"))
            color = _leak_color(lk.get("severity", "Medium"), norm_st)
            is_sel = selected_leak and selected_leak["leak_id"] == lk["leak_id"]
            jid = lk.get("junction_id", "")
            upd = lk.get("updated_at", lk.get("detected_at", ""))

            if norm_st not in ("RESOLVED", "FALSE_ALARM"):
                radius_m = max(110.0, min(620.0, max(10.0, loss) * 6.5))
                folium.Circle(
                    location=[lat, lon],
                    radius=radius_m,
                    color=color,
                    weight=3 if is_sel else 1.5,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.28 if is_sel else 0.16,
                    tooltip=f"{lk['leak_id']} ({jid}) | {norm_st} · {loss:.1f} m³/hr",
                ).add_to(leak_group)

            marker_tooltip = (
                f"{lk['leak_id']} | Junction {jid} | {norm_st} | "
                f"{loss:.1f} m³/hr | Updated: {upd}"
            )
            if norm_st == "UNDER_REPAIR":
                folium.Marker(
                    location=[lat, lon],
                    tooltip=marker_tooltip,
                    icon=folium.Icon(color="orange", icon="wrench", prefix="fa"),
                ).add_to(leak_group)
            else:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=11 if is_sel else (6 if norm_st in ("RESOLVED", "FALSE_ALARM") else 8),
                    color="#FFFFFF" if is_sel else color,
                    weight=3 if is_sel else 2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.65 if norm_st in ("RESOLVED", "FALSE_ALARM") else 0.95,
                    tooltip=marker_tooltip,
                ).add_to(leak_group)
        leak_group.add_to(fmap)

    # 5b. Core Facilities (DESAL-01 Coastal Plant & RES-01 Strategic Reservoir) + Main Transmission Spine
    from digital_twin.hydraulics import PLANT_DESAL01_LATLON, RESERVOIR_RES01_LATLON, TRUNK_HUB_LATLON

    fac_group = folium.FeatureGroup(name="Plant & Reservoir")
    spine_coords = [
        list(PLANT_DESAL01_LATLON),
        list(RESERVOIR_RES01_LATLON),
        list(TRUNK_HUB_LATLON["TRUNK-HUB-S5"]),
        list(TRUNK_HUB_LATLON["TRUNK-HUB-S4"]),
        list(TRUNK_HUB_LATLON["TRUNK-HUB-S3"]),
        list(TRUNK_HUB_LATLON["TRUNK-HUB-S2"]),
        list(TRUNK_HUB_LATLON["TRUNK-HUB-S1"]),
    ]
    folium.PolyLine(
        locations=spine_coords,
        color="#0284C7",
        weight=5,
        opacity=0.80,
        dash_array="8 4",
        tooltip="Main Transmission Trunk (DESAL-01 -> RES-01 -> Sectors 1-5)",
    ).add_to(fac_group)
    folium.PolyLine(
        locations=[list(RESERVOIR_RES01_LATLON), list(TRUNK_HUB_LATLON["TRUNK-HUB-S6"])],
        color="#0284C7",
        weight=5,
        opacity=0.80,
        dash_array="8 4",
        tooltip="RES-01 South Outlet Trunk -> Sector 6",
    ).add_to(fac_group)
    folium.Marker(
        location=list(PLANT_DESAL01_LATLON),
        tooltip="DESAL-01 Coastal Desalination Plant",
        icon=folium.Icon(color="darkblue", icon="industry", prefix="fa"),
    ).add_to(fac_group)
    folium.Marker(
        location=list(RESERVOIR_RES01_LATLON),
        tooltip="RES-01 Strategic Reservoir (500 ML)",
        icon=folium.Icon(color="blue", icon="tint", prefix="fa"),
    ).add_to(fac_group)
    fac_group.add_to(fmap)

    # 6. Floating Legend Box & Native Leaflet Layer Control
    legend = MacroElement()
    legend._template = Template(_build_map_legend_html())
    fmap.get_root().add_child(legend)
    folium.LayerControl(collapsed=True).add_to(fmap)

    # 7. Always fit map bounds with padding to what is shown (unless preserving manual camera during Live Sim)
    if fit_bounds and not (preserve_center is not None and preserve_zoom is not None):
        if hl_norm is not None and boundaries:
            for sec, binfo in boundaries.items():
                if sec.lower() == hl_norm:
                    fmap.fit_bounds(binfo["bounds"], padding=(25, 25))
                    break
        elif focus_point_latlon is not None:
            lat, lon = float(focus_point_latlon[0]), float(focus_point_latlon[1])
            fmap.fit_bounds([[lat - 0.010, lon - 0.012], [lat + 0.010, lon + 0.012]], padding=(25, 25))
        elif selected_leak is not None:
            lat = float(selected_leak["latitude"])
            lon = float(selected_leak["longitude"])
            fmap.fit_bounds([[lat - 0.008, lon - 0.010], [lat + 0.008, lon + 0.010]], padding=(25, 25))
        elif boundaries:
            all_lats = [b["bounds"][0][0] for b in boundaries.values()] + [b["bounds"][1][0] for b in boundaries.values()]
            all_lons = [b["bounds"][0][1] for b in boundaries.values()] + [b["bounds"][1][1] for b in boundaries.values()]
            if all_lats and all_lons:
                fmap.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]], padding=(20, 20))

    return fmap


def _render_leak_details_panel(leak: dict, segment: dict) -> None:
    """Render the synced Leak Details panel beside the map with 2h charts, AI suggestions & HITL buttons."""
    leak_id = leak["leak_id"]
    sev = leak["severity"]
    status = leak["status"]
    loss_hr = float(leak["estimated_loss_m3_per_hour"])
    conf = float(leak["confidence"])

    det_dt = _parse_utc(leak["detected_at"])
    hours_elapsed = max(0.05, (datetime.now(timezone.utc) - det_dt).total_seconds() / 3600.0)
    est_total_loss_m3 = round(loss_hr * hours_elapsed, 1)

    try:
        sugg = json.loads(leak.get("ai_suggestions_json") or "{}")
    except json.JSONDecodeError:
        sugg = {}

    section_header(f"Leak Details — {leak_id}", ["CODE", "LLM", "HUMAN"], extra_badge=status_badge(status))

    st.markdown(
        f"""
        <div class="scada-card-alert">
            <div style="font-size:0.95rem; font-weight:700; color:#F8FAFC;">
                {html.escape(leak['location_name'])}
            </div>
            <div style="margin-top:0.35rem; font-size:0.84rem; color:#CBD5E1;">
                <strong>Sector:</strong> {html.escape(leak['sector'])} &nbsp;|&nbsp;
                <strong>Severity:</strong> {status_badge(sev)} &nbsp;|&nbsp;
                <strong>Status:</strong> {status_badge(status)}
            </div>
            <div style="margin-top:0.3rem; font-size:0.83rem; color:#9FB3C8;" class="ag-mono">
                Coords: ({float(leak['latitude']):.5f}, {float(leak['longitude']):.5f}) &nbsp;|&nbsp;
                Detected: {html.escape(format_dubai_time(leak['detected_at']))}
            </div>
            <div style="margin-top:0.3rem; font-size:0.83rem; color:#CBD5E1;">
                <strong>Pipe Segment:</strong> <span class="ag-mono">{html.escape(segment.get('segment_id', leak['pipe_segment_id']))}</span>
                ({segment.get('diameter_mm', '-')}mm {html.escape(str(segment.get('material', '-')))}, installed {segment.get('install_year', '-')})
            </div>
            <div style="margin-top:0.3rem; font-size:0.84rem; color:#FDE68A;">
                <strong>Est. Loss Rate:</strong> {loss_hr:.1f} m³/hr &nbsp;|&nbsp;
                <strong>Est. Loss So Far:</strong> {est_total_loss_m3:.1f} m³ ({hours_elapsed:.1f}h) &nbsp;|&nbsp;
                <strong>Sensors:</strong> <span class="ag-mono">{html.escape(leak['related_sensor_ids'])}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header(f"Last 2 Hours — {leak['pipe_segment_id']} Pressure & Flow", ["CODE"])
    p_chart, f_chart = _build_segment_history_charts(leak["pipe_segment_id"])
    if p_chart and f_chart:
        c_p, c_f = st.columns(2)
        with c_p:
            st.altair_chart(p_chart, width="stretch")
        with c_f:
            st.altair_chart(f_chart, width="stretch")
    else:
        st.caption("No 2-hour history available for this pipe segment yet.")

    section_header("AI Diagnosis & Practical Suggestions", ["LLM", "CODE"])
    st.markdown(
        f"""
        <div class="scada-card">
            <div><strong>Diagnosis (Confidence: {conf:.0%}):</strong><br/>
                {html.escape(str(leak.get('ai_summary', '')))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if sugg:
        valve_id = sugg.get("isolation_valve_id", segment.get("isolation_valve_id", "V-12"))
        verify_badge = (
            '<span class="ag-badge ag-badge-warn">VERIFY FIRST</span>'
            if sugg.get("physical_verification_first")
            else '<span class="ag-badge ag-badge-ok">HIGH CONFIDENCE</span>'
        )
        st.markdown(
            f"""
            <div class="scada-card-pending">
                <div><strong>Isolation Proposal (HITL Required):</strong> {html.escape(str(sugg.get('proposed_action', '')))}</div>
                <div style="margin-top:0.3rem; font-size:0.85rem;">
                    <strong>Impact:</strong> ~{int(sugg.get('affected_customers', 0)):,} customers in {html.escape(str(sugg.get('affected_area', '')))}
                </div>
                <div style="margin-top:0.25rem; font-size:0.85rem;">
                    <strong>Repair Priority:</strong> {html.escape(str(sugg.get('repair_priority', '')))}
                </div>
                <div style="margin-top:0.25rem; font-size:0.85rem;">
                    <strong>Field Verification:</strong> {verify_badge} {html.escape(str(sugg.get('verification_note', '')))}
                </div>
                <div style="margin-top:0.25rem; font-size:0.85rem; color:#FCA5A5;">
                    <strong>Unfixed Loss Cost:</strong> {sugg.get('estimated_loss_m3_per_hour', loss_hr)} m³/hr
                    &asymp; <strong>${sugg.get('estimated_cost_usd_per_hour', 0):.2f}/hr</strong>
                    (<strong>AED {sugg.get('estimated_cost_aed_per_hour', 0):.2f}/hr</strong> @ ${sugg.get('tariff_used_per_kwh', 0.20)}/kWh)
                </div>
                <div style="margin-top:0.35rem; font-size:0.77rem; color:#9FB3C8;">
                    Generated: {html.escape(format_dubai_time(sugg.get('generated_at_utc')))} &nbsp;|&nbsp;
                    Confidence: {float(sugg.get('confidence', conf)):.0%} &nbsp;|&nbsp;
                    Based on: {html.escape(str(sugg.get('based_on', '')))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        valve_id = segment.get("isolation_valve_id", "V-12")

    section_header(f"Operator Controls — {leak_id} (Valve {valve_id})", ["HUMAN", "CODE"])
    operator_sig = st.session_state.get("operator_signature", "Lead_Operator_SCADA")

    conn = get_db_connection()
    try:
        pending_row = conn.execute(
            "SELECT id FROM pending_actions WHERE target_id = ? AND action_type = 'CLOSE_VALVE' AND status = 'PENDING' ORDER BY id DESC LIMIT 1;",
            (valve_id,),
        ).fetchone()
    finally:
        conn.close()
    pending_id = int(pending_row["id"]) if pending_row else None

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        approve_label = f"Approve Close {valve_id}" if pending_id else f"Stage Close {valve_id}"
        if st.button(approve_label, key=f"map_approve_{leak_id}", type="primary", width="stretch"):
            if pending_id:
                res = execute_human_approval(pending_id, approver=operator_sig)
                if res.get("status") == "EXECUTED":
                    update_leak_status_by_operator(leak_id, "Under Repair", operator_sig, f"Valve {valve_id} closed via Action #{pending_id}")
                    st.session_state["last_hitl_feedback"] = {
                        "kind": "success",
                        "message": f"Valve {valve_id} closed (Action #{pending_id}) and {leak_id} marked Under Repair.",
                    }
                else:
                    st.session_state["last_hitl_feedback"] = {
                        "kind": "error",
                        "message": f"Action #{pending_id} -> {res.get('status')}: {res.get('reason')}",
                    }
            else:
                gw = stage_gated_valve_or_setpoint_action(
                    action_type="CLOSE_VALVE",
                    target_id=valve_id,
                    proposed_value=0.0,
                    rationale=f"Operator-requested staging from Leak Map for {leak_id} on {segment.get('street_area', leak['location_name'])}.",
                )
                st.session_state["last_hitl_feedback"] = {
                    "kind": "info",
                    "message": f"{valve_id} closure staged ({gw.get('status')}). Click Approve again or use HITL Queue to execute.",
                }
            st.rerun()

    with b2:
        if st.button("Reject", key=f"map_reject_{leak_id}", width="stretch"):
            if pending_id:
                execute_human_rejection(pending_id, f"Operator declined isolation for {leak_id}", approver=operator_sig)
                st.session_state["last_hitl_feedback"] = {
                    "kind": "warning",
                    "message": f"Pending Action #{pending_id} ({valve_id}) REJECTED for {leak_id}. Valve unchanged.",
                }
            else:
                st.session_state["last_hitl_feedback"] = {
                    "kind": "info",
                    "message": f"No pending valve action for {valve_id} to reject.",
                }
            st.rerun()

    with b3:
        if st.button("False Alarm", key=f"map_false_{leak_id}", width="stretch"):
            update_leak_status_by_operator(leak_id, "False Alarm – Sensor Fault", operator_sig, "Marked via Leak Map")
            st.session_state["last_hitl_feedback"] = {
                "kind": "warning",
                "message": f"{leak_id} marked as 'False Alarm – Sensor Fault' by {operator_sig}.",
            }
            st.rerun()

    with b4:
        if st.button("Mark Resolved", key=f"map_resolve_{leak_id}", width="stretch"):
            update_leak_status_by_operator(leak_id, "Resolved", operator_sig, "Marked resolved via Leak Map")
            st.session_state["last_hitl_feedback"] = {
                "kind": "success",
                "message": f"{leak_id} marked as 'Resolved' by {operator_sig}.",
            }
            st.rerun()


def render_leak_map_view(snapshot: dict) -> None:
    """Render filters, interactive 2D OpenStreetMap Leak Map, synced details panel, and Leak Locations table."""
    section_header(
        f"2D Leak & Pipe Network Map — {REGION_CONFIG['region_name']} (OpenStreetMap)",
        ["CODE", "LLM", "HUMAN"],
    )

    conn = get_db_connection()
    try:
        seg_rows = conn.execute("SELECT * FROM pipe_segments ORDER BY sector, segment_id;").fetchall()
        leak_rows = conn.execute("SELECT * FROM leaks ORDER BY detected_at DESC;").fetchall()
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
    seg_by_id = {s["segment_id"]: s for s in segments}
    latest_pressures = {r["pipe_segment_id"]: float(r["pressure_psi"]) for r in p_rows if r["pressure_psi"] is not None}
    quarantined_ids = {s["sensor_id"] for s in snapshot.get("telemetry_data", {}).get("quarantined_sensors", [])}

    with st.expander("Show details — Map Filters & Layers", expanded=False):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            sev_filter = st.multiselect(
                "Severity",
                options=["Critical", "High", "Medium", "Low"],
                default=["Critical", "High", "Medium", "Low"],
                key="map_filter_severity",
            )
        with f2:
            status_filter = st.multiselect(
                "Status",
                options=["Suspected", "Confirmed", "Under Repair", "Resolved", "False Alarm – Sensor Fault"],
                default=["Suspected", "Confirmed", "Under Repair", "Resolved", "False Alarm – Sensor Fault"],
                key="map_filter_status",
            )
        with f3:
            all_sectors = sorted({s["sector"] for s in segments})
            sector_filter = st.multiselect(
                "Sector",
                options=all_sectors,
                default=all_sectors,
                key="map_filter_sector",
            )
        with f4:
            time_range = st.selectbox(
                "Time Range",
                options=["All", "Last 1h", "Last 6h", "Last 24h"],
                index=0,
                key="map_filter_time",
            )

        l1, l2, l3, l4 = st.columns([1.0, 1.2, 1.0, 1.3])
        with l1:
            show_pipes = st.checkbox("Pipe Network", value=True, key="map_layer_pipes")
        with l2:
            show_leaks = st.checkbox("Leaks & Loss Circles", value=True, key="map_layer_leaks")
        with l3:
            show_sensors = st.checkbox("Sensors", value=True, key="map_layer_sensors")
        with l4:
            pipe_color_mode = st.selectbox(
                "Color Pipes By",
                options=["Sector", "Pressure Status"],
                index=0,
                key="map_pipe_color_mode",
                label_visibility="collapsed",
            )

    now_utc = datetime.now(timezone.utc)
    cutoff_hours = {"Last 1h": 1, "Last 6h": 6, "Last 24h": 24}.get(time_range)
    filtered_segments = [s for s in segments if s["sector"] in sector_filter]
    filtered_leaks = []
    for lk in all_leaks:
        if lk["severity"] not in sev_filter:
            continue
        if lk["status"] not in status_filter:
            continue
        if lk["sector"] not in sector_filter:
            continue
        if cutoff_hours is not None:
            if (now_utc - _parse_utc(lk["detected_at"])) > timedelta(hours=cutoff_hours):
                continue
        filtered_leaks.append(lk)

    filtered_ids = [lk["leak_id"] for lk in filtered_leaks]
    selected_id = st.session_state.get("selected_leak_id")
    if selected_id not in filtered_ids:
        selected_id = filtered_ids[0] if filtered_ids else None
        st.session_state["selected_leak_id"] = selected_id

    selected_leak = next((lk for lk in filtered_leaks if lk["leak_id"] == selected_id), None)

    col_map, col_detail = st.columns([1.35, 1.05])

    with col_map:
        fmap = build_leak_folium_map(
            segments=filtered_segments,
            leaks=filtered_leaks,
            quarantined_ids=quarantined_ids,
            latest_pressures=latest_pressures,
            selected_leak=selected_leak,
            show_pipes=show_pipes,
            pipe_color_mode=pipe_color_mode,
            show_leaks=show_leaks,
            show_sensors=show_sensors,
        )
        map_event = st_folium(
            fmap,
            height=520,
            width=None,
            returned_objects=["last_object_clicked_tooltip"],
            key="aquagrid_dubai_leak_map",
        )
        clicked_tip = (map_event or {}).get("last_object_clicked_tooltip") or ""
        if clicked_tip:
            for lid in filtered_ids:
                if clicked_tip.startswith(lid) and lid != selected_id:
                    st.session_state["selected_leak_id"] = lid
                    st.rerun()

        st.caption(
            "Map Legend — Leaks: Yellow = Low | Orange = Medium | Red = High | Dark Red = Critical | Grey = Resolved. "
            "Map data © OpenStreetMap contributors."
        )

    with col_detail:
        if selected_leak:
            _render_leak_details_panel(selected_leak, seg_by_id.get(selected_leak["pipe_segment_id"], {}))
        else:
            st.info("No leaks match the active filters.")

    with st.expander(f"Show details — Leak Locations Table ({len(filtered_leaks)})", expanded=False):
        if not filtered_leaks:
            st.info("No leaks match the active filters.")
            return

        table_df = pd.DataFrame(
            [
                {
                    "leak_id": lk["leak_id"],
                    "pipe_segment_id": lk["pipe_segment_id"],
                    "sector": lk["sector"],
                    "location_name": lk["location_name"],
                    "latitude": round(float(lk["latitude"]), 5),
                    "longitude": round(float(lk["longitude"]), 5),
                    "detected_at (Asia/Dubai)": format_dubai_time(lk["detected_at"]),
                    "severity": lk["severity"],
                    "estimated_loss_m3_per_hour": round(float(lk["estimated_loss_m3_per_hour"]), 1),
                    "confidence": f"{float(lk['confidence']):.0%}",
                    "status": lk["status"],
                    "related_sensor_ids": lk["related_sensor_ids"],
                    "ai_summary": lk["ai_summary"],
                }
                for lk in filtered_leaks
            ]
        )

        selection = st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="leak_locations_table",
        )
        selected_rows = (selection or {}).get("selection", {}).get("rows", [])
        if selected_rows:
            chosen_lid = table_df.iloc[selected_rows[0]]["leak_id"]
            if chosen_lid != st.session_state.get("selected_leak_id"):
                st.session_state["selected_leak_id"] = chosen_lid
                st.rerun()
