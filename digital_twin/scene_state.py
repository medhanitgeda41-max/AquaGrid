"""Pure-Python 3D Digital Twin scene-state builder & view-command helpers.

Reads live SCADA data from SQLite (read-only) and emits one deterministic JSON-serializable
scene state with local x/z metre coordinates around the Dubai network centre.
Never mutates valves, plant setpoints, or any safety guard state.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import re
from typing import Optional

from config.constants import DEFAULT_SCADA_HOUR, RESERVOIR_FLOOR_PERCENT
from config.region_config import SECTOR_COLOR_NAMES, SECTOR_COLORS, now_local
from database.connection import get_db_connection
from database.dubai_network import (
    DUBAI_PIPE_SEGMENTS,
    JUNCTION_COORDINATES,
    JUNCTION_SECTOR_MAP,
    VALVE_COORDINATES,
    VALVE_SECTOR_MAP,
    compute_sector_boundaries,
    point_in_polygon,
)
from database.leak_registry import get_active_leaks, get_all_leaks, normalize_leak_status
from digital_twin.hydraulics import (
    build_connected_network_geometry,
    compute_network_flows,
    get_effective_valve_for_segment,
)
from tools.read_feeds import read_tariff_and_quality_feeds
from tools.read_telemetry import read_network_telemetry


def _compute_network_center() -> tuple[float, float]:
    """Compute fixed geographic centre (lat, lon) across all 28 Dubai pipe segments."""
    lats: list[float] = []
    lons: list[float] = []
    for seg in DUBAI_PIPE_SEGMENTS:
        for pt in seg["geometry"]:
            lats.append(float(pt[0]))
            lons.append(float(pt[1]))
    return round(sum(lats) / len(lats), 6), round(sum(lons) / len(lons), 6)


NETWORK_CENTER_LATLON: tuple[float, float] = _compute_network_center()

# Coastal plant DESAL-01 & strategic reservoir RES-01 coordinates (west coast near Jebel Ali / Marina intake)
DESAL_PLANT_LATLON: tuple[float, float] = (25.0760, 55.1060)
RESERVOIR_RES01_LATLON: tuple[float, float] = (25.0980, 55.1260)

# Stylized Dubai western coastline points (west of Sectors 1-6, separating Persian Gulf sea from land)
DUBAI_COASTLINE_LATLON: list[tuple[float, float]] = [
    (25.3050, 55.3180),
    (25.2780, 55.2880),
    (25.2520, 55.2640),
    (25.2260, 55.2380),
    (25.1980, 55.2180),
    (25.1680, 55.1920),
    (25.1380, 55.1660),
    (25.1080, 55.1360),
    (25.0800, 55.1120),
    (25.0480, 55.0920),
    (25.0200, 55.0750),
]

SECTOR_ARCHITECTURAL_STYLES: dict[str, dict] = {
    "Sector 1": {"style": "historic_midrise", "height_scale": 1.05, "base_hex": "#4B5563"},
    "Sector 2": {"style": "courtyard_commercial", "height_scale": 1.15, "base_hex": "#475569"},
    "Sector 3": {"style": "downtown_towers", "height_scale": 1.85, "base_hex": "#334155"},
    "Sector 4": {"style": "coastal_villas", "height_scale": 0.85, "base_hex": "#52525B"},
    "Sector 5": {"style": "urban_blocks", "height_scale": 1.10, "base_hex": "#3F4E5A"},
    "Sector 6": {"style": "marina_skyscrapers", "height_scale": 1.75, "base_hex": "#2E3B4E"},
}


def latlon_to_local_xz(
    lat: float,
    lon: float,
    center_latlon: tuple[float, float] = NETWORK_CENTER_LATLON,
) -> tuple[float, float]:
    """
    Convert geographic (lat, lon) to local planar (x, z) in metres around center_latlon.
    +X = East, -X = West (sea/coastline), -Z = North, +Z = South.
    """
    ref_lat, ref_lon = float(center_latlon[0]), float(center_latlon[1])
    cos_lat = math.cos(math.radians(ref_lat))
    x_m = (float(lon) - ref_lon) * 111_320.0 * cos_lat
    z_m = -(float(lat) - ref_lat) * 111_320.0
    return round(x_m, 2), round(z_m, 2)


def generate_sector_buildings(
    boundaries: dict[str, dict],
    low_quality: bool = False,
) -> list[dict]:
    """
    Deterministically generate seeded procedural city building blocks inside each sector polygon.
    Same coordinates, dimensions, styles, and heights on every call.
    """
    buildings: list[dict] = []
    target_per_sector = 16 if low_quality else 46

    for sec_idx in range(1, 7):
        sec_name = f"Sector {sec_idx}"
        binfo = boundaries.get(sec_name)
        if not binfo:
            continue
        poly = binfo["polygon"]
        (min_lat, min_lon), (max_lat, max_lon) = binfo["bounds"]
        rng = random.Random(4200 + sec_idx * 101)
        style_cfg = SECTOR_ARCHITECTURAL_STYLES.get(sec_name, {"style": "urban", "height_scale": 1.0, "base_hex": "#475569"})
        height_scale = float(style_cfg["height_scale"])

        placed = 0
        attempts = 0
        max_attempts = target_per_sector * 14
        while placed < target_per_sector and attempts < max_attempts:
            attempts += 1
            lat = rng.uniform(min_lat, max_lat)
            lon = rng.uniform(min_lon, max_lon)
            if not point_in_polygon(lat, lon, poly):
                continue
            x_m, z_m = latlon_to_local_xz(lat, lon)
            is_tower = (sec_idx in (3, 6)) and (rng.random() < 0.38)
            w_m = round(rng.uniform(110.0, 220.0) if is_tower else rng.uniform(150.0, 310.0), 1)
            d_m = round(rng.uniform(110.0, 220.0) if is_tower else rng.uniform(150.0, 310.0), 1)
            h_m = round((rng.uniform(220.0, 460.0) if is_tower else rng.uniform(85.0, 260.0)) * height_scale, 1)
            buildings.append(
                {
                    "x": x_m,
                    "z": z_m,
                    "w": w_m,
                    "d": d_m,
                    "h": h_m,
                    "sector": sec_name,
                    "style": style_cfg["style"],
                    "base_hex": style_cfg["base_hex"],
                    "color": SECTOR_COLORS.get(sec_name, "#64748B"),
                }
            )
            placed += 1

    return buildings


def generate_city_roads_and_trees(
    boundaries: dict[str, dict],
    low_quality: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministically generate seeded roads between city blocks, green park patches, and trees."""
    roads: list[dict] = []
    parks: list[dict] = []
    trees: list[dict] = []
    trees_per_sector = 6 if low_quality else 18

    for sec_idx in range(1, 7):
        sec_name = f"Sector {sec_idx}"
        binfo = boundaries.get(sec_name)
        if not binfo:
            continue
        poly = binfo["polygon"]
        (min_lat, min_lon), (max_lat, max_lon) = binfo["bounds"]
        cx_m, cz_m = latlon_to_local_xz(binfo["centroid"][0], binfo["centroid"][1])
        rng = random.Random(7700 + sec_idx * 53)

        # Two orthogonal boulevard roads crossing each sector block grid
        lat_mid = 0.5 * (min_lat + max_lat)
        lon_mid = 0.5 * (min_lon + max_lon)
        r1a = list(latlon_to_local_xz(min_lat + 0.12 * (max_lat - min_lat), lon_mid))
        r1b = list(latlon_to_local_xz(max_lat - 0.12 * (max_lat - min_lat), lon_mid))
        r2a = list(latlon_to_local_xz(lat_mid, min_lon + 0.12 * (max_lon - min_lon)))
        r2b = list(latlon_to_local_xz(lat_mid, max_lon - 0.12 * (max_lon - min_lon)))
        roads.append({"sector": sec_name, "start_xz": r1a, "end_xz": r1b, "width_m": 65.0})
        roads.append({"sector": sec_name, "start_xz": r2a, "end_xz": r2b, "width_m": 55.0})

        # Seeded community park patch near centroid offset
        px = round(cx_m + rng.uniform(-550.0, 550.0), 1)
        pz = round(cz_m + rng.uniform(-550.0, 550.0), 1)
        parks.append({"sector": sec_name, "x": px, "z": pz, "radius_m": round(rng.uniform(210.0, 340.0), 1)})

        # Seeded park & boulevard trees inside the sector polygon
        placed = 0
        attempts = 0
        while placed < trees_per_sector and attempts < trees_per_sector * 10:
            attempts += 1
            lat = rng.uniform(min_lat, max_lat)
            lon = rng.uniform(min_lon, max_lon)
            if not point_in_polygon(lat, lon, poly):
                continue
            tx, tz = latlon_to_local_xz(lat, lon)
            trees.append({"x": tx, "z": tz, "scale": round(rng.uniform(0.8, 1.35), 2), "sector": sec_name})
            placed += 1

    return roads, parks, trees


def _normalize_focus(focus: Optional[str]) -> tuple[str, Optional[str]]:
    """Classify focus string into (focus_type, normalized_id)."""
    raw = (focus or "all").strip()
    if not raw or raw.lower() in ("all", "all sectors", "network", "overview"):
        return "all", None
    low = raw.lower()
    if low in ("reservoir_inside", "inside_reservoir", "inside reservoir", "look inside reservoir", "res-01-inside"):
        return "reservoir_inside", "RES-01"
    if low in ("reservoir", "res-01", "res01", "tank"):
        return "reservoir", "RES-01"
    if low in ("intake", "intake-01", "intake-02", "intake 1", "intake 2", "intakes"):
        return "intake", "INTAKE-02"
    if low in ("plant", "desal", "desal-01", "desal01", "desalination"):
        return "plant", "DESAL-01"
    m_v = re.match(r"^v-?0?(\d{1,2})$", low)
    if m_v:
        return "valve", f"V-{int(m_v.group(1)):02d}"
    m_l = re.match(r"^leak-?(\d+)$", low)
    if m_l:
        return "leak", f"LEAK-{int(m_l.group(1)):03d}"
    m_j = re.match(r"^j-?0?(\d{1,2})$", low)
    if m_j:
        return "junction", f"J-{int(m_j.group(1)):02d}"
    m_s = re.search(r"sector[\s-]*0?([1-6])", low)
    if m_s:
        return "sector", f"Sector {m_s.group(1)}"
    return "all", None


def _compute_camera_target(
    focus_type: str,
    focus_id: Optional[str],
    sectors: list[dict],
    valves: list[dict],
    leaks: list[dict],
    junctions: list[dict],
    plant: dict,
    reservoir: dict,
) -> dict:
    """Compute camera look-at target (x, z in metres) and camera position (x, y, z in metres)."""
    if focus_type == "sector" and focus_id:
        for s in sectors:
            if s["name"].lower() == focus_id.lower():
                cx, cz = s["centroid_xz"]
                return {
                    "target_xz": [cx, cz],
                    "target_y": 150.0,
                    "position_xyz": [round(cx - 4200.0, 1), 3400.0, round(cz + 4600.0, 1)],
                    "label": s["name"],
                    "look_inside": False,
                }
    elif focus_type == "valve" and focus_id:
        for v in valves:
            if v["valve_id"].upper() == focus_id.upper():
                vx, vz = v["pos_xz"]
                return {
                    "target_xz": [vx, vz],
                    "target_y": 160.0,
                    "position_xyz": [round(vx - 1900.0, 1), 1650.0, round(vz + 2200.0, 1)],
                    "label": f"Valve {v['valve_id']}",
                    "look_inside": False,
                }
    elif focus_type == "leak" and focus_id:
        for lk in leaks:
            if lk["leak_id"].upper() == focus_id.upper():
                lx, lz = lk["pos_xz"]
                return {
                    "target_xz": [lx, lz],
                    "target_y": 180.0,
                    "position_xyz": [round(lx - 1850.0, 1), 1550.0, round(lz + 2150.0, 1)],
                    "label": f"{lk['leak_id']} ({lk['sector']})",
                    "look_inside": False,
                }
    elif focus_type == "junction" and focus_id:
        for j in junctions:
            if j["junction_id"].upper() == focus_id.upper():
                jx, jz = j["pos_xz"]
                return {
                    "target_xz": [jx, jz],
                    "target_y": 160.0,
                    "position_xyz": [round(jx - 2100.0, 1), 1750.0, round(jz + 2300.0, 1)],
                    "label": f"Junction {j['junction_id']} ({j['sector']})",
                    "look_inside": False,
                }
    elif focus_type in ("reservoir", "reservoir_inside"):
        rx, rz = reservoir["pos_xz"]
        # Close cutaway view looking right inside RES-01 (level rings, rippling water, inlet/outlet pipes)
        return {
            "target_xz": [rx, rz],
            "target_y": 360.0,
            "position_xyz": [round(rx - 950.0, 1), 720.0, round(rz + 1250.0, 1)],
            "label": "Inside Reservoir RES-01" if focus_type == "reservoir_inside" else "Reservoir RES-01",
            "look_inside": True,
        }
    elif focus_type == "intake":
        px, pz = plant["pos_xz"]
        return {
            "target_xz": [round(px - 650.0, 1), pz],
            "target_y": 180.0,
            "position_xyz": [round(px - 2200.0, 1), 1200.0, round(pz + 1800.0, 1)],
            "label": "Seawater Intakes (DESAL-01)",
            "look_inside": False,
        }
    elif focus_type == "plant":
        px, pz = plant["pos_xz"]
        return {
            "target_xz": [px, pz],
            "target_y": 240.0,
            "position_xyz": [round(px - 1950.0, 1), 1450.0, round(pz + 2250.0, 1)],
            "label": "Plant DESAL-01",
            "look_inside": False,
        }

    # Default overview: fit all 6 sectors + coastal plant/reservoir nicely
    return {
        "target_xz": [0.0, 1200.0],
        "target_y": 0.0,
        "position_xyz": [-14500.0, 12800.0, 16200.0],
        "label": "All 6 Sectors (Dubai Network)",
        "look_inside": False,
    }


def build_twin_view_command(
    focus: str = "all",
    event_type: str = "focus",
    note: str = "",
    highlight_pipe_ids: Optional[list[str]] = None,
    seq: int = 1,
) -> dict:
    """Build a structured view command stored in st.session_state['twin_view'] and sent to the persistent 3D twin."""
    f_type, f_id = _normalize_focus(focus)
    return {
        "focus": focus or "all",
        "focus_type": f_type,
        "focus_id": f_id,
        "event_type": event_type,
        "note": note,
        "highlight_pipe_ids": list(highlight_pipe_ids or []),
        "seq": int(seq),
    }


def build_twin_caption(scene_state: dict, twin_view: Optional[dict] = None) -> str:
    """Build a concise one-line operator caption describing what the top 3D digital twin currently shows."""
    tf = scene_state.get("telemetry_status") or {}
    if tf.get("is_telemetry_loss"):
        return "Showing: TELEMETRY LOSS — holding last safe state"

    hl = scene_state.get("highlight") or {}
    f_type = hl.get("type", "all")
    f_id = hl.get("id")
    tv_note = (twin_view or {}).get("note") or ""
    suffix = f" · {tv_note}" if tv_note else ""
    upd_str = str(scene_state.get("updated_at_local") or "")
    upd_tag = f" · Last update {upd_str}" if upd_str else ""

    if f_type == "sector" and f_id:
        sec = next((s for s in scene_state.get("sectors", []) if s["name"].lower() == f_id.lower()), None)
        if sec:
            cl_txt = "chlorine NON_COMPLIANT" if sec["compliance_status"] == "NON_COMPLIANT" else "chlorine OK"
            fire_txt = ", FIRE EMERGENCY" if sec.get("has_fire_emergency") else ""
            lk_cnt = int(sec.get("active_leaks", 0))
            lk_txt = f"{lk_cnt} active leak" + ("" if lk_cnt == 1 else "s")
            return f"Showing: {sec['name']} — {lk_txt}, {cl_txt}{fire_txt}{suffix}{upd_tag}"

    if f_type in ("reservoir", "reservoir_inside"):
        res = scene_state.get("reservoir") or {}
        rate_str = str(res.get("rate_str") or "")
        eta_str = str(res.get("floor_eta_str") or "")
        trend_part = f" ({rate_str}, {eta_str})" if rate_str and eta_str else ""
        em_txt = f", dynamic floor {res.get('safety_floor_percent', 35.0):.1f}%" if res.get("has_active_emergency") else f", floor {res.get('safety_floor_percent', 35.0):.1f}%"
        prefix = "Inside Reservoir RES-01" if f_type == "reservoir_inside" else "Reservoir RES-01"
        return f"Showing: {prefix} — {res.get('level_percent', 48.0):.1f}%{trend_part}{em_txt}{suffix}{upd_tag}"

    if f_type in ("plant", "intake"):
        pl = scene_state.get("plant") or {}
        cap_txt = f" (algae cap {pl.get('algae_cap_mld', 120):.0f} MLD)" if float(pl.get("algae_cap_mld", 120)) < float(pl.get("max_capacity_mld", 120)) else ""
        return f"Showing: DESAL-01 — {pl.get('output_mld', 100):.0f} MLD{cap_txt}{suffix}{upd_tag}"

    if f_type == "valve" and f_id:
        v = next((x for x in scene_state.get("valves", []) if x["valve_id"].upper() == f_id.upper()), None)
        if v:
            return f"Showing: Valve {v['valve_id']} ({v['sector']}) — {v['status']}{suffix}{upd_tag}"

    if f_type == "leak" and f_id:
        lk = next((x for x in scene_state.get("leaks", []) if x["leak_id"].upper() == f_id.upper()), None)
        if lk:
            st_txt = "ISOLATED" if lk.get("isolated") else f"{lk.get('estimated_loss_m3_per_hour', 0):.1f} m³/hr"
            return f"Showing: {lk['leak_id']} at {lk.get('junction_id', '')} ({lk['sector']}) — {lk['status']}, {st_txt}{suffix}{upd_tag}"

    if f_type == "junction" and f_id:
        j = next((x for x in scene_state.get("junctions", []) if x["junction_id"].upper() == f_id.upper()), None)
        if j:
            return f"Showing: Junction {j['junction_id']} ({j['sector']}) — {j['pressure_psi']:.1f} psi, {j['flow_lps']:.1f} L/s{suffix}{upd_tag}"

    res = scene_state.get("reservoir") or {}
    pl = scene_state.get("plant") or {}
    act_leaks = int(scene_state.get("active_leak_count", sum(1 for l in scene_state.get("leaks", []) if l.get("active"))))
    non_comp = [s["name"] for s in scene_state.get("sectors", []) if s.get("compliance_status") == "NON_COMPLIANT"]
    wq_txt = f"NON_COMPLIANT: {', '.join(non_comp)}" if non_comp else "chlorine OK"
    rate_str = str(res.get("rate_str") or "")
    eta_str = str(res.get("floor_eta_str") or "")
    res_trend = f" {rate_str} ({eta_str})" if rate_str and eta_str else ""
    return (
        f"Showing: All 6 Sectors — RES-01 {res.get('level_percent', 48.0):.1f}%{res_trend}, "
        f"DESAL-01 {pl.get('output_mld', 100.0):.0f} MLD, {act_leaks} active leak(s), {wq_txt}{suffix}{upd_tag}"
    )


def build_3d_scene_state(
    focus: str = "all",
    current_hour: int = DEFAULT_SCADA_HOUR,
    db_path: Optional[str] = None,
    low_quality: bool = False,
    highlight_pipe_ids: Optional[list[str]] = None,
    flow_override_by_segment: Optional[dict[str, float]] = None,
    valve_override: Optional[dict[str, str]] = None,
    show_leak_history: bool = False,
) -> dict:
    """
    Build one complete, JSON-serializable 3D scene state from live SQLite tables.
    All coordinates are local (x, z) metres around NETWORK_CENTER_LATLON.
    """
    conn = get_db_connection(db_path)
    try:
        seg_rows = conn.execute("SELECT * FROM pipe_segments ORDER BY sector, segment_id;").fetchall()
        valve_rows = conn.execute("SELECT valve_id, sector, status, last_updated FROM valves ORDER BY valve_id ASC;").fetchall()
        leak_rows = conn.execute("SELECT * FROM leaks ORDER BY detected_at DESC;").fetchall()
        plant_row = conn.execute("SELECT * FROM plant_setpoints WHERE plant_id = 'DESAL-01';").fetchone()
        wq_rows = conn.execute(
            """
            SELECT sector, chlorine_mg_l
            FROM sector_water_quality
            WHERE id IN (SELECT MAX(id) FROM sector_water_quality GROUP BY sector);
            """
        ).fetchall()
        hist_rows = conn.execute(
            """
            SELECT pipe_segment_id, pressure_psi, flow_lps
            FROM scada_history
            WHERE id IN (SELECT MAX(id) FROM scada_history GROUP BY pipe_segment_id);
            """
        ).fetchall()
        latest_tel_rows = conn.execute(
            """
            SELECT sensor_id, junction_id, sector, value, unit, status
            FROM telemetry
            WHERE id IN (SELECT MAX(id) FROM telemetry GROUP BY sensor_id)
            ORDER BY sensor_id ASC;
            """
        ).fetchall()
    finally:
        conn.close()

    segments = [dict(r) for r in seg_rows] if seg_rows else list(DUBAI_PIPE_SEGMENTS)
    valve_status_map = {r["valve_id"]: str(r["status"]).upper() for r in valve_rows}
    if valve_override:
        for vid, vst in valve_override.items():
            valve_status_map[vid] = str(vst).upper()

    wq_by_sec = {r["sector"]: float(r["chlorine_mg_l"]) for r in wq_rows}
    hist_by_seg = {
        r["pipe_segment_id"]: {
            "pressure_psi": float(r["pressure_psi"]) if r["pressure_psi"] is not None else 55.0,
            "flow_lps": float(r["flow_lps"]) if r["flow_lps"] is not None else None,
        }
        for r in hist_rows
    }

    tel_data = read_network_telemetry(current_hour=current_hour, db_path=db_path, log_audit=False)
    feeds_data = read_tariff_and_quality_feeds(horizon_hours=6, current_hour=current_hour, db_path=db_path, log_audit=False)

    res_info = feeds_data.get("reservoir") or {}
    wq_info = feeds_data.get("water_quality") or {}
    tf_info = tel_data.get("telemetry_freshness") or {}
    plant_output_mld = float(plant_row["production_mld"]) if plant_row else float(feeds_data.get("current_plant_output_mld", 100.0))
    plant_max_mld = float(plant_row["max_capacity_mld"]) if plant_row else 120.0
    plant_status = str(plant_row["status"]) if plant_row else "ONLINE"

    quarantined_map = {
        s["sensor_id"]: list(s.get("reasons") or [])
        for s in (tel_data.get("quarantined_sensors") or [])
    }

    # Pure-Python directed-graph hydraulic propagation
    computed_flows: dict[str, float] = dict(flow_override_by_segment or {})
    isolated_leaks: set[str] = set()
    rerouted_segments: set[str] = set()
    zero_flow_segments: set[str] = set()
    if not computed_flows:
        hyd = compute_network_flows(
            segments=segments,
            valve_statuses=valve_status_map,
            plant_output_mld=plant_output_mld,
            leaks=[dict(r) for r in leak_rows],
        )
        computed_flows = hyd.get("pipe_flows_lps") or {}
        isolated_leaks = set(hyd.get("isolated_leak_ids") or [])
        rerouted_segments = set(hyd.get("rerouted_segment_ids") or [])
        zero_flow_segments = set(hyd.get("zero_flow_segment_ids") or [])
    else:
        for sid, fl in computed_flows.items():
            if float(fl) <= 0.05:
                zero_flow_segments.add(sid)

    focus_type, focus_id = _normalize_focus(focus)
    hl_pipes_set = set(highlight_pipe_ids or [])

    # 1. Sector boundaries (500 m buffer)
    boundaries = compute_sector_boundaries(segments, buffer_meters=500.0)

    # 2. Pipes in local (x, z) metres
    pipes_out: list[dict] = []
    sector_flow_totals: dict[str, float] = {f"Sector {i}": 0.0 for i in range(1, 7)}
    pipe_by_id: dict[str, dict] = {}
    for seg in segments:
        seg_id = seg["segment_id"]
        sec = seg["sector"]
        geom = json.loads(seg["geometry_json"]) if seg.get("geometry_json") else seg.get("geometry", [])
        pts_xz = [list(latlon_to_local_xz(p[0], p[1])) for p in geom]
        if len(pts_xz) < 2:
            continue
        diam = int(seg.get("diameter_mm", 500))
        v_id = get_effective_valve_for_segment(seg)
        hist_item = hist_by_seg.get(seg_id, {})
        flow_lps = round(float(computed_flows.get(seg_id, 0.0)), 1)

        sector_flow_totals[sec] = round(sector_flow_totals.get(sec, 0.0) + flow_lps, 1)
        is_hl = (
            seg_id in hl_pipes_set
            or (focus_type == "sector" and focus_id and sec.lower() == focus_id.lower())
            or (focus_type == "valve" and focus_id and v_id.upper() == focus_id.upper())
        )
        p_obj = {
            "segment_id": seg_id,
            "name": seg.get("name", seg_id),
            "sector": sec,
            "color": SECTOR_COLORS.get(sec, "#2563EB"),
            "diameter_mm": diam,
            "is_trunk": diam >= 600,
            "points_xz": pts_xz,
            "start_xz": pts_xz[0],
            "end_xz": pts_xz[-1],
            "isolation_valve_id": v_id,
            "flow_lps": flow_lps,
            "rerouted": seg_id in rerouted_segments,
            "pressure_psi": round(float(hist_item.get("pressure_psi", 55.0)), 1),
            "highlighted": bool(is_hl),
        }
        pipes_out.append(p_obj)
        pipe_by_id[seg_id] = p_obj

    # 3. Valves (V-01..V-20)
    valves_out: list[dict] = []
    for v_id, (v_lat, v_lon) in VALVE_COORDINATES.items():
        v_sec = VALVE_SECTOR_MAP.get(v_id, "Sector 1")
        v_stat = valve_status_map.get(v_id, "OPEN")
        vx, vz = latlon_to_local_xz(v_lat, v_lon)
        dir_xz = [1.0, 0.0]
        linked_seg = None
        for p in pipes_out:
            if p["isolation_valve_id"] == v_id:
                linked_seg = p["segment_id"]
                dx = p["end_xz"][0] - p["start_xz"][0]
                dz = p["end_xz"][1] - p["start_xz"][1]
                L = math.hypot(dx, dz) or 1.0
                dir_xz = [round(dx / L, 4), round(dz / L, 4)]
                break
        valves_out.append(
            {
                "valve_id": v_id,
                "sector": v_sec,
                "status": v_stat,
                "pos_xz": [vx, vz],
                "pipe_segment_id": linked_seg,
                "pipe_dir_xz": dir_xz,
                "highlighted": bool(
                    (focus_type == "valve" and focus_id and v_id.upper() == focus_id.upper())
                    or (focus_type == "sector" and focus_id and v_sec.lower() == focus_id.lower())
                ),
            }
        )

    # 4. Junctions & Sensors
    tel_by_junc_p: dict[str, dict] = {}
    tel_by_junc_f: dict[str, dict] = {}
    for row in latest_tel_rows:
        sid = str(row["sensor_id"])
        jid = str(row["junction_id"])
        if sid.startswith("P-"):
            tel_by_junc_p[jid] = dict(row)
        elif sid.startswith("F-"):
            tel_by_junc_f[jid] = dict(row)

    junctions_out: list[dict] = []
    sensors_out: list[dict] = []
    for j_id, (j_lat, j_lon) in JUNCTION_COORDINATES.items():
        j_sec = JUNCTION_SECTOR_MAP.get(j_id, "Sector 1")
        jx, jz = latlon_to_local_xz(j_lat, j_lon)
        p_row = tel_by_junc_p.get(j_id)
        f_row = tel_by_junc_f.get(j_id)
        p_val = round(float(p_row["value"]), 2) if p_row else 55.0
        f_val = round(float(f_row["value"]), 2) if f_row else 110.0

        junctions_out.append(
            {
                "junction_id": j_id,
                "sector": j_sec,
                "pos_xz": [jx, jz],
                "pressure_psi": p_val,
                "flow_lps": f_val,
                "highlighted": bool(focus_type == "junction" and focus_id and j_id.upper() == focus_id.upper()),
            }
        )

        if j_id.startswith("J-"):
            s_id = j_id.replace("J-", "P-")
            s_val = p_val
            s_unit = "psi"
            s_status = str(p_row["status"]) if p_row else "ACTIVE"
        else:
            s_id = "T-INTAKE-1" if j_id == "INTAKE-01" else "T-INTAKE-2"
            t_match = next((dict(r) for r in latest_tel_rows if r["sensor_id"] == s_id), None)
            s_val = round(float(t_match["value"]), 2) if t_match else 1.25
            s_unit = "NTU"
            s_status = str(t_match["status"]) if t_match else "ACTIVE"

        is_quar = (s_id in quarantined_map) or (s_status.upper() == "QUARANTINED")
        sensors_out.append(
            {
                "sensor_id": s_id,
                "junction_id": j_id,
                "sector": j_sec,
                "pos_xz": [jx, jz],
                "value": s_val,
                "unit": s_unit,
                "status": "QUARANTINED" if is_quar else "ACTIVE",
                "quarantined": is_quar,
                "reasons": quarantined_map.get(s_id, []),
            }
        )

    # 5. Leaks — sourced ONLY from the single `leaks` registry (`get_active_leaks` / `get_all_leaks`)
    canonical_active_leaks = get_active_leaks(db_path=db_path)
    source_leaks = get_all_leaks(include_history=True, db_path=db_path) if show_leak_history else canonical_active_leaks
    leaks_out: list[dict] = []
    active_leaks_by_sec: dict[str, int] = {f"Sector {i}": 0 for i in range(1, 7)}
    for lk in source_leaks:
        lid = lk["leak_id"]
        l_sec = lk["sector"]
        norm_status = normalize_leak_status(lk["status"])
        lx, lz = latlon_to_local_xz(float(lk["latitude"]), float(lk["longitude"]))
        seg_ref = pipe_by_id.get(lk["pipe_segment_id"])
        pipe_flow = float(seg_ref["flow_lps"]) if seg_ref else 80.0
        is_history = norm_status in ("RESOLVED", "FALSE_ALARM")
        is_isolated = (lid in isolated_leaks) or (pipe_flow <= 0.1) or bool(lk.get("is_isolated")) or is_history
        is_active = not is_history
        if is_active:
            active_leaks_by_sec[l_sec] = active_leaks_by_sec.get(l_sec, 0) + 1
        eff_loss = 0.0 if is_isolated else round(float(lk["raw_loss_rate_m3h"]), 1)
        leaks_out.append(
            {
                "leak_id": lid,
                "junction_id": lk["junction_id"],
                "pipe_segment_id": lk["pipe_segment_id"],
                "sector": l_sec,
                "location_name": lk["location_name"],
                "pos_xz": [lx, lz],
                "severity": lk["severity"],
                "status": norm_status,
                "isolating_valve_id": lk["isolating_valve_id"],
                "valve_status": lk["valve_status"],
                "updated_at": lk["updated_at"],
                "detected_by": lk["detected_by"],
                "estimated_loss_m3_per_hour": eff_loss,
                "loss_rate_m3h": eff_loss,
                "raw_loss_m3_per_hour": round(float(lk["raw_loss_rate_m3h"]), 1),
                "confidence": round(float(lk["confidence"]), 2),
                "active": is_active,
                "isolated": is_isolated,
                "is_history": is_history,
                "highlighted": bool(focus_type == "leak" and focus_id and lid.upper() == focus_id.upper()),
            }
        )

    # 6. Sectors summary (with chlorine compliance & Sector 6 fire emergency flag)
    has_fire_emergency = bool(res_info.get("has_active_emergency", False))
    sectors_out: list[dict] = []
    for i in range(1, 7):
        sec_name = f"Sector {i}"
        binfo = boundaries.get(sec_name)
        if not binfo:
            continue
        poly_xz = [list(latlon_to_local_xz(p[0], p[1])) for p in binfo["polygon"]]
        cx, cz = latlon_to_local_xz(binfo["centroid"][0], binfo["centroid"][1])
        xs = [p[0] for p in poly_xz]
        zs = [p[1] for p in poly_xz]
        cl_val = wq_by_sec.get(sec_name, 0.50)
        comp = "NON_COMPLIANT" if cl_val < 0.20 else "COMPLIANT"
        sectors_out.append(
            {
                "id": f"SECTOR-0{i}",
                "name": sec_name,
                "color": SECTOR_COLORS[sec_name],
                "color_name": SECTOR_COLOR_NAMES[sec_name],
                "polygon_xz": poly_xz,
                "centroid_xz": [cx, cz],
                "bounds_xz": [[min(xs), min(zs)], [max(xs), max(zs)]],
                "chlorine_mg_l": round(cl_val, 3),
                "compliance_status": comp,
                "active_leaks": active_leaks_by_sec.get(sec_name, 0),
                "has_fire_emergency": bool(has_fire_emergency and sec_name == "Sector 6"),
                "pipe_count": sum(1 for p in pipes_out if p["sector"] == sec_name),
                "valve_count": sum(1 for v in valves_out if v["sector"] == sec_name),
                "total_flow_lps": sector_flow_totals.get(sec_name, 0.0),
                "highlighted": bool(focus_type == "sector" and focus_id and sec_name.lower() == focus_id.lower()),
            }
        )

    # 7. Detailed DESAL-01 Coastal Desalination Plant (with Intake 1 & Intake 2 status)
    px, pz = latlon_to_local_xz(*DESAL_PLANT_LATLON)
    bloom_sev = str(wq_info.get("bloom_severity", "NONE")).upper()
    algae_cap = round(float(wq_info.get("max_allowable_output_mld", 120.0)), 1)
    intake2_isolated = (bloom_sev in ("CRITICAL", "HIGH", "MODERATE")) or (algae_cap < plant_max_mld - 0.5)
    t_int1 = next((s["value"] for s in sensors_out if s["sensor_id"] == "T-INTAKE-1"), 1.2)
    t_int2 = next((s["value"] for s in sensors_out if s["sensor_id"] == "T-INTAKE-2"), 1.25)
    plant_label = (
        f"DESAL-01 — {plant_output_mld:.0f} MLD (algae cap {algae_cap:.0f})"
        if intake2_isolated
        else f"DESAL-01 — {plant_output_mld:.0f} MLD"
    )
    plant_out = {
        "id": "DESAL-01",
        "pos_xz": [px, pz],
        "output_mld": round(plant_output_mld, 1),
        "max_capacity_mld": round(plant_max_mld, 1),
        "algae_cap_mld": algae_cap,
        "bloom_severity": bloom_sev,
        "status": plant_status,
        "label": plant_label,
        "intakes": [
            {"id": "Intake 1", "status": "ACTIVE", "algae_tint": False, "turbidity_ntu": t_int1},
            {"id": "Intake 2", "status": "ISOLATED" if intake2_isolated else "ACTIVE", "algae_tint": bloom_sev != "NONE", "turbidity_ntu": t_int2},
        ],
        "highlighted": focus_type in ("plant", "intake"),
    }

    # 8. Strategic Reservoir RES-01 (with base floor, dynamic floor, warning floor rings & live depletion rate/ETA)
    rx, rz = latlon_to_local_xz(*RESERVOIR_RES01_LATLON)
    base_floor_pct = round(RESERVOIR_FLOOR_PERCENT * 100.0, 2)
    eff_floor_pct = round(float(res_info.get("safety_floor_percent", base_floor_pct)), 2)
    reservoir_out = {
        "id": "RES-01",
        "pos_xz": [rx, rz],
        "level_percent": round(float(res_info.get("current_level_percent", 48.0)), 2),
        "volume_ml": round(float(res_info.get("current_volume_ml", 240.0)), 1),
        "capacity_ml": round(float(res_info.get("capacity_ml", 500.0)), 1),
        "base_floor_percent": base_floor_pct,
        "safety_floor_percent": eff_floor_pct,
        "warning_floor_percent": round(eff_floor_pct + 5.0, 2),
        "has_active_emergency": has_fire_emergency,
        "floor_banner": str(res_info.get("floor_banner", "")),
        "effective_production_mld": round(float(res_info.get("effective_production_mld", plant_output_mld)), 1),
        "current_demand_mld": round(float(res_info.get("current_demand_mld", 85.0)), 1),
        "rate_ml_per_hour": round(float(res_info.get("rate_ml_per_hour", 0.0)), 3),
        "rate_pct_per_hour": round(float(res_info.get("rate_pct_per_hour", 0.0)), 3),
        "trend_arrow": str(res_info.get("trend_arrow", "→")),
        "rate_str": str(res_info.get("rate_str", "→ 0.0 %/h")),
        "hours_to_floor": res_info.get("hours_to_floor"),
        "floor_eta_str": str(res_info.get("floor_eta_str", "stable")),
        "warning_state": str(res_info.get("warning_state", "NORMAL")),
        "within_warning_band": bool(res_info.get("within_warning_band", False)),
        "at_or_below_floor": bool(res_info.get("at_or_below_floor", False)),
        "look_inside": focus_type in ("reservoir", "reservoir_inside"),
        "highlighted": focus_type in ("reservoir", "reservoir_inside"),
    }

    # 8b. 100% Connected Geometric & Hydraulic Pipe Network Graph (Intakes -> DESAL-01 -> RES-01 -> Trunk Spine -> Valves -> Sectors)
    connected_net = build_connected_network_geometry(
        segments=segments,
        valve_statuses=valve_status_map,
        plant_output_mld=float(res_info.get("effective_production_mld", plant_output_mld)),
        intake2_isolated=intake2_isolated,
        leaks=[dict(lk) for lk in leak_rows],
        latlon_to_xz=latlon_to_local_xz,
        pipe_flows_lps={p["segment_id"]: p["flow_lps"] for p in pipes_out},
        rerouted_segments=rerouted_segments,
        highlight_pipe_ids=hl_pipes_set,
        focus_type=focus_type,
        focus_id=focus_id,
    )

    # 9. Deterministic procedural city (buildings, roads, parks, trees) & western coastline
    buildings_out = generate_sector_buildings(boundaries, low_quality=low_quality)
    roads_out, parks_out, trees_out = generate_city_roads_and_trees(boundaries, low_quality=low_quality)
    coastline_xz = [list(latlon_to_local_xz(lat, lon)) for lat, lon in DUBAI_COASTLINE_LATLON]

    # 10. Highlight & camera target
    camera_target = _compute_camera_target(
        focus_type=focus_type,
        focus_id=focus_id,
        sectors=sectors_out,
        valves=valves_out,
        leaks=leaks_out,
        junctions=junctions_out,
        plant=plant_out,
        reservoir=reservoir_out,
    )

    telemetry_status = {
        "status": str(tf_info.get("status", "FRESH")),
        "is_telemetry_loss": bool(tel_data.get("telemetry_loss", False) or tf_info.get("is_telemetry_loss", False)),
        "age_seconds": int(tf_info.get("age_seconds", 0) or 0),
        "banner": str(tf_info.get("banner", "Telemetry fresh.")),
    }

    # Deterministic state hash/version so the bidirectional 3D component applies ONLY changes (no reload, camera kept)
    hash_payload = {
        "hour": int(current_hour),
        "valves": [(v["valve_id"], v["status"]) for v in valves_out],
        "pipes": [(p["segment_id"], p["flow_lps"]) for p in pipes_out],
        "leaks": [(l["leak_id"], l["status"], l["isolated"], l["loss_rate_m3h"], l["updated_at"]) for l in leaks_out],
        "plant": (plant_out["output_mld"], plant_out["algae_cap_mld"], plant_out["bloom_severity"]),
        "reservoir": (reservoir_out["level_percent"], reservoir_out["safety_floor_percent"], reservoir_out["rate_pct_per_hour"]),
        "sectors": [(s["name"], s["compliance_status"], s["has_fire_emergency"]) for s in sectors_out],
        "telemetry_loss": telemetry_status["is_telemetry_loss"],
        "show_leak_history": bool(show_leak_history),
    }
    scene_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    return {
        "version": "2.1",
        "scene_hash": scene_hash,
        "scene_version": scene_hash,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_at_local": now_local().strftime("%H:%M:%S"),
        "current_hour": int(current_hour),
        "active_leak_count": len(canonical_active_leaks),
        "show_leak_history": bool(show_leak_history),
        "center_latlon": list(NETWORK_CENTER_LATLON),
        "scale_m_per_unit": 100.0,
        "low_quality": bool(low_quality),
        "sectors": sectors_out,
        "pipes": pipes_out,
        "connected_network": connected_net,
        "valves": valves_out,
        "junctions": junctions_out,
        "sensors": sensors_out,
        "leaks": leaks_out,
        "plant": plant_out,
        "reservoir": reservoir_out,
        "buildings": buildings_out,
        "roads": roads_out,
        "parks": parks_out,
        "trees": trees_out,
        "coastline_xz": coastline_xz,
        "telemetry_status": telemetry_status,
        "highlight": {
            "type": focus_type,
            "id": focus_id,
            "downstream_pipe_ids": list(hl_pipes_set),
        },
        "hydraulics": {
            "rerouted_segment_ids": sorted(rerouted_segments),
            "zero_flow_segment_ids": sorted(zero_flow_segments),
            "isolated_leak_ids": sorted(isolated_leaks),
        },
        "camera_target": camera_target,
    }
