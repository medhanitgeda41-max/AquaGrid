"""Pure-Python 3D Digital Twin scene-state builder.

Reads live SCADA data from SQLite (read-only) and emits one deterministic JSON-serializable
scene state with local x/z metre coordinates around the Dubai network centre.
Never mutates valves, plant setpoints, or any safety guard state.
"""
from datetime import datetime, timezone
import json
import math
import random
import re
from typing import Optional

from config.constants import DEFAULT_SCADA_HOUR
from config.region_config import SECTOR_COLOR_NAMES, SECTOR_COLORS
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
    Same coordinates, dimensions, and heights on every call.
    """
    buildings: list[dict] = []
    target_per_sector = 16 if low_quality else 44

    for sec_idx in range(1, 7):
        sec_name = f"Sector {sec_idx}"
        binfo = boundaries.get(sec_name)
        if not binfo:
            continue
        poly = binfo["polygon"]
        (min_lat, min_lon), (max_lat, max_lon) = binfo["bounds"]
        rng = random.Random(4200 + sec_idx * 101)

        placed = 0
        attempts = 0
        max_attempts = target_per_sector * 14
        # Slightly taller towers in Downtown (Sector 3) and Dubai Marina (Sector 6)
        height_scale = 1.65 if sec_idx in (3, 6) else (1.2 if sec_idx in (1, 2) else 1.0)

        while placed < target_per_sector and attempts < max_attempts:
            attempts += 1
            lat = rng.uniform(min_lat, max_lat)
            lon = rng.uniform(min_lon, max_lon)
            if not point_in_polygon(lat, lon, poly):
                continue
            x_m, z_m = latlon_to_local_xz(lat, lon)
            w_m = round(rng.uniform(140.0, 310.0), 1)
            d_m = round(rng.uniform(140.0, 310.0), 1)
            h_m = round(rng.uniform(95.0, 360.0) * height_scale, 1)
            buildings.append(
                {
                    "x": x_m,
                    "z": z_m,
                    "w": w_m,
                    "d": d_m,
                    "h": h_m,
                    "sector": sec_name,
                    "color": SECTOR_COLORS.get(sec_name, "#64748B"),
                }
            )
            placed += 1

    return buildings


def _normalize_focus(focus: Optional[str]) -> tuple[str, Optional[str]]:
    """Classify focus string into (focus_type, normalized_id)."""
    raw = (focus or "all").strip()
    if not raw or raw.lower() in ("all", "all sectors", "network", "overview"):
        return "all", None
    low = raw.lower()
    if low in ("reservoir", "res-01", "res01", "tank"):
        return "reservoir", "RES-01"
    if low in ("plant", "desal", "desal-01", "desal01"):
        return "plant", "DESAL-01"
    m_v = re.match(r"^v-?0?(\d{1,2})$", low)
    if m_v:
        return "valve", f"V-{int(m_v.group(1)):02d}"
    m_l = re.match(r"^leak-?(\d+)$", low)
    if m_l:
        return "leak", f"LEAK-{int(m_l.group(1)):03d}"
    m_s = re.search(r"sector[\s-]*0?([1-6])", low)
    if m_s:
        return "sector", f"Sector {m_s.group(1)}"
    m_j = re.match(r"^j-?0?(\d{1,2})$", low)
    if m_j:
        j_id = f"J-{int(m_j.group(1)):02d}"
        sec = JUNCTION_SECTOR_MAP.get(j_id)
        if sec:
            return "sector", sec
    return "all", None


def _compute_camera_target(
    focus_type: str,
    focus_id: Optional[str],
    sectors: list[dict],
    valves: list[dict],
    leaks: list[dict],
    plant: dict,
    reservoir: dict,
) -> dict:
    """Compute camera look-at target (x, z in metres) and camera offset (x, y, z in metres)."""
    if focus_type == "sector" and focus_id:
        for s in sectors:
            if s["name"].lower() == focus_id.lower():
                cx, cz = s["centroid_xz"]
                return {
                    "target_xz": [cx, cz],
                    "position_xyz": [round(cx - 4200.0, 1), 3600.0, round(cz + 4600.0, 1)],
                    "label": s["name"],
                }
    elif focus_type == "valve" and focus_id:
        for v in valves:
            if v["valve_id"].upper() == focus_id.upper():
                vx, vz = v["pos_xz"]
                return {
                    "target_xz": [vx, vz],
                    "position_xyz": [round(vx - 2200.0, 1), 1900.0, round(vz + 2500.0, 1)],
                    "label": f"Valve {v['valve_id']}",
                }
    elif focus_type == "leak" and focus_id:
        for lk in leaks:
            if lk["leak_id"].upper() == focus_id.upper():
                lx, lz = lk["pos_xz"]
                return {
                    "target_xz": [lx, lz],
                    "position_xyz": [round(lx - 2100.0, 1), 1800.0, round(lz + 2400.0, 1)],
                    "label": f"{lk['leak_id']} ({lk['sector']})",
                }
    elif focus_type == "reservoir":
        rx, rz = reservoir["pos_xz"]
        return {
            "target_xz": [rx, rz],
            "position_xyz": [round(rx - 2200.0, 1), 1900.0, round(rz + 2600.0, 1)],
            "label": "Reservoir RES-01",
        }
    elif focus_type == "plant":
        px, pz = plant["pos_xz"]
        return {
            "target_xz": [px, pz],
            "position_xyz": [round(px - 2600.0, 1), 2100.0, round(pz + 2800.0, 1)],
            "label": "Plant DESAL-01",
        }

    # Default overview: fit all 6 sectors + coastal plant/reservoir nicely
    return {
        "target_xz": [0.0, 1200.0],
        "position_xyz": [-14500.0, 13500.0, 16500.0],
        "label": "All 6 Sectors (Dubai Network)",
    }


def _baseline_pipe_flow_lps(diameter_mm: int, plant_output_mld: float, valve_status: str) -> float:
    """Compute realistic baseline flow (L/s) for a pipe segment based on diameter, plant output, and valve state."""
    if str(valve_status).upper() == "CLOSED":
        return 0.0
    scale = max(0.0, float(plant_output_mld)) / 100.0
    # Nominal flow by diameter (900mm ~140 L/s, 700mm ~105 L/s, 600mm ~85 L/s, 500mm ~65 L/s, 400mm ~45 L/s)
    base = max(25.0, (float(diameter_mm) / 600.0) * 85.0)
    return round(base * scale, 1)


def build_3d_scene_state(
    focus: str = "all",
    current_hour: int = DEFAULT_SCADA_HOUR,
    db_path: Optional[str] = None,
    low_quality: bool = False,
    highlight_pipe_ids: Optional[list[str]] = None,
    flow_override_by_segment: Optional[dict[str, float]] = None,
    valve_override: Optional[dict[str, str]] = None,
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
    plant_output_mld = float(plant_row["production_mld"]) if plant_row else float(feeds_data.get("current_plant_output_mld", 100.0))
    plant_max_mld = float(plant_row["max_capacity_mld"]) if plant_row else 120.0
    plant_status = str(plant_row["status"]) if plant_row else "ONLINE"

    quarantined_map = {
        s["sensor_id"]: list(s.get("reasons") or [])
        for s in (tel_data.get("quarantined_sensors") or [])
    }

    # Optional hydraulic propagation hook (populated in Part 2; cleanly falls back to baseline in Part 1)
    computed_flows: dict[str, float] = dict(flow_override_by_segment or {})
    isolated_leaks: set[str] = set()
    if not computed_flows:
        try:
            from digital_twin.hydraulics import compute_network_flows

            hyd = compute_network_flows(
                segments=segments,
                valve_statuses=valve_status_map,
                plant_output_mld=plant_output_mld,
                leaks=[dict(r) for r in leak_rows],
            )
            computed_flows = hyd.get("pipe_flows_lps") or {}
            isolated_leaks = set(hyd.get("isolated_leak_ids") or [])
        except ImportError:
            pass

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
        v_id = seg.get("isolation_valve_id", "V-01")
        v_stat = valve_status_map.get(v_id, "OPEN")
        hist_item = hist_by_seg.get(seg_id, {})

        if seg_id in computed_flows:
            flow_lps = round(float(computed_flows[seg_id]), 1)
        else:
            flow_lps = _baseline_pipe_flow_lps(diam, plant_output_mld, v_stat)

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
        # Find nearest matching pipe direction for wheel orientation
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
            }
        )

        # Primary representative sensor at this junction (P-xx or T-INTAKE-x)
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

    # 5. Leaks
    leaks_out: list[dict] = []
    active_leaks_by_sec: dict[str, int] = {f"Sector {i}": 0 for i in range(1, 7)}
    for lk in leak_rows:
        lid = lk["leak_id"]
        l_sec = lk["sector"]
        l_status = str(lk["status"])
        lx, lz = latlon_to_local_xz(float(lk["latitude"]), float(lk["longitude"]))
        seg_ref = pipe_by_id.get(lk["pipe_segment_id"])
        pipe_flow = float(seg_ref["flow_lps"]) if seg_ref else 80.0
        is_isolated = (lid in isolated_leaks) or (pipe_flow <= 0.1) or (l_status.lower() in ("resolved", "under repair"))
        is_active = (l_status.lower() not in ("resolved", "false alarm – sensor fault")) and (not is_isolated)
        if is_active:
            active_leaks_by_sec[l_sec] = active_leaks_by_sec.get(l_sec, 0) + 1
        leaks_out.append(
            {
                "leak_id": lid,
                "pipe_segment_id": lk["pipe_segment_id"],
                "sector": l_sec,
                "location_name": lk["location_name"],
                "pos_xz": [lx, lz],
                "severity": lk["severity"],
                "status": l_status,
                "estimated_loss_m3_per_hour": 0.0 if is_isolated else round(float(lk["estimated_loss_m3_per_hour"]), 1),
                "raw_loss_m3_per_hour": round(float(lk["estimated_loss_m3_per_hour"]), 1),
                "confidence": round(float(lk["confidence"]), 2),
                "active": is_active,
                "isolated": is_isolated,
                "highlighted": bool(focus_type == "leak" and focus_id and lid.upper() == focus_id.upper()),
            }
        )

    # 6. Sectors summary
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
                "pipe_count": sum(1 for p in pipes_out if p["sector"] == sec_name),
                "valve_count": sum(1 for v in valves_out if v["sector"] == sec_name),
                "total_flow_lps": sector_flow_totals.get(sec_name, 0.0),
                "highlighted": bool(focus_type == "sector" and focus_id and sec_name.lower() == focus_id.lower()),
            }
        )

    # 7. Plant DESAL-01 & Reservoir RES-01
    px, pz = latlon_to_local_xz(*DESAL_PLANT_LATLON)
    plant_out = {
        "id": "DESAL-01",
        "pos_xz": [px, pz],
        "output_mld": round(plant_output_mld, 1),
        "max_capacity_mld": round(plant_max_mld, 1),
        "algae_cap_mld": round(float(wq_info.get("max_allowable_output_mld", 120.0)), 1),
        "bloom_severity": str(wq_info.get("bloom_severity", "NONE")),
        "status": plant_status,
        "highlighted": focus_type == "plant",
    }

    rx, rz = latlon_to_local_xz(*RESERVOIR_RES01_LATLON)
    reservoir_out = {
        "id": "RES-01",
        "pos_xz": [rx, rz],
        "level_percent": round(float(res_info.get("current_level_percent", 48.0)), 2),
        "volume_ml": round(float(res_info.get("current_volume_ml", 240.0)), 1),
        "capacity_ml": round(float(res_info.get("capacity_ml", 500.0)), 1),
        "safety_floor_percent": round(float(res_info.get("safety_floor_percent", 35.0)), 2),
        "has_active_emergency": bool(res_info.get("has_active_emergency", False)),
        "highlighted": focus_type == "reservoir",
    }

    # 8. Deterministic procedural city & western coastline
    buildings_out = generate_sector_buildings(boundaries, low_quality=low_quality)
    coastline_xz = [list(latlon_to_local_xz(lat, lon)) for lat, lon in DUBAI_COASTLINE_LATLON]

    # 9. Highlight & camera target
    camera_target = _compute_camera_target(
        focus_type=focus_type,
        focus_id=focus_id,
        sectors=sectors_out,
        valves=valves_out,
        leaks=leaks_out,
        plant=plant_out,
        reservoir=reservoir_out,
    )

    return {
        "version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "center_latlon": list(NETWORK_CENTER_LATLON),
        "scale_m_per_unit": 100.0,
        "low_quality": bool(low_quality),
        "sectors": sectors_out,
        "pipes": pipes_out,
        "valves": valves_out,
        "junctions": junctions_out,
        "sensors": sensors_out,
        "leaks": leaks_out,
        "plant": plant_out,
        "reservoir": reservoir_out,
        "buildings": buildings_out,
        "coastline_xz": coastline_xz,
        "highlight": {
            "type": focus_type,
            "id": focus_id,
            "downstream_pipe_ids": list(hl_pipes_set),
        },
        "camera_target": camera_target,
    }
