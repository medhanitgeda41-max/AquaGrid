"""Pure-Python directed-graph hydraulic propagation model using NetworkX.

Models water flow from Sea Intakes -> DESAL-01 -> RES-01 -> outlet trunk mains ->
sector headers & valves -> junctions -> distribution pipe segments.
When a valve is CLOSED, downstream pipes lose flow (or receive reduced rerouted flow ONLY
where a valid alternative open path exists), and isolated leaks stop losing water.
Read-only calculation only; never mutates database state or safety guards.
"""
import json
import math
from typing import Callable, Optional
import networkx as nx

from config.region_config import SECTOR_COLORS
from database.connection import get_db_connection
from database.dubai_network import (
    DUBAI_PIPE_SEGMENTS,
    JUNCTION_COORDINATES,
    JUNCTION_SECTOR_MAP,
    VALVE_COORDINATES,
    VALVE_SECTOR_MAP,
    distance_to_polyline_meters,
)

# Coastal plant DESAL-01, strategic reservoir RES-01, and main north-south transmission corridor hubs
PLANT_DESAL01_LATLON: tuple[float, float] = (25.0760, 55.1060)
RESERVOIR_RES01_LATLON: tuple[float, float] = (25.0980, 55.1260)

TRUNK_HUB_LATLON: dict[str, tuple[float, float]] = {
    "TRUNK-HUB-S6": (25.0780, 55.1440),  # Sector 6 Marina / JLT trunk manifold
    "TRUNK-HUB-S5": (25.1140, 55.1880),  # Sector 5 Al Barsha / SZR manifold
    "TRUNK-HUB-S4": (25.1600, 55.2240),  # Sector 4 Al Wasl / Jumeirah manifold
    "TRUNK-HUB-S3": (25.1960, 55.2680),  # Sector 3 Downtown / Business Bay manifold
    "TRUNK-HUB-S2": (25.2420, 55.3010),  # Sector 2 Bur Dubai / Karama manifold
    "TRUNK-HUB-S1": (25.2640, 55.3240),  # Sector 1 Deira / Al Rigga manifold
}

# Primary valve controlling each pipe segment (aligned with DUBAI_PIPE_SEGMENTS & V-01..V-20 placements)
SEGMENT_VALVE_MAP: dict[str, str] = {
    # Sector 1
    "SEG-S1-01": "V-01",
    "SEG-S1-02": "V-01",
    "SEG-S1-03": "V-02",
    "SEG-S1-04": "V-02",
    "SEG-S1-05": "V-03",
    # Sector 2
    "SEG-S2-01": "V-04",
    "SEG-S2-02": "V-04",
    "SEG-S2-03": "V-05",
    "SEG-S2-04": "V-05",
    "SEG-S2-05": "V-06",
    # Sector 3
    "SEG-S3-01": "V-07",
    "SEG-S3-02": "V-08",
    "SEG-S3-03": "V-08",
    "SEG-S3-04": "V-09",
    "SEG-S3-05": "V-09",
    # Sector 4 (V-12 isolates Sector 4 Main SEG-S4-01 & downstream SEG-S4-05)
    "SEG-S4-01": "V-12",
    "SEG-S4-02": "V-10",
    "SEG-S4-03": "V-11",
    "SEG-S4-04": "V-11",
    "SEG-S4-05": "V-12",
    # Sector 5 (V-13..V-16 colocated on Sector 5 pipes)
    "SEG-S5-01": "V-13",
    "SEG-S5-02": "V-14",
    "SEG-S5-03": "V-15",
    "SEG-S5-04": "V-16",
    # Sector 6 (V-17..V-20 colocated on Sector 6 pipes — V-17 controls Marina Main SEG-S6-01)
    "SEG-S6-01": "V-17",
    "SEG-S6-02": "V-18",
    "SEG-S6-03": "V-19",
    "SEG-S6-04": "V-20",
}

# Directed hydraulic topology per segment:
# (upstream_node, downstream_node, alternative_upstream_node_or_None)
# An alternative upstream node models a real physical ring-main cross-tie where flow can be rerouted
# if the primary upstream branch is closed (at reduced capacity). Branches with alt=None have NO bypass
# (e.g. Sector 4 Main SEG-S4-01 & downstream SEG-S4-05 under V-12 isolate completely to 0 L/s).
SEGMENT_TOPOLOGY: dict[str, tuple[str, str, Optional[str]]] = {
    # Sector 1: V-03 spine can partially backfeed SEG-S1-04 if V-02 closes; V-01 trunk has no bypass
    "SEG-S1-05": ("HDR-S1", "N-S1-SPINE", None),
    "SEG-S1-01": ("HDR-S1", "N-S1-01", None),
    "SEG-S1-02": ("N-S1-01", "N-S1-02", None),
    "SEG-S1-03": ("HDR-S1", "N-S1-03", None),
    "SEG-S1-04": ("N-S1-03", "N-S1-04", "N-S1-SPINE"),
    # Sector 2: ring tie from SEG-S2-03 (V-05) into SEG-S2-05 (V-06)
    "SEG-S2-01": ("HDR-S2", "N-S2-01", None),
    "SEG-S2-02": ("N-S2-01", "N-S2-02", None),
    "SEG-S2-03": ("HDR-S2", "N-S2-03", None),
    "SEG-S2-04": ("N-S2-03", "N-S2-04", None),
    "SEG-S2-05": ("HDR-S2", "N-S2-05", "N-S2-03"),
    # Sector 3: Downtown MBR Blvd loop (SEG-S3-03) has primary feed via SEG-S3-02 (V-08) AND alt loop tie via SEG-S3-04 (V-09)
    "SEG-S3-01": ("HDR-S3", "N-S3-TRUNK", None),
    "SEG-S3-02": ("N-S3-TRUNK", "N-S3-02", None),
    "SEG-S3-04": ("N-S3-TRUNK", "N-S3-04", None),
    "SEG-S3-03": ("N-S3-02", "N-S3-03", "N-S3-04"),
    "SEG-S3-05": ("N-S3-04", "N-S3-05", None),
    # Sector 4: V-12 isolates SEG-S4-01 (J-14 Main) and its downstream branch SEG-S4-05 completely (no bypass!)
    "SEG-S4-01": ("HDR-S4", "N-S4-J14", None),
    "SEG-S4-05": ("N-S4-J14", "N-S4-05", None),
    "SEG-S4-02": ("HDR-S4", "N-S4-02", None),
    "SEG-S4-03": ("N-S4-02", "N-S4-03", None),
    "SEG-S4-04": ("N-S4-03", "N-S4-04", None),
    # Sector 5: Sheikh Zayed South-Central (SEG-S5-04) can cross-feed Al Barsha local loop (SEG-S5-02)
    "SEG-S5-04": ("HDR-S5", "N-S5-TRUNK", None),
    "SEG-S5-01": ("HDR-S5", "N-S5-01", None),
    "SEG-S5-02": ("N-S5-01", "N-S5-02", "N-S5-TRUNK"),
    "SEG-S5-03": ("N-S5-TRUNK", "N-S5-03", None),
    # Sector 6: V-17 isolates Marina Main SEG-S6-01 (no bypass); V-18 feeds SEG-S6-02
    "SEG-S6-04": ("HDR-S6", "N-S6-TRUNK", None),
    "SEG-S6-01": ("N-S6-TRUNK", "N-S6-01", None),
    "SEG-S6-02": ("N-S6-TRUNK", "N-S6-02", None),
    "SEG-S6-03": ("N-S6-TRUNK", "N-S6-03", None),
}


def _nominal_segment_flow_lps(diameter_mm: int, plant_output_mld: float) -> float:
    """Compute nominal flow (L/s) for an open, energized pipe scaled by plant output MLD."""
    scale = max(0.0, float(plant_output_mld)) / 100.0
    base = max(28.0, (float(diameter_mm) / 600.0) * 84.0)
    return round(base * scale, 1)


def get_effective_valve_for_segment(seg: dict) -> str:
    """Return the controlling valve ID for a pipe segment."""
    seg_id = seg["segment_id"]
    return SEGMENT_VALVE_MAP.get(seg_id, seg.get("isolation_valve_id", "V-01"))


def build_hydraulic_graph(
    segments: list[dict],
    valve_statuses: dict[str, str],
) -> nx.DiGraph:
    """
    Construct a NetworkX directed hydraulic graph from DESAL-01 -> RES-01 -> sector headers -> pipe segments.
    Closed valve edges are marked open=False and excluded from active flow paths.
    """
    G = nx.DiGraph()
    G.add_node("DESAL-01", kind="plant")
    G.add_node("RES-01", kind="reservoir")
    G.add_edge("DESAL-01", "RES-01", open=True, kind="supply_main")

    for i in range(1, 7):
        hdr = f"HDR-S{i}"
        G.add_node(hdr, kind="sector_header", sector=f"Sector {i}")
        G.add_edge("RES-01", hdr, open=True, kind="transmission_trunk")

    for seg in segments:
        seg_id = seg["segment_id"]
        v_id = get_effective_valve_for_segment(seg)
        is_open = str(valve_statuses.get(v_id, "OPEN")).upper() == "OPEN"
        u, v, alt_u = SEGMENT_TOPOLOGY.get(seg_id, (f"HDR-S{seg['sector'][-1]}", f"N-{seg_id}", None))

        G.add_edge(
            u,
            v,
            segment_id=seg_id,
            valve_id=v_id,
            diameter_mm=int(seg.get("diameter_mm", 500)),
            open=is_open,
            is_primary=True,
        )
        if alt_u is not None:
            # Alternative ring-main cross-tie: only usable if alt_u itself is reachable AND this segment's own valve is OPEN
            # (or if this segment's primary upstream was blocked by a different upstream valve, allowing bypass around upstream closure).
            G.add_edge(
                alt_u,
                v,
                segment_id=f"TIE-{seg_id}",
                target_segment_id=seg_id,
                valve_id=v_id,
                diameter_mm=int(seg.get("diameter_mm", 500)),
                open=is_open,
                is_primary=False,
            )

    return G


def compute_network_flows(
    segments: Optional[list[dict]] = None,
    valve_statuses: Optional[dict[str, str]] = None,
    plant_output_mld: float = 100.0,
    leaks: Optional[list[dict]] = None,
) -> dict:
    """
    Propagate directed flow from DESAL-01 -> RES-01 across open edges in the NetworkX graph.

    Rules:
    - Closed valve cuts flow on its controlled pipe and any strictly downstream pipes without an alternative path to 0.0 L/s.
    - If a segment's primary upstream node is cut off BUT its own valve is OPEN and an alternative open path
      from DESAL-01 reaches its alt_u cross-tie, flow is rerouted at reduced hydraulic capacity (~65% of nominal).
    - Any leak on a segment whose propagated flow drops to 0.0 L/s is marked isolated (0.0 m3/hr loss).
    """
    segs = list(segments) if segments is not None else list(DUBAI_PIPE_SEGMENTS)
    v_map = {k: str(v).upper() for k, v in (valve_statuses or {}).items()}

    G = build_hydraulic_graph(segs, v_map)

    # Subgraph containing only OPEN edges
    open_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("open", True)]
    G_open = G.edge_subgraph(open_edges)
    reachable_any = set(nx.descendants(G_open, "DESAL-01")) | {"DESAL-01"} if "DESAL-01" in G_open else set()

    # Subgraph containing only OPEN PRIMARY edges (to distinguish primary supply vs rerouted alternate path)
    primary_open_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("open", True) and d.get("is_primary", True)
    ]
    G_primary = G.edge_subgraph(primary_open_edges)
    reachable_primary = set(nx.descendants(G_primary, "DESAL-01")) | {"DESAL-01"} if "DESAL-01" in G_primary else set()

    pipe_flows_lps: dict[str, float] = {}
    rerouted_segment_ids: list[str] = []
    zero_flow_segment_ids: list[str] = []

    for seg in segs:
        seg_id = seg["segment_id"]
        diam = int(seg.get("diameter_mm", 500))
        v_id = get_effective_valve_for_segment(seg)
        v_open = v_map.get(v_id, "OPEN") == "OPEN"
        u, v, alt_u = SEGMENT_TOPOLOGY.get(seg_id, (f"HDR-S{seg['sector'][-1]}", f"N-{seg_id}", None))
        nominal = _nominal_segment_flow_lps(diam, plant_output_mld)

        if not v_open or plant_output_mld <= 0.05:
            pipe_flows_lps[seg_id] = 0.0
            zero_flow_segment_ids.append(seg_id)
            continue

        # Primary upstream path energized
        if u in reachable_primary:
            pipe_flows_lps[seg_id] = nominal
        elif u in reachable_any:
            # Upstream node was itself rerouted via an alternate ring path -> reduced pressure/flow
            pipe_flows_lps[seg_id] = round(nominal * 0.65, 1)
            rerouted_segment_ids.append(seg_id)
        elif alt_u is not None and alt_u in reachable_any:
            # Primary upstream is blocked, but an open alternative cross-tie reaches this segment!
            pipe_flows_lps[seg_id] = round(nominal * 0.65, 1)
            rerouted_segment_ids.append(seg_id)
        else:
            # Completely cut off downstream of a closed upstream valve with no alternative path
            pipe_flows_lps[seg_id] = 0.0
            zero_flow_segment_ids.append(seg_id)

    # Leak isolation check: leaks on zero-flow segments stop losing water
    isolated_leak_ids: list[str] = []
    active_leak_loss: dict[str, float] = {}
    for lk in (leaks or []):
        lid = lk["leak_id"]
        seg_id = lk.get("pipe_segment_id", "")
        raw_loss = round(float(lk.get("estimated_loss_m3_per_hour", 0.0)), 1)
        l_status = str(lk.get("status", "")).lower()
        if pipe_flows_lps.get(seg_id, 0.0) <= 0.05 or l_status in ("resolved", "false alarm – sensor fault"):
            isolated_leak_ids.append(lid)
            active_leak_loss[lid] = 0.0
        else:
            active_leak_loss[lid] = raw_loss

    return {
        "pipe_flows_lps": pipe_flows_lps,
        "rerouted_segment_ids": rerouted_segment_ids,
        "zero_flow_segment_ids": zero_flow_segment_ids,
        "isolated_leak_ids": isolated_leak_ids,
        "active_leak_loss_m3_hr": active_leak_loss,
        "graph_node_count": G.number_of_nodes(),
        "graph_edge_count": G.number_of_edges(),
    }


def simulate_valve_change_impact(
    valve_id: str,
    new_status: str,
    segments: Optional[list[dict]] = None,
    valve_statuses: Optional[dict[str, str]] = None,
    plant_output_mld: Optional[float] = None,
    leaks: Optional[list[dict]] = None,
    db_path: Optional[str] = None,
) -> dict:
    """
    Compute BEFORE vs AFTER hydraulic flow changes if `valve_id` is switched to `new_status` ('CLOSED' or 'OPEN').
    Read-only simulation used for HITL approval previews and impact explanations.
    """
    if segments is None or valve_statuses is None or plant_output_mld is None or leaks is None:
        conn = get_db_connection(db_path)
        try:
            if segments is None:
                rows = conn.execute("SELECT * FROM pipe_segments ORDER BY sector, segment_id;").fetchall()
                segments = [dict(r) for r in rows] if rows else list(DUBAI_PIPE_SEGMENTS)
            if valve_statuses is None:
                v_rows = conn.execute("SELECT valve_id, status FROM valves;").fetchall()
                valve_statuses = {r["valve_id"]: str(r["status"]).upper() for r in v_rows}
            if plant_output_mld is None:
                p_row = conn.execute("SELECT production_mld FROM plant_setpoints WHERE plant_id = 'DESAL-01';").fetchone()
                plant_output_mld = float(p_row["production_mld"]) if p_row else 100.0
            if leaks is None:
                l_rows = conn.execute("SELECT * FROM leaks;").fetchall()
                leaks = [dict(r) for r in l_rows]
        finally:
            conn.close()

    norm_vid = str(valve_id).strip().upper()
    norm_status = str(new_status).strip().upper()

    before = compute_network_flows(segments, valve_statuses, plant_output_mld, leaks)
    after_valves = dict(valve_statuses)
    after_valves[norm_vid] = norm_status
    after = compute_network_flows(segments, after_valves, plant_output_mld, leaks)

    seg_by_id = {s["segment_id"]: s for s in segments}
    affected_pipes: list[dict] = []
    summary_lines: list[str] = []

    for seg_id, b_flow in before["pipe_flows_lps"].items():
        a_flow = after["pipe_flows_lps"].get(seg_id, 0.0)
        if abs(a_flow - b_flow) >= 0.1:
            s_info = seg_by_id.get(seg_id, {})
            name = s_info.get("name", seg_id)
            item = {
                "segment_id": seg_id,
                "name": name,
                "sector": s_info.get("sector", ""),
                "before_lps": round(b_flow, 1),
                "after_lps": round(a_flow, 1),
                "delta_lps": round(a_flow - b_flow, 1),
            }
            affected_pipes.append(item)
            summary_lines.append(f"{name}: {b_flow:.0f} -> {a_flow:.0f} L/s")

    # Sort biggest drops first
    affected_pipes.sort(key=lambda x: abs(x["delta_lps"]), reverse=True)
    summary_lines = [f"{p['name']}: {p['before_lps']:.0f} -> {p['after_lps']:.0f} L/s" for p in affected_pipes]

    return {
        "valve_id": norm_vid,
        "new_status": norm_status,
        "affected_pipes": affected_pipes,
        "affected_pipe_ids": [p["segment_id"] for p in affected_pipes],
        "before_flows_lps": before["pipe_flows_lps"],
        "after_flows_lps": after["pipe_flows_lps"],
        "isolated_leaks_before": before["isolated_leak_ids"],
        "isolated_leaks_after": after["isolated_leak_ids"],
        "summary_lines": summary_lines,
    }


def _split_polyline_at_coord(
    pts_xz: list[list[float]],
    split_xz: list[float],
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Split a polyline `pts_xz` cleanly at the closest projection of `split_xz` so
    upstream ends strictly at `split_xz` and downstream starts strictly at `split_xz`.
    """
    if len(pts_xz) < 2:
        return [list(split_xz), list(split_xz)], [list(split_xz), list(split_xz)]

    sx, sz = float(split_xz[0]), float(split_xz[1])
    best_i = 0
    best_dist = float("inf")
    for i in range(len(pts_xz) - 1):
        ax, az = float(pts_xz[i][0]), float(pts_xz[i][1])
        bx, bz = float(pts_xz[i + 1][0]), float(pts_xz[i + 1][1])
        dx, dz = bx - ax, bz - az
        seg_sq = dx * dx + dz * dz
        t = 0.0 if seg_sq <= 1e-9 else max(0.0, min(1.0, ((sx - ax) * dx + (sz - az) * dz) / seg_sq))
        px, pz = ax + t * dx, az + t * dz
        d = math.hypot(sx - px, sz - pz)
        if d < best_dist:
            best_dist = d
            best_i = i

    up = [list(p) for p in pts_xz[: best_i + 1]]
    if math.hypot(up[-1][0] - sx, up[-1][1] - sz) > 0.05:
        up.append([sx, sz])
    else:
        up[-1] = [sx, sz]
    if len(up) < 2:
        up = [list(pts_xz[0]), [sx, sz]]

    down = [[sx, sz]] + [list(p) for p in pts_xz[best_i + 1 :]]
    if len(down) >= 2 and math.hypot(down[0][0] - down[1][0], down[0][1] - down[1][1]) <= 0.05:
        down = [[sx, sz]] + down[2:]
    if len(down) < 2:
        down = [[sx, sz], list(pts_xz[-1])]

    return up, down


def build_connected_network_geometry(
    segments: Optional[list[dict]] = None,
    valve_statuses: Optional[dict[str, str]] = None,
    plant_output_mld: float = 100.0,
    intake2_isolated: bool = False,
    leaks: Optional[list[dict]] = None,
    latlon_to_xz: Optional[Callable[[float, float], tuple[float, float]]] = None,
    pipe_flows_lps: Optional[dict[str, float]] = None,
    rerouted_segments: Optional[set[str]] = None,
    highlight_pipe_ids: Optional[set[str]] = None,
    focus_type: str = "all",
    focus_id: Optional[str] = None,
) -> dict:
    """
    Build ONE 100% connected geometric & hydraulic pipe network from:
        Sea Intakes (INTAKE-01/02) -> DESAL-01 -> RES-01 -> Outlet Trunk Spine (TRUNK-HUB-S6..S1)
        -> Sector Trunks -> Isolation Valves (V-01..V-20 placed ON exact nodes)
        -> All 28 Dubai Street Pipes & Junctions -> Distribution Branches.

    Every pipe edge starts and ends at an exact node position in `nodes_by_id` (zero floating segments).
    Pipe roles control 3D diameter: 'intake'/'product_water'/'trunk_main' (thick), 'sector_trunk' (medium), 'distribution' (thin).
    """
    segs = list(segments) if segments is not None else list(DUBAI_PIPE_SEGMENTS)
    v_map = {k: str(v).upper() for k, v in (valve_statuses or {}).items()}
    hl_set = set(highlight_pipe_ids or set())
    rerouted = set(rerouted_segments or set())

    if latlon_to_xz is None:
        lats = [float(p[0]) for s in DUBAI_PIPE_SEGMENTS for p in s["geometry"]]
        lons = [float(p[1]) for s in DUBAI_PIPE_SEGMENTS for p in s["geometry"]]
        c_lat, c_lon = round(sum(lats) / len(lats), 6), round(sum(lons) / len(lons), 6)
        cos_lat = math.cos(math.radians(c_lat))

        def _conv(lat: float, lon: float) -> tuple[float, float]:
            return round((float(lon) - c_lon) * 111_320.0 * cos_lat, 2), round(-(float(lat) - c_lat) * 111_320.0, 2)

        latlon_to_xz = _conv

    if pipe_flows_lps is None:
        hyd = compute_network_flows(segs, v_map, plant_output_mld=plant_output_mld, leaks=leaks)
        pipe_flows_lps = hyd["pipe_flows_lps"]
        rerouted = set(hyd["rerouted_segment_ids"])

    nodes_by_id: dict[str, dict] = {}

    def _add_node(node_id: str, kind: str, lat: float, lon: float, sector: str = "Network", extra: Optional[dict] = None) -> list[float]:
        xz = list(latlon_to_xz(lat, lon))
        if node_id not in nodes_by_id:
            rec = {
                "node_id": node_id,
                "kind": kind,
                "sector": sector,
                "latlon": [round(float(lat), 6), round(float(lon), 6)],
                "pos_xz": xz,
            }
            if extra:
                rec.update(extra)
            nodes_by_id[node_id] = rec
        return list(nodes_by_id[node_id]["pos_xz"])

    # 1. Core Plant, Intakes, Reservoir & Main Transmission Spine Hubs
    p_int1 = _add_node("INTAKE-01", "intake", *JUNCTION_COORDINATES["INTAKE-01"], sector="Plant")
    p_int2 = _add_node("INTAKE-02", "intake", *JUNCTION_COORDINATES["INTAKE-02"], sector="Plant")
    p_plant = _add_node("DESAL-01", "plant", *PLANT_DESAL01_LATLON, sector="Plant")
    p_res = _add_node("RES-01", "reservoir", *RESERVOIR_RES01_LATLON, sector="Reservoir")

    hub_pos: dict[str, list[float]] = {}
    for hub_id, (hlat, hlon) in TRUNK_HUB_LATLON.items():
        sec_num = hub_id[-1]
        hub_pos[hub_id] = _add_node(hub_id, "trunk_hub", hlat, hlon, sector=f"Sector {sec_num}")

    # Register every Valve (V-01..V-20) as a first-class graph node
    for vid, (vlat, vlon) in VALVE_COORDINATES.items():
        _add_node(
            vid,
            "valve",
            vlat,
            vlon,
            sector=VALVE_SECTOR_MAP.get(vid, "Sector 1"),
            extra={"status": v_map.get(vid, "OPEN")},
        )

    pipes_out: list[dict] = []

    def _add_pipe(
        pipe_id: str,
        name: str,
        from_node: str,
        to_node: str,
        points_xz: list[list[float]],
        role: str,
        diameter_mm: int,
        sector: str,
        color: str,
        flow_lps: float,
        valve_id: Optional[str] = None,
        segment_id: Optional[str] = None,
        is_rerouted: bool = False,
        highlighted: bool = False,
    ) -> None:
        # Enforce exact mathematical endpoint snapping to nodes_by_id (0.00m gap!)
        start_pt = list(nodes_by_id[from_node]["pos_xz"])
        end_pt = list(nodes_by_id[to_node]["pos_xz"])
        clean_pts = [start_pt] + [list(p) for p in points_xz[1:-1]] + [end_pt] if len(points_xz) >= 2 else [start_pt, end_pt]
        elbows = [list(p) for p in clean_pts[1:-1]]
        pipes_out.append(
            {
                "pipe_id": pipe_id,
                "segment_id": segment_id or pipe_id,
                "name": name,
                "from_node": from_node,
                "to_node": to_node,
                "role": role,
                "diameter_mm": int(diameter_mm),
                "is_trunk": role in ("intake", "product_water", "trunk_main", "sector_trunk"),
                "sector": sector,
                "color": color,
                "isolation_valve_id": valve_id,
                "flow_lps": round(max(0.0, float(flow_lps)), 1),
                "rerouted": bool(is_rerouted),
                "highlighted": bool(highlighted),
                "points_xz": clean_pts,
                "start_xz": start_pt,
                "end_xz": end_pt,
                "elbows_xz": elbows,
            }
        )

    # 2. Main Backbone Edges: Sea Intakes -> DESAL-01 -> RES-01 -> Outlet Trunk Spine (S6 & S5->S4->S3->S2->S1)
    plant_flow_lps = round((float(plant_output_mld) * 1_000_000.0) / 86_400.0, 1)  # 100 MLD ≈ 1157.4 L/s (display scaled cleanly)
    backbone_lps = round(max(0.0, float(plant_output_mld)) * 1.4, 1)
    int1_lps = backbone_lps if intake2_isolated else round(backbone_lps * 0.55, 1)
    int2_lps = 0.0 if intake2_isolated else round(backbone_lps * 0.55, 1)

    _add_pipe(
        "PIPE-INTAKE-01", "Sea Intake 1 Main", "INTAKE-01", "DESAL-01",
        [p_int1, p_plant], role="intake", diameter_mm=1200, sector="Plant",
        color="#0284C7", flow_lps=int1_lps, highlighted=(focus_type in ("plant", "intake")),
    )
    _add_pipe(
        "PIPE-INTAKE-02", "Sea Intake 2 Main", "INTAKE-02", "DESAL-01",
        [p_int2, p_plant], role="intake", diameter_mm=1200, sector="Plant",
        color="#22C55E" if intake2_isolated else "#0284C7", flow_lps=int2_lps,
        highlighted=(focus_type in ("plant", "intake")),
    )
    _add_pipe(
        "PIPE-PRODUCT-01", "DESAL-01 Product Water Trunk -> RES-01", "DESAL-01", "RES-01",
        [p_plant, p_res], role="product_water", diameter_mm=1200, sector="Reservoir",
        color="#38BDF8", flow_lps=backbone_lps, highlighted=(focus_type in ("plant", "reservoir", "reservoir_inside")),
    )

    # Compute total open flow in each sector so spine trunk carries realistic positive flow whenever downstream is open
    sec_flow_totals: dict[int, float] = {i: 0.0 for i in range(1, 7)}
    for seg in segs:
        sec_str = str(seg.get("sector", "Sector 1"))
        s_num = int(sec_str.split()[-1]) if sec_str.split()[-1].isdigit() else 1
        sec_flow_totals[s_num] += float(pipe_flows_lps.get(seg["segment_id"], 0.0))

    spine_edges = [
        ("TRUNK-RES-S6", "RES-01 South Outlet Trunk -> Sector 6", "RES-01", "TRUNK-HUB-S6", 1200, "Sector 6", sec_flow_totals[6]),
        ("TRUNK-RES-S5", "RES-01 North Outlet Trunk -> Sector 5", "RES-01", "TRUNK-HUB-S5", 1200, "Sector 5", sum(sec_flow_totals[k] for k in (5, 4, 3, 2, 1))),
        ("TRUNK-S5-S4", "Sheikh Zayed Transmission Main (S5 -> S4)", "TRUNK-HUB-S5", "TRUNK-HUB-S4", 1100, "Sector 4", sum(sec_flow_totals[k] for k in (4, 3, 2, 1))),
        ("TRUNK-S4-S3", "Sheikh Zayed Transmission Main (S4 -> S3)", "TRUNK-HUB-S4", "TRUNK-HUB-S3", 1000, "Sector 3", sum(sec_flow_totals[k] for k in (3, 2, 1))),
        ("TRUNK-S3-S2", "Sheikh Zayed Transmission Main (S3 -> S2)", "TRUNK-HUB-S3", "TRUNK-HUB-S2", 950, "Sector 2", sum(sec_flow_totals[k] for k in (2, 1))),
        ("TRUNK-S2-S1", "Sheikh Zayed Transmission Main (S2 -> S1)", "TRUNK-HUB-S2", "TRUNK-HUB-S1", 900, "Sector 1", sec_flow_totals[1]),
    ]
    for pid, pname, fn, tn, diam, sec_lbl, cum_flow in spine_edges:
        eff_flow = min(140.0, max(35.0 if (plant_output_mld > 0.1 and cum_flow > 0.1) else 0.0, cum_flow * 0.25))
        _add_pipe(
            pid, pname, fn, tn,
            [nodes_by_id[fn]["pos_xz"], nodes_by_id[tn]["pos_xz"]],
            role="trunk_main", diameter_mm=diam, sector=sec_lbl,
            color="#38BDF8", flow_lps=eff_flow,
            highlighted=(focus_type == "reservoir" or (focus_type == "sector" and focus_id == sec_lbl)),
        )

    # 3. Map each hydraulic topology downstream node -> actual geometric end node of its producing segment
    # First register every street segment's start & end nodes, and identify which segment hosts each valve V-01..V-20
    seg_geom_xz: dict[str, list[list[float]]] = {}
    seg_start_node: dict[str, str] = {}
    seg_end_node: dict[str, str] = {}
    topo_node_to_geom_node: dict[str, str] = {f"HDR-S{i}": f"TRUNK-HUB-S{i}" for i in range(1, 7)}

    valve_host_segment: dict[str, str] = {}
    for vid, (vlat, vlon) in VALVE_COORDINATES.items():
        best_seg = None
        best_d = float("inf")
        for seg in segs:
            geom = seg.get("geometry") or json.loads(seg.get("geometry_json", "[]"))
            if not geom:
                continue
            d = distance_to_polyline_meters(vlat, vlon, geom)
            if d < best_d:
                best_d = d
                best_seg = seg["segment_id"]
        if best_seg is not None:
            valve_host_segment[vid] = best_seg

    host_seg_to_valve: dict[str, str] = {sid: vid for vid, sid in valve_host_segment.items()}

    for seg in segs:
        seg_id = seg["segment_id"]
        sec = seg["sector"]
        geom = seg.get("geometry") or json.loads(seg.get("geometry_json", "[]"))
        pts = [list(latlon_to_xz(float(p[0]), float(p[1]))) for p in geom]
        seg_geom_xz[seg_id] = pts
        s_nid = f"{seg_id}-START"
        e_nid = f"{seg_id}-END"
        _add_node(s_nid, "pipe_node", float(geom[0][0]), float(geom[0][1]), sector=sec)
        _add_node(e_nid, "pipe_node", float(geom[-1][0]), float(geom[-1][1]), sector=sec)
        seg_start_node[seg_id] = s_nid
        seg_end_node[seg_id] = e_nid

        _, v_topo, _ = SEGMENT_TOPOLOGY.get(seg_id, (f"HDR-S{sec[-1]}", f"N-{seg_id}", None))
        topo_node_to_geom_node[v_topo] = e_nid

    # Register junctions J-01..J-20 as nodes
    for jid, (jlat, jlon) in JUNCTION_COORDINATES.items():
        if jid.startswith("INTAKE-"):
            continue
        _add_node(jid, "junction", jlat, jlon, sector=JUNCTION_SECTOR_MAP.get(jid, "Sector 1"))

    # 4. Build connected sector trunks, valve-split street pipes, and inter-segment connectors
    for seg in segs:
        seg_id = seg["segment_id"]
        sec = seg["sector"]
        diam = int(seg.get("diameter_mm", 500))
        v_id = get_effective_valve_for_segment(seg)
        flow_lps = float(pipe_flows_lps.get(seg_id, 0.0))
        is_rr = seg_id in rerouted
        col = SECTOR_COLORS.get(sec, "#38BDF8")
        is_hl = (
            (seg_id in hl_set)
            or (focus_type == "sector" and focus_id == sec)
            or (focus_type == "valve" and focus_id == v_id)
        )
        u_topo, _, alt_u_topo = SEGMENT_TOPOLOGY.get(seg_id, (f"HDR-S{sec[-1]}", f"N-{seg_id}", None))
        upstream_geom_node = topo_node_to_geom_node.get(u_topo, f"TRUNK-HUB-S{sec[-1]}")
        s_nid = seg_start_node[seg_id]
        e_nid = seg_end_node[seg_id]
        pts = seg_geom_xz[seg_id]

        # Role by hydraulic function & diameter
        is_sector_feeder = u_topo.startswith("HDR-")
        seg_role = "sector_trunk" if (diam >= 600 or is_sector_feeder) else "distribution"

        # Upstream connector from TRUNK-HUB-Si (or parent segment end node) -> segment start node
        # Note: if segment is downstream of a closed valve, dead-end connector also carries 0 L/s
        conn_role = "sector_trunk" if is_sector_feeder else "distribution"
        conn_diam = max(diam, 650) if is_sector_feeder else max(350, min(diam, 500))
        _add_pipe(
            pipe_id=f"FEED-{seg_id}" if is_sector_feeder else f"CONN-{seg_id}",
            name=f"{sec} Feeder -> {seg['name']}" if is_sector_feeder else f"Branch Tie -> {seg['name']}",
            from_node=upstream_geom_node,
            to_node=s_nid,
            points_xz=[nodes_by_id[upstream_geom_node]["pos_xz"], nodes_by_id[s_nid]["pos_xz"]],
            role=conn_role,
            diameter_mm=conn_diam,
            sector=sec,
            color=col,
            flow_lps=flow_lps,
            valve_id=v_id,
            segment_id=seg_id,
            is_rerouted=is_rr,
            highlighted=is_hl,
        )

        # Optional cross-tie ring connector where alt_u exists in SEGMENT_TOPOLOGY (makes ring loops physically connected too!)
        if alt_u_topo and alt_u_topo in topo_node_to_geom_node:
            alt_node = topo_node_to_geom_node[alt_u_topo]
            tie_flow = flow_lps if is_rr else round(flow_lps * 0.25, 1)
            _add_pipe(
                pipe_id=f"TIE-{seg_id}",
                name=f"{sec} Ring Cross-Tie -> {seg['name']}",
                from_node=alt_node,
                to_node=s_nid,
                points_xz=[nodes_by_id[alt_node]["pos_xz"], nodes_by_id[s_nid]["pos_xz"]],
                role="distribution",
                diameter_mm=400,
                sector=sec,
                color=col,
                flow_lps=tie_flow,
                valve_id=v_id,
                segment_id=seg_id,
                is_rerouted=is_rr,
                highlighted=is_hl,
            )

        # If this street segment hosts a valve V-xx on its polyline, split cleanly AT the valve node
        # so the valve sits directly ON the pipe node (start -> V-xx -> end)!
        hosted_vid = host_seg_to_valve.get(seg_id)
        if hosted_vid and hosted_vid in nodes_by_id:
            v_pos = nodes_by_id[hosted_vid]["pos_xz"]
            up_pts, down_pts = _split_polyline_at_coord(pts, v_pos)
            _add_pipe(
                pipe_id=f"{seg_id}-UP",
                name=f"{seg['name']} (approach to {hosted_vid})",
                from_node=s_nid,
                to_node=hosted_vid,
                points_xz=up_pts,
                role=seg_role,
                diameter_mm=diam,
                sector=sec,
                color=col,
                flow_lps=flow_lps,
                valve_id=v_id,
                segment_id=seg_id,
                is_rerouted=is_rr,
                highlighted=is_hl,
            )
            _add_pipe(
                pipe_id=seg_id,
                name=seg["name"],
                from_node=hosted_vid,
                to_node=e_nid,
                points_xz=down_pts,
                role=seg_role,
                diameter_mm=diam,
                sector=sec,
                color=col,
                flow_lps=flow_lps,
                valve_id=v_id,
                segment_id=seg_id,
                is_rerouted=is_rr,
                highlighted=is_hl,
            )
        else:
            _add_pipe(
                pipe_id=seg_id,
                name=seg["name"],
                from_node=s_nid,
                to_node=e_nid,
                points_xz=pts,
                role=seg_role,
                diameter_mm=diam,
                sector=sec,
                color=col,
                flow_lps=flow_lps,
                valve_id=v_id,
                segment_id=seg_id,
                is_rerouted=is_rr,
                highlighted=is_hl,
            )

    connectivity = validate_network_connectivity(list(nodes_by_id.values()), pipes_out)
    return {
        "nodes": list(nodes_by_id.values()),
        "pipes": pipes_out,
        "connectivity": connectivity,
        "plant_flow_lps": plant_flow_lps,
    }


def validate_network_connectivity(nodes: list[dict], pipes: list[dict]) -> dict:
    """
    Pure-Python NetworkX validation check verifying:
    1. Every pipe's from_node and to_node exist in `nodes` AND geometry endpoints match node positions.
    2. DESAL-01 reaches RES-01, and RES-01 reaches every single sector (Sectors 1-6).
    3. Every valve V-01..V-20 exists as a connected node on the pipe graph.
    """
    node_map = {n["node_id"]: n for n in nodes}
    G = nx.DiGraph()
    for nid, n in node_map.items():
        G.add_node(nid, **n)

    missing_endpoints: list[str] = []
    misaligned_pipes: list[str] = []

    for p in pipes:
        pid = p["pipe_id"]
        fn = p["from_node"]
        tn = p["to_node"]
        if fn not in node_map or tn not in node_map:
            missing_endpoints.append(pid)
            continue
        pts = p.get("points_xz") or []
        if len(pts) < 2:
            misaligned_pipes.append(pid)
            continue
        s_node = node_map[fn]["pos_xz"]
        e_node = node_map[tn]["pos_xz"]
        if math.hypot(pts[0][0] - s_node[0], pts[0][1] - s_node[1]) > 0.1 or math.hypot(pts[-1][0] - e_node[0], pts[-1][1] - e_node[1]) > 0.1:
            misaligned_pipes.append(pid)
        G.add_edge(fn, tn, pipe_id=pid, role=p.get("role"), flow_lps=p.get("flow_lps", 0.0))

    desal_reaches_res = ("DESAL-01" in G and "RES-01" in G and nx.has_path(G, "DESAL-01", "RES-01"))
    reachable_from_res = (set(nx.descendants(G, "RES-01")) | {"RES-01"}) if "RES-01" in G else set()
    reachable_sectors = sorted(
        {
            node_map[nid]["sector"]
            for nid in reachable_from_res
            if nid in node_map and str(node_map[nid].get("sector", "")).startswith("Sector ")
        }
    )
    missing_sectors = [f"Sector {i}" for i in range(1, 7) if f"Sector {i}" not in reachable_sectors]

    disconnected_valves = [
        f"V-{i:02d}"
        for i in range(1, 21)
        if f"V-{i:02d}" not in G or G.degree(f"V-{i:02d}") < 2
    ]

    is_connected = (
        len(missing_endpoints) == 0
        and len(misaligned_pipes) == 0
        and desal_reaches_res
        and len(missing_sectors) == 0
        and len(disconnected_valves) == 0
    )

    return {
        "is_connected": is_connected,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "desal_reaches_res": desal_reaches_res,
        "reachable_sectors": reachable_sectors,
        "missing_sectors": missing_sectors,
        "missing_endpoints": missing_endpoints,
        "misaligned_pipes": misaligned_pipes,
        "disconnected_valves": disconnected_valves,
    }

