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

# Strictly THREE standardized pipe classes — fixed diameter, color, 3D radius, and non-colliding elevation layer
PIPE_CLASS_SPECS: dict[str, dict] = {
    "TRUNK_MAIN": {
        "class_id": "TRUNK_MAIN",
        "label": "Trunk Main",
        "diameter_mm": 900,
        "color": "#1D4ED8",      # Dark Blue
        "radius_3d": 0.46,
        "base_y": 2.00,
    },
    "SECTOR_TRUNK": {
        "class_id": "SECTOR_TRUNK",
        "label": "Sector Trunk",
        "diameter_mm": 500,
        "color": "#2563EB",      # Blue
        "radius_3d": 0.30,
        "base_y": 1.15,
    },
    "DISTRIBUTION": {
        "class_id": "DISTRIBUTION",
        "label": "Distribution",
        "diameter_mm": 200,
        "color": "#38BDF8",      # Light Blue
        "radius_3d": 0.18,
        "base_y": 0.50,
    },
}


def classify_pipe_class(diameter_mm: int, role: str = "") -> str:
    """Map database diameter & role into strictly one of the 3 standardized pipe classes."""
    r = str(role or "").lower()
    d = int(diameter_mm or 500)
    if r in ("intake", "product_water", "trunk_main") or d >= 800:
        return "TRUNK_MAIN"
    if r == "sector_trunk" or d >= 500:
        return "SECTOR_TRUNK"
    return "DISTRIBUTION"

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


def _pts_close(p1: list[float], p2: list[float], tol: float = 0.25) -> bool:
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1])) <= tol


def _segments_intersect_2d(
    a1: list[float],
    a2: list[float],
    b1: list[float],
    b2: list[float],
    eps: float = 1e-6,
) -> bool:
    """Return True if 2D segments [a1, a2] and [b1, b2] intersect (including proper crossings & collinear overlaps)."""
    x1, z1 = float(a1[0]), float(a1[1])
    x2, z2 = float(a2[0]), float(a2[1])
    x3, z3 = float(b1[0]), float(b1[1])
    x4, z4 = float(b2[0]), float(b2[1])

    # Bounding-box fast reject
    if max(x1, x2) < min(x3, x4) - eps or max(x3, x4) < min(x1, x2) - eps:
        return False
    if max(z1, z2) < min(z3, z4) - eps or max(z3, z4) < min(z1, z2) - eps:
        return False

    def _orient(px: float, pz: float, qx: float, qz: float, rx: float, rz: float) -> float:
        return (qx - px) * (rz - pz) - (qz - pz) * (rx - px)

    o1 = _orient(x1, z1, x2, z2, x3, z3)
    o2 = _orient(x1, z1, x2, z2, x4, z4)
    o3 = _orient(x3, z3, x4, z4, x1, z1)
    o4 = _orient(x3, z3, x4, z4, x2, z2)

    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True

    def _on_seg(px: float, pz: float, qx: float, qz: float, rx: float, rz: float) -> bool:
        return (
            min(px, rx) - eps <= qx <= max(px, rx) + eps
            and min(pz, rz) - eps <= qz <= max(pz, rz) + eps
        )

    if abs(o1) <= eps and _on_seg(x1, z1, x3, z3, x2, z2):
        return True
    if abs(o2) <= eps and _on_seg(x1, z1, x4, z4, x2, z2):
        return True
    if abs(o3) <= eps and _on_seg(x3, z3, x1, z1, x4, z4):
        return True
    if abs(o4) <= eps and _on_seg(x3, z3, x2, z2, x4, z4):
        return True
    return False


def _choose_clean_orthogonal_route(
    start_pt: list[float],
    end_pt: list[float],
    placed_subsegs: list[tuple[list[float], list[float], str, str]],
    from_node: str,
    to_node: str,
) -> list[list[float]]:
    """
    Build a crisp orthogonal street-grid route (only horizontal X or vertical Z segments with 90-degree elbows)
    from `start_pt` to `end_pt`, selecting the L-route (HV vs VH) that minimizes 2D crossings with existing pipes.
    """
    sx, sz = float(start_pt[0]), float(start_pt[1])
    ex, ez = float(end_pt[0]), float(end_pt[1])
    if abs(sx - ex) <= 0.5 or abs(sz - ez) <= 0.5:
        return [[sx, sz], [ex, ez]]

    candidates = [
        [[sx, sz], [ex, sz], [ex, ez]],  # Horizontal-then-Vertical (90° elbow)
        [[sx, sz], [sx, ez], [ex, ez]],  # Vertical-then-Horizontal (90° elbow)
    ]

    best_route = candidates[0]
    best_score = float("inf")
    for cand in candidates:
        crossings = 0
        for i in range(len(cand) - 1):
            a1, a2 = cand[i], cand[i + 1]
            for b1, b2, b_fn, b_tn in placed_subsegs:
                # Ignore shared endpoint touches with adjacent pipes
                if b_fn in (from_node, to_node) or b_tn in (from_node, to_node):
                    if _pts_close(a1, b1) or _pts_close(a1, b2) or _pts_close(a2, b1) or _pts_close(a2, b2):
                        continue
                if _segments_intersect_2d(a1, a2, b1, b2):
                    crossings += 1
        if crossings < best_score:
            best_score = crossings
            best_route = cand
            if crossings == 0:
                break
    return best_route


def _pipes_intersect_3d_unconnected(p1: dict, p2: dict) -> bool:
    """Return True if two non-identical pipes intersect in 3D at any point other than a shared node."""
    shared_nodes = {p1["from_node"], p1["to_node"]} & {p2["from_node"], p2["to_node"]}
    y1, r1 = float(p1["y_level"]), float(p1["radius_3d"])
    y2, r2 = float(p2["y_level"]), float(p2["radius_3d"])
    # Vertical clearance check: pipes at different depths with daylight gap >= r1 + r2 + 0.04 NEVER intersect!
    if abs(y1 - y2) >= (r1 + r2 + 0.04):
        return False

    pts1 = p1.get("points_xz") or []
    pts2 = p2.get("points_xz") or []
    for i in range(len(pts1) - 1):
        a1, a2 = pts1[i], pts1[i + 1]
        for j in range(len(pts2) - 1):
            b1, b2 = pts2[j], pts2[j + 1]
            if not _segments_intersect_2d(a1, a2, b1, b2):
                continue
            # If they share a graph node, verify this touch is ONLY at the shared endpoint (not a mid-segment crossing)
            if shared_nodes:
                touch_only_at_end = (
                    (_pts_close(a1, b1) or _pts_close(a1, b2) or _pts_close(a2, b1) or _pts_close(a2, b2))
                    and not (
                        abs(a1[0] - a2[0]) <= 0.1 and abs(b1[0] - b2[0]) <= 0.1 and abs(a1[0] - b1[0]) <= 0.1
                        and (min(a1[1], a2[1]) < max(b1[1], b2[1]) - 0.5 and min(b1[1], b2[1]) < max(a1[1], a2[1]) - 0.5)
                    )
                    and not (
                        abs(a1[1] - a2[1]) <= 0.1 and abs(b1[1] - b2[1]) <= 0.1 and abs(a1[1] - b1[1]) <= 0.1
                        and (min(a1[0], a2[0]) < max(b1[0], b2[0]) - 0.5 and min(b1[0], b2[0]) < max(a1[0], a2[0]) - 0.5)
                    )
                )
                if touch_only_at_end:
                    continue
            return True
    return False


def _resolve_pipe_crossings_3d(pipes_out: list[dict]) -> None:
    """
    Guarantee ZERO unconnected 3D pipe-pipe intersections:
    Pipes already sit on distinct base elevations per class (TRUNK_MAIN=2.00, SECTOR_TRUNK=1.15, DISTRIBUTION=0.50).
    If any two pipes still intersect in 3D (e.g. same-class crossing), assign a clean vertical overpass depth offset.
    """
    candidate_offsets = [0.0, 0.55, 1.10, 1.65, 2.20, 2.75, 3.30, 3.85]
    for idx, p in enumerate(pipes_out):
        base_y = float(PIPE_CLASS_SPECS[p["pipe_class"]]["base_y"])
        for off in candidate_offsets:
            p["y_level"] = round(base_y + off, 3)
            p["has_crossing_offset"] = off > 0.0
            conflict = False
            for j in range(idx):
                if _pipes_intersect_3d_unconnected(p, pipes_out[j]):
                    conflict = True
                    break
            if not conflict:
                break


def _compute_fittings_and_reducers(nodes_by_id: dict[str, dict], pipes_out: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Build explicit junction fittings (T-piece / Cross / Riser) and tapered Reducers at EVERY node
    where connected pipes change diameter class (e.g., TRUNK_MAIN 900mm -> SECTOR_TRUNK 500mm -> DISTRIBUTION 200mm).
    """
    incident: dict[str, list[dict]] = {nid: [] for nid in nodes_by_id}
    for p in pipes_out:
        if p["from_node"] in incident:
            incident[p["from_node"]].append({"pipe": p, "is_outgoing": True})
        if p["to_node"] in incident:
            incident[p["to_node"]].append({"pipe": p, "is_outgoing": False})

    fittings: list[dict] = []
    reducers: list[dict] = []

    for nid, inc_list in incident.items():
        if not inc_list:
            continue
        nd = nodes_by_id[nid]
        deg = len(inc_list)
        ys = [float(item["pipe"]["y_level"]) for item in inc_list]
        radii = [float(item["pipe"]["radius_3d"]) for item in inc_list]
        classes = {item["pipe"]["pipe_class"] for item in inc_list}
        max_r = max(radii)
        y_min, y_max = min(ys), max(ys)

        fitting_type = "cross" if deg >= 4 else ("tee" if deg == 3 else ("inline" if deg == 2 else "endcap"))
        fittings.append(
            {
                "node_id": nid,
                "kind": nd["kind"],
                "sector": nd["sector"],
                "pos_xz": list(nd["pos_xz"]),
                "degree": deg,
                "fitting_type": fitting_type,
                "y_min": round(y_min, 3),
                "y_max": round(y_max, 3),
                "radius_3d": round(max_r, 3),
                "label": nid if (nd["kind"] in ("junction", "trunk_hub") or deg >= 3) else "",
            }
        )

        # Wherever >1 pipe class meets at this node, emit a tapered reducer cone transitioning from the larger pipe to each smaller pipe!
        if len(classes) > 1:
            largest_item = max(inc_list, key=lambda it: PIPE_CLASS_SPECS[it["pipe"]["pipe_class"]]["diameter_mm"])
            from_cls = largest_item["pipe"]["pipe_class"]
            from_spec = PIPE_CLASS_SPECS[from_cls]
            from_y = float(largest_item["pipe"]["y_level"])
            for item in inc_list:
                to_p = item["pipe"]
                to_cls = to_p["pipe_class"]
                if to_cls == from_cls:
                    continue
                to_spec = PIPE_CLASS_SPECS[to_cls]
                to_y = float(to_p["y_level"])
                pts = to_p["points_xz"]
                if item["is_outgoing"] and len(pts) >= 2:
                    dx, dz = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
                elif (not item["is_outgoing"]) and len(pts) >= 2:
                    dx, dz = pts[-2][0] - pts[-1][0], pts[-2][1] - pts[-1][1]
                else:
                    dx, dz = 1.0, 0.0
                dlen = max(1e-6, math.hypot(dx, dz))
                reducers.append(
                    {
                        "node_id": nid,
                        "pipe_id": to_p["pipe_id"],
                        "pos_xz": list(nd["pos_xz"]),
                        "from_class": from_cls,
                        "to_class": to_cls,
                        "from_diameter_mm": from_spec["diameter_mm"],
                        "to_diameter_mm": to_spec["diameter_mm"],
                        "from_radius": from_spec["radius_3d"],
                        "to_radius": to_spec["radius_3d"],
                        "from_y": round(from_y, 3),
                        "to_y": round(to_y, 3),
                        "dir_xz": [round(dx / dlen, 4), round(dz / dlen, 4)],
                    }
                )

    return fittings, reducers


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
    Build ONE 100% connected, clean orthogonal street-grid pipe network from:
        Sea Intakes (INTAKE-01/02) -> DESAL-01 -> Product Pipe -> RES-01 -> Main Trunk (TRUNK-HUB-S6..S1)
        -> Sector Trunks -> Isolation Valves (V-01..V-20 placed ON exact nodes)
        -> All 28 Dubai Street Pipes & Junctions -> Distribution Branches.

    Guarantees:
    - Strictly 3 pipe classes: TRUNK_MAIN (900mm dark blue), SECTOR_TRUNK (500mm blue), DISTRIBUTION (200mm light blue).
    - Every connected run keeps the same class end to end; tapered reducers sit at EVERY class change.
    - Orthogonal 90-degree street-aligned segments only; ZERO unconnected 3D pipe-pipe crossings.
    - Real junction fittings (T-piece / Cross) at connected nodes; valves sized to their pipe.
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
    _add_node("INTAKE-01", "intake", *JUNCTION_COORDINATES["INTAKE-01"], sector="Plant")
    _add_node("INTAKE-02", "intake", *JUNCTION_COORDINATES["INTAKE-02"], sector="Plant")
    _add_node("DESAL-01", "plant", *PLANT_DESAL01_LATLON, sector="Plant")
    _add_node("RES-01", "reservoir", *RESERVOIR_RES01_LATLON, sector="Reservoir")

    for hub_id, (hlat, hlon) in TRUNK_HUB_LATLON.items():
        sec_num = hub_id[-1]
        _add_node(hub_id, "trunk_hub", hlat, hlon, sector=f"Sector {sec_num}")

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
    placed_subsegs: list[tuple[list[float], list[float], str, str]] = []

    def _add_pipe(
        pipe_id: str,
        name: str,
        from_node: str,
        to_node: str,
        role: str,
        raw_diameter_mm: int,
        sector: str,
        flow_lps: float,
        pipe_class_override: Optional[str] = None,
        valve_id: Optional[str] = None,
        segment_id: Optional[str] = None,
        is_rerouted: bool = False,
        highlighted: bool = False,
    ) -> None:
        start_pt = list(nodes_by_id[from_node]["pos_xz"])
        end_pt = list(nodes_by_id[to_node]["pos_xz"])
        clean_pts = _choose_clean_orthogonal_route(start_pt, end_pt, placed_subsegs, from_node, to_node)
        for i in range(len(clean_pts) - 1):
            placed_subsegs.append((clean_pts[i], clean_pts[i + 1], from_node, to_node))

        p_cls = pipe_class_override or classify_pipe_class(raw_diameter_mm, role)
        spec = PIPE_CLASS_SPECS[p_cls]
        elbows = [list(p) for p in clean_pts[1:-1]]
        f_val = round(max(0.0, float(flow_lps)), 1)
        pipe_status = "NO_FLOW" if f_val <= 0.05 else ("REROUTED" if is_rerouted else "FLOWING")

        run_id_val = pipe_id if pipe_id.startswith("TIE-") else (segment_id or pipe_id)

        pipes_out.append(
            {
                "pipe_id": pipe_id,
                "segment_id": segment_id or pipe_id,
                "run_id": run_id_val,
                "name": name,
                "from_node": from_node,
                "to_node": to_node,
                "role": role,
                "pipe_class": p_cls,
                "class_label": spec["label"],
                "diameter_mm": spec["diameter_mm"],
                "db_diameter_mm": int(raw_diameter_mm),
                "radius_3d": spec["radius_3d"],
                "y_level": spec["base_y"],
                "has_crossing_offset": False,
                "is_trunk": p_cls in ("TRUNK_MAIN", "SECTOR_TRUNK"),
                "sector": sector,
                "color": spec["color"],
                "isolation_valve_id": valve_id,
                "flow_lps": f_val,
                "status": pipe_status,
                "rerouted": bool(is_rerouted),
                "highlighted": bool(highlighted),
                "points_xz": clean_pts,
                "start_xz": start_pt,
                "end_xz": end_pt,
                "elbows_xz": elbows,
            }
        )

    # 2. Main Backbone Edges (TRUNK_MAIN — 900mm dark blue): Sea Intakes -> DESAL-01 -> RES-01 -> Spine Hubs
    plant_flow_lps = round((float(plant_output_mld) * 1_000_000.0) / 86_400.0, 1)
    backbone_lps = round(max(0.0, float(plant_output_mld)) * 1.4, 1)
    int1_lps = backbone_lps if intake2_isolated else round(backbone_lps * 0.55, 1)
    int2_lps = 0.0 if intake2_isolated else round(backbone_lps * 0.55, 1)

    _add_pipe(
        "PIPE-INTAKE-01", "Sea Intake 1 Main", "INTAKE-01", "DESAL-01",
        role="intake", raw_diameter_mm=900, sector="Plant",
        flow_lps=int1_lps, pipe_class_override="TRUNK_MAIN", highlighted=(focus_type in ("plant", "intake")),
    )
    _add_pipe(
        "PIPE-INTAKE-02", "Sea Intake 2 Main", "INTAKE-02", "DESAL-01",
        role="intake", raw_diameter_mm=900, sector="Plant",
        flow_lps=int2_lps, pipe_class_override="TRUNK_MAIN", highlighted=(focus_type in ("plant", "intake")),
    )
    _add_pipe(
        "PIPE-PRODUCT-01", "DESAL-01 Product Water Trunk -> RES-01", "DESAL-01", "RES-01",
        role="product_water", raw_diameter_mm=900, sector="Reservoir",
        flow_lps=backbone_lps, pipe_class_override="TRUNK_MAIN", highlighted=(focus_type in ("plant", "reservoir", "reservoir_inside")),
    )

    sec_flow_totals: dict[int, float] = {i: 0.0 for i in range(1, 7)}
    for seg in segs:
        sec_str = str(seg.get("sector", "Sector 1"))
        s_num = int(sec_str.split()[-1]) if sec_str.split()[-1].isdigit() else 1
        sec_flow_totals[s_num] += float(pipe_flows_lps.get(seg["segment_id"], 0.0))

    spine_edges = [
        ("TRUNK-RES-S6", "RES-01 South Outlet Trunk -> Sector 6", "RES-01", "TRUNK-HUB-S6", 900, "Sector 6", sec_flow_totals[6]),
        ("TRUNK-RES-S5", "RES-01 North Outlet Trunk -> Sector 5", "RES-01", "TRUNK-HUB-S5", 900, "Sector 5", sum(sec_flow_totals[k] for k in (5, 4, 3, 2, 1))),
        ("TRUNK-S5-S4", "Main Transmission Trunk (S5 -> S4)", "TRUNK-HUB-S5", "TRUNK-HUB-S4", 900, "Sector 4", sum(sec_flow_totals[k] for k in (4, 3, 2, 1))),
        ("TRUNK-S4-S3", "Main Transmission Trunk (S4 -> S3)", "TRUNK-HUB-S4", "TRUNK-HUB-S3", 900, "Sector 3", sum(sec_flow_totals[k] for k in (3, 2, 1))),
        ("TRUNK-S3-S2", "Main Transmission Trunk (S3 -> S2)", "TRUNK-HUB-S3", "TRUNK-HUB-S2", 900, "Sector 2", sum(sec_flow_totals[k] for k in (2, 1))),
        ("TRUNK-S2-S1", "Main Transmission Trunk (S2 -> S1)", "TRUNK-HUB-S2", "TRUNK-HUB-S1", 900, "Sector 1", sec_flow_totals[1]),
    ]
    for pid, pname, fn, tn, diam, sec_lbl, cum_flow in spine_edges:
        eff_flow = min(140.0, max(35.0 if (plant_output_mld > 0.1 and cum_flow > 0.1) else 0.0, cum_flow * 0.25))
        _add_pipe(
            pid, pname, fn, tn,
            role="trunk_main", raw_diameter_mm=diam, sector=sec_lbl,
            flow_lps=eff_flow, pipe_class_override="TRUNK_MAIN",
            highlighted=(focus_type == "reservoir" or (focus_type == "sector" and focus_id == sec_lbl)),
        )

    # 3. Register street segment start/end nodes and map valves + junctions cleanly onto host segments
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

    # Map each street segment -> its primary junction node (e.g. SEG-S4-01 -> J-14)
    seg_to_junction: dict[str, str] = {}
    for seg in segs:
        seg_id = seg["segment_id"]
        sec = seg["sector"]
        geom = seg.get("geometry") or json.loads(seg.get("geometry_json", "[]"))
        s_nid = f"{seg_id}-START"
        e_nid = f"{seg_id}-END"
        _add_node(s_nid, "pipe_node", float(geom[0][0]), float(geom[0][1]), sector=sec)
        _add_node(e_nid, "pipe_node", float(geom[-1][0]), float(geom[-1][1]), sector=sec)
        seg_start_node[seg_id] = s_nid
        seg_end_node[seg_id] = e_nid

        rel = [x.strip() for x in str(seg.get("related_sensor_ids", "")).split(",") if x.strip()]
        if rel and rel[0].startswith("P-"):
            cand_j = rel[0].replace("P-", "J-")
            if cand_j in JUNCTION_COORDINATES and cand_j not in seg_to_junction.values():
                seg_to_junction[seg_id] = cand_j
                jlat, jlon = JUNCTION_COORDINATES[cand_j]
                _add_node(cand_j, "junction", jlat, jlon, sector=sec)

        _, v_topo, _ = SEGMENT_TOPOLOGY.get(seg_id, (f"HDR-S{sec[-1]}", f"N-{seg_id}", None))
        topo_node_to_geom_node[v_topo] = e_nid

    # Register any remaining junctions J-01..J-20
    for jid, (jlat, jlon) in JUNCTION_COORDINATES.items():
        if jid.startswith("INTAKE-"):
            continue
        _add_node(jid, "junction", jlat, jlon, sector=JUNCTION_SECTOR_MAP.get(jid, "Sector 1"))

    # 4. Build connected sector trunks & distribution runs (consistent class end-to-end along every run!)
    for seg in segs:
        seg_id = seg["segment_id"]
        sec = seg["sector"]
        diam = int(seg.get("diameter_mm", 500))
        v_id = get_effective_valve_for_segment(seg)
        flow_lps = float(pipe_flows_lps.get(seg_id, 0.0))
        is_rr = seg_id in rerouted
        is_hl = (
            (seg_id in hl_set)
            or (focus_type == "sector" and focus_id == sec)
            or (focus_type == "valve" and focus_id == v_id)
        )
        u_topo, _, alt_u_topo = SEGMENT_TOPOLOGY.get(seg_id, (f"HDR-S{sec[-1]}", f"N-{seg_id}", None))
        upstream_geom_node = topo_node_to_geom_node.get(u_topo, f"TRUNK-HUB-S{sec[-1]}")
        s_nid = seg_start_node[seg_id]
        e_nid = seg_end_node[seg_id]

        # Consistent run class determined from DB diameter & hydraulic role (SECTOR_TRUNK=500mm or DISTRIBUTION=200mm)
        is_sector_feeder = u_topo.startswith("HDR-")
        run_class = "SECTOR_TRUNK" if (is_sector_feeder or diam >= 500) else "DISTRIBUTION"
        run_role = "sector_trunk" if run_class == "SECTOR_TRUNK" else "distribution"

        # Upstream connector keeps the EXACT same run_class so reducers occur cleanly at hub/branch nodes!
        _add_pipe(
            pipe_id=f"FEED-{seg_id}" if is_sector_feeder else f"CONN-{seg_id}",
            name=f"{sec} Trunk Feeder -> {seg['name']}" if is_sector_feeder else f"{sec} Distribution Branch -> {seg['name']}",
            from_node=upstream_geom_node,
            to_node=s_nid,
            role=run_role,
            raw_diameter_mm=diam,
            sector=sec,
            flow_lps=flow_lps,
            pipe_class_override=run_class,
            valve_id=v_id,
            segment_id=seg_id,
            is_rerouted=is_rr,
            highlighted=is_hl,
        )

        # Optional distribution ring cross-tie where alt_u exists in SEGMENT_TOPOLOGY
        if alt_u_topo and alt_u_topo in topo_node_to_geom_node:
            alt_node = topo_node_to_geom_node[alt_u_topo]
            tie_flow = flow_lps if is_rr else round(flow_lps * 0.25, 1)
            _add_pipe(
                pipe_id=f"TIE-{seg_id}",
                name=f"{sec} Ring Cross-Tie -> {seg['name']}",
                from_node=alt_node,
                to_node=s_nid,
                role="distribution",
                raw_diameter_mm=200,
                sector=sec,
                flow_lps=tie_flow,
                pipe_class_override="DISTRIBUTION",
                valve_id=v_id,
                segment_id=seg_id,
                is_rerouted=is_rr,
                highlighted=is_hl,
            )

        # Route street segment through its hosted valve (V-xx) and hosted junction (J-xx) cleanly with the SAME run_class!
        hosted_vid = host_seg_to_valve.get(seg_id)
        hosted_jid = seg_to_junction.get(seg_id)

        chain_nodes = [s_nid]
        if hosted_vid and hosted_vid in nodes_by_id:
            chain_nodes.append(hosted_vid)
        if hosted_jid and hosted_jid in nodes_by_id and hosted_jid not in chain_nodes:
            chain_nodes.append(hosted_jid)
        chain_nodes.append(e_nid)

        for idx_c in range(len(chain_nodes) - 1):
            fn = chain_nodes[idx_c]
            tn = chain_nodes[idx_c + 1]
            # Keep the primary segment_id on the final downstream leg so tests looking for pipe_id == seg_id find it cleanly
            is_main_leg = (idx_c == len(chain_nodes) - 2)
            pid_c = seg_id if is_main_leg else (f"{seg_id}-UP" if idx_c == 0 else f"{seg_id}-MID")
            _add_pipe(
                pipe_id=pid_c,
                name=seg["name"],
                from_node=fn,
                to_node=tn,
                role=run_role,
                raw_diameter_mm=diam,
                sector=sec,
                flow_lps=flow_lps,
                pipe_class_override=run_class,
                valve_id=v_id,
                segment_id=seg_id,
                is_rerouted=is_rr,
                highlighted=is_hl,
            )

    # 5. Resolve ANY remaining 3D crossings between unconnected pipes via clean elevation separation
    _resolve_pipe_crossings_3d(pipes_out)

    # 6. Build explicit Junction Fittings (Tee / Cross / Riser) & Tapered Reducers at every class change
    fittings, reducers = _compute_fittings_and_reducers(nodes_by_id, pipes_out)

    # Attach exact pipe radius & y_level onto each valve node so valves size & sit strictly ON their pipe!
    for p in pipes_out:
        for end_nid in (p["from_node"], p["to_node"]):
            nd = nodes_by_id.get(end_nid)
            if nd and nd.get("kind") == "valve":
                nd["pipe_radius_3d"] = p["radius_3d"]
                nd["pipe_y_level"] = p["y_level"]
                nd["pipe_class"] = p["pipe_class"]

    connectivity = validate_network_connectivity(list(nodes_by_id.values()), pipes_out)
    crossings_check = validate_no_unconnected_crossings(pipes_out)
    reducers_check = validate_reducers_at_class_changes(list(nodes_by_id.values()), pipes_out, reducers)

    return {
        "nodes": list(nodes_by_id.values()),
        "pipes": pipes_out,
        "fittings": fittings,
        "reducers": reducers,
        "pipe_classes": PIPE_CLASS_SPECS,
        "connectivity": connectivity,
        "crossings_check": crossings_check,
        "reducers_check": reducers_check,
        "plant_flow_lps": plant_flow_lps,
    }


def validate_no_unconnected_crossings(pipes: list[dict]) -> dict:
    """Verify ZERO pairs of pipes intersect in 3D except at shared graph nodes."""
    conflicts: list[tuple[str, str]] = []
    n = len(pipes)
    for i in range(n):
        for j in range(i + 1, n):
            if _pipes_intersect_3d_unconnected(pipes[i], pipes[j]):
                conflicts.append((pipes[i]["pipe_id"], pipes[j]["pipe_id"]))
    return {
        "has_unconnected_crossings": len(conflicts) > 0,
        "crossing_count": len(conflicts),
        "conflicts": conflicts,
    }


def validate_reducers_at_class_changes(nodes: list[dict], pipes: list[dict], reducers: list[dict]) -> dict:
    """Verify EVERY node where connected pipes change class (e.g. TRUNK_MAIN -> SECTOR_TRUNK -> DISTRIBUTION) has a reducer."""
    node_classes: dict[str, set[str]] = {n["node_id"]: set() for n in nodes}
    for p in pipes:
        if p["from_node"] in node_classes:
            node_classes[p["from_node"]].add(p["pipe_class"])
        if p["to_node"] in node_classes:
            node_classes[p["to_node"]].add(p["pipe_class"])

    reducer_nodes = {r["node_id"] for r in reducers}
    missing: list[str] = []
    class_change_nodes: list[str] = []
    for nid, clss in node_classes.items():
        if len(clss) > 1:
            class_change_nodes.append(nid)
            if nid not in reducer_nodes:
                missing.append(nid)
    return {
        "all_class_changes_have_reducers": len(missing) == 0,
        "class_change_node_count": len(class_change_nodes),
        "reducer_count": len(reducers),
        "missing_reducer_nodes": missing,
    }


def _segment_intersects_aabb(
    a1: list[float],
    a2: list[float],
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
) -> bool:
    """Return True if 2D segment [a1, a2] intersects or enters axis-aligned box [min_x, max_x] x [min_z, max_z]."""
    x1, z1 = float(a1[0]), float(a1[1])
    x2, z2 = float(a2[0]), float(a2[1])
    if max(x1, x2) < min_x or min(x1, x2) > max_x or max(z1, z2) < min_z or min(z1, z2) > max_z:
        return False
    if (min_x <= x1 <= max_x and min_z <= z1 <= max_z) or (min_x <= x2 <= max_x and min_z <= z2 <= max_z):
        return True
    box_edges = [
        ([min_x, min_z], [max_x, min_z]),
        ([max_x, min_z], [max_x, max_z]),
        ([max_x, max_z], [min_x, max_z]),
        ([min_x, max_z], [min_x, min_z]),
    ]
    return any(_segments_intersect_2d(a1, a2, e1, e2) for e1, e2 in box_edges)


def validate_pipes_clear_buildings(pipes: list[dict], buildings: list[dict], margin_m: float = 0.0) -> dict:
    """Verify NO pipe segment passes through ANY building footprint [x - w/2, x + w/2] x [z - d/2, z + d/2]."""
    collisions: list[dict] = []
    for p in pipes:
        pts = p.get("points_xz") or []
        for i in range(len(pts) - 1):
            a1, a2 = pts[i], pts[i + 1]
            for b_idx, b in enumerate(buildings):
                hx = float(b["w"]) * 0.5 + margin_m
                hz = float(b["d"]) * 0.5 + margin_m
                bx, bz = float(b["x"]), float(b["z"])
                if _segment_intersects_aabb(a1, a2, bx - hx, bx + hx, bz - hz, bz + hz):
                    collisions.append({"pipe_id": p["pipe_id"], "building_index": b_idx, "sector": b.get("sector")})
                    break
    return {
        "all_pipes_clear_buildings": len(collisions) == 0,
        "collision_count": len(collisions),
        "collisions": collisions[:20],
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

