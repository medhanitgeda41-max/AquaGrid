"""Pure-Python directed-graph hydraulic propagation model using NetworkX.

Models water flow from DESAL-01 -> RES-01 -> sector distribution headers -> pipe segments.
When a valve is CLOSED, downstream pipes lose flow (or receive reduced rerouted flow ONLY
where a valid alternative open path exists), and isolated leaks stop losing water.
Read-only calculation only; never mutates database state or safety guards.
"""
from typing import Optional
import networkx as nx

from database.connection import get_db_connection
from database.dubai_network import DUBAI_PIPE_SEGMENTS

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
