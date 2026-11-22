"""Tests for Part 2 — Clean 3D Pipe Network (street-aligned orthogonal pipes, 3 classes, reducers, fittings, clearance)."""
from pathlib import Path
import pytest

from database.seed_data import seed_database
from digital_twin import (
    PIPE_CLASS_SPECS,
    build_3d_scene_state,
    validate_network_connectivity,
    validate_no_unconnected_crossings,
    validate_pipes_clear_buildings,
    validate_reducers_at_class_changes,
)


@pytest.fixture
def clean_db(tmp_path):
    db_file = str(tmp_path / "clean_pipe_net.db")
    seed_database(db_path=db_file)
    return db_file


def test_every_pipe_endpoint_is_a_node_and_orthogonal(clean_db):
    """Every pipe endpoint must match a graph node and every sub-segment must be strictly 90° street-aligned."""
    state = build_3d_scene_state(db_path=clean_db)
    cn = state["connected_network"]

    conn = validate_network_connectivity(cn["nodes"], cn["pipes"])
    assert conn["is_connected"] is True
    assert conn["missing_endpoints"] == []
    assert conn["misaligned_pipes"] == []
    assert conn["disconnected_valves"] == []

    # Verify strictly orthogonal (axis-aligned horizontal X or vertical Z) segments only
    for p in cn["pipes"]:
        pts = p["points_xz"]
        assert len(pts) >= 2
        for a, b in zip(pts, pts[1:]):
            dx = abs(float(a[0]) - float(b[0]))
            dz = abs(float(a[1]) - float(b[1]))
            assert dx <= 0.5 or dz <= 0.5, f"Non-orthogonal diagonal pipe {p['pipe_id']}: {a} -> {b}"


def test_zero_unconnected_pipe_pipe_intersections_in_3d(clean_db):
    """No two pipes may intersect in 3D except at shared graph nodes."""
    state = build_3d_scene_state(db_path=clean_db)
    cn = state["connected_network"]

    check = validate_no_unconnected_crossings(cn["pipes"])
    assert check["has_unconnected_crossings"] is False
    assert check["crossing_count"] == 0
    assert check["conflicts"] == []


def test_zero_pipes_pass_through_any_building_footprint(clean_db):
    """Buildings must sit cleanly inside blocks between streets with zero pipe-building collisions."""
    for low_q in (False, True):
        state = build_3d_scene_state(db_path=clean_db, low_quality=low_q)
        cn = state["connected_network"]
        buildings = state["buildings"]
        assert len(buildings) > 0

        clearance = validate_pipes_clear_buildings(cn["pipes"], buildings, margin_m=0.0)
        assert clearance["all_pipes_clear_buildings"] is True
        assert clearance["collision_count"] == 0
        assert cn["building_clearance"]["all_pipes_clear_buildings"] is True


def test_three_pipe_classes_consistent_runs_and_reducers_at_every_class_change(clean_db):
    """Strictly 3 pipe classes only; consistent diameter along every run; tapered reducer at EVERY class change."""
    state = build_3d_scene_state(db_path=clean_db)
    cn = state["connected_network"]

    expected_classes = {"TRUNK_MAIN", "SECTOR_TRUNK", "DISTRIBUTION"}
    seen_classes = {p["pipe_class"] for p in cn["pipes"]}
    assert seen_classes == expected_classes

    # Verify exact fixed diameters & colors per class
    for p in cn["pipes"]:
        spec = PIPE_CLASS_SPECS[p["pipe_class"]]
        assert p["diameter_mm"] == spec["diameter_mm"]
        assert p["color"] == spec["color"]
        assert p["radius_3d"] == spec["radius_3d"]

    # Every connected run_id keeps the exact same diameter & class end to end
    runs: dict[str, set[tuple[str, int]]] = {}
    for p in cn["pipes"]:
        runs.setdefault(p["run_id"], set()).add((p["pipe_class"], p["diameter_mm"]))
    inconsistent = {rid: vals for rid, vals in runs.items() if len(vals) > 1}
    assert inconsistent == {}

    # Every node where pipe class changes has a tapered reducer
    red_check = validate_reducers_at_class_changes(cn["nodes"], cn["pipes"], cn["reducers"])
    assert red_check["all_class_changes_have_reducers"] is True
    assert red_check["missing_reducer_nodes"] == []
    assert red_check["reducer_count"] > 0

    # Real junction fittings (tee/cross) exist at multi-way nodes
    fitting_types = {f["fitting_type"] for f in cn["fittings"]}
    assert "tee" in fitting_types or "cross" in fitting_types

    # Every valve inherits its host pipe radius & elevation
    for v in state["valves"]:
        assert v["pipe_class"] in expected_classes
        assert v["pipe_radius_3d"] == PIPE_CLASS_SPECS[v["pipe_class"]]["radius_3d"]
        assert v["pipe_y_level"] >= PIPE_CLASS_SPECS["DISTRIBUTION"]["base_y"]


def test_res01_reaches_every_sector_and_closing_trunk_valve_stops_downstream_flow(clean_db):
    """RES-01 reaches all 6 sectors; closing V-12 or V-17 stops downstream flow to 0.0 L/s."""
    baseline = build_3d_scene_state(db_path=clean_db)
    conn = baseline["connected_network"]["connectivity"]
    assert conn["desal_reaches_res"] is True
    assert conn["reachable_sectors"] == [f"Sector {i}" for i in range(1, 7)]

    base_s4 = next(p for p in baseline["connected_network"]["pipes"] if p["pipe_id"] == "SEG-S4-01")
    base_s4_down = next(p for p in baseline["connected_network"]["pipes"] if p["pipe_id"] == "SEG-S4-05")
    assert base_s4["flow_lps"] > 0.0
    assert base_s4_down["flow_lps"] > 0.0

    closed_v12 = build_3d_scene_state(db_path=clean_db, valve_override={"V-12": "CLOSED"})
    after_s4 = next(p for p in closed_v12["connected_network"]["pipes"] if p["pipe_id"] == "SEG-S4-01")
    after_s4_down = next(p for p in closed_v12["connected_network"]["pipes"] if p["pipe_id"] == "SEG-S4-05")
    assert after_s4["flow_lps"] == 0.0
    assert after_s4["status"] == "NO_FLOW"
    assert after_s4_down["flow_lps"] == 0.0

    closed_v17 = build_3d_scene_state(db_path=clean_db, valve_override={"V-17": "CLOSED"})
    after_s6 = next(p for p in closed_v17["connected_network"]["pipes"] if p["pipe_id"] == "SEG-S6-01")
    assert after_s6["flow_lps"] == 0.0
    assert after_s6["status"] == "NO_FLOW"


def test_twin_frontend_has_clean_pipe_classes_reducers_fittings_and_arrows():
    """Verify the 3D frontend renders the 3 pipe classes, reducers, T/Cross fittings, flow arrows, and run hover."""
    html = Path("app/components/twin_frontend/index.html").read_text(encoding="utf-8")
    for token in (
        "Pipe Classes (3)",
        "Trunk Main",
        "Sector Trunk",
        "Distribution",
        "#1D4ED8",
        "#2563EB",
        "#38BDF8",
        "#67E8F9",
        "Tapered Reducer",
        "Junction Fitting (T / Cross)",
        "Flow Arrow",
        "arrowInst",
        "setHoveredRun",
        "pipeRunId",
    ):
        assert token in html, f"Expected token {token!r} in twin_frontend/index.html"
    assert "carto" not in html.lower()
