"""Tests for Part 1 (3D Digital Twin scene state, deterministic city, Three.js HTML & WebGL 2D fallback)."""
import json
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from database.scenarios import inject_scenario_1, reset_database
from digital_twin.scene_state import (
    NETWORK_CENTER_LATLON,
    build_3d_scene_state,
    latlon_to_local_xz,
)


@pytest.fixture(autouse=True)
def clean_db():
    reset_database()
    yield
    reset_database()


def test_scene_state_builds_complete_json_with_local_xz_metres():
    """build_3d_scene_state produces one valid JSON state with all network entities in local x/z metres."""
    inject_scenario_1()  # adds active J-14 leak and quarantined P-17 sensor
    state = build_3d_scene_state(focus="all", current_hour=14, low_quality=False)

    # Must round-trip cleanly through JSON
    encoded = json.dumps(state)
    parsed = json.loads(encoded)

    assert len(parsed["sectors"]) == 6
    assert [s["color_name"] for s in parsed["sectors"]] == ["blue", "green", "orange", "purple", "teal", "red"]
    assert len(parsed["pipes"]) == 28
    assert all("start_xz" in p and "end_xz" in p and "flow_lps" in p for p in parsed["pipes"])
    assert any(p["is_trunk"] for p in parsed["pipes"])
    assert len(parsed["valves"]) == 20
    assert all(v["status"] in ("OPEN", "CLOSED") for v in parsed["valves"])
    assert len(parsed["junctions"]) == 22
    assert len(parsed["sensors"]) == 22
    assert any(s["sensor_id"] == "P-17" and s["quarantined"] is True for s in parsed["sensors"])
    assert len(parsed["leaks"]) >= 1
    assert any(lk["active"] for lk in parsed["leaks"])
    assert parsed["reservoir"]["id"] == "RES-01"
    assert 0.0 <= parsed["reservoir"]["level_percent"] <= 100.0
    assert parsed["plant"]["id"] == "DESAL-01"
    assert parsed["plant"]["output_mld"] > 0

    # Centre maps to (0, 0) metres; western coastline/plant sit to the west (-X)
    cx, cz = latlon_to_local_xz(*NETWORK_CENTER_LATLON)
    assert cx == 0.0 and cz == 0.0
    assert parsed["plant"]["pos_xz"][0] < 0.0
    assert len(parsed["coastline_xz"]) >= 5


def test_procedural_city_is_deterministic_and_low_quality_reduces_buildings():
    """Procedural city buildings inside each sector are identical every time, and Low quality reduces count."""
    s1 = build_3d_scene_state(focus="all", low_quality=False)
    s2 = build_3d_scene_state(focus="all", low_quality=False)
    s_low = build_3d_scene_state(focus="all", low_quality=True)

    assert s1["buildings"] == s2["buildings"]
    assert len(s1["buildings"]) > 150
    assert 0 < len(s_low["buildings"]) < len(s1["buildings"])


def test_focus_targets_compute_highlight_and_camera_target():
    """Focusing on Sector 4, V-12, reservoir, or plant updates highlight and camera_target accurately."""
    s4 = build_3d_scene_state(focus="Sector 4")
    assert s4["highlight"]["type"] == "sector"
    assert s4["highlight"]["id"] == "Sector 4"
    assert s4["camera_target"]["label"] == "Sector 4"

    sv = build_3d_scene_state(focus="V-12")
    assert sv["highlight"]["type"] == "valve"
    assert sv["highlight"]["id"] == "V-12"
    assert "V-12" in sv["camera_target"]["label"]

    sr = build_3d_scene_state(focus="reservoir")
    assert sr["highlight"]["type"] == "reservoir"
    assert sr["camera_target"]["target_xz"] == sr["reservoir"]["pos_xz"]

    sp = build_3d_scene_state(focus="plant")
    assert sp["highlight"]["type"] == "plant"
    assert sp["camera_target"]["target_xz"] == sp["plant"]["pos_xz"]


def test_3d_html_uses_https_jsdelivr_cdn_instanced_mesh_and_webgl_2d_fallback():
    """Rendered 3D HTML loads Three.js + OrbitControls over HTTPS jsDelivr, zero CARTO/API keys, with WebGL->2D OSM fallback."""
    from components.digital_twin_3d import build_3d_digital_twin_html

    state = build_3d_scene_state(focus="Sector 4")
    html_out = build_3d_digital_twin_html(state, widget_id="test_twin_1", height=480)

    assert "https://cdn.jsdelivr.net/npm/three@" in html_out
    assert "OrbitControls.js" in html_out
    assert "InstancedMesh" in html_out
    assert "Reset view" in html_out
    assert "detectWebGL" in html_out
    assert "render2DFallback" in html_out
    assert "https://tile.openstreetmap.org/{z}/{x}/{y}.png" in html_out
    assert "carto" not in html_out.lower()

    html_no_webgl = build_3d_digital_twin_html(state, widget_id="test_twin_fb", force_no_webgl=True)
    assert "window.__AQUAGRID_FORCE_NO_WEBGL__ = true;" in html_no_webgl


def test_hydraulic_graph_closing_v12_stops_downstream_flow_and_isolates_j14_leak():
    """Closing V-12 in the NetworkX hydraulic graph cuts SEG-S4-01 & downstream SEG-S4-05 to 0 L/s and stops J-14 leak loss."""
    from database.dubai_network import DUBAI_PIPE_SEGMENTS
    from digital_twin.hydraulics import compute_network_flows

    all_open = {f"V-{i:02d}": "OPEN" for i in range(1, 21)}
    leaks = [{"leak_id": "LEAK-001", "pipe_segment_id": "SEG-S4-01", "estimated_loss_m3_per_hour": 42.5, "status": "Confirmed"}]

    res_open = compute_network_flows(DUBAI_PIPE_SEGMENTS, all_open, plant_output_mld=100.0, leaks=leaks)
    assert res_open["pipe_flows_lps"]["SEG-S4-01"] > 80.0
    assert res_open["pipe_flows_lps"]["SEG-S4-05"] > 50.0
    assert "LEAK-001" not in res_open["isolated_leak_ids"]
    assert res_open["active_leak_loss_m3_hr"]["LEAK-001"] == 42.5

    # Close V-12 -> downstream pipes drop to 0 L/s and J-14 leak is isolated
    v12_closed = dict(all_open)
    v12_closed["V-12"] = "CLOSED"
    res_closed = compute_network_flows(DUBAI_PIPE_SEGMENTS, v12_closed, plant_output_mld=100.0, leaks=leaks)
    assert res_closed["pipe_flows_lps"]["SEG-S4-01"] == 0.0
    assert res_closed["pipe_flows_lps"]["SEG-S4-05"] == 0.0
    assert res_closed["pipe_flows_lps"]["SEG-S4-02"] > 0.0  # Unaffected coastal branch stays supplied
    assert "LEAK-001" in res_closed["isolated_leak_ids"]
    assert res_closed["active_leak_loss_m3_hr"]["LEAK-001"] == 0.0

    # Scene state reflects computed 0 L/s flow and isolated leak when V-12 is overridden CLOSED
    inject_scenario_1()
    st_closed = build_3d_scene_state(focus="V-12", valve_override={"V-12": "CLOSED"})
    p_s4_01 = next(p for p in st_closed["pipes"] if p["segment_id"] == "SEG-S4-01")
    assert p_s4_01["flow_lps"] == 0.0
    s4_leaks = [l for l in st_closed["leaks"] if l["pipe_segment_id"] == "SEG-S4-01"]
    assert s4_leaks and all(l["isolated"] is True and l["estimated_loss_m3_per_hour"] == 0.0 for l in s4_leaks)


def test_hydraulic_rerouting_only_where_alternative_path_exists():
    """Flow reroutes at reduced capacity where an open alternate ring path exists, and drops to 0 when no bypass exists."""
    from database.dubai_network import DUBAI_PIPE_SEGMENTS
    from digital_twin.hydraulics import compute_network_flows

    all_open = {f"V-{i:02d}": "OPEN" for i in range(1, 21)}
    base = compute_network_flows(DUBAI_PIPE_SEGMENTS, all_open, plant_output_mld=100.0)
    base_s5_02 = base["pipe_flows_lps"]["SEG-S5-02"]
    assert base_s5_02 > 40.0

    # Close V-13 (blocks primary feeder SEG-S5-01), but V-14 & V-16 stay OPEN -> SEG-S5-02 reroutes via SEG-S5-04!
    v13_closed = dict(all_open)
    v13_closed["V-13"] = "CLOSED"
    rerouted = compute_network_flows(DUBAI_PIPE_SEGMENTS, v13_closed, plant_output_mld=100.0)
    assert rerouted["pipe_flows_lps"]["SEG-S5-01"] == 0.0
    assert 0.0 < rerouted["pipe_flows_lps"]["SEG-S5-02"] < base_s5_02
    assert "SEG-S5-02" in rerouted["rerouted_segment_ids"]

    # Now also close V-16 (cuts the alternate path too) -> SEG-S5-02 drops to 0.0 L/s
    both_closed = dict(v13_closed)
    both_closed["V-16"] = "CLOSED"
    no_alt = compute_network_flows(DUBAI_PIPE_SEGMENTS, both_closed, plant_output_mld=100.0)
    assert no_alt["pipe_flows_lps"]["SEG-S5-02"] == 0.0

    # Closing Sector 3 root trunk V-07 has NO bypass -> all Sector 3 downstream pipes drop to 0.0 L/s
    v07_closed = dict(all_open)
    v07_closed["V-07"] = "CLOSED"
    s3_cut = compute_network_flows(DUBAI_PIPE_SEGMENTS, v07_closed, plant_output_mld=100.0)
    for sid in ("SEG-S3-01", "SEG-S3-02", "SEG-S3-03", "SEG-S3-04", "SEG-S3-05"):
        assert s3_cut["pipe_flows_lps"][sid] == 0.0


def test_simulate_valve_change_impact_and_part2_animation_html():
    """simulate_valve_change_impact computes exact before->after flows, and 3D HTML includes particles, spray, puddle & localStorage diff/camera."""
    from components.digital_twin_3d import build_3d_digital_twin_html
    from digital_twin.hydraulics import simulate_valve_change_impact

    impact = simulate_valve_change_impact("V-12", "CLOSED")
    assert "SEG-S4-01" in impact["affected_pipe_ids"]
    assert "SEG-S4-05" in impact["affected_pipe_ids"]
    assert any("-> 0 L/s" in line for line in impact["summary_lines"])

    inject_scenario_1()
    state = build_3d_scene_state(focus="V-12")
    html_out = build_3d_digital_twin_html(state, widget_id="anim_test", preserve_camera=True)

    assert "pipeFlowTracks" in html_out
    assert "flowInst" in html_out
    assert "sprayDrops" in html_out
    assert "puddle" in html_out
    assert "valveAnims" in html_out
    assert "aquagrid_3d_prev_state_v1" in html_out
    assert "aquagrid_3d_camera_v1" in html_out
    assert "window.localStorage" in html_out
    assert "window.__AQUAGRID_PRESERVE_CAMERA__ = true;" in html_out


def test_show_3d_tool_default_routing_and_2d_override():
    """'show Sector 4', 'where is the leak', 'show the reservoir', 'show valve V-12' use show_3d by default; '2D' uses 2D map."""
    from unittest.mock import patch
    from agent.chat_assistant import build_3d_render_summary, run_chat_turn

    inject_scenario_1()

    with patch("agent.chat_assistant.llm_client.chat_with_tools", return_value=(None, "Deterministic Offline Engine", [])):
        r_s4 = run_chat_turn("show Sector 4", history=[], current_hour=14)
        assert any(w["type"] == "twin_3d" and w["focus"] == "Sector 4" for w in r_s4["widgets"])
        assert "Sector 4" in r_s4["content"]

        r_lk = run_chat_turn("where is the leak", history=[], current_hour=14)
        assert any(w["type"] == "twin_3d" for w in r_lk["widgets"])

        r_res = run_chat_turn("show the reservoir", history=[], current_hour=14)
        assert any(w["type"] == "twin_3d" and w["focus"] == "RES-01" for w in r_res["widgets"])

        r_v12 = run_chat_turn("show valve V-12", history=[], current_hour=14)
        assert any(w["type"] == "twin_3d" and w["focus"] == "V-12" for w in r_v12["widgets"])

        # Explicit 2D request uses the 2D OpenStreetMap widget
        r_2d = run_chat_turn("show Sector 4 in 2D", history=[], current_hour=14)
        assert any(w["type"] == "map" and w["sector"] == "Sector 4" for w in r_2d["widgets"])

    # Direct tool summary matches live DB
    s_v12 = build_3d_render_summary("V-12", current_hour=14)
    assert s_v12["displayed"] is True and s_v12["view"] == "3D" and s_v12["focus_id"] == "V-12"
    assert "V-12" in s_v12["summary_text"]


def test_approval_card_3d_preview_and_post_approve_vs_reject():
    """Approval card computes 3D valve preview + flow delta; Approve attaches post-action 3D view; Reject changes nothing."""
    from unittest.mock import patch
    import components.chat_widgets as cw
    from database.connection import get_db_connection
    from tools.write_gateway import stage_gated_valve_or_setpoint_action

    # Stage CLOSE_VALVE V-12
    gw = stage_gated_valve_or_setpoint_action(
        action_type="CLOSE_VALVE",
        target_id="V-12",
        proposed_value=0.0,
        rationale="Isolate Sector 4 J-14 rupture",
    )
    assert gw["status"] == "STAGED_PENDING_APPROVAL"
    action_id = gw["action_id"]

    # 1. Preview includes V-12 focus, highlighted downstream pipes, and computed flow drop
    prev = cw.build_approval_impact_preview("CLOSE_VALVE", "V-12")
    assert prev["focus"] == "V-12"
    assert "SEG-S4-01" in prev["highlight_pipe_ids"]
    assert any("-> 0 L/s" in line for line in prev["summary_lines"])

    # 2. Clicking Approve executes V-12 and posts 3D view focused on V-12
    fake_state: dict = {"operator_signature": "Test_Operator", "ask_aquagrid_messages": []}

    class _Col:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _btn_approve(label, **kwargs):
        return label.startswith("Approve")

    with (
        patch.object(cw.st, "session_state", fake_state),
        patch.object(cw.st, "markdown"),
        patch.object(cw.st, "columns", return_value=(_Col(), _Col(), _Col())),
        patch.object(cw.st, "button", side_effect=_btn_approve),
        patch.object(cw.st, "rerun"),
        patch.object(cw, "render_3d_digital_twin"),
    ):
        cw.render_inline_approvals("test_ap")

    assert len(fake_state["ask_aquagrid_messages"]) == 1
    dec = fake_state["ask_aquagrid_messages"][0]
    assert "EXECUTED" in dec["content"]
    assert any(w["type"] == "twin_3d" and w["focus"] == "V-12" for w in dec["widgets"])

    conn = get_db_connection()
    try:
        assert conn.execute("SELECT status FROM valves WHERE valve_id = 'V-12';").fetchone()["status"] == "CLOSED"
    finally:
        conn.close()

    # 3. Clicking Reject on a fresh pending action leaves valve unchanged and attaches no 3D widget
    reset_database()
    gw2 = stage_gated_valve_or_setpoint_action(
        action_type="CLOSE_VALVE",
        target_id="V-12",
        proposed_value=0.0,
        rationale="Second test staging",
    )
    assert gw2["status"] == "STAGED_PENDING_APPROVAL"
    fake_state2: dict = {"operator_signature": "Test_Operator", "ask_aquagrid_messages": []}

    with (
        patch.object(cw.st, "session_state", fake_state2),
        patch.object(cw.st, "markdown"),
        patch.object(cw.st, "columns", return_value=(_Col(), _Col(), _Col())),
        patch.object(cw.st, "button", side_effect=lambda label, **kw: label.startswith("Reject")),
        patch.object(cw.st, "rerun"),
        patch.object(cw, "render_3d_digital_twin"),
    ):
        cw.render_inline_approvals("test_rej")

    assert len(fake_state2["ask_aquagrid_messages"]) == 1
    rej_msg = fake_state2["ask_aquagrid_messages"][0]
    assert "REJECTED" in rej_msg["content"]
    assert rej_msg["widgets"] == []
    conn = get_db_connection()
    try:
        assert conn.execute("SELECT status FROM valves WHERE valve_id = 'V-12';").fetchone()["status"] == "OPEN"
    finally:
        conn.close()


def test_scenarios_show_relevant_3d_views_in_result():
    """Scenarios attach targeted 3D views: J-14 leak (R1-S1), reservoir (R1-S2), V-12 (R1-S3), Sector 2 (U-5.1), Sector 6 (U-5.2), V-17 (U-5.3)."""
    from unittest.mock import patch
    from agent.chat_assistant import run_scenario_for_chat
    from agent.llm_factory import generate_offline_mock_response

    expected = {
        "R1-S1": ("LEAK-", "Sector 4"),
        "R1-S2": ("reservoir",),
        "R1-S3": ("V-12",),
        "U-5.1": ("Sector 2",),
        "U-5.2": ("Sector 6",),
        "U-5.3": ("V-17",),
    }

    with patch("agent.nodes.llm_client.query", side_effect=lambda sp, up: (generate_offline_mock_response(sp, up), "Deterministic Offline Engine")):
        for sc_id, valid_prefixes in expected.items():
            msg = run_scenario_for_chat(sc_id, current_hour=14)
            w3d = [w for w in msg.get("widgets", []) if w.get("type") == "twin_3d"]
            assert len(w3d) == 1, f"{sc_id} missing twin_3d widget"
            focus_val = str(w3d[0]["focus"])
            assert any(focus_val.startswith(p) for p in valid_prefixes), f"{sc_id} got unexpected 3D focus {focus_val}"


def test_requirements_has_networkx_and_all_3d_cdn_urls_use_https():
    """requirements.txt includes networkx and every 3D/fallback CDN URL loads over HTTPS (Streamlit Cloud compatible)."""
    import components.digital_twin_3d as dt3d

    req_txt = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "networkx" in req_txt.lower()

    for url in (dt3d.THREE_MODULE_URL, dt3d.THREE_ADDONS_BASE, dt3d.LEAFLET_CSS_URL, dt3d.LEAFLET_JS_URL, dt3d.OSM_TILE_URL):
        assert url.startswith("https://"), f"CDN URL must use https:// for Streamlit Cloud: {url}"



