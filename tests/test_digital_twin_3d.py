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


def test_show_3d_sets_twin_view_for_natural_operator_questions():
    """show_3d and chat routing set twin_view for sectors, look inside reservoir, plant, valves, and chlorine."""
    from unittest.mock import patch
    from agent.chat_assistant import build_3d_render_summary, run_chat_turn

    # Direct build_3d_render_summary checks
    inside = build_3d_render_summary("reservoir_inside", current_hour=14)
    assert inside["twin_view"]["focus"] == "reservoir_inside"
    assert inside["camera_target"]["look_inside"] is True

    plant = build_3d_render_summary("plant", current_hour=14)
    assert plant["twin_view"]["focus"] == "plant"
    assert plant["focus"] == "DESAL-01"

    intake = build_3d_render_summary("intake", current_hour=14)
    assert intake["twin_view"]["focus"] == "intake"

    # Natural-language chat routing checks (mock LLM)
    prompts_and_expected = [
        ("show Sector 4", "Sector 4"),
        ("look inside the reservoir", "reservoir_inside"),
        ("show the desalination plant", "plant"),
        ("check valve V-12", "V-12"),
        ("what about Sector 2 chlorine", "Sector 2"),
    ]
    with patch("agent.chat_assistant.llm_client.chat_with_tools", return_value=(None, "Deterministic Offline Engine", [])):
        for prompt, expected_focus in prompts_and_expected:
            reply = run_chat_turn(prompt, history=[], current_hour=14)
            tv = reply.get("twin_view")
            assert tv is not None, f"Missing twin_view for {prompt!r}"
            assert tv["focus"] == expected_focus, f"For {prompt!r}, expected focus {expected_focus!r}, got {tv['focus']!r}"


def test_approve_emits_valve_close_and_zero_downstream_flow_while_reject_leaves_unchanged():
    """Approve emits valve_close in twin_view with 0 L/s downstream flow; Reject emits valve_reject and changes nothing."""
    from unittest.mock import patch
    import components.chat_widgets as cw
    from database.connection import get_db_connection
    from digital_twin.hydraulics import compute_network_flows
    from tools.write_gateway import stage_gated_valve_or_setpoint_action

    class _Col:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    reset_database()
    gw = stage_gated_valve_or_setpoint_action(
        action_type="CLOSE_VALVE",
        target_id="V-12",
        proposed_value=0.0,
        rationale="Isolate Sector 4 J-14 branch",
    )
    assert gw["status"] == "STAGED_PENDING_APPROVAL"

    state_approve: dict = {"operator_signature": "Control_Op", "ask_aquagrid_messages": []}
    with (
        patch.object(cw.st, "session_state", state_approve),
        patch.object(cw.st, "markdown"),
        patch.object(cw.st, "columns", return_value=(_Col(), _Col(), _Col())),
        patch.object(cw.st, "button", side_effect=lambda label, **kw: label.startswith("Approve")),
        patch.object(cw.st, "rerun"),
    ):
        cw.render_inline_approvals("ev_ap")

    tv = state_approve.get("twin_view")
    assert tv is not None
    assert tv["focus"] == "V-12"
    assert tv["event_type"] == "valve_close"

    # Computed downstream flow on SEG-S4-01 and SEG-S4-05 is now 0.0 L/s
    conn = get_db_connection()
    try:
        v_map = {r["valve_id"]: r["status"] for r in conn.execute("SELECT valve_id, status FROM valves;").fetchall()}
        assert v_map["V-12"] == "CLOSED"
        hyd = compute_network_flows(valve_statuses=v_map)
        assert hyd["pipe_flows_lps"]["SEG-S4-01"] == 0.0
        assert hyd["pipe_flows_lps"]["SEG-S4-05"] == 0.0
    finally:
        conn.close()

    # Now Reject on a fresh pending action
    reset_database()
    gw2 = stage_gated_valve_or_setpoint_action(
        action_type="CLOSE_VALVE",
        target_id="V-12",
        proposed_value=0.0,
        rationale="Reject test",
    )
    assert gw2["status"] == "STAGED_PENDING_APPROVAL"

    state_reject: dict = {"operator_signature": "Control_Op", "ask_aquagrid_messages": []}
    with (
        patch.object(cw.st, "session_state", state_reject),
        patch.object(cw.st, "markdown"),
        patch.object(cw.st, "columns", return_value=(_Col(), _Col(), _Col())),
        patch.object(cw.st, "button", side_effect=lambda label, **kw: label.startswith("Reject")),
        patch.object(cw.st, "rerun"),
    ):
        cw.render_inline_approvals("ev_rej")

    tv_rej = state_reject.get("twin_view")
    assert tv_rej is not None
    assert tv_rej["focus"] == "V-12"
    assert tv_rej["event_type"] == "valve_reject"
    assert "Rejected" in tv_rej["note"]

    conn = get_db_connection()
    try:
        v_map2 = {r["valve_id"]: r["status"] for r in conn.execute("SELECT valve_id, status FROM valves;").fetchall()}
        assert v_map2["V-12"] == "OPEN"
        hyd2 = compute_network_flows(valve_statuses=v_map2)
        assert hyd2["pipe_flows_lps"]["SEG-S4-01"] > 0.0
    finally:
        conn.close()


def test_new_leak_emits_new_leak_event_from_scenario_and_alerts():
    """R1-S1 scenario and collect_system_alerts emit event_type='new_leak' in twin_view."""
    from unittest.mock import patch
    from agent.chat_assistant import collect_system_alerts, run_scenario_for_chat
    from agent.llm_factory import generate_offline_mock_response
    from tools.read_feeds import read_tariff_and_quality_feeds
    from tools.read_telemetry import read_network_telemetry

    reset_database()
    with patch("agent.nodes.llm_client.query", side_effect=lambda sp, up: (generate_offline_mock_response(sp, up), "Deterministic Offline Engine")):
        msg = run_scenario_for_chat("R1-S1", current_hour=14)
    assert msg.get("twin_view") is not None
    assert msg["twin_view"]["event_type"] == "new_leak"

    # System alert for confirmed leak also emits new_leak
    tel = read_network_telemetry(current_hour=14, log_audit=False)
    feeds = read_tariff_and_quality_feeds(horizon_hours=6, current_hour=14, log_audit=False)
    alerts, _ = collect_system_alerts(tel, feeds, seen_alert_keys=set())
    leak_alerts = [a for a in alerts if a.get("twin_view", {}).get("event_type") == "new_leak"]
    assert len(leak_alerts) >= 1


def test_persistent_top_twin_uses_fixed_component_key_without_reload():
    """render_persistent_top_twin passes updates to the SAME fixed component key across reruns (no iframe reload)."""
    from unittest.mock import patch
    import components.digital_twin_3d as dt3d

    reset_database()
    calls = []

    def _fake_component(**kwargs):
        calls.append(kwargs)
        return None

    class _Col:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake_state: dict = {"twin_view": {"focus": "all", "event_type": "focus", "note": "", "seq": 1}}

    with (
        patch.object(dt3d, "_aquagrid_3d_component", side_effect=_fake_component),
        patch.object(dt3d.st, "session_state", fake_state),
        patch.object(dt3d.st, "markdown"),
        patch.object(dt3d.st, "columns", side_effect=lambda spec, **kw: tuple(_Col() for _ in range(len(spec) if isinstance(spec, (list, tuple)) else int(spec)))),
        patch.object(dt3d.st, "button", return_value=False),
        patch.object(dt3d.st, "rerun"),
    ):
        dt3d.render_persistent_top_twin(display_hour=14, height=450)
        # Update focus via set_twin_view (simulating a chat turn) and re-render
        dt3d.set_twin_view(fake_state, focus="Sector 4", event_type="focus", note="Sector 4")
        dt3d.render_persistent_top_twin(display_hour=14, height=450)
        dt3d.set_twin_view(fake_state, focus="reservoir_inside", event_type="focus", note="Inside RES-01")
        dt3d.render_persistent_top_twin(display_hour=14, height=450)

    assert len(calls) == 3
    keys = [c["key"] for c in calls]
    assert keys == [dt3d.TWIN_COMPONENT_KEY] * 3
    assert calls[0]["view_command"]["focus"] == "all"
    assert calls[1]["view_command"]["focus"] == "Sector 4"
    assert calls[2]["view_command"]["focus"] == "reservoir_inside"
    assert calls[2]["view_command"]["seq"] > calls[1]["view_command"]["seq"]


def test_twin_frontend_index_html_has_realistic_3d_features_and_no_carto():
    """Custom component frontend index.html implements bidirectional Streamlit messaging, Sky/Water/DESAL/RES cutaway, and OSM fallback with zero CARTO."""
    frontend_html = (PROJECT_ROOT / "app" / "components" / "twin_frontend" / "index.html").read_text(encoding="utf-8")

    # Bidirectional Streamlit component protocol
    assert "streamlit:componentReady" in frontend_html
    assert "streamlit:render" in frontend_html
    assert "streamlit:setFrameHeight" in frontend_html

    # Realistic Three.js modules over HTTPS (no CARTO, no API keys)
    assert "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js" in frontend_html
    assert "three/addons/controls/OrbitControls.js" in frontend_html
    assert "three/addons/objects/Sky.js" in frontend_html
    assert "three/addons/objects/Water.js" in frontend_html
    assert "https://tile.openstreetmap.org/{z}/{x}/{y}.png" in frontend_html
    assert "carto" not in frontend_html.lower()

    # Required realistic features
    assert "DESAL-01" in frontend_html
    assert "RO Hall" in frontend_html
    assert "Intake 2" in frontend_html
    assert "Brine Outfall" in frontend_html
    assert "TELEMETRY LOSS" in frontend_html
    assert "render2DFallback" in frontend_html
    assert "InstancedMesh" in frontend_html
    assert "applySceneUpdate" in frontend_html


def test_scene_state_includes_detailed_plant_reservoir_rings_and_city_features():
    """build_3d_scene_state includes detailed DESAL-01 intakes, RES-01 floor rings, seeded roads/trees, and telemetry_status."""
    from digital_twin.scene_state import build_twin_caption

    reset_database()
    state = build_3d_scene_state(current_hour=14, focus="reservoir_inside")

    assert len(state["plant"]["intakes"]) == 2
    assert state["plant"]["intakes"][0]["id"] == "Intake 1"
    assert state["plant"]["intakes"][1]["id"] == "Intake 2"
    assert state["reservoir"]["base_floor_percent"] == 35.0
    assert state["reservoir"]["safety_floor_percent"] >= 35.0
    assert state["reservoir"]["warning_floor_percent"] == round(state["reservoir"]["safety_floor_percent"] + 5.0, 2)
    assert state["reservoir"]["look_inside"] is True
    assert state["camera_target"]["look_inside"] is True
    assert len(state["roads"]) > 0
    assert len(state["parks"]) > 0
    assert len(state["trees"]) > 0
    assert state["telemetry_status"]["is_telemetry_loss"] is False

    caption = build_twin_caption(state, {"focus": "reservoir_inside", "event_type": "focus"})
    assert "Inside Reservoir RES-01" in caption


def test_unified_reservoir_model_matches_schedule_validator_and_floor_eta():
    """Unified reservoir model: prod < dem falls, prod > dem rises, matches schedule validator exactly, and floor ETA is exact."""
    from guards.reservoir_model import (
        evaluate_reservoir_status,
        project_reservoir_series,
        step_reservoir_volume_ml,
    )
    from guards.schedule_validator import validate_production_schedule

    # 1. Production < demand -> falls; production > demand -> rises; clamped [0, 500]
    v_fall = step_reservoir_volume_ml(240.0, production_mld=60.0, demand_mld=96.0, dt_hours=1.0, capacity_ml=500.0)
    assert v_fall == pytest.approx(238.5, abs=1e-6)
    v_rise = step_reservoir_volume_ml(240.0, production_mld=120.0, demand_mld=96.0, dt_hours=1.0, capacity_ml=500.0)
    assert v_rise == pytest.approx(241.0, abs=1e-6)
    assert step_reservoir_volume_ml(0.5, production_mld=0.0, demand_mld=120.0, dt_hours=2.0, capacity_ml=500.0) == 0.0
    assert step_reservoir_volume_ml(499.5, production_mld=120.0, demand_mld=0.0, dt_hours=2.0, capacity_ml=500.0) == 500.0

    # 2. Exact match with schedule validator hour-by-hour over 24 hours
    demand_24h = {h: (96.0 if 12 <= h <= 20 else 72.0) for h in range(24)}
    val = validate_production_schedule(
        proposed_schedule="00:00-24:00 at 80 MLD",
        current_volume_ml=240.0,
        demand_by_hour=demand_24h,
        current_hour=14,
        total_capacity_ml=500.0,
        min_floor_percent=0.35,
    )
    vol = 240.0
    for step_idx in range(24):
        hr = (14 + step_idx) % 24
        vol = step_reservoir_volume_ml(vol, production_mld=80.0, demand_mld=demand_24h[hr], dt_hours=1.0, capacity_ml=500.0)
        traj_pt = val["cumulative_trajectory"][step_idx + 1]
        assert traj_pt["volume_ml"] == round(vol, 2)
        assert traj_pt["level_percent"] == round((vol / 500.0) * 100.0, 2)

    # 3. Exact "floor in X h Y m" calculation (240 ML -> 175 ML = 65 ML at -1.5 ML/h => 43 h 20 m)
    st_fall = evaluate_reservoir_status(current_volume_ml=240.0, production_mld=60.0, demand_mld=96.0, floor_percent=35.0, capacity_ml=500.0)
    assert st_fall["trend_arrow"] == "↓"
    assert st_fall["rate_pct_per_hour"] == pytest.approx(-0.3, abs=1e-3)
    assert st_fall["hours_to_floor"] == pytest.approx(65.0 / 1.5, abs=0.01)
    assert st_fall["floor_eta_str"] == "35% floor in 43 h 20 m"
    assert st_fall["warning_state"] == "NORMAL"

    # Dynamic floor 35.4% (177 ML -> 63 ML / 1.5 ML/h = exactly 42 h)
    st_dyn = evaluate_reservoir_status(current_volume_ml=240.0, production_mld=60.0, demand_mld=96.0, floor_percent=35.4, capacity_ml=500.0)
    assert st_dyn["floor_eta_str"] == "35.4% floor in 42 h"

    # Rising & warning / critical bands
    st_rise = evaluate_reservoir_status(current_volume_ml=240.0, production_mld=110.0, demand_mld=80.0, floor_percent=35.0, capacity_ml=500.0)
    assert st_rise["trend_arrow"] == "↑" and st_rise["floor_eta_str"] == "rising"

    st_warn = evaluate_reservoir_status(current_volume_ml=190.0, production_mld=60.0, demand_mld=96.0, floor_percent=35.0, capacity_ml=500.0)
    assert st_warn["warning_state"] == "WARNING" and st_warn["within_warning_band"] is True

    st_crit = evaluate_reservoir_status(current_volume_ml=175.0, production_mld=60.0, demand_mld=96.0, floor_percent=35.0, capacity_ml=500.0)
    assert st_crit["warning_state"] == "CRITICAL" and st_crit["at_or_below_floor"] is True

    # 6h history + 12h projection series includes both segments & floor
    series = project_reservoir_series(240.0, current_hour=14, production_mld=60.0, demand_by_hour=demand_24h, floor_percent=35.0)
    assert any(p["segment"] == "History (last 6h)" for p in series)
    assert any(p["segment"] == "Projection (next 12h)" for p in series)


def test_approving_plant_output_100_to_60_mld_updates_depletion_rate_immediately():
    """Approving SET_PLANT_OUTPUT 100 -> 60 MLD changes live reservoir rate immediately and posts trend line."""
    from unittest.mock import patch
    import components.chat_widgets as cw
    from tools.read_feeds import read_tariff_and_quality_feeds
    from tools.write_gateway import stage_gated_valve_or_setpoint_action

    class _Col:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    reset_database()
    before_feeds = read_tariff_and_quality_feeds(horizon_hours=6, current_hour=14, log_audit=False)
    rate_before = float(before_feeds["reservoir"]["rate_pct_per_hour"])

    gw = stage_gated_valve_or_setpoint_action(
        action_type="SET_PLANT_OUTPUT",
        target_id="DESAL-01",
        proposed_value=60.0,
        rationale="Peak tariff reduction to 60 MLD",
    )
    assert gw["status"] == "STAGED_PENDING_APPROVAL"

    fake_state: dict = {"operator_signature": "Control_Op", "ask_aquagrid_messages": []}
    with (
        patch.object(cw.st, "session_state", fake_state),
        patch.object(cw.st, "markdown"),
        patch.object(cw.st, "columns", return_value=(_Col(), _Col(), _Col())),
        patch.object(cw.st, "button", side_effect=lambda label, **kw: label.startswith("Approve")),
        patch.object(cw.st, "rerun"),
    ):
        cw.render_inline_approvals("plant_ap")

    after_feeds = read_tariff_and_quality_feeds(horizon_hours=6, current_hour=14, log_audit=False)
    rate_after = float(after_feeds["reservoir"]["rate_pct_per_hour"])
    assert after_feeds["current_plant_output_mld"] == 60.0
    assert rate_after < rate_before, f"Expected rate to drop after 100 -> 60 MLD ({rate_before} -> {rate_after})"
    assert after_feeds["reservoir"]["trend_arrow"] == "↓"
    assert "floor in" in after_feeds["reservoir"]["floor_eta_str"]

    assert len(fake_state["ask_aquagrid_messages"]) == 1
    msg = fake_state["ask_aquagrid_messages"][0]
    assert "DESAL-01 100 -> 60 MLD" in msg["content"]
    assert "reservoir now" in msg["content"]
    assert fake_state["twin_view"]["event_type"] == "plant_change"


def test_connected_pipe_network_connectivity_and_valve_cutoff():
    """Every pipe endpoint is a graph node, RES-01 reaches all 6 sectors, valves sit on nodes, and closing V-12 cuts downstream flow to 0."""
    from digital_twin.hydraulics import build_connected_network_geometry, validate_network_connectivity

    reset_database()
    state_open = build_3d_scene_state(current_hour=14, focus="all")
    cnet = state_open["connected_network"]
    check = validate_network_connectivity(cnet["nodes"], cnet["pipes"])

    assert check["is_connected"] is True, f"Network connectivity failed: {check}"
    assert check["missing_endpoints"] == []
    assert check["misaligned_pipes"] == []
    assert check["desal_reaches_res"] is True
    assert check["missing_sectors"] == []
    assert check["disconnected_valves"] == []
    assert len(check["reachable_sectors"]) == 6

    # Closing sector trunk valve V-12 makes downstream connected pipes SEG-S4-01 & SEG-S4-05 drop to 0.0 L/s
    all_open = {f"V-{i:02d}": "OPEN" for i in range(1, 21)}
    v12_closed = dict(all_open)
    v12_closed["V-12"] = "CLOSED"
    cnet_closed = build_connected_network_geometry(valve_statuses=v12_closed, plant_output_mld=100.0)
    s4_01_pipes = [p for p in cnet_closed["pipes"] if p["segment_id"] in ("SEG-S4-01", "SEG-S4-05")]
    assert len(s4_01_pipes) >= 2
    assert all(p["flow_lps"] == 0.0 for p in s4_01_pipes)
    # Main transmission trunk and Sector 6 remain pressurized
    s6_pipes = [p for p in cnet_closed["pipes"] if p["pipe_id"] == "SEG-S6-01"]
    assert s6_pipes and s6_pipes[0]["flow_lps"] > 0.0


def test_focusing_sector_4_updates_shared_twin_view_for_both_3d_and_2d():
    """Focusing 'Sector 4' updates st.session_state['twin_view'] used identically by both 3D twin and 2D OpenStreetMap."""
    from unittest.mock import patch
    from agent.chat_assistant import run_chat_turn
    from components.digital_twin_3d import resolve_2d_map_focus_from_twin_view, set_twin_view

    reset_database()
    with patch("agent.chat_assistant.llm_client.chat_with_tools", return_value=(None, "Deterministic Offline Engine", [])):
        reply = run_chat_turn("show Sector 4", history=[], current_hour=14)

    tv = reply["twin_view"]
    state_dict: dict = {}
    cmd = set_twin_view(state_dict, focus=tv["focus"], event_type=tv["event_type"], note=tv["note"])

    # 3D scene state reads the exact shared twin_view focus
    scene_3d = build_3d_scene_state(focus=cmd["focus"], current_hour=14)
    assert scene_3d["highlight"]["type"] == "sector"
    assert scene_3d["highlight"]["id"] == "Sector 4"

    # 2D map resolver reads the exact same shared twin_view focus
    focus_2d = resolve_2d_map_focus_from_twin_view(state_dict["twin_view"])
    assert focus_2d["highlight_sector"] == "Sector 4"

    # Also verify shared sync for valve, reservoir_inside, and plant
    cmd_v = set_twin_view(state_dict, focus="V-12")
    assert resolve_2d_map_focus_from_twin_view(cmd_v)["focus_point_latlon"] is not None
    cmd_r = set_twin_view(state_dict, focus="reservoir_inside")
    assert resolve_2d_map_focus_from_twin_view(cmd_r)["focus_point_latlon"] is not None
    cmd_p = set_twin_view(state_dict, focus="plant")
    assert resolve_2d_map_focus_from_twin_view(cmd_p)["focus_point_latlon"] is not None





