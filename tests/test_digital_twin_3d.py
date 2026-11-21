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
