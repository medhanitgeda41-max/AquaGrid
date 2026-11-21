"""Tests for sector boundary polygons, show_map tool truthfulness, Folium map layers, and 'How I got this' bullets."""
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

from agent.chat_assistant import build_map_render_summary, run_chat_turn
from config.region_config import SECTOR_COLOR_NAMES, SECTOR_COLORS
from database.dubai_network import (
    DUBAI_PIPE_SEGMENTS,
    JUNCTION_COORDINATES,
    JUNCTION_SECTOR_MAP,
    VALVE_COORDINATES,
    VALVE_SECTOR_MAP,
    compute_sector_boundaries,
    point_in_polygon,
)
from database.scenarios import reset_database

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


@pytest.fixture(autouse=True)
def clean_db():
    reset_database()
    yield
    reset_database()


def test_sector_boundaries_computed_for_all_6_sectors_and_contain_all_assets():
    """All 6 sectors get buffered (~300m) convex-hull polygons that strictly contain every pipe vertex, valve, junction, and sensor."""
    assert SECTOR_COLOR_NAMES == {
        "Sector 1": "blue",
        "Sector 2": "green",
        "Sector 3": "orange",
        "Sector 4": "purple",
        "Sector 5": "teal",
        "Sector 6": "red",
    }
    assert set(SECTOR_COLORS.keys()) == set(SECTOR_COLOR_NAMES.keys())

    boundaries = compute_sector_boundaries(DUBAI_PIPE_SEGMENTS, buffer_meters=300.0)
    assert set(boundaries.keys()) == {f"Sector {i}" for i in range(1, 7)}

    for sec, binfo in boundaries.items():
        poly = binfo["polygon"]
        assert len(poly) >= 10
        assert poly[0] == poly[-1], f"{sec} polygon must be closed"
        assert len(binfo["asset_points"]) > 0

        # Every asset point belonging to this sector must lie strictly inside its buffered polygon
        for lat, lon in binfo["asset_points"]:
            assert point_in_polygon(lat, lon, poly), f"Asset ({lat}, {lon}) outside {sec} boundary!"

    # Explicit cross-check: every pipe vertex, junction, and valve V-01..V-20 is inside its sector boundary
    for seg in DUBAI_PIPE_SEGMENTS:
        poly = boundaries[seg["sector"]]["polygon"]
        for lat, lon in seg["geometry"]:
            assert point_in_polygon(lat, lon, poly), f"Pipe {seg['segment_id']} point outside {seg['sector']}"

    for j_id, (lat, lon) in JUNCTION_COORDINATES.items():
        sec = JUNCTION_SECTOR_MAP[j_id]
        assert point_in_polygon(lat, lon, boundaries[sec]["polygon"]), f"Junction {j_id} outside {sec}"

    for v_id, (lat, lon) in VALVE_COORDINATES.items():
        sec = VALVE_SECTOR_MAP[v_id]
        assert point_in_polygon(lat, lon, boundaries[sec]["polygon"]), f"Valve {v_id} outside {sec}"


def test_show_map_all_draws_6_boundaries_and_sector_4_zooms_and_highlights():
    """show_map('all') draws 6 boundaries; show_map('Sector 4') zooms to Sector 4 and highlights it; Folium map renders all layers."""
    from components.leak_map_view import build_leak_folium_map

    all_summary = build_map_render_summary(sector="all", show_boundaries=True)
    assert all_summary["boundaries_drawn"] == 6
    assert all_summary["highlight_sector"] is None
    assert "All sectors" in all_summary["zoom_target"]
    assert all_summary["layers_drawn"] == ["Sector boundaries", "Pipes", "Sensors", "Valves", "Leaks"]
    assert all_summary["basemap"] == "CartoDB Positron (light_all)"
    assert all_summary["total_pipes"] == 28
    assert all_summary["total_valves"] == 20
    assert set(all_summary["per_sector"].keys()) == {f"Sector {i}" for i in range(1, 7)}
    for i in range(1, 7):
        assert all_summary["per_sector"][f"Sector {i}"]["boundary_drawn"] is True

    s4_summary = build_map_render_summary(sector="Sector 4", show_boundaries=True)
    assert s4_summary["highlight_sector"] == "Sector 4"
    assert s4_summary["zoom_target"] == "Sector 4"
    assert s4_summary["boundaries_drawn"] == 6
    assert s4_summary["zoom_bounds"] is not None
    assert "Sector 4" in s4_summary["summary_text"] and "purple" in s4_summary["summary_text"]

    # Build real Folium map HTML and verify CartoDB Positron basemap, legend, sector labels, boundaries & valves render
    fmap_all = build_leak_folium_map(
        segments=DUBAI_PIPE_SEGMENTS,
        leaks=[],
        quarantined_ids=set(),
        latest_pressures={},
        selected_leak=None,
        show_boundaries=True,
        show_pipes=True,
        show_sensors=True,
        show_valves=True,
        show_leaks=True,
        highlight_sector=None,
        basemap="CartoDB Positron",
    )
    html_all = fmap_all.get_root().render()
    assert "basemaps.cartocdn.com/light_all" in html_all
    assert "Sector Boundaries" in html_all
    assert "Sectors &amp; Assets" in html_all
    for i in range(1, 7):
        assert f"Sector {i}" in html_all

    fmap_s4 = build_leak_folium_map(
        segments=DUBAI_PIPE_SEGMENTS,
        leaks=[],
        quarantined_ids=set(),
        latest_pressures={},
        selected_leak=None,
        show_boundaries=True,
        highlight_sector="Sector 4",
        basemap="CartoDB Positron",
    )
    html_s4 = fmap_s4.get_root().render()
    assert "fitBounds" in html_s4
    assert "Sector 4 (purple boundary)" in html_s4


def test_truthful_map_description_replaces_untrue_boundary_claims():
    """If the LLM claims boundaries were drawn when show_map had show_boundaries=False, reply is replaced by the true tool summary."""
    def _mock_untruthful_llm(messages, tools, tool_executor, max_tool_steps=5, fallback_context_builder=None):
        tool_executor("show_map", {"sector": "all", "show_boundaries": False})
        return "Here is the map with each sector's boundary drawn in a distinct color line.", "Groq (openai/gpt-oss-120b)", []

    with patch("agent.chat_assistant.llm_client.chat_with_tools", side_effect=_mock_untruthful_llm):
        reply = run_chat_turn("Show the map without boundaries", history=[], current_hour=14)

    assert "no sector boundaries drawn" in reply["content"].lower()
    assert "distinct color line" not in reply["content"].lower()


def test_how_i_got_this_has_no_empty_bullets_and_shows_model_fallback():
    """'How I got this' never emits empty bullets and clearly explains primary -> fallback model transitions."""
    from components.ai_prompt_panel import build_how_i_got_this_summary

    with patch("agent.chat_assistant.llm_client.chat_with_tools", return_value=(None, "Deterministic Offline Engine", [])):
        reply = run_chat_turn("Show all sectors with separate borders", history=[], current_hour=14)

    # Simulate a 120b -> 20b fallback + stray empty dict in tools_called
    reply["provider"] = "Groq (openai/gpt-oss-20b)"
    reply["tools_called"].append({"tool": "", "summary": ""})
    reply["providers_tried"] = [
        {"provider": "Groq (openai/gpt-oss-120b)", "status": "FAILED", "error": "429 rate_limit_exceeded"},
        {"provider": "Groq (openai/gpt-oss-20b)", "status": "OK", "error": ""},
    ]

    hgt = build_how_i_got_this_summary(reply)
    assert len(hgt["tool_bullets"]) >= 2
    for bullet in hgt["tool_bullets"]:
        assert bullet.startswith("- `")
        parts = bullet.split(" — ", 1)
        assert len(parts) == 2 and parts[0].strip("- `") != "" and parts[1].strip() != ""

    assert hgt["provider"] == "Groq (openai/gpt-oss-20b)"
    assert any(
        "gpt-oss-120b" in note and "gpt-oss-20b" in note and "rate_limit_exceeded" in note
        for note in hgt["fallback_notes"]
    )
