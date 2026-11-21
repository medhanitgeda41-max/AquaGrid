"""Interactive 3D Digital Twin component using Three.js + OrbitControls (jsDelivr HTTPS CDN, zero API keys).

Displays Python-calculated scene state only; never mutates SCADA actuators or safety guards.
Includes automatic WebGL detection with graceful fallback to a 2D OpenStreetMap view,
instanced procedural buildings, Low quality toggle, Reset view button, and live hover tooltips.
"""
import html as html_lib
import json
from pathlib import Path
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components

from config.constants import DEFAULT_SCADA_HOUR
from digital_twin.scene_state import (
    build_3d_scene_state,
    build_twin_caption,
    build_twin_view_command,
)

FRONTEND_DIR = Path(__file__).resolve().parent / "twin_frontend"
TWIN_COMPONENT_KEY = "aquagrid_main_3d_twin"

# Bidirectional Streamlit custom component — mounted ONCE with a fixed key so it updates in place without reloading!
_aquagrid_3d_component = components.declare_component(
    "aquagrid_3d_twin",
    path=str(FRONTEND_DIR),
)

THREE_MODULE_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"
THREE_ADDONS_BASE = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
LEAFLET_CSS_URL = "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css"
LEAFLET_JS_URL = "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


def set_twin_view(
    session_state: dict,
    focus: str = "all",
    event_type: str = "focus",
    note: str = "",
    highlight_pipe_ids: Optional[list[str]] = None,
) -> dict:
    """Update st.session_state['twin_view'] with an incrementing sequence number so the top 3D twin updates without reload."""
    seq = int(session_state.get("_twin_cmd_seq", 0)) + 1
    session_state["_twin_cmd_seq"] = seq
    cmd = build_twin_view_command(
        focus=focus,
        event_type=event_type,
        note=note,
        highlight_pipe_ids=highlight_pipe_ids,
        seq=seq,
    )
    session_state["twin_view"] = cmd
    return cmd


def build_3d_digital_twin_html(
    scene_state: dict,
    widget_id: str = "aquagrid_3d_main",
    height: int = 520,
    force_no_webgl: bool = False,
    preserve_camera: bool = False,
) -> str:
    """
    Build self-contained HTML/JS for the Three.js 3D Digital Twin viewport.
    Uses HTTPS jsDelivr CDN scripts (no API keys) and falls back cleanly to 2D OpenStreetMap if WebGL is unavailable.
    """
    # Escape </script> defensively when embedding JSON into HTML
    state_json = json.dumps(scene_state, separators=(",", ":")).replace("</", "<\\/")
    safe_wid = html_lib.escape(str(widget_id))
    h_px = max(240, int(height))
    force_flag = "true" if force_no_webgl else "false"
    preserve_cam_flag = "true" if preserve_camera else "false"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 100%;
    height: {h_px}px;
    overflow: hidden;
    background: #0B1220;
    color: #F8FAFC;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #ag-root {{
    position: relative;
    width: 100%;
    height: {h_px}px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #1E293B;
    background: radial-gradient(circle at 30% 20%, #111D35 0%, #090F1C 100%);
  }}
  #ag-canvas-wrap {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }}
  #ag-labels {{
    position: absolute;
    inset: 0;
    pointer-events: none;
    overflow: hidden;
  }}
  .ag-label {{
    position: absolute;
    transform: translate(-50%, -100%);
    white-space: nowrap;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    background: rgba(15, 23, 42, 0.82);
    border: 1px solid rgba(148, 163, 184, 0.35);
    color: #F8FAFC;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.7);
    user-select: none;
  }}
  .ag-label-valve {{
    font-size: 9px;
    padding: 1px 4px;
    background: rgba(15, 23, 42, 0.75);
  }}
  .ag-label-warn {{
    font-size: 10px;
    padding: 1px 5px;
    background: rgba(245, 158, 11, 0.92);
    color: #0F172A;
    border-color: #FEF3C7;
  }}
  #ag-hud-top {{
    position: absolute;
    top: 10px;
    left: 10px;
    right: 10px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    pointer-events: none;
    z-index: 20;
  }}
  .ag-pill {{
    pointer-events: auto;
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 11px;
    color: #E2E8F0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.35);
  }}
  .ag-btn {{
    pointer-events: auto;
    cursor: pointer;
    background: rgba(30, 41, 59, 0.92);
    color: #F8FAFC;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 11px;
    font-weight: 600;
    transition: background 0.15s ease;
  }}
  .ag-btn:hover {{
    background: #334155;
  }}
  #ag-tooltip {{
    position: absolute;
    display: none;
    pointer-events: none;
    z-index: 30;
    max-width: 260px;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid #38BDF8;
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 11px;
    line-height: 1.4;
    color: #F8FAFC;
    box-shadow: 0 4px 16px rgba(0,0,0,0.45);
  }}
  #ag-legend {{
    position: absolute;
    bottom: 10px;
    right: 10px;
    z-index: 20;
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 10px;
    max-width: 185px;
    color: #E2E8F0;
  }}
  #ag-legend summary {{
    cursor: pointer;
    font-weight: 700;
    user-select: none;
  }}
  .ag-leg-row {{
    display: flex;
    align-items: center;
    gap: 5px;
    margin: 2px 0;
  }}
  .ag-swatch {{
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 2px;
  }}
  #ag-fallback {{
    display: none;
    position: absolute;
    inset: 0;
    flex-direction: column;
    background: #0F172A;
    z-index: 40;
  }}
  #ag-fallback-banner {{
    padding: 8px 12px;
    background: #1E293B;
    border-bottom: 1px solid #334155;
    font-size: 12px;
    color: #FDE68A;
  }}
  #ag-fallback-map {{
    flex: 1;
    width: 100%;
    height: 100%;
  }}
</style>
<script type="importmap">
{{
  "imports": {{
    "three": "{THREE_MODULE_URL}",
    "three/addons/": "{THREE_ADDONS_BASE}"
  }}
}}
</script>
</head>
<body>
<div id="ag-root" data-widget-id="{safe_wid}">
  <div id="ag-canvas-wrap"></div>
  <div id="ag-labels"></div>
  <div id="ag-hud-top">
    <div class="ag-pill" id="ag-status-pill">
      <strong>AquaGrid 3D Twin</strong> · <span id="ag-focus-label">Loading...</span>
    </div>
    <button class="ag-btn" id="ag-reset-btn" type="button">Reset view</button>
  </div>
  <div id="ag-tooltip"></div>
  <details id="ag-legend">
    <summary>3D Legend</summary>
    <div style="margin-top:4px; padding-top:4px; border-top:1px solid #334155;" id="ag-legend-body"></div>
  </details>
  <div id="ag-fallback">
    <div id="ag-fallback-banner">
      WebGL is unavailable in this browser — showing the 2D OpenStreetMap fallback view.
    </div>
    <div id="ag-fallback-map"></div>
  </div>
</div>

<script>
window.__AQUAGRID_SCENE__ = {state_json};
window.__AQUAGRID_FORCE_NO_WEBGL__ = {force_flag};
window.__AQUAGRID_PRESERVE_CAMERA__ = {preserve_cam_flag};
window.__AQUAGRID_WIDGET_ID__ = "{safe_wid}";

function detectWebGL() {{
  if (window.__AQUAGRID_FORCE_NO_WEBGL__) return false;
  try {{
    const c = document.createElement('canvas');
    return !!(window.WebGLRenderingContext && (c.getContext('webgl2') || c.getContext('webgl')));
  }} catch (e) {{
    return false;
  }}
}}

function render2DFallback(reasonMsg) {{
  const fb = document.getElementById('ag-fallback');
  const banner = document.getElementById('ag-fallback-banner');
  if (reasonMsg) banner.textContent = reasonMsg;
  fb.style.display = 'flex';
  const state = window.__AQUAGRID_SCENE__ || {{}};
  const center = state.center_latlon || [25.1972, 55.2744];

  // Load Leaflet over HTTPS to render OpenStreetMap 2D fallback
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = '{LEAFLET_CSS_URL}';
  document.head.appendChild(link);

  const script = document.createElement('script');
  script.src = '{LEAFLET_JS_URL}';
  script.onload = () => {{
    try {{
      const map = L.map('ag-fallback-map').setView(center, 11);
      L.tileLayer('{OSM_TILE_URL}', {{
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 18
      }}).addTo(map);
      const refLat = center[0], refLon = center[1];
      const cosLat = Math.cos(refLat * Math.PI / 180);
      const xzToLatLon = (x, z) => [refLat - z / 111320.0, refLon + x / (111320.0 * cosLat)];
      const bounds = [];
      (state.sectors || []).forEach(sec => {{
        const latlngs = (sec.polygon_xz || []).map(p => xzToLatLon(p[0], p[1]));
        if (latlngs.length >= 3) {{
          L.polygon(latlngs, {{ color: sec.color, weight: 2.5, fillOpacity: 0.14 }}).bindTooltip(sec.name).addTo(map);
          latlngs.forEach(ll => bounds.push(ll));
        }}
      }});
      (state.pipes || []).forEach(pipe => {{
        const latlngs = (pipe.points_xz || []).map(p => xzToLatLon(p[0], p[1]));
        if (latlngs.length >= 2) {{
          L.polyline(latlngs, {{ color: pipe.color, weight: pipe.is_trunk ? 4 : 2.5 }}).bindTooltip(pipe.segment_id + ' (' + pipe.flow_lps + ' L/s)').addTo(map);
        }}
      }});
      (state.leaks || []).forEach(lk => {{
        if (!lk.active) return;
        const ll = xzToLatLon(lk.pos_xz[0], lk.pos_xz[1]);
        L.circleMarker(ll, {{ radius: 8, color: '#EF4444', fillColor: '#EF4444', fillOpacity: 0.9 }}).bindTooltip(lk.leak_id + ' · ' + lk.severity).addTo(map);
      }});
      if (bounds.length > 0) map.fitBounds(bounds, {{ padding: [18, 18] }});
    }} catch (err) {{
      banner.textContent = 'WebGL is unavailable; switch the toggle above to "2D Map" to view the network.';
    }}
  }};
  script.onerror = () => {{
    banner.textContent = 'WebGL is unavailable; switch the toggle above to "2D Map" to view the network.';
  }};
  document.head.appendChild(script);
}}

if (!detectWebGL()) {{
  render2DFallback('WebGL is unavailable in this browser — showing the 2D OpenStreetMap fallback view.');
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

if (!window.detectWebGL || !window.detectWebGL()) {{
  // Fallback already mounted
}} else {{
  try {{
    const state = window.__AQUAGRID_SCENE__;
    const SCALE = 1.0 / (state.scale_m_per_unit || 100.0); // 100m -> 1 world unit
    const lowQ = !!state.low_quality;
    const container = document.getElementById('ag-canvas-wrap');
    const labelsLayer = document.getElementById('ag-labels');
    const tooltipEl = document.getElementById('ag-tooltip');
    const focusLabelEl = document.getElementById('ag-focus-label');
    const legendBody = document.getElementById('ag-legend-body');

    focusLabelEl.textContent = (state.camera_target && state.camera_target.label) || 'All 6 Sectors';

    // Populate legend
    const legHtml = (state.sectors || []).map(s =>
      `<div class="ag-leg-row"><span class="ag-swatch" style="background:${{s.color}}55; border:2px solid ${{s.color}};"></span><span><strong>${{s.name}}</strong> (${{s.color_name}})</span></div>`
    ).join('') +
    `<div style="margin-top:4px; padding-top:4px; border-top:1px solid #334155;">
      <div class="ag-leg-row"><span class="ag-swatch" style="background:#22C55E; border-radius:50%;"></span><span>Valve OPEN · <span style="color:#EF4444;">●</span> CLOSED</span></div>
      <div class="ag-leg-row"><span class="ag-swatch" style="background:#38BDF8; border-radius:50%;"></span><span>Sensor · ⚠ Quarantined</span></div>
      <div class="ag-leg-row"><span class="ag-swatch" style="background:#EF4444; border-radius:50%;"></span><span>Active Leak</span></div>
    </div>`;
    legendBody.innerHTML = legHtml;

    // Renderer
    const renderer = new THREE.WebGLRenderer({{ antialias: !lowQ, powerPreference: 'high-performance' }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, lowQ ? 1.25 : 2.0));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    container.appendChild(renderer.domElement);

    // Scene & Camera
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0B1220');
    scene.fog = new THREE.FogExp2('#0B1220', 0.0014);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.5, 1800);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.maxPolarAngle = Math.PI * 0.485;
    controls.minDistance = 8;
    controls.maxDistance = 520;

    const camSpec = state.camera_target || {{ target_xz: [0, 0], position_xyz: [-14500, 13500, 16500] }};
    const defaultTarget = new THREE.Vector3(camSpec.target_xz[0] * SCALE, 0, camSpec.target_xz[1] * SCALE);
    const defaultCamPos = new THREE.Vector3(
      camSpec.position_xyz[0] * SCALE,
      camSpec.position_xyz[1] * SCALE,
      camSpec.position_xyz[2] * SCALE
    );

    // Load previous state & saved camera from window.localStorage for smooth diff animations & Live Sim camera persistence
    const PREV_STATE_KEY = 'aquagrid_3d_prev_state_v1';
    const CAM_STATE_KEY = 'aquagrid_3d_camera_v1';
    const currentFocusKey = `${{(state.highlight && state.highlight.type) || 'all'}}:${{(state.highlight && state.highlight.id) || 'all'}}`;

    let prevState = null;
    try {{
      const rawPrev = window.localStorage.getItem(PREV_STATE_KEY);
      if (rawPrev) prevState = JSON.parse(rawPrev);
    }} catch (e) {{
      prevState = null;
    }}

    let savedCam = null;
    try {{
      const rawCam = window.localStorage.getItem(CAM_STATE_KEY);
      if (rawCam) savedCam = JSON.parse(rawCam);
    }} catch (e) {{
      savedCam = null;
    }}

    let flyAnim = null;
    if (savedCam && savedCam.pos && savedCam.target && (window.__AQUAGRID_PRESERVE_CAMERA__ || savedCam.focusKey === currentFocusKey)) {{
      camera.position.set(savedCam.pos[0], savedCam.pos[1], savedCam.pos[2]);
      controls.target.set(savedCam.target[0], savedCam.target[1], savedCam.target[2]);
    }} else if (savedCam && savedCam.pos && savedCam.target && savedCam.focusKey !== currentFocusKey) {{
      // Smoothly fly from previous camera to the new focus target
      camera.position.set(savedCam.pos[0], savedCam.pos[1], savedCam.pos[2]);
      controls.target.set(savedCam.target[0], savedCam.target[1], savedCam.target[2]);
      flyAnim = {{
        startPos: camera.position.clone(),
        endPos: defaultCamPos.clone(),
        startTarget: controls.target.clone(),
        endTarget: defaultTarget.clone(),
        startTime: 0.0,
        duration: 1.15
      }};
    }} else {{
      camera.position.copy(defaultCamPos);
      controls.target.copy(defaultTarget);
    }}
    controls.update();

    let camSaveTimer = null;
    function saveCameraNow() {{
      try {{
        window.localStorage.setItem(CAM_STATE_KEY, JSON.stringify({{
          focusKey: currentFocusKey,
          pos: [camera.position.x, camera.position.y, camera.position.z],
          target: [controls.target.x, controls.target.y, controls.target.z]
        }}));
      }} catch (e) {{}}
    }}
    controls.addEventListener('change', () => {{
      if (camSaveTimer) clearTimeout(camSaveTimer);
      camSaveTimer = setTimeout(saveCameraNow, 150);
    }});

    // Persist compact snapshot of current state for the next update's diff animations
    try {{
      const nextSnapshot = {{
        valves: Object.fromEntries((state.valves || []).map(v => [v.valve_id, String(v.status).toUpperCase()])),
        pipeFlows: Object.fromEntries((state.pipes || []).map(p => [p.segment_id, Number(p.flow_lps || 0)])),
        reservoirLevel: Number((state.reservoir && state.reservoir.level_percent) || 48.0),
        plantOutput: Number((state.plant && state.plant.output_mld) || 100.0),
        activeLeaks: Object.fromEntries((state.leaks || []).map(l => [l.leak_id, !!l.active]))
      }};
      window.localStorage.setItem(PREV_STATE_KEY, JSON.stringify(nextSnapshot));
    }} catch (e) {{}}

    // Lighting
    scene.add(new THREE.HemisphereLight('#DBEAFE', '#1E293B', 0.85));
    const sun = new THREE.DirectionalLight('#FFFBEB', 1.15);
    sun.position.set(120, 220, 90);
    scene.add(sun);
    const rim = new THREE.DirectionalLight('#38BDF8', 0.45);
    rim.position.set(-160, 90, -110);
    scene.add(rim);

    // 1. Ground plane + subtle grid + Western Sea / Coastline
    const landGeo = new THREE.PlaneGeometry(760, 760);
    const landMat = new THREE.MeshStandardMaterial({{ color: '#141E30', roughness: 0.92, metalness: 0.05 }});
    const landMesh = new THREE.Mesh(landGeo, landMat);
    landMesh.rotation.x = -Math.PI / 2;
    landMesh.position.y = -0.15;
    scene.add(landMesh);

    const grid = new THREE.GridHelper(480, 48, '#1E3A5F', '#162238');
    grid.position.y = -0.08;
    scene.add(grid);

    // Build Western Sea polygon along coastline_xz extending far west (-X)
    const coastPts = (state.coastline_xz || []).map(p => [p[0] * SCALE, p[1] * SCALE]);
    if (coastPts.length >= 2) {{
      const seaShape = new THREE.Shape();
      // Shape is in XY plane, rotated -PI/2 around X so shape.y -> -world.z
      seaShape.moveTo(-360, -coastPts[0][1]);
      for (const [cx, cz] of coastPts) {{
        seaShape.lineTo(cx, -cz);
      }}
      seaShape.lineTo(-360, -coastPts[coastPts.length - 1][1]);
      seaShape.closePath();

      const seaGeo = new THREE.ShapeGeometry(seaShape);
      const seaMat = new THREE.MeshStandardMaterial({{
        color: '#0A4B78',
        emissive: '#06263F',
        roughness: 0.25,
        metalness: 0.35,
        transparent: true,
        opacity: 0.92
      }});
      const seaMesh = new THREE.Mesh(seaGeo, seaMat);
      seaMesh.rotation.x = -Math.PI / 2;
      seaMesh.position.y = -0.02;
      scene.add(seaMesh);

      // Sandy coastline rim
      const shoreCurve = new THREE.CatmullRomCurve3(
        coastPts.map(([cx, cz]) => new THREE.Vector3(cx, 0.04, cz))
      );
      const shoreTube = new THREE.Mesh(
        new THREE.TubeGeometry(shoreCurve, 48, 0.75, 6, false),
        new THREE.MeshStandardMaterial({{ color: '#C2A878', roughness: 0.9 }})
      );
      scene.add(shoreTube);
    }}

    // Hoverable objects registry & projected 2D labels registry
    const hoverables = [];
    const labelAnchors = [];
    const hlInfo = state.highlight || {{ type: 'all', id: null, downstream_pipe_ids: [] }};
    const isFocused = hlInfo.type && hlInfo.type !== 'all';

    function registerLabel(text, worldPos, clsName = 'ag-label', borderColor = null) {{
      const el = document.createElement('div');
      el.className = clsName;
      el.textContent = text;
      if (borderColor) el.style.borderColor = borderColor;
      labelsLayer.appendChild(el);
      labelAnchors.push({{ el, pos: worldPos.clone() }});
      return el;
    }}

    // 2. Sector zones (coloured semi-transparent ground areas + borders + floating labels)
    (state.sectors || []).forEach(sec => {{
      const poly = sec.polygon_xz || [];
      if (poly.length < 3) return;
      const shape = new THREE.Shape();
      shape.moveTo(poly[0][0] * SCALE, -poly[0][1] * SCALE);
      for (let i = 1; i < poly.length; i++) {{
        shape.lineTo(poly[i][0] * SCALE, -poly[i][1] * SCALE);
      }}
      shape.closePath();

      const secHl = !!sec.highlighted;
      const secDim = isFocused && hlInfo.type === 'sector' && !secHl;
      const fillOp = secHl ? 0.28 : (secDim ? 0.05 : 0.15);

      const zoneMesh = new THREE.Mesh(
        new THREE.ShapeGeometry(shape),
        new THREE.MeshBasicMaterial({{
          color: sec.color,
          transparent: true,
          opacity: fillOp,
          side: THREE.DoubleSide,
          depthWrite: false
        }})
      );
      zoneMesh.rotation.x = -Math.PI / 2;
      zoneMesh.position.y = 0.06;
      zoneMesh.userData.tooltip = `<strong>${{sec.name}}</strong> (${{sec.color_name}})<br/>Chlorine: ${{sec.chlorine_mg_l}} mg/L (${{sec.compliance_status}})<br/>Total Flow: ${{sec.total_flow_lps}} L/s · Pipes: ${{sec.pipe_count}} · Active Leaks: ${{sec.active_leaks}}`;
      scene.add(zoneMesh);
      hoverables.push(zoneMesh);

      // Sector outline
      const borderPts = poly.map(p => new THREE.Vector3(p[0] * SCALE, 0.14, p[1] * SCALE));
      const borderGeo = new THREE.BufferGeometry().setFromPoints(borderPts);
      const borderMat = new THREE.LineBasicMaterial({{
        color: sec.color,
        transparent: true,
        opacity: secDim ? 0.30 : 0.95
      }});
      scene.add(new THREE.Line(borderGeo, borderMat));

      // Floating sector label
      const cx = sec.centroid_xz[0] * SCALE;
      const cz = sec.centroid_xz[1] * SCALE;
      const lbl = registerLabel(sec.name, new THREE.Vector3(cx, 6.5, cz), 'ag-label', sec.color);
      if (secDim) lbl.style.opacity = '0.45';
    }});

    // 3. Procedural simulated city — InstancedMesh for smooth laptop FPS
    const buildings = state.buildings || [];
    if (buildings.length > 0) {{
      const boxGeo = new THREE.BoxGeometry(1, 1, 1);
      const boxMat = new THREE.MeshStandardMaterial({{
        color: '#FFFFFF',
        roughness: 0.72,
        metalness: 0.22
      }});
      const inst = new THREE.InstancedMesh(boxGeo, boxMat, buildings.length);
      const dummy = new THREE.Object3D();
      const tmpColor = new THREE.Color();
      const baseSlate = new THREE.Color('#273549');

      buildings.forEach((b, idx) => {{
        const w = Math.max(0.8, b.w * SCALE);
        const d = Math.max(0.8, b.d * SCALE);
        const h = Math.max(0.9, b.h * SCALE);
        dummy.position.set(b.x * SCALE, h * 0.5, b.z * SCALE);
        dummy.scale.set(w, h, d);
        dummy.updateMatrix();
        inst.setMatrixAt(idx, dummy.matrix);

        const bSecDim = isFocused && hlInfo.type === 'sector' && b.sector !== hlInfo.id;
        tmpColor.copy(baseSlate).lerp(new THREE.Color(b.color || '#64748B'), bSecDim ? 0.08 : 0.28);
        inst.setColorAt(idx, tmpColor);
      }});
      inst.instanceMatrix.needsUpdate = true;
      if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
      scene.add(inst);
    }}

    // 4. DESAL-01 Coastal Desalination Plant
    const plant = state.plant || {{ id: 'DESAL-01', pos_xz: [-13000, 11000], output_mld: 100, max_capacity_mld: 120, algae_cap_mld: 120, status: 'ONLINE' }};
    const plantPos = new THREE.Vector3(plant.pos_xz[0] * SCALE, 0, plant.pos_xz[1] * SCALE);
    const plantGroup = new THREE.Group();
    plantGroup.position.copy(plantPos);

    const plantHall = new THREE.Mesh(
      new THREE.BoxGeometry(7.2, 3.2, 4.8),
      new THREE.MeshStandardMaterial({{ color: '#334155', roughness: 0.5, metalness: 0.45 }})
    );
    plantHall.position.y = 1.6;
    plantHall.userData.tooltip = `<strong>${{plant.id}} Desalination Plant</strong><br/>Output: ${{plant.output_mld}} / ${{plant.max_capacity_mld}} MLD<br/>Algae Cap: ${{plant.algae_cap_mld}} MLD · Status: ${{plant.status}}`;
    plantGroup.add(plantHall);
    hoverables.push(plantHall);

    // Output gauge glow bar & status beacon on top of plant (animates smoothly from previous output)
    const plantMaxCap = Math.max(1, plant.max_capacity_mld || 120);
    const targetPlantOut = Number(plant.output_mld || 0);
    const prevPlantOut = (prevState && typeof prevState.plantOutput === 'number') ? prevState.plantOutput : targetPlantOut;
    const initPlantRatio = Math.max(0.05, Math.min(1.0, prevPlantOut / plantMaxCap));

    const gaugeBg = new THREE.Mesh(
      new THREE.BoxGeometry(6.2, 0.72, 1.05),
      new THREE.MeshStandardMaterial({{ color: '#1E293B', roughness: 0.7 }})
    );
    gaugeBg.position.set(0, 3.6, 0);
    plantGroup.add(gaugeBg);

    const gaugeMat = new THREE.MeshStandardMaterial({{ color: '#38BDF8', emissive: '#0284C7', emissiveIntensity: 0.9 }});
    const gaugeBar = new THREE.Mesh(new THREE.BoxGeometry(6.0, 0.76, 1.1), gaugeMat);
    gaugeBar.scale.x = initPlantRatio;
    gaugeBar.position.set(-3.0 + 3.0 * initPlantRatio, 3.6, 0);
    plantGroup.add(gaugeBar);

    const plantBeacon = new THREE.Mesh(
      new THREE.SphereGeometry(0.55, 12, 12),
      new THREE.MeshStandardMaterial({{ color: '#38BDF8', emissive: '#38BDF8', emissiveIntensity: 1.0 }})
    );
    plantBeacon.position.set(3.2, 3.9, 1.6);
    plantGroup.add(plantBeacon);

    // Sea intake pipes extending west into sea
    [-1.2, 1.2].forEach(zOff => {{
      const intake = new THREE.Mesh(
        new THREE.CylinderGeometry(0.28, 0.28, 7.0, 10),
        new THREE.MeshStandardMaterial({{ color: '#0EA5E9', metalness: 0.5, roughness: 0.35 }})
      );
      intake.rotation.z = Math.PI / 2;
      intake.position.set(-6.5, 0.45, zOff);
      plantGroup.add(intake);
    }});
    scene.add(plantGroup);
    const plantLblEl = registerLabel(`DESAL-01 · ${{targetPlantOut}} MLD`, new THREE.Vector3(plantPos.x, 5.4, plantPos.z), 'ag-label', '#38BDF8');

    // 5. Reservoir RES-01 (cylindrical tank with smooth animated inner water level + safety floor ring)
    const res = state.reservoir || {{ id: 'RES-01', pos_xz: [-11000, 8800], level_percent: 48, safety_floor_percent: 35, volume_ml: 240, capacity_ml: 500 }};
    const resPos = new THREE.Vector3(res.pos_xz[0] * SCALE, 0, res.pos_xz[1] * SCALE);
    const tankRadius = 4.2;
    const tankHeight = 6.2;

    const tankOuter = new THREE.Mesh(
      new THREE.CylinderGeometry(tankRadius, tankRadius, tankHeight, 28, 1, true),
      new THREE.MeshPhysicalMaterial({{
        color: '#CBD5E1',
        transparent: true,
        opacity: 0.28,
        roughness: 0.15,
        metalness: 0.1,
        side: THREE.DoubleSide
      }})
    );
    tankOuter.position.set(resPos.x, tankHeight * 0.5, resPos.z);
    tankOuter.userData.tooltip = `<strong>${{res.id}} Strategic Reservoir</strong><br/>Level: ${{res.level_percent}}% (${{res.volume_ml}} / ${{res.capacity_ml}} ML)<br/>Safety Floor: ${{res.safety_floor_percent}}%${{res.has_active_emergency ? ' (EMERGENCY RAISED)' : ''}}`;
    scene.add(tankOuter);
    hoverables.push(tankOuter);

    const tankBase = new THREE.Mesh(
      new THREE.CylinderGeometry(tankRadius + 0.3, tankRadius + 0.4, 0.4, 28),
      new THREE.MeshStandardMaterial({{ color: '#475569', roughness: 0.8 }})
    );
    tankBase.position.set(resPos.x, 0.2, resPos.z);
    scene.add(tankBase);

    const targetResPct = Number(res.level_percent || 48.0);
    const prevResPct = (prevState && typeof prevState.reservoirLevel === 'number') ? prevState.reservoirLevel : targetResPct;
    const waterMat = new THREE.MeshStandardMaterial({{
      color: targetResPct < res.safety_floor_percent ? '#EF4444' : '#0EA5E9',
      emissive: '#0369A1',
      emissiveIntensity: 0.45,
      transparent: true,
      opacity: 0.86,
      roughness: 0.2
    }});
    // Unit-height cylinder scaled dynamically on Y for smooth level animation
    const waterMesh = new THREE.Mesh(
      new THREE.CylinderGeometry(tankRadius - 0.25, tankRadius - 0.25, 1.0, 28),
      waterMat
    );
    const initWaterH = tankHeight * Math.max(0.03, Math.min(0.98, prevResPct / 100.0));
    waterMesh.scale.y = initWaterH;
    waterMesh.position.set(resPos.x, initWaterH * 0.5 + 0.2, resPos.z);
    scene.add(waterMesh);

    // Safety floor ring around tank
    const floorY = 0.2 + tankHeight * Math.max(0.05, Math.min(0.95, (res.safety_floor_percent || 35) / 100.0));
    const floorRing = new THREE.Mesh(
      new THREE.TorusGeometry(tankRadius + 0.08, 0.12, 10, 36),
      new THREE.MeshBasicMaterial({{ color: '#F59E0B' }})
    );
    floorRing.rotation.x = Math.PI / 2;
    floorRing.position.set(resPos.x, floorY, resPos.z);
    scene.add(floorRing);

    const resLblEl = registerLabel(`RES-01 · ${{targetResPct.toFixed(1)}}%`, new THREE.Vector3(resPos.x, tankHeight + 1.8, resPos.z), 'ag-label', '#0EA5E9');

    // Supply transmission pipe connecting DESAL-01 -> RES-01
    const plantToResCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(plantPos.x + 2.5, 0.8, plantPos.z),
      new THREE.Vector3(resPos.x, 0.8, resPos.z)
    ]);
    scene.add(new THREE.Mesh(
      new THREE.TubeGeometry(plantToResCurve, 12, 0.55, 8, false),
      new THREE.MeshStandardMaterial({{ color: '#38BDF8', emissive: '#0369A1', emissiveIntensity: 0.3 }})
    ));

    // 6. Pipes as 3D tubes coloured by sector + live water-flow particle tracks
    const pipeFlowTracks = [];
    // Add supply main flow track
    pipeFlowTracks.push({{
      curve: plantToResCurve,
      prevFlow: prevPlantOut * 1.1,
      targetFlow: targetPlantOut * 1.1,
      currentFlow: prevPlantOut * 1.1,
      mat: null,
      activeColor: new THREE.Color('#38BDF8'),
      dryColor: new THREE.Color('#475569'),
      delay: 0.0,
      particles: Array.from({{ length: lowQ ? 3 : 6 }}, (_, i) => ({{ phase: i / (lowQ ? 3 : 6) }}))
    }});

    (state.pipes || []).forEach(pipe => {{
      const pts = (pipe.points_xz || []).map(p => new THREE.Vector3(p[0] * SCALE, 0.85, p[1] * SCALE));
      if (pts.length < 2) return;
      const curve = new THREE.CatmullRomCurve3(pts);
      const radius = pipe.is_trunk ? 0.52 : 0.28;
      const pHl = !!pipe.highlighted;
      const pDim = isFocused && !pHl && (hlInfo.type === 'sector' || hlInfo.type === 'valve');

      const targetFlow = Number(pipe.flow_lps || 0);
      const prevFlow = (prevState && prevState.pipeFlows && typeof prevState.pipeFlows[pipe.segment_id] === 'number')
        ? prevState.pipeFlows[pipe.segment_id]
        : targetFlow;
      const activeCol = new THREE.Color(pipe.color || '#2563EB');
      const dryCol = new THREE.Color('#475569');

      const mat = new THREE.MeshStandardMaterial({{
        color: prevFlow <= 0.05 ? dryCol : activeCol,
        emissive: pHl ? activeCol : new THREE.Color('#000000'),
        emissiveIntensity: pHl ? 0.5 : 0.0,
        roughness: 0.35,
        metalness: 0.3,
        transparent: pDim,
        opacity: pDim ? 0.28 : 0.96
      }});
      const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(12, pts.length * 6), radius, 8, false), mat);
      const rerouteTag = pipe.rerouted ? ' · REROUTED' : '';
      tube.userData.tooltip = `<strong>${{pipe.segment_id}}</strong> (${{pipe.sector}})<br/>${{pipe.name}}<br/>${{pipe.diameter_mm}}mm ${{pipe.is_trunk ? 'Trunk' : 'Distribution'}} · Valve: ${{pipe.isolation_valve_id}}<br/>Flow: <strong>${{pipe.flow_lps}} L/s</strong>${{rerouteTag}} · Pressure: ${{pipe.pressure_psi}} psi`;
      scene.add(tube);
      hoverables.push(tube);

      // Particle count proportional to max(prevFlow, targetFlow); 0 particles if flow is 0 both before and after
      const maxFlow = Math.max(prevFlow, targetFlow);
      const maxCount = lowQ ? 4 : 9;
      const pCount = maxFlow <= 0.05 ? 0 : Math.max(2, Math.min(maxCount, Math.round(maxFlow / (lowQ ? 35 : 16))));
      const flowChanged = Math.abs(targetFlow - prevFlow) > 0.5;
      pipeFlowTracks.push({{
        curve,
        prevFlow,
        targetFlow,
        currentFlow: prevFlow,
        mat,
        activeColor: activeCol,
        dryColor: dryCol,
        delay: flowChanged ? 0.65 : 0.0,
        particles: Array.from({{ length: pCount }}, (_, i) => ({{ phase: i / Math.max(1, pCount) }}))
      }});
    }});

    // Single InstancedMesh for all moving water flow particles across the whole network
    const totalFlowParticles = pipeFlowTracks.reduce((acc, tr) => acc + tr.particles.length, 0);
    let flowInst = null;
    if (totalFlowParticles > 0) {{
      flowInst = new THREE.InstancedMesh(
        new THREE.SphereGeometry(0.32, 8, 8),
        new THREE.MeshBasicMaterial({{ color: '#BAE6FD' }}),
        totalFlowParticles
      );
      scene.add(flowInst);
    }}

    // 7. Valves as 3D handwheels on pipes (green OPEN, red CLOSED) with ~2s diff turn + color transition
    const valveAnims = [];
    const colOpen = new THREE.Color('#22C55E');
    const colClosed = new THREE.Color('#EF4444');
    (state.valves || []).forEach(v => {{
      const vx = v.pos_xz[0] * SCALE;
      const vz = v.pos_xz[1] * SCALE;
      const targetOpen = String(v.status).toUpperCase() === 'OPEN';
      const prevStatStr = (prevState && prevState.valves && prevState.valves[v.valve_id]) ? prevState.valves[v.valve_id] : v.status;
      const prevOpen = String(prevStatStr).toUpperCase() === 'OPEN';
      const changed = prevOpen !== targetOpen;

      const initColor = prevOpen ? colOpen.clone() : colClosed.clone();
      const targetColor = targetOpen ? colOpen.clone() : colClosed.clone();
      const vColorHex = targetOpen ? '#22C55E' : '#EF4444';

      const vGroup = new THREE.Group();
      vGroup.position.set(vx, 1.55, vz);

      const stem = new THREE.Mesh(
        new THREE.CylinderGeometry(0.14, 0.14, 1.1, 8),
        new THREE.MeshStandardMaterial({{ color: '#94A3B8', metalness: 0.6 }})
      );
      stem.position.y = -0.35;
      vGroup.add(stem);

      const wheelGroup = new THREE.Group();
      const wheelMat = new THREE.MeshStandardMaterial({{
        color: initColor.clone(),
        emissive: initColor.clone(),
        emissiveIntensity: v.highlighted ? 0.65 : 0.3,
        roughness: 0.3,
        metalness: 0.4
      }});
      const wheelRing = new THREE.Mesh(new THREE.TorusGeometry(0.72, 0.16, 10, 20), wheelMat);
      wheelRing.rotation.x = Math.PI / 2;
      wheelGroup.add(wheelRing);

      const spoke1 = new THREE.Mesh(new THREE.BoxGeometry(1.35, 0.12, 0.12), wheelMat);
      const spoke2 = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.12, 1.35), wheelMat);
      wheelGroup.add(spoke1, spoke2);
      wheelGroup.position.y = 0.2;
      vGroup.add(wheelGroup);

      const hit = new THREE.Mesh(
        new THREE.SphereGeometry(1.4, 8, 8),
        new THREE.MeshBasicMaterial({{ transparent: true, opacity: 0.0, depthWrite: false }})
      );
      hit.userData.tooltip = `<strong>Valve ${{v.valve_id}}</strong> (${{v.sector}})<br/>Status: <span style="color:${{vColorHex}}; font-weight:700;">${{v.status}}</span>${{v.pipe_segment_id ? '<br/>Segment: ' + v.pipe_segment_id : ''}}`;
      vGroup.add(hit);
      hoverables.push(hit);

      scene.add(vGroup);
      const lblEl = registerLabel(v.valve_id, new THREE.Vector3(vx, 2.7, vz), 'ag-label ag-label-valve', vColorHex);

      if (changed) {{
        valveAnims.push({{
          wheelGroup,
          wheelMat,
          lblEl,
          fromColor: initColor,
          toColor: targetColor,
          spinSign: targetOpen ? -1.0 : 1.0,
          duration: 2.0
        }});
      }}
    }});

    // 8. Sensors as small glowing points (quarantined = grey + warning icon)
    (state.sensors || []).forEach(s => {{
      const sx = s.pos_xz[0] * SCALE;
      const sz = s.pos_xz[1] * SCALE;
      const isQ = !!s.quarantined;
      const sColor = isQ ? '#94A3B8' : '#38BDF8';
      const sMesh = new THREE.Mesh(
        new THREE.SphereGeometry(isQ ? 0.55 : 0.42, 12, 12),
        new THREE.MeshStandardMaterial({{
          color: sColor,
          emissive: isQ ? '#475569' : '#0EA5E9',
          emissiveIntensity: isQ ? 0.25 : 0.95
        }})
      );
      sMesh.position.set(sx, 1.6, sz);
      sMesh.userData.tooltip = `<strong>Sensor ${{s.sensor_id}}</strong> (${{s.junction_id}}, ${{s.sector}})<br/>Reading: ${{s.value}} ${{s.unit}}<br/>Status: <strong>${{s.status}}</strong>${{isQ && s.reasons && s.reasons.length ? '<br/>Reason: ' + s.reasons[0] : ''}}`;
      scene.add(sMesh);
      hoverables.push(sMesh);

      if (isQ) {{
        registerLabel(`⚠ ${{s.sensor_id}}`, new THREE.Vector3(sx, 2.8, sz), 'ag-label ag-label-warn');
      }}
    }});

    // 9. Leaks with water spray fountain + expanding street puddle (spray stops when isolated!)
    const leakEffects = [];
    const dropletGeo = new THREE.SphereGeometry(0.18, 6, 6);
    const dropletMat = new THREE.MeshBasicMaterial({{ color: '#7DD3FC', transparent: true, opacity: 0.9 }});

    (state.leaks || []).forEach(lk => {{
      const wasActive = !!(prevState && prevState.activeLeaks && prevState.activeLeaks[lk.leak_id]);
      const isActive = !!lk.active && !lk.isolated;
      const justIsolated = wasActive && !isActive;
      if (!isActive && !justIsolated) return;

      const lx = lk.pos_xz[0] * SCALE;
      const lz = lk.pos_xz[1] * SCALE;

      const coreMat = new THREE.MeshStandardMaterial({{
        color: isActive ? '#EF4444' : '#94A3B8',
        emissive: isActive ? '#DC2626' : '#475569',
        emissiveIntensity: isActive ? 1.0 : 0.3
      }});
      const core = new THREE.Mesh(new THREE.SphereGeometry(0.9, 16, 16), coreMat);
      core.position.set(lx, 1.5, lz);
      core.userData.tooltip = `<strong>${{lk.leak_id}}</strong> (${{lk.severity}} · ${{isActive ? lk.status : 'ISOLATED'}})<br/>${{lk.location_name}}<br/>Loss: <strong>${{lk.estimated_loss_m3_per_hour}} m³/hr</strong>`;
      scene.add(core);
      hoverables.push(core);

      const pulseRing = new THREE.Mesh(
        new THREE.RingGeometry(1.1, 1.8, 28),
        new THREE.MeshBasicMaterial({{ color: '#EF4444', transparent: true, opacity: isActive ? 0.75 : 0.0, side: THREE.DoubleSide }})
      );
      pulseRing.rotation.x = -Math.PI / 2;
      pulseRing.position.set(lx, 0.25, lz);
      scene.add(pulseRing);

      // Street puddle beneath the leak
      const puddleMat = new THREE.MeshBasicMaterial({{
        color: '#0EA5E9',
        transparent: true,
        opacity: isActive ? 0.48 : 0.35,
        side: THREE.DoubleSide
      }});
      const puddle = new THREE.Mesh(new THREE.CircleGeometry(2.6, 24), puddleMat);
      puddle.rotation.x = -Math.PI / 2;
      puddle.position.set(lx, 0.16, lz);
      scene.add(puddle);

      // Parabolic water spray fountain droplets
      const dropCount = lowQ ? 8 : 18;
      const sprayDrops = [];
      for (let i = 0; i < dropCount; i++) {{
        const ang = (i / dropCount) * Math.PI * 2 + (i % 3) * 0.25;
        const speed = 1.6 + (i % 4) * 0.45;
        const drop = new THREE.Mesh(dropletGeo, dropletMat);
        drop.position.set(lx, 1.4, lz);
        scene.add(drop);
        sprayDrops.push({{ mesh: drop, ang, speed, offset: i / dropCount }});
      }}

      leakEffects.push({{
        lx, lz, core, ring: pulseRing, puddle, puddleMat, sprayDrops,
        isActive, justIsolated
      }});
      registerLabel(
        isActive ? `🚨 ${{lk.leak_id}}` : `✓ ${{lk.leak_id}} ISOLATED`,
        new THREE.Vector3(lx, 4.0, lz),
        'ag-label',
        isActive ? '#EF4444' : '#22C55E'
      );
    }});

    // Reset view button (smoothly flies back to default camera & updates localStorage)
    document.getElementById('ag-reset-btn').addEventListener('click', () => {{
      flyAnim = {{
        startPos: camera.position.clone(),
        endPos: defaultCamPos.clone(),
        startTarget: controls.target.clone(),
        endTarget: defaultTarget.clone(),
        startTime: clock.getElapsedTime(),
        duration: 0.85
      }};
    }});

    // Raycaster hover tooltips
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2(-99, -99);
    renderer.domElement.addEventListener('pointermove', (ev) => {{
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(hoverables, false);
      if (hits.length > 0 && hits[0].object.userData.tooltip) {{
        tooltipEl.innerHTML = hits[0].object.userData.tooltip;
        tooltipEl.style.display = 'block';
        const tx = Math.min(rect.width - 270, Math.max(8, ev.clientX - rect.left + 12));
        const ty = Math.min(rect.height - 90, Math.max(8, ev.clientY - rect.top + 12));
        tooltipEl.style.left = tx + 'px';
        tooltipEl.style.top = ty + 'px';
      }} else {{
        tooltipEl.style.display = 'none';
      }}
    }});
    renderer.domElement.addEventListener('pointerleave', () => {{
      tooltipEl.style.display = 'none';
    }});

    // Resize handling
    window.addEventListener('resize', () => {{
      const w = container.clientWidth || 640;
      const h = container.clientHeight || {h_px};
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }});

    // Animation loop (water particles, valve turn/color diff, reservoir level, plant glow, leak spray & camera fly-to)
    const projVec = new THREE.Vector3();
    const pDummy = new THREE.Object3D();
    const pPos = new THREE.Vector3();
    const clock = new THREE.Clock();
    let prevTime = 0;

    function easeOutCubic(x) {{
      return 1 - Math.pow(1 - Math.max(0, Math.min(1, x)), 3);
    }}

    function animate() {{
      requestAnimationFrame(animate);
      const t = clock.getElapsedTime();
      const dt = Math.min(0.05, Math.max(0.001, t - prevTime));
      prevTime = t;

      // 1. Camera fly-to transition (if focus changed or Reset clicked)
      if (flyAnim) {{
        const u = easeOutCubic((t - flyAnim.startTime) / flyAnim.duration);
        camera.position.lerpVectors(flyAnim.startPos, flyAnim.endPos, u);
        controls.target.lerpVectors(flyAnim.startTarget, flyAnim.endTarget, u);
        if (u >= 1.0) {{
          flyAnim = null;
          saveCameraNow();
        }}
      }}
      controls.update();

      // 2. Smooth reservoir level & plant output gauge animation (~2.0s)
      const lvlU = easeOutCubic(t / 2.0);
      const curResPct = prevResPct + (targetResPct - prevResPct) * lvlU;
      const curWaterH = tankHeight * Math.max(0.03, Math.min(0.98, curResPct / 100.0));
      waterMesh.scale.y = curWaterH;
      waterMesh.position.y = curWaterH * 0.5 + 0.2;
      resLblEl.textContent = `RES-01 · ${{curResPct.toFixed(1)}}%`;

      const curPlantOut = prevPlantOut + (targetPlantOut - prevPlantOut) * lvlU;
      const curPlantRatio = Math.max(0.05, Math.min(1.0, curPlantOut / plantMaxCap));
      gaugeBar.scale.x = curPlantRatio;
      gaugeBar.position.x = -3.0 + 3.0 * curPlantRatio;
      gaugeMat.emissiveIntensity = 0.65 + 0.35 * Math.sin(t * 2.5) * curPlantRatio;
      plantBeacon.scale.setScalar(0.85 + 0.25 * Math.sin(t * 3.5));
      plantLblEl.textContent = `DESAL-01 · ${{curPlantOut.toFixed(0)}} MLD`;

      // 3. Valve wheel rotation + green <-> red transition (~2.0s)
      for (const va of valveAnims) {{
        const vu = easeOutCubic(t / va.duration);
        va.wheelGroup.rotation.y = va.spinSign * Math.PI * 2.0 * vu;
        va.wheelMat.color.copy(va.fromColor).lerp(va.toColor, vu);
        va.wheelMat.emissive.copy(va.fromColor).lerp(va.toColor, vu);
      }}

      // 4. Water flow particles along each pipe (speed & visibility proportional to flow; none when flow == 0)
      if (flowInst) {{
        let instIdx = 0;
        for (const tr of pipeFlowTracks) {{
          const fu = tr.delay > 0 ? easeOutCubic((t - tr.delay) / 1.8) : 1.0;
          tr.currentFlow = tr.prevFlow + (tr.targetFlow - tr.prevFlow) * fu;
          if (tr.mat) {{
            const wetFrac = Math.min(1.0, tr.currentFlow / 25.0);
            tr.mat.color.copy(tr.dryColor).lerp(tr.activeColor, wetFrac);
          }}
          const speed = (tr.currentFlow / 100.0) * 0.28;
          const visibleScale = tr.currentFlow <= 0.05 ? 0.0 : Math.min(1.15, 0.55 + (tr.currentFlow / 120.0) * 0.6);

          for (const pt of tr.particles) {{
            if (visibleScale > 0.0) {{
              pt.phase = (pt.phase + dt * speed) % 1.0;
              tr.curve.getPointAt(pt.phase, pPos);
              pDummy.position.set(pPos.x, pPos.y + 0.42, pPos.z);
              pDummy.scale.setScalar(visibleScale);
            }} else {{
              pDummy.position.set(0, -100, 0);
              pDummy.scale.setScalar(0.0);
            }}
            pDummy.updateMatrix();
            flowInst.setMatrixAt(instIdx++, pDummy.matrix);
          }}
        }}
        flowInst.instanceMatrix.needsUpdate = true;
      }}

      // 5. Leak water spray fountain + puddle (stops smoothly when isolated!)
      for (const lk of leakEffects) {{
        const sprayIntensity = lk.isActive ? 1.0 : Math.max(0.0, 1.0 - easeOutCubic((t - 0.8) / 1.6));
        if (lk.isActive) {{
          const s = 1.0 + 0.3 * Math.sin(t * 4.5);
          lk.core.scale.setScalar(s);
          const rScale = 1.0 + ((t * 1.4) % 1.8);
          lk.ring.scale.setScalar(rScale);
          lk.ring.material.opacity = Math.max(0.0, 0.8 - (rScale - 1.0) * 0.45);
          lk.puddle.scale.setScalar(0.95 + 0.12 * Math.sin(t * 2.2));
        }} else {{
          lk.core.scale.setScalar(0.75);
          lk.ring.material.opacity = 0.0;
          lk.puddleMat.opacity = 0.15 + 0.25 * sprayIntensity;
        }}
        for (const dr of lk.sprayDrops) {{
          if (sprayIntensity <= 0.02) {{
            dr.mesh.visible = false;
          }} else {{
            dr.mesh.visible = true;
            const cyc = (t * 1.6 + dr.offset) % 1.0;
            const r = cyc * dr.speed * 1.6 * sprayIntensity;
            const y = 1.3 + (4.2 * cyc - 4.2 * cyc * cyc) * (2.6 * sprayIntensity);
            dr.mesh.position.set(lk.lx + Math.cos(dr.ang) * r, Math.max(0.2, y), lk.lz + Math.sin(dr.ang) * r);
            dr.mesh.scale.setScalar(0.8 * sprayIntensity);
          }}
        }}
      }}

      // 6. Project floating 3D labels to 2D screen positions
      const cw = container.clientWidth;
      const ch = container.clientHeight;
      for (const item of labelAnchors) {{
        projVec.copy(item.pos).project(camera);
        if (projVec.z > 1.0) {{
          item.el.style.display = 'none';
        }} else {{
          const sx = (projVec.x * 0.5 + 0.5) * cw;
          const sy = (-projVec.y * 0.5 + 0.5) * ch;
          if (sx < -40 || sx > cw + 40 || sy < -30 || sy > ch + 30) {{
            item.el.style.display = 'none';
          }} else {{
            item.el.style.display = 'block';
            item.el.style.left = sx + 'px';
            item.el.style.top = sy + 'px';
          }}
        }}
      }}

      renderer.render(scene, camera);
    }}
    animate();
  }} catch (err) {{
    console.error('AquaGrid 3D Twin WebGL initialization failed:', err);
    if (window.render2DFallback) {{
      window.render2DFallback('WebGL failed to initialize — showing the 2D OpenStreetMap fallback view.');
    }}
  }}
}}
</script>
</body>
</html>
"""


def render_3d_digital_twin(
    focus: str = "all",
    widget_key: str = "twin_3d_main",
    height: int = 460,
    current_hour: int = DEFAULT_SCADA_HOUR,
    highlight_pipe_ids: Optional[list[str]] = None,
    flow_override_by_segment: Optional[dict[str, float]] = None,
    valve_override: Optional[dict[str, str]] = None,
    show_controls: bool = True,
    preserve_camera: Optional[bool] = None,
) -> dict:
    """Render the interactive 3D Digital Twin inside Streamlit with Low quality toggle. Returns the scene state."""
    low_q = False
    if show_controls:
        c_info, c_lq = st.columns([3.2, 1.0])
        with c_lq:
            low_q = st.checkbox("Low quality", value=False, key=f"twin_lowq_{widget_key}")
        with c_info:
            st.caption("Drag to rotate · Scroll to zoom · Right-drag to pan · Hover assets for live values")

    # Automatically preserve user camera angle when Live Simulation is ON
    eff_preserve = bool(st.session_state.get("live_sim_toggle", False)) if preserve_camera is None else bool(preserve_camera)

    scene_state = build_3d_scene_state(
        focus=focus,
        current_hour=current_hour,
        low_quality=low_q,
        highlight_pipe_ids=highlight_pipe_ids,
        flow_override_by_segment=flow_override_by_segment,
        valve_override=valve_override,
    )
    html_str = build_3d_digital_twin_html(
        scene_state=scene_state,
        widget_id=widget_key,
        height=height,
        preserve_camera=eff_preserve,
    )
    components.html(html_str, height=height + 6, scrolling=False)
    return scene_state


def render_persistent_top_twin(
    display_hour: int = DEFAULT_SCADA_HOUR,
    snapshot: Optional[dict] = None,
    height: int = 450,
) -> dict:
    """
    Render the single persistent top-of-screen 3D Digital Twin (or 2D OpenStreetMap when sidebar '2D map' is ON)
    with a fixed component key so it never reloads across reruns, plus the one-line caption and camera/quality controls.
    """
    snapshot = snapshot or {}
    twin_view = st.session_state.get("twin_view")
    if not isinstance(twin_view, dict):
        twin_view = build_twin_view_command(focus="all", event_type="init", seq=0)
        st.session_state["twin_view"] = twin_view

    quality = st.session_state.get("twin_quality_mode", "High")
    low_q = (quality == "Low")
    follow_chat = bool(st.session_state.get("twin_follow_chat", True))
    show_2d_map = bool(st.session_state.get("twin_show_2d_map", False))

    scene_state = build_3d_scene_state(
        focus=str(twin_view.get("focus") or "all"),
        current_hour=display_hour,
        low_quality=low_q,
        highlight_pipe_ids=twin_view.get("highlight_pipe_ids") or [],
    )

    if show_2d_map:
        from streamlit_folium import st_folium
        from database.connection import get_db_connection
        from components.leak_map_view import build_leak_folium_map

        conn = get_db_connection()
        try:
            segs = [dict(r) for r in conn.execute("SELECT * FROM pipe_segments ORDER BY sector, segment_id;").fetchall()]
            leaks = [dict(r) for r in conn.execute("SELECT * FROM leaks ORDER BY detected_at DESC;").fetchall()]
            valves = {r["valve_id"]: r["status"] for r in conn.execute("SELECT valve_id, status FROM valves;").fetchall()}
            p_rows = conn.execute(
                "SELECT pipe_segment_id, pressure_psi FROM scada_history WHERE id IN (SELECT MAX(id) FROM scada_history GROUP BY pipe_segment_id);"
            ).fetchall()
        finally:
            conn.close()
        latest_p = {r["pipe_segment_id"]: float(r["pressure_psi"]) for r in p_rows if r["pressure_psi"] is not None}
        q_ids = {s["sensor_id"] for s in (snapshot.get("telemetry_data") or {}).get("quarantined_sensors", [])}
        hl_sec = twin_view.get("focus_id") if twin_view.get("focus_type") == "sector" else None
        sel_leak = None
        if twin_view.get("focus_type") == "leak" and twin_view.get("focus_id"):
            sel_leak = next((l for l in leaks if l["leak_id"].upper() == str(twin_view["focus_id"]).upper()), None)

        fmap = build_leak_folium_map(
            segments=segs,
            leaks=leaks,
            quarantined_ids=q_ids,
            latest_pressures=latest_p,
            selected_leak=sel_leak,
            show_boundaries=True,
            show_pipes=True,
            show_sensors=True,
            show_valves=True,
            show_leaks=True,
            highlight_sector=hl_sec,
            valve_statuses=valves,
            fit_bounds=True,
        )
        st_folium(fmap, height=height, width=None, returned_objects=[], key="aquagrid_top_2d_map")
    else:
        _aquagrid_3d_component(
            scene_state=scene_state,
            view_command=twin_view,
            quality=quality,
            follow_chat=follow_chat,
            height=height,
            force_no_webgl=False,
            key=TWIN_COMPONENT_KEY,
        )

    # One-line caption + twin controls bar directly underneath
    caption_text = build_twin_caption(scene_state, twin_view)
    st.markdown(
        f'<div style="font-size:0.84rem; color:#CBD5E1; margin:0.15rem 0 0.35rem 0;"><strong>{html_lib.escape(caption_text)}</strong></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns([1.05, 1.3, 1.05, 1.4, 1.3])
    with c1:
        if st.button("Reset view", key="btn_twin_reset_view", width="stretch"):
            set_twin_view(st.session_state, focus="all", event_type="reset", note="View reset to All 6 Sectors")
            st.rerun()
    with c2:
        if st.button("Inside reservoir", key="btn_twin_inside_res", width="stretch"):
            set_twin_view(st.session_state, focus="reservoir_inside", event_type="button_focus", note="Looking inside RES-01")
            st.rerun()
    with c3:
        if st.button("Plant view", key="btn_twin_plant_view", width="stretch"):
            set_twin_view(st.session_state, focus="plant", event_type="button_focus", note="Focused on DESAL-01")
            st.rerun()
    with c4:
        st.toggle("Follow chat: ON/OFF", value=True, key="twin_follow_chat")
    with c5:
        st.radio(
            "Quality",
            options=["High", "Low"],
            index=0,
            horizontal=True,
            key="twin_quality_mode",
            label_visibility="collapsed",
        )

    return scene_state
