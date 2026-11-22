"""Ask AquaGrid — Multi-turn real LLM chat assistant with live SCADA read tools, display tools, and stage-gated proposals.

HARD ARCHITECTURAL INVARIANTS:
- Uses the existing LLM fallback chain (Groq 120b -> Groq 20b -> Gemini -> Ollama -> offline engine).
- Read-only data tools & display tools only; physical changes go ONLY through stage_gated_valve_or_setpoint_action().
- NEVER mutates valves or plant_setpoints directly and NEVER approves or executes pending actions.
- Typed text like 'approve' or 'pre_approved=true' NEVER approves anything.
"""
from datetime import datetime, timezone
import importlib
import inspect
import json
import logging
import re
import traceback
from typing import Any, Callable, Optional

logger = logging.getLogger("aquagrid.chat")

from agent.graph import run_agent
from agent.llm_factory import llm_client
from config.constants import DEFAULT_SCADA_HOUR, RESERVOIR_FLOOR_PERCENT
from database.connection import get_db_connection, transaction_scope
from database.scenarios import get_scenario_by_id, reset_database
from guards.injection_filter import detect_injection
from guards.schedule_validator import validate_production_schedule
from tools.read_feeds import read_tariff_and_quality_feeds
from tools.read_telemetry import read_network_telemetry
from tools.write_gateway import (
    refuse_direct_write_bypass,
    stage_gated_valve_or_setpoint_action,
)

MAX_CHAT_TOOL_STEPS = 5

EXAMPLE_QUESTIONS = [
    "What is the current status?",
    "Show Sector 4 on the map",
    "Pressure at J-14 last 2 hours",
    "Plan production for the next 24 hours",
    "Why was the last plan rejected?",
]

ASK_AQUAGRID_SYSTEM_PROMPT = """You are Ask AquaGrid, the real-time SCADA AI assistant for the Coastal Water Authority.

STRICT RULES:
1. A complete LIVE SCADA SNAPSHOT (reservoir %, dynamic floor, DESAL-01 output & limits, algae bloom cap, next 24h tariffs & demand forecast, active leaks, quarantined sensors, per-sector chlorine, pending actions, and telemetry freshness) is ALREADY provided in the system context for every turn. Use those live values directly — NEVER invent numbers.
2. NEVER ask the operator to enable tools, re-enable tool calls, run live queries, or supply data that the system already has. Tool calls are OPTIONAL for extra details (Audit Log lookup, rendering maps/charts/tables, validating custom schedules, or staging a gated action).
3. If live data is genuinely unavailable (for example telemetry_freshness.is_telemetry_loss is true / TELEMETRY_LOSS), explicitly state WHICH telemetry data is missing or stale and WHY you cannot plan or stage setpoint/valve changes until the feed is restored.
4. Answer ANY operator question clearly and concisely (typically 2-6 short lines or compact bullets): plant status, leaks, quarantined sensors, reservoir & dynamic emergency floor, electricity tariffs, hourly production plans, past actions in the Audit Log, why a plan was rejected, what any technical status code means, and next steps. For simple greetings like "Hello" or "how are you", reply briefly in 1-2 friendly sentences (mentioning overall grid status) without dumping a full table unless asked.
5. Use prior conversation messages naturally for follow-up questions (e.g. "Why?", "What about Sector 2?", "And if demand rises?").
6. Use DISPLAY TOOLS to control the persistent 3D digital twin at the top of the screen and render charts/tables/approvals:
   - show_3d(focus): flies the top 3D digital twin to focus and highlights it (others dimmed). focus can be "all", "Sector 1".."Sector 6", valve_id ("V-12"), leak_id ("LEAK-001"), junction_id ("J-14"), "reservoir", "reservoir_inside", "plant", or "intake". Use automatically for any question about a sector, leak, valve, junction, reservoir, desalination plant, intake, or chlorine status.
   - show_map(sector, leak_id, show_boundaries): 2D OpenStreetMap view (use when the operator asks for "2D", "2D map", or "show the sectors" / "show the map").
   - show_chart(metric, asset, hours): metric in ["pressure", "flow", "reservoir", "tariff"].
   - show_table(what): what in ["leaks", "sensors", "schedule", "audit", "valves"].
   - show_schedule_validation(plan): validating a production schedule.
   - show_pending_approvals(): showing pending actions awaiting operator signoff.
   Never describe 3D or 2D features that are not listed in the tool result. Only cite targets, colours, flows, counts, and status values returned by the tool summary.
7. If the operator asks for a physical change (e.g. "reduce output to 70 MLD", "close V-12", "isolate and flush Sector 2"), you may ONLY call propose_gated_action. All Python safety guards run automatically inside the gateway. State the exact gateway status code (STAGED_PENDING_APPROVAL or REJECTED_* + reason) and remind the operator to click the Approve or Reject button on the approval card.
8. You can NEVER execute, auto-approve, or bypass operator approval. Keep exact technical status codes unchanged (STAGED_PENDING_APPROVAL, REJECTED_BY_SAFETY_GUARD, REJECTED_WATER_QUALITY, CANCELLED_BY_SAFETY_RECHECK, REJECTED_UNAUTHORIZED_APPROVAL_CLAIM, REJECTED_APPROVAL_MISMATCH, REJECTED_APPROVAL_ALREADY_USED, REJECTED_APPROVAL_EXPIRED, TELEMETRY_LOSS, REJECTED_STALE_TELEMETRY, CANCELLED_STALE_TELEMETRY, NON_COMPLIANT, UNTRUSTED_TEXT, VALID, GAP).
"""

CHAT_TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_network_telemetry",
            "description": "Read live verified sensor telemetry, confirmed pipe leaks, quarantined sensors, and telemetry freshness (FRESH vs TELEMETRY_LOSS).",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_hour": {"type": "integer", "description": "Hour of day (0-23). Defaults to active SCADA hour."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_tariff_and_quality_feeds",
            "description": "Read live reservoir level & dynamic safety floor, electricity tariffs, demand forecast, DESAL-01 output, algal bloom cap, per-sector chlorine compliance, sanitized alerts, field tickets, and emergency commitments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon_hours": {"type": "integer", "description": "Forecast horizon in hours (default 6)."},
                    "current_hour": {"type": "integer", "description": "Hour of day (0-23)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_valves_and_pending_actions",
            "description": "Read live isolation valve states (V-01..V-20), DESAL-01 plant setpoint, active emergency commitments, pending/resolved actions, and tracked map leaks.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_audit_log",
            "description": "Read recent immutable SCADA Audit Log entries to inspect past decisions, rejections, blocks, and operator actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max entries to return (default 20, max 50)."},
                    "event_filter": {"type": "string", "description": "Optional substring filter on event_type, status, or details."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_3d",
            "description": "Focus the persistent top 3D digital twin on a target, highlighting it (dimming others) and flying the camera to it. Returns an exact summary of what is shown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "'all', 'Sector 1'..'Sector 6', valve_id ('V-12'), leak_id ('LEAK-001'), junction_id ('J-14'), 'reservoir', 'reservoir_inside', 'plant', or 'intake'.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_map",
            "description": "Display the interactive Leaflet map with sector boundary polygons, pipes, valves, sensors, and leaks inside this chat reply. Returns exact layers, colours, asset counts, and zoom target drawn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "description": "'all' for all 6 sectors, a sector name like 'Sector 4', or a leak ID like 'LEAK-001'.",
                    },
                    "leak_id": {"type": "string", "description": "Optional explicit leak ID like 'LEAK-001'."},
                    "show_boundaries": {
                        "type": "boolean",
                        "description": "Draw buffered convex-hull sector boundary polygons (default true).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_chart",
            "description": "Display a live chart inside this chat reply for pressure, flow, reservoir, or tariff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["pressure", "flow", "reservoir", "tariff"],
                        "description": "Chart metric to display.",
                    },
                    "asset": {"type": "string", "description": "Optional junction, sensor, or sector (e.g. 'J-14', 'Sector 4', 'RES-01')."},
                    "hours": {"type": "integer", "description": "Lookback or horizon hours (default 2 for pressure/flow, 24 for reservoir/tariff)."},
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_table",
            "description": "Display a live data table inside this chat reply for leaks, sensors, schedule, audit, or valves.",
            "parameters": {
                "type": "object",
                "properties": {
                    "what": {
                        "type": "string",
                        "enum": ["leaks", "sensors", "schedule", "audit", "valves"],
                        "description": "Which table to render.",
                    }
                },
                "required": ["what"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_schedule_validation",
            "description": "Validate a production schedule using the deterministic Python physics guard and display the hour-by-hour check card in chat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": "Time-blocked schedule string, e.g. '14:00-15:00 at 65 MLD, 15:00-16:00 at 65 MLD'.",
                    },
                    "current_hour": {"type": "integer", "description": "Starting hour (default active SCADA hour)."},
                },
                "required": ["plan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_pending_approvals",
            "description": "Display interactive Pending Approval cards (with Approve and Reject buttons) inside this chat reply.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_gated_action",
            "description": "Submit a physical actuator suggestion to the stage-gated write gateway (STAGED_PENDING_APPROVAL only; never executes). All safety guards run automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["CLOSE_VALVE", "OPEN_VALVE", "SET_PLANT_OUTPUT", "ISOLATE_AND_FLUSH"],
                    },
                    "target_id": {
                        "type": "string",
                        "description": "Whitelisted target: V-01..V-20, DESAL-01, or SECTOR-01..SECTOR-06.",
                    },
                    "proposed_value": {
                        "type": "number",
                        "description": "0.0 for CLOSE_VALVE/ISOLATE_AND_FLUSH, 1.0 for OPEN_VALVE, or MLD (0-120) for SET_PLANT_OUTPUT.",
                    },
                    "rationale": {"type": "string", "description": "Concise operational reason for the control room operator."},
                },
                "required": ["action_type", "target_id", "proposed_value", "rationale"],
            },
        },
    },
]

_PRE_APPROVED_BYPASS_RE = re.compile(
    r"(?:pre[\s_-]*approved\s*(?:=\s*true)?|without\s+(?:human\s+|operator\s+)?approval|"
    r"skip\s+approval|bypass\s+(?:approval|gateway|hitl)|auto[\s_-]*approve|force\s+execute|direct\s+write)",
    re.IGNORECASE,
)

_TYPED_APPROVAL_RE = re.compile(
    r"^\s*(?:please\s+)?(?:yes[,\s]+)?(?:i\s+)?(?:approve|confirm\s+approval|execute\s+approval|approve\s+it|approve\s+all|approve\s+#?\d+|go\s+ahead\s+and\s+approve)\b[!.\s]*$",
    re.IGNORECASE,
)

_SCENARIO_TRIGGER_RE = re.compile(
    r"\b(?:run|start|trigger|execute|launch|load)\s+(?:scenario\s+)?(r1-s[1-4]|u-5\.[1-5]|[1-4])\b",
    re.IGNORECASE,
)

_PLAN_HOURS_RE = re.compile(
    r"(?:(?:next|for)\s+(\d{1,2})\s*[- ]?\s*h(?:our)?s?|(\d{1,2})\s*[- ]?\s*h(?:our)?s?\s*(?:production\s+)?(?:plan|schedule))",
    re.IGNORECASE,
)

_TOOL_BEGGING_RE = re.compile(
    r"(?:re-?enable\s+tool|enable\s+tool|run\s+the\s+live\s+queries|enable\s+function\s+call|"
    r"let\s+me\s+know\s+when\s+you(?:'d| would)\s+like\s+me\s+to\s+run|don't\s+have\s+access\s+to\s+live)",
    re.IGNORECASE,
)

_BOUNDARY_CLAIM_RE = re.compile(
    r"\b(?:boundar(?:y|ies)|border(?:s)?|polygon(?:s)?|distinct\s+colo(?:u)?r\s+(?:line|outline|border))\b",
    re.IGNORECASE,
)

_ASKING_WHICH_SECTOR_RE = re.compile(
    r"\b(?:which\s+sector|specify\s+(?:a\s+)?sector|please\s+tell\s+me\s+which\s+sector)\b",
    re.IGNORECASE,
)


def _emit_status(status_callback: Optional[Callable[[str], None]], message: str) -> None:
    """Safely send a live progress step to the UI status widget."""
    if callable(status_callback):
        try:
            status_callback(message)
        except Exception:
            pass


def _accepted_chat_optional_kwargs(fn: Any) -> tuple[bool, bool]:
    """Inspect fn (or its mock side_effect) to check whether fallback_context_builder and status_callback are accepted."""
    target = getattr(fn, "side_effect", None)
    check_fn = target if callable(target) else fn
    try:
        params = inspect.signature(check_fn).parameters
    except (TypeError, ValueError):
        return False, False
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    return (
        ("fallback_context_builder" in params or has_var_kw),
        ("status_callback" in params or has_var_kw),
    )


def _is_map_request(text: str) -> bool:
    """Return True when the operator explicitly asks for the 2D map or 'show the sectors'."""
    lower = (text or "").strip().lower()
    if not lower:
        return False
    if "3d" in lower:
        return False
    if "2d" in lower and any(k in lower for k in ("map", "sector", "leak", "valve", "reservoir", "plant", "show", "view")):
        return True
    if "map" in lower:
        return True
    if ("sectors" in lower or "all sector" in lower) and any(
        k in lower for k in ("show", "display", "draw", "view", "border", "boundar", "outline", "see")
    ):
        return True
    return False


def _parse_3d_focus_request(text: str) -> Optional[str]:
    """
    Detect queries that should focus the persistent top 3D digital twin automatically:
    'show Sector 4', 'where is the leak', 'show the reservoir', 'look inside the reservoir',
    'show the desalination plant', 'check valve V-12', 'what about Sector 2 chlorine', 'show intake', etc.
    Returns focus target string or None (returns None if operator explicitly asks for '2D').
    """
    cleaned = (text or "").strip()
    lower = cleaned.lower()
    if not lower or "2d" in lower:
        return None
    # Do not override pure 2D map queries like 'Show Sector 4 on the map' or 'show the sectors' unless '3d' is asked
    if "3d" not in lower and ("map" in lower or "sectors" in lower or "all sector" in lower):
        return None
    # Do not intercept action requests like "Close V-12" or "Open V-12"
    if re.search(r"\b(?:close|shut|open|isolate|flush|throttle|reduce|set)\b", lower) and not any(
        k in lower for k in ("show", "3d", "check", "inspect", "look")
    ):
        return None

    m_v = re.search(r"\b(v-?\d{1,2})\b", cleaned, re.IGNORECASE)
    if m_v and any(k in lower for k in ("show", "view", "where", "3d", "display", "focus", "zoom", "inspect", "twin", "check", "valve", "status")):
        digits = re.sub(r"\D", "", m_v.group(1))
        return f"V-{int(digits):02d}"

    m_l = re.search(r"\b(LEAK-\d+)\b", cleaned, re.IGNORECASE)
    if m_l and any(k in lower for k in ("show", "view", "where", "3d", "display", "focus", "zoom", "check", "inspect")):
        return m_l.group(1).upper()

    if "where" in lower and "leak" in lower:
        return "leak"

    if any(k in lower for k in ("reservoir", "res-01")) and any(
        k in lower for k in ("inside", "look", "show", "view", "3d", "display", "focus", "twin", "check")
    ):
        return "reservoir_inside"

    if "intake" in lower and any(k in lower for k in ("show", "view", "3d", "display", "focus", "twin", "check", "algae", "status")):
        return "intake"

    if any(k in lower for k in ("plant", "desal", "desal-01", "desalination")) and any(
        k in lower for k in ("show", "view", "3d", "display", "focus", "twin", "check", "inspect")
    ):
        return "plant"

    m_j = re.search(r"\b(j-?0?\d{1,2})\b", cleaned, re.IGNORECASE)
    if m_j and any(k in lower for k in ("show", "view", "where", "3d", "focus", "zoom", "inspect", "twin")):
        digits = re.sub(r"\D", "", m_j.group(1))
        return f"J-{int(digits):02d}"

    m_s = re.search(r"sector\s*([1-6])", cleaned, re.IGNORECASE)
    if m_s and any(
        k in lower for k in ("show", "view", "3d", "display", "focus", "zoom", "twin", "chlorine", "what about", "check", "status")
    ):
        return f"Sector {m_s.group(1)}"

    if any(k in lower for k in ("3d", "digital twin")):
        return "all"

    return None


def build_3d_render_summary(
    focus: str = "all",
    current_hour: int = DEFAULT_SCADA_HOUR,
    db_path: Optional[str] = None,
) -> dict:
    """Compute exact factual summary of what show_3d renders so the AI only describes what is actually shown."""
    from digital_twin.scene_state import build_3d_scene_state, build_twin_view_command

    raw_focus = (focus or "all").strip()
    if raw_focus.lower() == "leak":
        conn = get_db_connection(db_path)
        try:
            lk_row = conn.execute(
                "SELECT leak_id, sector FROM leaks WHERE LOWER(status) NOT IN ('resolved', 'false alarm – sensor fault') ORDER BY detected_at DESC LIMIT 1;"
            ).fetchone()
        finally:
            conn.close()
        raw_focus = lk_row["leak_id"] if lk_row else "Sector 4"

    state = build_3d_scene_state(focus=raw_focus, current_hour=current_hour, db_path=db_path)
    hl = state.get("highlight") or {}
    f_type = hl.get("type", "all")
    f_id = hl.get("id")
    cam_label = (state.get("camera_target") or {}).get("label", "All 6 Sectors")
    res = state.get("reservoir") or {}
    plant = state.get("plant") or {}
    pipes = state.get("pipes") or []
    valves = state.get("valves") or []
    leaks = state.get("leaks") or []
    junctions = state.get("junctions") or []
    active_leaks = [l for l in leaks if l.get("active")]
    sectors = state.get("sectors") or []
    hyd = state.get("hydraulics") or {}

    if f_type == "sector" and f_id:
        sec_obj = next((s for s in sectors if s["name"].lower() == f_id.lower()), None)
        if sec_obj:
            fire_note = ", FIRE EMERGENCY active" if sec_obj.get("has_fire_emergency") else ""
            summary_text = (
                f"3D digital twin focused on **{sec_obj['name']}** ({sec_obj['color_name']}, other sectors dimmed): "
                f"{sec_obj['pipe_count']} pipe(s), {sec_obj['valve_count']} valve(s), total flow **{sec_obj['total_flow_lps']:.1f} L/s**, "
                f"chlorine **{sec_obj['chlorine_mg_l']:.2f} mg/L** (`{sec_obj['compliance_status']}`), "
                f"**{sec_obj['active_leaks']}** active leak(s){fire_note}."
            )
        else:
            summary_text = f"3D digital twin focused on **{cam_label}**."
    elif f_type == "valve" and f_id:
        v_obj = next((v for v in valves if v["valve_id"].upper() == f_id.upper()), None)
        v_pipes = [p for p in pipes if p.get("isolation_valve_id", "").upper() == f_id.upper()]
        p_desc = ", ".join(f"{p['segment_id']} ({p['flow_lps']:.1f} L/s)" for p in v_pipes[:3]) or "no linked pipe"
        if v_obj:
            summary_text = (
                f"3D digital twin focused on **Valve {v_obj['valve_id']}** ({v_obj['sector']}, status **`{v_obj['status']}`**): "
                f"controlled pipe flow: {p_desc}."
            )
        else:
            summary_text = f"3D digital twin focused on **Valve {f_id}**."
    elif f_type == "leak" and f_id:
        l_obj = next((l for l in leaks if l["leak_id"].upper() == f_id.upper()), None)
        if l_obj:
            iso_note = "ISOLATED (0.0 m³/hr, spray stopped)" if l_obj.get("isolated") else f"ACTIVE ({l_obj['estimated_loss_m3_per_hour']:.1f} m³/hr spray)"
            summary_text = (
                f"3D digital twin focused on **{l_obj['leak_id']}** in **{l_obj['sector']}** "
                f"({l_obj['location_name']}, {l_obj['severity']} — {iso_note})."
            )
        else:
            summary_text = f"3D digital twin focused on **{f_id}** ({len(active_leaks)} active leak(s) network-wide)."
    elif f_type == "junction" and f_id:
        j_obj = next((j for j in junctions if j["junction_id"].upper() == f_id.upper()), None)
        if j_obj:
            summary_text = (
                f"3D digital twin focused on **Junction {j_obj['junction_id']}** ({j_obj['sector']}): "
                f"pressure **{j_obj['pressure_psi']:.1f} psi**, flow **{j_obj['flow_lps']:.1f} L/s**."
            )
        else:
            summary_text = f"3D digital twin focused on **Junction {f_id}**."
    elif f_type in ("reservoir", "reservoir_inside"):
        dyn_note = (
            f" (dynamic emergency floor **{res.get('safety_floor_percent', 35.0):.1f}%** active)"
            if res.get("has_active_emergency")
            else f" (safety floor **{res.get('safety_floor_percent', 35.0):.1f}%**, warning **{res.get('warning_floor_percent', 40.0):.1f}%**)"
        )
        summary_text = (
            f"3D digital twin looking inside **Reservoir RES-01**: water level **{res.get('level_percent', 48.0):.1f}%** "
            f"({res.get('volume_ml', 240.0):.1f} / {res.get('capacity_ml', 500.0):.0f} ML){dyn_note}; "
            f"DESAL-01 inlet feeding **{plant.get('output_mld', 100.0):.0f} MLD**."
        )
    elif f_type == "intake":
        intakes = plant.get("intakes") or []
        i_desc = ", ".join(f"{it['id']}: `{it['status']}` ({it['turbidity_ntu']} NTU)" for it in intakes)
        summary_text = (
            f"3D digital twin focused on **DESAL-01 Seawater Intakes**: {i_desc}; "
            f"bloom severity `{plant.get('bloom_severity', 'NONE')}`, algae cap **{plant.get('algae_cap_mld', 120.0):.0f} MLD**."
        )
    elif f_type == "plant":
        intakes = plant.get("intakes") or []
        int2_st = intakes[1]["status"] if len(intakes) > 1 else "ACTIVE"
        summary_text = (
            f"3D digital twin focused on **DESAL-01** coastal desalination plant (intakes, pre-treatment, RO hall, HP pumps, substation, brine outfall): "
            f"output **{plant.get('output_mld', 100.0):.0f} / {plant.get('max_capacity_mld', 120.0):.0f} MLD**, "
            f"algae cap **{plant.get('algae_cap_mld', 120.0):.0f} MLD**, Intake 2 `{int2_st}`, status `{plant.get('status', 'ONLINE')}`."
        )
    else:
        closed_valves = [v["valve_id"] for v in valves if v["status"] == "CLOSED"]
        summary_text = (
            f"3D digital twin overview of **All 6 Sectors** ({len(pipes)} pipes, {len(valves)} valves, "
            f"{len(closed_valves)} closed, {len(active_leaks)} active leak(s)), "
            f"RES-01 at **{res.get('level_percent', 48.0):.1f}%**, DESAL-01 at **{plant.get('output_mld', 100.0):.0f} MLD**."
        )

    tv_cmd = build_twin_view_command(
        focus=raw_focus,
        event_type="focus",
        note=cam_label,
    )

    return {
        "displayed": True,
        "view": "3D",
        "focus": f_id or "all",
        "raw_focus": raw_focus,
        "focus_type": f_type,
        "focus_id": f_id,
        "camera_label": cam_label,
        "camera_target": state.get("camera_target"),
        "highlight": hl,
        "twin_view": tv_cmd,
        "reservoir_percent": res.get("level_percent", 48.0),
        "plant_output_mld": plant.get("output_mld", 100.0),
        "active_leaks": len(active_leaks),
        "zero_flow_pipes": len(hyd.get("zero_flow_segment_ids") or []),
        "rerouted_pipes": len(hyd.get("rerouted_segment_ids") or []),
        "summary_text": summary_text,
    }


def build_map_render_summary(
    sector: str = "all",
    leak_id: Optional[str] = None,
    show_boundaries: bool = True,
    db_path: Optional[str] = None,
) -> dict:
    """Compute the exact structured summary of what show_map renders (layers, colours, counts, zoom target)."""
    from config.region_config import SECTOR_COLOR_NAMES, SECTOR_COLORS
    from database.dubai_network import JUNCTION_SECTOR_MAP, VALVE_SECTOR_MAP, compute_sector_boundaries

    raw_sec = (sector or "all").strip()
    norm_leak = (leak_id or "").strip().upper() or None
    highlight_sector: Optional[str] = None
    if raw_sec.upper().startswith("LEAK-"):
        norm_leak = raw_sec.upper()
    elif raw_sec.lower() not in ("", "all", "all sectors"):
        # Normalize e.g. "sector 4" -> "Sector 4"
        m_s = re.search(r"([1-6])", raw_sec)
        highlight_sector = f"Sector {m_s.group(1)}" if m_s else raw_sec

    conn = get_db_connection(db_path)
    try:
        segs = [dict(r) for r in conn.execute("SELECT segment_id, sector, geometry_json FROM pipe_segments;").fetchall()]
        valves = [dict(r) for r in conn.execute("SELECT valve_id, sector, status FROM valves;").fetchall()]
        leaks = [dict(r) for r in conn.execute("SELECT leak_id, sector, location_name, severity, status FROM leaks;").fetchall()]
        wq_rows = conn.execute(
            """
            SELECT sector, chlorine_mg_l
            FROM sector_water_quality
            WHERE id IN (SELECT MAX(id) FROM sector_water_quality GROUP BY sector);
            """
        ).fetchall()
    finally:
        conn.close()

    wq_by_sec = {r["sector"]: float(r["chlorine_mg_l"]) for r in wq_rows}
    tel = read_network_telemetry(current_hour=DEFAULT_SCADA_HOUR, db_path=db_path, log_audit=False)
    quarantined = tel.get("quarantined_sensors") or []

    boundaries = compute_sector_boundaries(segs, buffer_meters=500.0)
    boundaries_drawn = len(boundaries) if show_boundaries else 0

    layers_drawn: list[str] = []
    if show_boundaries:
        layers_drawn.append("Sector boundaries")
    layers_drawn.extend(["Pipes", "Sensors", "Valves", "Leaks"])

    per_sector: dict[str, dict] = {}
    sector_colors: dict[str, str] = {}
    for i in range(1, 7):
        sec_name = f"Sector {i}"
        cname = SECTOR_COLOR_NAMES[sec_name]
        chex = SECTOR_COLORS[sec_name]
        sector_colors[sec_name] = f"{cname} ({chex})"
        s_pipes = sum(1 for s in segs if s["sector"] == sec_name)
        s_valves = sum(1 for v in valves if v["sector"] == sec_name)
        s_juncs = sum(1 for _, sc in JUNCTION_SECTOR_MAP.items() if sc == sec_name)
        s_sensors = s_juncs * 2
        s_leaks = sum(1 for l in leaks if l["sector"] == sec_name and str(l.get("status", "")).lower() != "resolved")
        s_quar = sum(1 for q in quarantined if q.get("sector") == sec_name)
        cl_val = wq_by_sec.get(sec_name)
        s_status = "NON_COMPLIANT" if (cl_val is not None and cl_val < 0.20) else ("ALERT" if s_leaks > 0 else "NOMINAL")
        per_sector[sec_name] = {
            "color": cname,
            "hex": chex,
            "pipes": s_pipes,
            "sensors": s_sensors,
            "valves": s_valves,
            "leaks": s_leaks,
            "quarantined": s_quar,
            "chlorine_mg_l": round(cl_val, 3) if cl_val is not None else None,
            "status": s_status,
            "boundary_drawn": bool(show_boundaries and sec_name in boundaries),
        }

    matched_leaks = [
        l for l in leaks
        if (not highlight_sector or l["sector"] == highlight_sector)
        and (not norm_leak or l["leak_id"].upper() == norm_leak)
    ]

    if norm_leak:
        zoom_target = norm_leak
        zoom_bounds = None
    elif highlight_sector and highlight_sector in boundaries:
        zoom_target = highlight_sector
        zoom_bounds = boundaries[highlight_sector]["bounds"]
    else:
        zoom_target = "All sectors (Sectors 1-6)"
        all_lats = [b["bounds"][0][0] for b in boundaries.values()] + [b["bounds"][1][0] for b in boundaries.values()]
        all_lons = [b["bounds"][0][1] for b in boundaries.values()] + [b["bounds"][1][1] for b in boundaries.values()]
        zoom_bounds = [[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]] if all_lats else None

    colors_desc = ", ".join(f"{s} ({SECTOR_COLOR_NAMES[s]})" for s in ("Sector 1", "Sector 2", "Sector 3", "Sector 4", "Sector 5", "Sector 6"))
    if highlight_sector:
        info = per_sector.get(highlight_sector, {})
        b_phrase = f"highlighted {info.get('color', '')} sector boundary (other sectors dimmed)" if show_boundaries else "no sector boundaries drawn"
        summary_text = (
            f"Map zoomed to **{highlight_sector}** with {b_phrase}, "
            f"{info.get('pipes', 0)} pipe(s), {info.get('valves', 0)} valve(s), "
            f"{info.get('sensors', 0)} sensor(s), and {info.get('leaks', 0)} active leak(s) "
            f"(layers: {', '.join(layers_drawn)}; basemap: OpenStreetMap)."
        )
    else:
        b_phrase = f"{boundaries_drawn} buffered sector boundary polygons ({colors_desc})" if show_boundaries else "no sector boundaries drawn"
        summary_text = (
            f"Map fitted to **{zoom_target}** showing {b_phrase}, "
            f"{len(segs)} pipes, {len(valves)} valves, sensors, and {len(matched_leaks)} tracked leak(s) "
            f"(layers: {', '.join(layers_drawn)}; basemap: OpenStreetMap)."
        )

    return {
        "displayed": True,
        "sector": highlight_sector or "all",
        "highlight_sector": highlight_sector,
        "leak_id": norm_leak,
        "show_boundaries": bool(show_boundaries),
        "boundaries_drawn": boundaries_drawn,
        "layers_drawn": layers_drawn,
        "sector_colors": sector_colors,
        "per_sector": per_sector,
        "total_pipes": len(segs),
        "total_valves": len(valves),
        "total_sensors": sum(v["sensors"] for v in per_sector.values()),
        "matched_leaks": len(matched_leaks),
        "leaks": matched_leaks[:5],
        "zoom_target": zoom_target,
        "zoom_bounds": zoom_bounds,
        "basemap": "OpenStreetMap",
        "summary_text": summary_text,
    }


def parse_planning_horizon_hours(text: str) -> Optional[int]:
    """Detect production planning requests and return requested horizon in hours (1..24), or None."""
    cleaned = (text or "").strip()
    lower = cleaned.lower()
    if not cleaned:
        return None
    # Do not intercept questions asking WHY a previous plan was rejected/cancelled
    if any(k in lower for k in ("why ", "why?", "rejected", "cancelled", "last plan", "what does ", "meaning of")):
        return None
    if not any(k in lower for k in ("plan", "schedule")):
        return None
    # Explicit N hours (e.g. "make a 5-hour production plan", "plan the next 8 hours", "24-hour plan")
    m = _PLAN_HOURS_RE.search(cleaned)
    if m:
        raw_n = m.group(1) or m.group(2)
        if raw_n and raw_n.isdigit():
            return max(1, min(int(raw_n), 24))
    if "tomorrow" in lower or "next day" in lower or "24h" in lower:
        return 24
    if any(k in lower for k in ("make a", "create a", "build a", "generate a", "draft a", "plan production", "production plan", "production schedule")):
        return 24
    return None


def build_live_scada_snapshot(
    current_hour: int = DEFAULT_SCADA_HOUR,
    db_path: Optional[str] = None,
) -> dict:
    """Fetch a compact, complete live SCADA snapshot BEFORE calling the LLM so it always has real data."""
    tel = read_network_telemetry(current_hour=current_hour, db_path=db_path, log_audit=False)
    feeds = read_tariff_and_quality_feeds(horizon_hours=24, current_hour=current_hour, db_path=db_path, log_audit=False)
    vp = _read_valves_and_pending_snapshot(db_path=db_path)

    conn = get_db_connection(db_path)
    try:
        t_rows = conn.execute("SELECT hour, rate_per_kwh, is_peak FROM tariffs ORDER BY hour ASC;").fetchall()
        d_rows = conn.execute("SELECT hour, demand_mld FROM demand_forecast ORDER BY hour ASC;").fetchall()
    finally:
        conn.close()

    tariff_by_h = {int(r["hour"]): {"rate_per_kwh": float(r["rate_per_kwh"]), "is_peak": bool(r["is_peak"])} for r in t_rows}
    demand_by_h = {int(r["hour"]): float(r["demand_mld"]) for r in d_rows}

    tariffs_24h = []
    demand_24h = []
    for off in range(24):
        h = (current_hour + off) % 24
        t_info = tariff_by_h.get(h, {"rate_per_kwh": 0.20, "is_peak": False})
        tariffs_24h.append({"hour": f"{h:02d}:00", "rate_per_kwh": t_info["rate_per_kwh"], "is_peak": t_info["is_peak"]})
        demand_24h.append({"hour": f"{h:02d}:00", "demand_mld": demand_by_h.get(h, 90.0)})

    res = feeds.get("reservoir") or {}
    wq = feeds.get("water_quality") or {}
    wq_comp = feeds.get("sector_water_quality_compliance") or {}
    plant = vp.get("plant_setpoint") or {}

    return {
        "current_hour": current_hour,
        "reservoir": {
            "level_percent": float(res.get("current_level_percent", 48.0)),
            "volume_ml": float(res.get("current_volume_ml", 240.0)),
            "capacity_ml": float(res.get("capacity_ml", 500.0)),
            "safety_floor_percent": float(res.get("safety_floor_percent", 35.0)),
            "has_active_emergency": bool(res.get("has_active_emergency", False)),
            "floor_banner": res.get("floor_banner", ""),
            "rate_pct_per_hour": float(res.get("rate_pct_per_hour", 0.0)),
            "rate_str": str(res.get("rate_str", "→ 0.0 %/h")),
            "floor_eta_str": str(res.get("floor_eta_str", "stable")),
            "warning_state": str(res.get("warning_state", "NORMAL")),
            "effective_production_mld": float(res.get("effective_production_mld", feeds.get("current_plant_output_mld", 100.0))),
            "current_demand_mld": float(res.get("current_demand_mld", 85.0)),
        },
        "desal_01": {
            "current_output_mld": float(feeds.get("current_plant_output_mld", plant.get("production_mld", 100.0))),
            "min_mld": 0.0,
            "max_capacity_mld": float(plant.get("max_capacity_mld", 120.0)),
            "algae_cap_mld": float(wq.get("max_allowable_output_mld", 120.0)),
            "bloom_severity": str(wq.get("bloom_severity", "NONE")),
            "status": str(plant.get("status", "ONLINE")),
        },
        "tariffs_next_24h": tariffs_24h,
        "demand_next_24h": demand_24h,
        "tariff_by_hour": tariff_by_h,
        "demand_by_hour": demand_by_h,
        "active_leaks": tel.get("leak_alerts") or [],
        "quarantined_sensors": [
            {"sensor_id": s.get("sensor_id"), "junction_id": s.get("junction_id"), "sector": s.get("sector"), "reasons": s.get("reasons", [])}
            for s in (tel.get("quarantined_sensors") or [])
        ],
        "sector_chlorine": wq_comp.get("sectors") or [],
        "excluded_sectors": feeds.get("excluded_sectors") or [],
        "pending_actions": [
            {
                "id": p["id"],
                "action_type": p["action_type"],
                "target_id": p["target_id"],
                "current_value": p["current_value"],
                "proposed_value": p["proposed_value"],
                "rationale": p["rationale"],
                "provenance": p.get("provenance", "AGENT_ANALYSIS"),
            }
            for p in (vp.get("pending_actions") or [])
        ],
        "telemetry_freshness": tel.get("telemetry_freshness") or {},
        "raw_feeds": feeds,
        "raw_telemetry": tel,
    }


def _compact_snapshot_json(snapshot: dict) -> str:
    """Serialize live SCADA snapshot in a compact token-efficient JSON format preserving all exact live values."""
    t24 = snapshot.get("tariffs_next_24h") or []
    d24 = snapshot.get("demand_next_24h") or []
    tariffs_compact = ",".join(
        f"{str(t.get('hour', ''))[:2]}:${float(t.get('rate_per_kwh', 0.2)):.2f}{'*' if t.get('is_peak') else ''}"
        for t in t24
    )
    demand_compact = ",".join(
        f"{str(d.get('hour', ''))[:2]}:{round(float(d.get('demand_mld', 90.0)), 1):g}"
        for d in d24
    )
    cl_compact = {
        s.get("sector"): f"{float(s.get('chlorine_mg_l', 0.5)):.2f}({s.get('status', 'OK')})"
        for s in (snapshot.get("sector_chlorine") or [])
        if isinstance(s, dict) and s.get("sector")
    }
    tf = snapshot.get("telemetry_freshness") or {}
    compact = {
        "current_hour": snapshot.get("current_hour"),
        "reservoir": snapshot.get("reservoir"),
        "desal_01": snapshot.get("desal_01"),
        "tariffs_next_24h": tariffs_compact,
        "demand_next_24h": demand_compact,
        "active_leaks": snapshot.get("active_leaks") or [],
        "quarantined_sensors": snapshot.get("quarantined_sensors") or [],
        "sector_chlorine": cl_compact,
        "excluded_sectors": snapshot.get("excluded_sectors") or [],
        "pending_actions": snapshot.get("pending_actions") or [],
        "telemetry_freshness": {
            "status": tf.get("status", "FRESH"),
            "is_telemetry_loss": bool(tf.get("is_telemetry_loss", False)),
            "age_seconds": tf.get("age_seconds", 0),
            "banner": tf.get("banner", ""),
        },
    }
    return json.dumps(compact, separators=(",", ":"), default=str)


def _format_hourly_blocks(start_hour: int, hourly_outputs: list[float]) -> str:
    """Format exact hourly blocks '14:00-15:00 at 65 MLD, 15:00-16:00 at 65 MLD, ...'."""
    parts = []
    for idx, mld in enumerate(hourly_outputs):
        h0 = (start_hour + idx) % 24
        h1 = (start_hour + idx + 1) % 24
        h1_str = "24:00" if (h0 == 23 and h1 == 0) else f"{h1:02d}:00"
        parts.append(f"{h0:02d}:00-{h1_str} at {round(float(mld), 1):g} MLD")
    return ", ".join(parts)


def _deterministic_hourly_plan(
    snapshot: dict,
    horizon_hours: int,
    boost_mld: float = 0.0,
) -> list[float]:
    """Compute a tariff-aware, bloom-capped, water-quality-safe hourly output sequence for exact N hours."""
    start_h = int(snapshot["current_hour"])
    desal = snapshot["desal_01"]
    algae_cap = float(desal["algae_cap_mld"])
    curr_mld = float(desal["current_output_mld"])
    res = snapshot["reservoir"]
    eff_floor = float(res["safety_floor_percent"])
    start_vol = float(res["volume_ml"])
    cap_ml = float(res["capacity_ml"])
    tariff_by_h = snapshot["tariff_by_hour"]
    demand_by_h = snapshot["demand_by_hour"]
    excluded = snapshot["excluded_sectors"]

    outputs: list[float] = []
    sim_vol = start_vol
    floor_ml = (eff_floor / 100.0) * cap_ml + 1.5  # small safety margin above floor

    for off in range(horizon_hours):
        h = (start_h + off) % 24
        t_info = tariff_by_h.get(h, {"rate_per_kwh": 0.20, "is_peak": False})
        dem = float(demand_by_h.get(h, 90.0))

        if excluded:
            # Water quality priority: never throttle below safe baseline during non-compliance
            target = min(max(curr_mld, dem), algae_cap)
        elif t_info["is_peak"]:
            # Peak tariff ($0.48/kWh): save energy while staying safely above dynamic floor
            target = min(65.0 + boost_mld, algae_cap)
        elif t_info["rate_per_kwh"] <= 0.10:
            # Off-peak ($0.09/kWh): recharge reservoir
            target = min(115.0, algae_cap)
        else:
            # Mid-peak ($0.20/kWh): match demand smoothly
            target = min(max(95.0, dem) + boost_mld, algae_cap)

        # Lookahead physics check for this hour: ensure reservoir never dips below floor_ml
        projected_vol = sim_vol + (target - dem) / 24.0
        if projected_vol < floor_ml:
            needed = dem + (floor_ml - sim_vol) * 24.0
            target = min(algae_cap, max(target, round(needed + 2.0, 1)))
            projected_vol = sim_vol + (target - dem) / 24.0

        target = round(max(0.0, min(algae_cap, target)), 1)
        outputs.append(target)
        sim_vol = max(0.0, min(cap_ml, sim_vol + (target - dem) / 24.0))

    return outputs


def _parse_llm_hourly_plan(raw_text: str, start_hour: int, horizon_hours: int) -> Optional[list[float]]:
    """Extract an N-hour list of MLD outputs from LLM JSON or time-block text."""
    if not raw_text:
        return None
    # 1. Try JSON {"hourly_plan": [{"hour": 14, "output_mld": 70}, ...]} or {"outputs_mld": [70, ...]}
    try:
        m_json = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if m_json:
            obj = json.loads(m_json.group(0))
            if isinstance(obj.get("outputs_mld"), list) and len(obj["outputs_mld"]) == horizon_hours:
                return [float(x) for x in obj["outputs_mld"]]
            if isinstance(obj.get("hourly_plan"), list) and len(obj["hourly_plan"]) == horizon_hours:
                return [float(item.get("output_mld", item.get("mld", 0.0))) for item in obj["hourly_plan"]]
    except Exception:
        pass
    # 2. Try HH:MM-HH:MM at X MLD blocks
    matches = re.findall(r"(\d{1,2}):00\s*[-–]\s*(\d{1,2}):00\s*(?:at|@|:)?\s*(\d+(?:\.\d+)?)\s*mld", raw_text, re.IGNORECASE)
    if matches:
        by_hour: dict[int, float] = {}
        for s_h, e_h, mld_s in matches:
            sh, eh, mld_v = int(s_h) % 24, int(e_h), float(mld_s)
            if eh == 24:
                eh = 24
            else:
                eh = eh % 24
            cur = sh
            steps = 0
            while steps < 24:
                by_hour[cur] = mld_v
                cur = (cur + 1) % 24
                steps += 1
                if (eh == 24 and cur == 0) or (eh != 24 and cur == eh):
                    break
        if all(((start_hour + off) % 24) in by_hour for off in range(horizon_hours)):
            return [by_hour[(start_hour + off) % 24] for off in range(horizon_hours)]
    return None


def run_production_planner_for_chat(
    user_message: str,
    horizon_hours: int,
    snapshot: dict,
    current_hour: int = DEFAULT_SCADA_HOUR,
    db_path: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Real N-hour production planner flow:
    1. Checks TELEMETRY_LOSS first (refuses planning with exact missing-data explanation).
    2. Proposes exact N-hour schedule from current_hour using live snapshot + LLM (with offline fallback).
    3. Validates via pure-Python validate_production_schedule() with dynamic floor & algae cap.
    4. Replans up to 2 times if rejected.
    5. Stages first-hour SET_PLANT_OUTPUT via write gateway (STAGED_PENDING_APPROVAL only).
    6. Returns plan table, block validation, reservoir chart, and approval card widgets.
    """
    tf = snapshot["telemetry_freshness"]
    res = snapshot["reservoir"]
    desal = snapshot["desal_01"]
    excluded = snapshot["excluded_sectors"]
    eff_floor = float(res["safety_floor_percent"])
    algae_cap = float(desal["algae_cap_mld"])

    tools_called = [
        {"tool": "read_network_telemetry", "args": {"current_hour": current_hour}, "summary": f"freshness={tf.get('status', 'FRESH')}, leaks={len(snapshot['active_leaks'])}, quarantined={len(snapshot['quarantined_sensors'])}"},
        {"tool": "read_tariff_and_quality_feeds", "args": {"horizon_hours": 24, "current_hour": current_hour}, "summary": f"RES-01={res['level_percent']:.1f}% (floor {eff_floor:.1f}%), DESAL-01={desal['current_output_mld']:.0f} MLD, cap={algae_cap:.0f} MLD"},
    ]
    code_checks = [
        {"check": "Prompt Injection Filter", "status": "PASS", "code": "ALLOWED", "detail": "No prompt-injection syntax."},
        {
            "check": "Telemetry Freshness",
            "status": "BLOCKED" if tf.get("is_telemetry_loss") else "PASS",
            "code": tf.get("status", "FRESH"),
            "detail": tf.get("banner", "Telemetry fresh."),
        },
        {
            "check": "Sector Water Quality",
            "status": "BLOCKED" if excluded else "PASS",
            "code": "NON_COMPLIANT" if excluded else "COMPLIANT",
            "detail": "All 6 sectors >= 0.20 mg/L chlorine." if not excluded else f"NON_COMPLIANT: {', '.join(excluded)}",
        },
    ]

    # 1. Hard stop on TELEMETRY_LOSS — clearly explain what data is missing
    if tf.get("is_telemetry_loss"):
        _audit("CHAT_PLANNER_BLOCKED", "DESAL-01", "TELEMETRY_LOSS", f"{horizon_hours}h plan blocked: {tf.get('banner')}", actor="CODE_GUARD", db_path=db_path)
        return {
            "role": "assistant",
            "kind": "refusal",
            "content": (
                f"Cannot generate a **{horizon_hours}-hour production plan** right now because live SCADA sensor telemetry "
                f"is missing/stale (`TELEMETRY_LOSS`: {tf.get('banner', 'feed silent > 180s')}). "
                f"Without fresh pressure, flow, and reservoir timestamps, AquaGrid holds the last safe state "
                f"(`DESAL-01` at {desal['current_output_mld']:.0f} MLD) and refuses new setpoint proposals. "
                f"Restore the telemetry feed in the sidebar to resume planning."
            ),
            "provider": "CODE_GUARD (Telemetry Freshness)",
            "tools_called": tools_called,
            "code_checks": code_checks,
            "widgets": [{"type": "table", "what": "sensors"}],
            "staged_action": None,
            "providers_tried": [],
        }

    # 2. Propose -> Validate -> Replan (up to MAX_REPLANS = 2)
    max_replans = 2
    active_llm_client = llm_client
    if not hasattr(active_llm_client, "chat_with_tools"):
        import agent.llm_factory as _lf_mod

        _lf_mod = importlib.reload(_lf_mod)
        active_llm_client = _lf_mod.llm_client

    provider_used = "Deterministic Offline Engine"
    providers_tried: list[dict] = []
    violations_history: list[str] = []
    best_outputs: list[float] = []
    best_schedule_str = ""
    best_validation: dict = {}
    llm_rationale = ""

    raw_feeds = snapshot["raw_feeds"]
    wq_raw = raw_feeds.get("water_quality") or {}
    demand_by_h = snapshot["demand_by_hour"]
    tariff_by_h = snapshot["tariff_by_hour"]
    demand_list = [float(demand_by_h.get((current_hour + i) % 24, 90.0)) for i in range(horizon_hours)]

    for attempt in range(max_replans + 1):
        proposed_outputs: Optional[list[float]] = None

        # Try real LLM planner first with full live snapshot
        planner_prompt = (
            f"Create an exact {horizon_hours}-hour desalination production plan for DESAL-01 starting at {current_hour:02d}:00 "
            f"(hours {', '.join(f'{(current_hour + i) % 24:02d}:00' for i in range(horizon_hours))}).\n"
            f"Operator request: {user_message}\n"
            f"Constraints from live snapshot: DESAL-01 range 0-{desal['max_capacity_mld']:.0f} MLD, "
            f"algae cap {algae_cap:.0f} MLD, RES-01 currently {res['level_percent']:.1f}% with mandatory safety floor {eff_floor:.1f}%."
            + (f"\nPrevious attempt was REJECTED by Python physics guard: {'; '.join(violations_history)}. Increase output so reservoir stays >= {eff_floor:.1f}% and <= {algae_cap:.0f} MLD." if violations_history else "")
            + f"\nReturn ONLY valid JSON: {{\"outputs_mld\": [<exactly {horizon_hours} floats>], \"summary\": \"1-2 sentence explanation\"}}."
        )
        planner_msgs = [
            {"role": "system", "content": ASK_AQUAGRID_SYSTEM_PROMPT},
            {"role": "system", "content": "LIVE SCADA SNAPSHOT:\n" + _compact_snapshot_json(snapshot)},
            {"role": "user", "content": planner_prompt},
        ]
        try:
            chat_fn = active_llm_client.chat_with_tools
            accepts_ctx, accepts_status = _accepted_chat_optional_kwargs(chat_fn)
            call_kwargs: dict[str, Any] = {
                "messages": planner_msgs,
                "tools": [],
                "tool_executor": lambda n, a: {},
                "max_tool_steps": 1,
            }
            if accepts_ctx:
                call_kwargs["fallback_context_builder"] = lambda: _compact_snapshot_json(snapshot)
            if accepts_status:
                call_kwargs["status_callback"] = status_callback
            raw_reply, prov, _ = chat_fn(**call_kwargs)
            providers_tried = list(getattr(active_llm_client, "last_chat_providers_tried", []) or [])
            if raw_reply:
                parsed = _parse_llm_hourly_plan(raw_reply, current_hour, horizon_hours)
                if parsed is not None:
                    proposed_outputs = parsed
                    provider_used = prov
                    try:
                        m_j = re.search(r"\{.*\}", raw_reply, flags=re.DOTALL)
                        if m_j:
                            llm_rationale = str(json.loads(m_j.group(0)).get("summary", ""))
                    except Exception:
                        pass
        except Exception:
            logger.exception("Planner LLM attempt #%d failed; falling back to live-data deterministic planner", attempt + 1)

        if proposed_outputs is None:
            proposed_outputs = _deterministic_hourly_plan(snapshot, horizon_hours, boost_mld=attempt * 15.0)
            if provider_used == "Deterministic Offline Engine":
                provider_used = "Deterministic Offline Engine"

        _emit_status(status_callback, "Validating the plan hour by hour...")
        schedule_str = _format_hourly_blocks(current_hour, proposed_outputs)
        validation = validate_production_schedule(
            proposed_schedule=schedule_str,
            current_volume_ml=float(res["volume_ml"]),
            demand_by_hour=demand_by_h,
            fallback_demand_mld=demand_list,
            bloom_severity=str(wq_raw.get("bloom_severity", "NONE")),
            turbidity_ntu=float(wq_raw.get("intake_turbidity_ntu", 1.2)),
            chlorine_ppm=float(wq_raw.get("intake_chlorine_ppm", 1.8)),
            min_floor_percent=eff_floor / 100.0,
            current_hour=current_hour,
            current_plant_output_mld=float(desal["current_output_mld"]),
            excluded_sectors=excluded,
        )
        tools_called.append(
            {
                "tool": "show_schedule_validation",
                "args": {"attempt": attempt + 1, "horizon_hours": horizon_hours},
                "summary": f"Attempt #{attempt + 1}: {validation.get('status')} — {validation.get('summary', '')[:100]}",
            }
        )

        best_outputs = proposed_outputs
        best_schedule_str = schedule_str
        best_validation = validation

        if validation.get("status") == "VALID":
            break

        # Collect rejection reasons for replan
        rej_reasons = [validation.get("summary", "Safety guard violation")]
        for blk in validation.get("rejected_blocks") or []:
            rej_reasons.extend(blk.get("reasons", []))
        violations_history.append(" | ".join(rej_reasons[:2]))

    # Build exact N-hour Plan Table rows from cumulative trajectory
    cum_traj = best_validation.get("cumulative_trajectory") or []
    plan_rows: list[dict] = []
    sim_vol = float(res["volume_ml"])
    cap_ml = float(res["capacity_ml"])
    for idx in range(horizon_hours):
        h0 = (current_hour + idx) % 24
        h1 = (current_hour + idx + 1) % 24
        h1_str = "24:00" if (h0 == 23 and h1 == 0) else f"{h1:02d}:00"
        mld = float(best_outputs[idx])
        dem = float(demand_by_h.get(h0, 90.0))
        t_info = tariff_by_h.get(h0, {"rate_per_kwh": 0.20, "is_peak": False})
        if idx + 1 < len(cum_traj):
            proj_pct = float(cum_traj[idx + 1]["level_percent"])
        else:
            sim_vol = max(0.0, min(cap_ml, sim_vol + (mld - dem) / 24.0))
            proj_pct = round((sim_vol / cap_ml) * 100.0, 2)
        hr_ok = (mld <= algae_cap + 1e-6) and (proj_pct >= eff_floor - 1e-6) and (not (excluded and mld < float(desal["current_output_mld"])))
        plan_rows.append(
            {
                "Window": f"{h0:02d}:00-{h1_str}",
                "Output (MLD)": round(mld, 1),
                "Demand (MLD)": round(dem, 1),
                "Tariff ($/kWh)": f"${t_info['rate_per_kwh']:.2f}" + (" (PEAK)" if t_info["is_peak"] else ""),
                "Projected RES-01 (%)": round(proj_pct, 2),
                "Status": "VALID" if hr_ok else "REJECTED_BY_CODE",
            }
        )

    sv_status = best_validation.get("status", "UNKNOWN")
    code_checks.append(
        {
            "check": f"{horizon_hours}h Schedule Validator (floor {eff_floor:.1f}%)",
            "status": "PASS" if sv_status == "VALID" else "BLOCKED",
            "code": sv_status,
            "detail": best_validation.get("summary", ""),
        }
    )

    # Stage first-hour SET_PLANT_OUTPUT through write gateway (suggest-only!)
    _emit_status(status_callback, "Running safety checks...")
    first_mld = float(best_outputs[0])
    gw_resp = stage_gated_valve_or_setpoint_action(
        action_type="SET_PLANT_OUTPUT",
        target_id="DESAL-01",
        proposed_value=first_mld,
        rationale=(
            f"{horizon_hours}-hour production plan starting {current_hour:02d}:00 "
            f"(first hour {first_mld:.1f} MLD; floor {eff_floor:.1f}%, cap {algae_cap:.0f} MLD)."
        ),
        db_path=db_path,
        provenance="AGENT_ANALYSIS",
        provenance_ref="ASK_AQUAGRID_PLANNER",
    )
    gw_status = str(gw_resp.get("status", "UNKNOWN"))
    tools_called.append(
        {
            "tool": "propose_gated_action",
            "args": {"action_type": "SET_PLANT_OUTPUT", "target_id": "DESAL-01", "proposed_value": first_mld},
            "summary": f"{gw_status}: DESAL-01 -> {first_mld:.1f} MLD",
        }
    )
    code_checks.append(
        {
            "check": "Write Gateway & Safety Guards",
            "status": "PASS" if gw_status == "STAGED_PENDING_APPROVAL" else "BLOCKED",
            "code": gw_status,
            "detail": gw_resp.get("reason") or gw_resp.get("message") or "",
        }
    )

    widgets: list[dict] = [
        {"type": "plan_table", "horizon_hours": horizon_hours, "rows": plan_rows},
        {"type": "schedule_validation", "plan": best_schedule_str, "validation": best_validation},
        {"type": "chart", "metric": "reservoir", "asset": "RES-01", "hours": horizon_hours},
    ]
    if gw_status in ("STAGED_PENDING_APPROVAL", "REJECTED_DUPLICATE_PENDING"):
        widgets.append({"type": "approvals"})

    min_proj = min(r["Projected RES-01 (%)"] for r in plan_rows) if plan_rows else res["level_percent"]
    replan_note = f" (replanned {len(violations_history)} time(s) after guard feedback)" if violations_history else ""
    gw_line = (
        f"First-hour setpoint **DESAL-01 → {first_mld:.1f} MLD** staged as `{gw_status}` (Action #{gw_resp.get('action_id')}) — "
        f"click **Approve** or **Reject** on the approval card below."
        if gw_status == "STAGED_PENDING_APPROVAL"
        else f"Gateway status for **DESAL-01 → {first_mld:.1f} MLD**: `{gw_status}` — {gw_resp.get('reason') or gw_resp.get('message', '')}"
    )
    rationale_line = f"\n\n{llm_rationale}" if llm_rationale and not _TOOL_BEGGING_RE.search(llm_rationale) else ""
    answer_text = (
        f"**{horizon_hours}-Hour Production Plan ({current_hour:02d}:00–{(current_hour + horizon_hours) % 24:02d}:00)**{replan_note}: "
        f"Schedule validation **`{sv_status}`** (min projected RES-01 **{min_proj:.2f}%** vs **{eff_floor:.1f}%** floor, "
        f"algae cap **{algae_cap:.0f} MLD**).\n\n{gw_line}{rationale_line}"
    )

    _audit(
        "CHAT_PLANNER_COMPLETED",
        "DESAL-01",
        gw_status,
        f"{horizon_hours}h plan ({sv_status}): {best_schedule_str[:220]} -> gateway={gw_status}",
        actor="LLM_AGENT",
        db_path=db_path,
    )

    return {
        "role": "assistant",
        "kind": "text",
        "content": answer_text,
        "provider": provider_used,
        "tools_called": tools_called,
        "code_checks": code_checks,
        "widgets": widgets,
        "staged_action": gw_resp,
        "providers_tried": providers_tried,
        "plan_rows": plan_rows,
        "schedule_validation": best_validation,
    }

STATUS_CODE_GLOSSARY = {
    "STAGED_PENDING_APPROVAL": "Action passed deterministic Python safety guards and is queued as PENDING awaiting operator Approve/Reject button click.",
    "REJECTED_BY_SAFETY_GUARD": "Blocked because the proposal breaches the 35% (or raised emergency) reservoir floor, algal-bloom output cap, or physical limits.",
    "REJECTED_BY_CODE": "24h production schedule rejected by the deterministic physics validator (floor breach, capacity limit, bloom cap, or unparseable blocks).",
    "REJECTED_WATER_QUALITY": "Blocked because a sector has non-compliant chlorine (< 0.20 mg/L); water quality outranks peak-tariff production cuts.",
    "NON_COMPLIANT": "Sector free chlorine residual dropped below the mandatory 0.20 mg/L potable minimum.",
    "CANCELLED_BY_SAFETY_RECHECK": "A previously staged PENDING action was cancelled automatically because conditions changed (e.g. fire emergency raised the safety floor).",
    "REJECTED_UNAUTHORIZED_APPROVAL_CLAIM": "Gateway refused an unauthorized pre_approved/approval parameter; only an operator button click can approve.",
    "REJECTED_APPROVAL_MISMATCH": "Single-use approval token refused because its bound target/action/parameters do not match the requested actuation.",
    "REJECTED_APPROVAL_ALREADY_USED": "Single-use approval token refused because it was already consumed once.",
    "REJECTED_APPROVAL_EXPIRED": "Single-use approval token refused because its 15-minute TTL expired.",
    "TELEMETRY_LOSS": "SCADA telemetry is older than 180 seconds; AI planning and new proposals are paused while holding the last safe state.",
    "REJECTED_STALE_TELEMETRY": "New proposal refused by the write gateway while TELEMETRY_LOSS is active.",
    "CANCELLED_STALE_TELEMETRY": "Approval click refused because telemetry is stale (> 180s); restore and re-validate feed first.",
    "UNTRUSTED_TEXT": "Data originated from an external alert or field work ticket and is never treated as trusted operator authorization.",
    "VALID": "Schedule or check passed every deterministic physics and water-quality rule.",
    "GAP": "Hours not explicitly covered by a schedule block; simulated using the current bloom-capped plant setpoint.",
    "BLOCKED": "Prompt injection or malicious payload intercepted and neutralized before reaching the LLM.",
}


def _audit(
    event_type: str,
    target_id: str,
    status: str,
    details: str,
    actor: str = "LLM_AGENT",
    db_path: Optional[str] = None,
) -> None:
    try:
        with transaction_scope(db_path) as conn:
            conn.execute(
                """
                INSERT INTO audit_log (event_type, target_id, details, actor, status)
                VALUES (?, ?, ?, ?, ?);
                """,
                (event_type, target_id, str(details)[:500], actor, status),
            )
    except Exception:
        pass


def parse_scenario_command(text: str) -> Optional[str]:
    """Detect typed scenario commands like 'Run scenario 2', 'run R1-S1', 'run U-5.3'. Returns registry ID or None."""
    m = _SCENARIO_TRIGGER_RE.search((text or "").strip())
    if not m:
        return None
    token = m.group(1).upper()
    if token in ("1", "2", "3", "4"):
        return f"R1-S{token}"
    if get_scenario_by_id(token):
        return token
    return None


def run_scenario_for_chat(
    scenario_key: str,
    current_hour: int = DEFAULT_SCADA_HOUR,
    db_path: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """Execute a Round 1 or Round 2 scenario and return a rich chat message dict with result card & widgets."""
    _emit_status(status_callback, "Checking your request (injection filter)...")
    entry = get_scenario_by_id(scenario_key)
    if not entry or not entry.get("enabled") or not callable(entry.get("injector")):
        return {
            "role": "assistant",
            "kind": "text",
            "content": f"Scenario `{scenario_key}` is not available.",
            "provider": "CODE_GUARD",
            "tools_called": [],
            "code_checks": [],
            "widgets": [],
            "staged_action": None,
        }

    _emit_status(status_callback, "Reading live plant data...")
    reset_database(db_path)
    info = entry["injector"](db_path) if db_path else entry["injector"]()

    scenario_num = entry.get("scenario_num")
    agent_scenario_id = scenario_num if scenario_num is not None else entry["id"]
    _emit_status(status_callback, "Running safety checks...")
    result = run_agent(scenario_id=agent_scenario_id, current_hour=current_hour)

    bypass_result = None
    flagged = [a for a in result.get("sanitized_alerts", []) if a.get("injection_detected")]
    if scenario_num == 4 or flagged:
        bypass_result = refuse_direct_write_bypass(target_valve_id="V-12", attempted_status="OPEN", db_path=db_path)

    pre_approved_refusal = None
    if entry["id"] == "U-5.3":
        pre_approved_refusal = stage_gated_valve_or_setpoint_action(
            action_type="CLOSE_VALVE",
            target_id="V-17",
            proposed_value=0.0,
            rationale="Direct call claiming pre_approved=True from supervisor ticket",
            db_path=db_path,
            pre_approved=True,
        )

    widgets: list[dict] = [
        {
            "type": "scenario_result",
            "scenario_id": entry["id"],
            "scenario_title": entry["title"],
            "scenario_num": scenario_num,
            "scenario_info": info,
            "agent_result": result,
            "bypass_result": bypass_result,
            "pre_approved_refusal": pre_approved_refusal,
        }
    ]

    # Attach relevant 3D Digital Twin view command + contextual charts for every Round 1 & Round 2 scenario!
    from digital_twin.scene_state import build_twin_view_command

    scenario_twin_view = build_twin_view_command(focus="all", event_type="scenario")
    if entry["id"] == "R1-S1":
        _emit_status(status_callback, "Drawing the map...")
        conn_l = get_db_connection(db_path)
        try:
            lk_row = conn_l.execute("SELECT leak_id FROM leaks WHERE sector = 'Sector 4' ORDER BY detected_at DESC LIMIT 1;").fetchone()
        finally:
            conn_l.close()
        j14_focus = lk_row["leak_id"] if lk_row else "Sector 4"
        widgets.append({"type": "twin_3d", "focus": j14_focus})
        scenario_twin_view = build_twin_view_command(focus=j14_focus, event_type="new_leak", note="J-14 pipe leak detected")
    elif entry["id"] == "R1-S3":
        _emit_status(status_callback, "Drawing the map...")
        widgets.append({"type": "twin_3d", "focus": "V-12"})
        scenario_twin_view = build_twin_view_command(focus="V-12", event_type="focus", note="V-12 isolation staged")
    elif entry["id"] == "R1-S4":
        scenario_twin_view = build_twin_view_command(focus="V-12", event_type="focus", note="Injection blocked — V-12 untouched")
    elif entry["id"] == "R1-S2":
        _emit_status(status_callback, "Validating the plan hour by hour...")
        widgets.append({"type": "twin_3d", "focus": "reservoir"})
        widgets.append({"type": "chart", "metric": "reservoir", "asset": "RES-01", "hours": 24})
        scenario_twin_view = build_twin_view_command(focus="reservoir_inside", event_type="focus", note="Algae cap & RES-01 safety floor")
        if result.get("schedule_validation"):
            widgets.append(
                {
                    "type": "schedule_validation",
                    "plan": result.get("proposed_schedule", ""),
                    "validation": result["schedule_validation"],
                }
            )
    elif entry["id"] == "U-5.1":
        _emit_status(status_callback, "Drawing the map...")
        widgets.append({"type": "twin_3d", "focus": "Sector 2"})
        scenario_twin_view = build_twin_view_command(focus="Sector 2", event_type="water_quality", note="Sector 2 chlorine NON_COMPLIANT")
    elif entry["id"] == "U-5.2":
        _emit_status(status_callback, "Validating the plan hour by hour...")
        widgets.append({"type": "twin_3d", "focus": "Sector 6"})
        widgets.append({"type": "chart", "metric": "reservoir", "asset": "RES-01", "hours": 24})
        scenario_twin_view = build_twin_view_command(focus="Sector 6", event_type="fire_emergency", note="Sector 6 fire — RES-01 floor 35.4%")
        if result.get("schedule_validation"):
            widgets.append(
                {
                    "type": "schedule_validation",
                    "plan": result.get("proposed_schedule", ""),
                    "validation": result["schedule_validation"],
                }
            )
    elif entry["id"] in ("U-5.3", "U-5.4"):
        _emit_status(status_callback, "Drawing the map...")
        widgets.append({"type": "twin_3d", "focus": "V-17"})
        note_txt = "Untrusted ticket on V-17" if entry["id"] == "U-5.3" else "Token replay blocked — V-18 OPEN"
        scenario_twin_view = build_twin_view_command(focus="V-17", event_type="focus", note=note_txt)
    elif entry["id"] == "U-5.5":
        scenario_twin_view = build_twin_view_command(focus="all", event_type="telemetry_loss", note="TELEMETRY LOSS — holding last safe state")

    # Check if any PENDING actions now exist -> show inline Approval Card with buttons!
    conn = get_db_connection(db_path)
    try:
        pending_cnt = conn.execute("SELECT COUNT(*) AS cnt FROM pending_actions WHERE status = 'PENDING';").fetchone()["cnt"]
    finally:
        conn.close()
    if pending_cnt > 0:
        widgets.append({"type": "approvals"})

    gw = result.get("gateway_response")
    code_checks = [
        {
            "check": "Telemetry Freshness",
            "status": "BLOCKED" if result.get("telemetry_loss") else "PASS",
            "code": (result.get("telemetry_freshness") or {}).get("status", "FRESH"),
            "detail": (result.get("telemetry_freshness") or {}).get("banner") or "Readings within 180s freshness limit.",
        },
        {
            "check": "Water Quality",
            "status": "BLOCKED" if result.get("excluded_sectors") else "PASS",
            "code": "NON_COMPLIANT" if result.get("excluded_sectors") else "COMPLIANT",
            "detail": (
                f"Excluded: {', '.join(result.get('excluded_sectors', []))}"
                if result.get("excluded_sectors")
                else "All sectors chlorine compliant (>= 0.20 mg/L)."
            ),
        },
    ]
    if gw:
        code_checks.append(
            {
                "check": "Write Gateway",
                "status": "PASS" if gw.get("status") == "STAGED_PENDING_APPROVAL" else "BLOCKED",
                "code": gw.get("status", "UNKNOWN"),
                "detail": gw.get("reason") or gw.get("message") or "",
            }
        )

    return {
        "role": "assistant",
        "kind": "scenario",
        "content": f"**{entry['id']} — {entry['title']}** completed.",
        "provider": result.get("llm_provider", "Deterministic Offline Engine"),
        "tools_called": [
            {"tool": "read_network_telemetry", "args": {"current_hour": current_hour}, "summary": f"{len(result.get('leak_alerts', []))} leak(s), {len(result.get('quarantined_sensors', []))} quarantined"},
            {"tool": "read_tariff_and_quality_feeds", "args": {"current_hour": current_hour}, "summary": f"Reservoir {float((result.get('reservoir_status') or {}).get('current_level_percent', 48.0)):.1f}%"},
        ],
        "code_checks": code_checks,
        "widgets": widgets,
        "staged_action": gw,
        "twin_view": scenario_twin_view,
        "scenario_meta": {
            "id": entry["id"],
            "scenario_num": scenario_num,
            "name": f"{entry['id']} {entry['title']}",
            "info": info,
            "result": result,
            "bypass_result": bypass_result,
            "pre_approved_refusal": pre_approved_refusal,
        },
    }


def _read_valves_and_pending_snapshot(db_path: Optional[str] = None) -> dict:
    conn = get_db_connection(db_path)
    try:
        valves = [dict(r) for r in conn.execute("SELECT valve_id, sector, status, last_updated FROM valves ORDER BY valve_id ASC;").fetchall()]
        plant = conn.execute("SELECT * FROM plant_setpoints WHERE plant_id = 'DESAL-01';").fetchone()
        pending = [dict(r) for r in conn.execute("SELECT * FROM pending_actions WHERE status = 'PENDING' ORDER BY id DESC;").fetchall()]
        recent_resolved = [dict(r) for r in conn.execute("SELECT * FROM pending_actions WHERE status != 'PENDING' ORDER BY id DESC LIMIT 10;").fetchall()]
        commitments = [dict(r) for r in conn.execute("SELECT * FROM emergency_commitments ORDER BY id DESC LIMIT 10;").fetchall()]
        leaks = [dict(r) for r in conn.execute("SELECT leak_id, pipe_segment_id, sector, location_name, severity, estimated_loss_m3_per_hour, status FROM leaks ORDER BY detected_at DESC LIMIT 15;").fetchall()]
    finally:
        conn.close()
    return {
        "plant_setpoint": dict(plant) if plant else {},
        "valves": valves,
        "pending_actions": pending,
        "recent_resolved_actions": recent_resolved,
        "emergency_commitments": commitments,
        "tracked_leaks": leaks,
    }


def _read_audit_entries(limit: int = 20, event_filter: str = "", db_path: Optional[str] = None) -> list[dict]:
    lim = max(1, min(int(limit or 20), 50))
    conn = get_db_connection(db_path)
    try:
        if event_filter:
            pat = f"%{event_filter}%"
            rows = conn.execute(
                """
                SELECT id, timestamp, event_type, target_id, actor, status, details
                FROM audit_log
                WHERE event_type LIKE ? OR status LIKE ? OR details LIKE ? OR target_id LIKE ?
                ORDER BY id DESC LIMIT ?;
                """,
                (pat, pat, pat, pat, lim),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, timestamp, event_type, target_id, actor, status, details FROM audit_log ORDER BY id DESC LIMIT ?;",
                (lim,),
            ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _summarize_tool_output(tool_name: str, result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)[:140]
    if tool_name == "read_network_telemetry":
        tf = (result.get("telemetry_freshness") or {}).get("status", "FRESH")
        return (
            f"{len(result.get('verified_telemetry', []))} sensors verified, "
            f"{len(result.get('leak_alerts', []))} leak(s), "
            f"{len(result.get('quarantined_sensors', []))} quarantined, freshness={tf}"
        )
    if tool_name == "read_tariff_and_quality_feeds":
        res = result.get("reservoir") or {}
        tariffs = result.get("tariffs") or []
        rate = tariffs[0]["rate_per_kwh"] if tariffs else 0.48
        excl = result.get("excluded_sectors") or []
        excl_s = f", NON_COMPLIANT={excl}" if excl else ", all sectors compliant"
        return (
            f"RES-01={float(res.get('current_level_percent', 0)):.1f}% "
            f"(floor {float(res.get('safety_floor_percent', 35.0)):.1f}%), "
            f"DESAL-01={float(result.get('current_plant_output_mld', 0)):.0f} MLD, "
            f"tariff=${rate:.2f}/kWh{excl_s}"
        )
    if tool_name == "read_valves_and_pending_actions":
        return (
            f"{len(result.get('valves', []))} valves, "
            f"{len(result.get('pending_actions', []))} PENDING action(s), "
            f"{len(result.get('tracked_leaks', []))} tracked leak(s)"
        )
    if tool_name == "read_audit_log":
        entries = result.get("entries") or []
        return f"Read {len(entries)} recent audit log entries"
    if tool_name == "show_3d":
        return (
            f"3D focus: {result.get('camera_label', result.get('focus', 'all'))} · "
            f"RES-01={result.get('reservoir_percent', 48.0):.1f}% · "
            f"DESAL-01={result.get('plant_output_mld', 100.0):.0f} MLD · "
            f"active leaks={result.get('active_leaks', 0)}"
        )
    if tool_name == "show_map":
        b_cnt = result.get("boundaries_drawn", 0)
        layers = ", ".join(result.get("layers_drawn") or [])
        return (
            f"Zoom: {result.get('zoom_target', result.get('sector', 'All'))} · "
            f"boundaries={b_cnt} · layers=[{layers}] · "
            f"pipes={result.get('total_pipes', 0)}, valves={result.get('total_valves', 0)}, leaks={result.get('matched_leaks', 0)}"
        )
    if tool_name == "show_chart":
        return f"Displayed {result.get('metric')} chart (asset={result.get('asset') or 'Grid'}, hours={result.get('hours')})"
    if tool_name == "show_table":
        return f"Displayed {result.get('what')} table ({result.get('row_count', 0)} rows)"
    if tool_name == "show_schedule_validation":
        return f"Schedule validation: {result.get('status')} — {str(result.get('summary', ''))[:100]}"
    if tool_name == "show_pending_approvals":
        return f"Displayed approval card(s): {result.get('pending_count', 0)} PENDING"
    if tool_name == "propose_gated_action":
        return f"{result.get('status')}: {result.get('action_type', '')} {result.get('target_id', '')} — {str(result.get('reason') or result.get('message') or '')[:100]}"
    return json.dumps(result, default=str)[:140]


def _append_unique_widget(widgets: list[dict], widget: dict) -> None:
    sig = (widget.get("type"), widget.get("metric"), widget.get("asset"), widget.get("sector"), widget.get("leak_id"), widget.get("what"), widget.get("focus"))
    for w in widgets:
        w_sig = (w.get("type"), w.get("metric"), w.get("asset"), w.get("sector"), w.get("leak_id"), w.get("what"), w.get("focus"))
        if w_sig == sig:
            return
    widgets.append(widget)


def _offline_chat_fallback(
    user_message: str,
    history: list[dict],
    tool_executor,
    current_hour: int,
) -> str:
    """Live-tool-backed conversational fallback when external/local LLM endpoints are offline."""
    text = user_message.strip()
    lower = text.lower()

    # Read live data first (never invent numbers!)
    tel = tool_executor("read_network_telemetry", {"current_hour": current_hour})
    feeds = tool_executor("read_tariff_and_quality_feeds", {"horizon_hours": 6, "current_hour": current_hour})

    res = feeds.get("reservoir") or {}
    res_pct = float(res.get("current_level_percent", 48.0))
    eff_floor = float(res.get("safety_floor_percent", 35.0))
    plant_mld = float(feeds.get("current_plant_output_mld", 100.0))
    tariffs = feeds.get("tariffs") or []
    curr_rate = float(tariffs[0]["rate_per_kwh"]) if tariffs else 0.48
    is_peak = bool(feeds.get("is_current_peak_tariff", False))
    wq = feeds.get("water_quality") or {}
    max_cap = float(wq.get("max_allowable_output_mld", 120.0))
    leaks = tel.get("leak_alerts") or []
    quarantined = tel.get("quarantined_sensors") or []
    excluded = feeds.get("excluded_sectors") or []
    is_loss = bool(tel.get("telemetry_loss", False))

    # Extract prior assistant context for follow-ups ("Why?", "What about Sector 2?", "And if demand rises?")
    last_assistant = ""
    for m in reversed(history):
        if m.get("role") == "assistant" and m.get("content"):
            last_assistant = str(m["content"])
            break

    # 0. Status code explanation
    for code, desc in STATUS_CODE_GLOSSARY.items():
        if code.lower() in lower and any(k in lower for k in ("what", "mean", "explain", "define")):
            return f"`{code}`: {desc}"

    # 1. Explicit change request -> ALWAYS stage via write gateway (never execute)
    m_valve = re.search(r"\b(close|shut|isolate|open)\b[^\n]{0,35}?\b(v-\d{1,2})\b", text, re.IGNORECASE)
    if not m_valve:
        m_rev = re.search(r"\b(v-\d{1,2})\b[^\n]{0,35}?\b(close|shut|isolate|open)\b", text, re.IGNORECASE)
        if m_rev:
            verb, vid = m_rev.group(2).lower(), m_rev.group(1).upper()
        else:
            verb, vid = None, None
    else:
        verb, vid = m_valve.group(1).lower(), m_valve.group(2).upper()

    if verb and vid:
        parts = vid.split("-")
        if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 1:
            vid = f"V-0{parts[1]}"
        a_type = "OPEN_VALVE" if verb == "open" else "CLOSE_VALVE"
        val = 1.0 if verb == "open" else 0.0
        gw = tool_executor(
            "propose_gated_action",
            {
                "action_type": a_type,
                "target_id": vid,
                "proposed_value": val,
                "rationale": f"Operator chat request: '{text[:120]}'",
            },
        )
        st_code = gw.get("status", "UNKNOWN")
        if st_code == "STAGED_PENDING_APPROVAL":
            return (
                f"Staged **{a_type} {vid}** as `{st_code}` (Action #{gw.get('action_id')}). "
                f"No valve has been changed yet — click **Approve** or **Reject** on the approval card below."
            )
        return f"Request **{a_type} {vid}** returned `{st_code}`: {gw.get('reason') or gw.get('message', '')}"

    m_flush = re.search(r"\b(?:flush|isolate\s+and\s+flush)\b[^\n]{0,40}?\bsector[\s-]*0?([1-6])\b", text, re.IGNORECASE)
    if m_flush:
        sec_num = int(m_flush.group(1))
        target_sec = f"SECTOR-0{sec_num}"
        gw = tool_executor(
            "propose_gated_action",
            {
                "action_type": "ISOLATE_AND_FLUSH",
                "target_id": target_sec,
                "proposed_value": 0.0,
                "rationale": f"Operator requested isolate and flush for Sector {sec_num}.",
            },
        )
        st_code = gw.get("status", "UNKNOWN")
        if st_code == "STAGED_PENDING_APPROVAL":
            return f"Staged **ISOLATE_AND_FLUSH {target_sec}** as `{st_code}` (Action #{gw.get('action_id')}). Use the **Approve / Reject** buttons below."
        return f"Request **ISOLATE_AND_FLUSH {target_sec}** returned `{st_code}`: {gw.get('reason', '')}"

    m_mld = re.search(r"(\d+(?:\.\d+)?)\s*mld\b", text, re.IGNORECASE)
    if m_mld and any(k in lower for k in ("set", "reduce", "throttle", "cut", "increase", "change", "output", "desal", "plant")):
        target_mld = float(m_mld.group(1))
        gw = tool_executor(
            "propose_gated_action",
            {
                "action_type": "SET_PLANT_OUTPUT",
                "target_id": "DESAL-01",
                "proposed_value": target_mld,
                "rationale": f"Operator chat request: set DESAL-01 to {target_mld:.1f} MLD.",
            },
        )
        st_code = gw.get("status", "UNKNOWN")
        if st_code == "STAGED_PENDING_APPROVAL":
            return (
                f"Staged **SET_PLANT_OUTPUT DESAL-01 → {target_mld:.1f} MLD** as `{st_code}` (Action #{gw.get('action_id')}, "
                f"min projected reservoir {gw.get('min_projected_percent')}% vs {eff_floor:.1f}% floor). "
                f"Click **Approve** or **Reject** below to decide."
            )
        return f"Request **SET_PLANT_OUTPUT DESAL-01 → {target_mld:.1f} MLD** returned `{st_code}`: {gw.get('reason', '')}"

    # 2. Telemetry loss note (read-only questions STILL get answered; proposals are refused by gateway!)
    loss_suffix = ""
    if is_loss:
        loss_banner = (tel.get("telemetry_freshness") or {}).get("banner", "TELEMETRY LOSS")
        loss_suffix = f"\n\n*Note: `{loss_banner}` (`TELEMETRY_LOSS`) — new change proposals are blocked until the feed is restored.*"

    # 3a. Default 3D Digital Twin requests ("show Sector 4", "where is the leak", "show the reservoir", "show valve V-12")
    focus_3d = _parse_3d_focus_request(text)
    if focus_3d is not None:
        v3d = tool_executor("show_3d", {"focus": focus_3d})
        return f"{v3d.get('summary_text', 'Displayed 3D digital twin.')}{loss_suffix}"

    # 3b. Explicit 2D Map display requests ("Show Sector 4 on the map", "show the sectors", "2D map")
    if _is_map_request(text):
        m_sec = re.search(r"sector\s*([1-6])", text, re.IGNORECASE)
        sec_arg = f"Sector {m_sec.group(1)}" if m_sec else "all"
        m_lid = re.search(r"(LEAK-\d+)", text, re.IGNORECASE)
        lid_arg = m_lid.group(1).upper() if m_lid else ""
        show_b = not ("without boundar" in lower or "no boundar" in lower or "hide boundar" in lower)
        mp = tool_executor("show_map", {"sector": sec_arg, "leak_id": lid_arg, "show_boundaries": show_b})
        return f"{mp.get('summary_text', 'Displayed live map.')}{loss_suffix}"

    # 3b. Direct reservoir level / trend question ("what is the reservoir level", "show reservoir trend")
    if ("reservoir" in lower or "res-01" in lower) and not any(k in lower for k in ("plan", "schedule", "optim", "tariff", "pressure", "flow")):
        res_vol = float(res.get("current_volume_ml", 240.0))
        rate_str = str(res.get("rate_str") or "→ 0.0 %/h")
        eta_str = str(res.get("floor_eta_str") or "stable")
        if any(k in lower for k in ("trend", "deplet", "project", "forecast", "chart", "graph", "when")):
            tool_executor("show_chart", {"metric": "reservoir_trend", "asset": "RES-01", "hours": 18})
        else:
            tool_executor("show_chart", {"metric": "reservoir", "asset": "RES-01", "hours": 24})
        tool_executor("show_3d", {"focus": "reservoir_inside"})
        em_note = " (fire emergency reserve active)" if res.get("has_active_emergency") else ""
        return (
            f"Strategic Reservoir **RES-01** is currently at **{res_pct:.1f}% ({res_vol:.1f} ML)** "
            f"(**{rate_str}**, **{eta_str}**) against the mandatory **{eff_floor:.1f}%** safety floor{em_note}. "
            f"DESAL-01 output is **{plant_mld:.0f} MLD** (cap {max_cap:.0f} MLD).{loss_suffix}"
        )

    # 4. Chart display requests ("Pressure at J-14 last 2 hours", "flow chart", "reservoir chart", "tariff chart")
    if any(k in lower for k in ("pressure", "flow", "chart", "graph", "trend", "last 2 hour", "last 6 hour")):
        metric = "pressure"
        if "flow" in lower:
            metric = "flow"
        elif "reservoir" in lower:
            metric = "reservoir"
        elif "tariff" in lower:
            metric = "tariff"
        m_j = re.search(r"\b([JPFS]-\d{1,2})\b", text, re.IGNORECASE)
        m_sec = re.search(r"sector\s*([1-6])", text, re.IGNORECASE)
        asset = m_j.group(1).upper() if m_j else (f"Sector {m_sec.group(1)}" if m_sec else "J-14")
        m_hr = re.search(r"(\d+)\s*h(?:our)?s?", lower)
        hrs = int(m_hr.group(1)) if m_hr else (2 if metric in ("pressure", "flow") else 24)
        ch = tool_executor("show_chart", {"metric": metric, "asset": asset, "hours": hrs})
        return f"Showing live **{metric}** chart for **{asset}** ({ch.get('summary', 'live SCADA data')})."

    # 5. Audit / "Why was the last plan rejected?" / follow-up "Why?"
    if any(k in lower for k in ("why", "rejected", "cancelled", "audit", "last plan", "history")):
        aud = tool_executor("read_audit_log", {"limit": 15, "event_filter": ""})
        entries = aud.get("entries") or []
        rej = [
            e for e in entries
            if any(tok in str(e.get("status", "")).upper() for tok in ("REJECTED", "CANCELLED", "BLOCKED", "REFUSED", "TELEMETRY_LOSS", "NON_COMPLIANT"))
        ]
        if "audit" in lower or "log" in lower:
            tool_executor("show_table", {"what": "audit"})
        if rej:
            latest = rej[0]
            ctx_note = f" (following up on: {last_assistant[:80]}...)" if lower.strip("?!") in ("why", "why?") and last_assistant else ""
            return (
                f"Most recent rejection/block{ctx_note}: **`{latest['status']}`** on `{latest['target_id']}` "
                f"({latest['event_type']}) — {latest['details']}"
            )
        if lower.strip("?!") in ("why", "why?") and last_assistant:
            return f"Based on our previous discussion ({last_assistant[:140]}): Reservoir RES-01 is at **{res_pct:.1f}%** (floor **{eff_floor:.1f}%**) and DESAL-01 is at **{plant_mld:.0f} MLD**."
        return f"No rejected or cancelled plans are currently in the recent Audit Log ({len(entries)} events checked)."

    # 6. Sector-specific follow-up ("What about Sector 2?", "Sector 6?")
    m_sec_q = re.search(r"sector\s*([1-6])", text, re.IGNORECASE)
    if m_sec_q:
        sec_name = f"Sector {m_sec_q.group(1)}"
        wq_comp = feeds.get("sector_water_quality_compliance") or {}
        sec_rows = [s for s in wq_comp.get("sectors", []) if s.get("sector") == sec_name]
        cl_txt = (
            f"chlorine **{sec_rows[0]['chlorine_mg_l']:.2f} mg/L** (`{sec_rows[0]['status']}`)"
            if sec_rows
            else "chlorine nominal"
        )
        sec_leaks = [l for l in leaks if l.get("sector") == sec_name]
        leak_txt = f"{len(sec_leaks)} confirmed leak(s)" if sec_leaks else "no confirmed hydraulic leaks"
        tool_executor("show_map", {"sector": sec_name})
        return f"**{sec_name}** live status: {cl_txt}, {leak_txt}. Map focused on {sec_name} below."

    # 7. What-if demand rises follow-up ("And if demand rises?")
    if "demand" in lower and any(k in lower for k in ("rise", "increase", "higher", "spike", "what if")):
        plan = f"00:00-06:00 at {min(115, int(max_cap))} MLD, 06:00-14:00 at {min(100, int(max_cap))} MLD, 14:00-18:00 at {min(85, int(max_cap))} MLD, 18:00-24:00 at {min(100, int(max_cap))} MLD"
        sv = tool_executor("show_schedule_validation", {"plan": plan, "current_hour": current_hour})
        tool_executor("show_chart", {"metric": "reservoir", "asset": "RES-01", "hours": 24})
        return (
            f"If demand rises, reducing DESAL-01 during peak hours drains RES-01 faster toward the **{eff_floor:.1f}%** floor "
            f"(currently **{res_pct:.1f}%**, bloom cap **{max_cap:.0f} MLD**). "
            f"Validating a higher-output safeguard plan (`{plan}`): **`{sv.get('status')}`** — {sv.get('summary', '')}"
        )

    # 8. Schedule / 24h plan / Save money at 2 PM
    if any(k in lower for k in ("plan", "schedule", "save money", "2 pm", "14:00", "tariff", "optim")):
        if excluded:
            return (
                f"Water quality outranks tariff savings: **{', '.join(excluded)}** is `NON_COMPLIANT` "
                f"(chlorine < 0.20 mg/L) and excluded from peak-tariff cuts. Resolve water quality before throttling."
            )
        peak_mld = min(65 if res.get("has_active_emergency") else 60, int(max_cap))
        off_mld = min(115, int(max_cap))
        mid_mld = min(95, int(max_cap))
        plan = f"00:00-06:00 at {off_mld} MLD, 06:00-14:00 at {mid_mld} MLD, 14:00-18:00 at {peak_mld} MLD, 18:00-24:00 at {mid_mld} MLD"
        sv = tool_executor("show_schedule_validation", {"plan": plan, "current_hour": current_hour})
        tool_executor("show_chart", {"metric": "tariff", "asset": "Grid", "hours": 24})
        return (
            f"At {current_hour:02d}:00 the tariff is **${curr_rate:.2f}/kWh** ({'PEAK' if is_peak else 'OFF/MID-PEAK'}), "
            f"RES-01 is at **{res_pct:.1f}%** (floor **{eff_floor:.1f}%**), and max allowable output is **{max_cap:.0f} MLD**. "
            f"Validated 24h plan: `{sv.get('status')}` — {sv.get('summary', '')}. "
            f"Ask me to **set output to {peak_mld} MLD** if you want to stage it for approval."
        )

    # 9. Leaks question
    if "leak" in lower:
        tool_executor("show_map", {"sector": leaks[0]["sector"] if leaks else ""})
        tool_executor("show_table", {"what": "leaks"})
        if leaks:
            l_desc = "; ".join(f"{l['junction_id']} ({l['sector']}): {l['diagnostic']}" for l in leaks)
            return f"**{len(leaks)} active confirmed pipe leak(s):** {l_desc}."
        return f"No active hydraulic pipe leak signatures detected in live telemetry ({len(quarantined)} quarantined sensor(s))."

    # 10. Pending approvals question ("what are the approvals waiting for me")
    if "pending" in lower or "approval" in lower or "queue" in lower or "waiting for me" in lower:
        ap = tool_executor("show_pending_approvals", {})
        pend_list = ap.get("pending_actions") or []
        if not pend_list:
            return f"There are currently **0** pending approvals waiting for you.{loss_suffix}"
        items = "; ".join(
            f"#{p['id']} `{p['action_type']}` on `{p['target_id']}`"
            for p in pend_list[:5]
        )
        return (
            f"There are **{len(pend_list)}** pending approval(s) waiting for you: {items}. "
            f"Use the **Approve** or **Reject** buttons on the card below to decide.{loss_suffix}"
        )

    # 11. Table requests
    for tbl_key in ("sensors", "valves", "leaks"):
        if tbl_key in lower and any(k in lower for k in ("show", "list", "table", "all")):
            t = tool_executor("show_table", {"what": tbl_key})
            return f"Showing live **{tbl_key}** table ({t.get('row_count', 0)} rows)."

    # 12. Default: clear live plant status
    snap = tool_executor("read_valves_and_pending_actions", {})
    pend_n = len(snap.get("pending_actions", []))
    if pend_n > 0:
        tool_executor("show_pending_approvals", {})
    wq_note = f"NON_COMPLIANT sectors: {', '.join(excluded)}" if excluded else "all 6 sectors chlorine compliant"
    return (
        f"**Live Grid Status ({current_hour:02d}:00):** Reservoir RES-01 is at **{res_pct:.1f}%** "
        f"(safety floor **{eff_floor:.1f}%**), DESAL-01 is at **{plant_mld:.0f} MLD** (cap {max_cap:.0f} MLD), "
        f"tariff is **${curr_rate:.2f}/kWh** ({'PEAK' if is_peak else 'OFF/MID-PEAK'}), "
        f"**{len(leaks)}** active leak(s), **{len(quarantined)}** quarantined sensor(s), "
        f"**{pend_n}** pending approval(s), and {wq_note}."
    )


def run_chat_turn(
    user_message: str,
    history: Optional[list[dict]] = None,
    current_hour: int = DEFAULT_SCADA_HOUR,
    db_path: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Process one operator chat message with injection filtering, multi-turn history, live read & display tools,
    and stage-gated write proposals. Emits live progress updates via status_callback.
    """
    cleaned = (user_message or "").strip()
    prior = list(history or [])
    if not cleaned:
        return {
            "role": "assistant",
            "kind": "text",
            "content": "Ask me anything about the plant, leaks, tariffs, schedules, or request a change.",
            "provider": "System",
            "tools_called": [],
            "code_checks": [],
            "widgets": [],
            "staged_action": None,
        }

    # Log every incoming operator question
    _audit("CHAT_QUESTION", "ASK_AQUAGRID", "ALLOWED", f"Operator asked: {cleaned[:300]}", actor="HUMAN_OPERATOR", db_path=db_path)

    # ── SAFETY 1: Prompt Injection Filter ─────────────────────────────────────
    _emit_status(status_callback, "Checking your request (injection filter)...")
    is_inj, patterns = detect_injection(cleaned)
    if is_inj:
        _audit(
            "CHAT_INJECTION_BLOCKED",
            "ASK_AQUAGRID",
            "BLOCKED",
            f"Blocked injection in chat message ({patterns}): {cleaned[:200]}",
            actor="CODE_GUARD",
            db_path=db_path,
        )
        return {
            "role": "assistant",
            "kind": "refusal",
            "content": (
                "Request refused (`BLOCKED`): prompt-injection or safety-override syntax was detected "
                "and blocked before reaching the AI. All safety guards remain active and no actuators were changed."
            ),
            "provider": "CODE_GUARD (Injection Filter)",
            "tools_called": [],
            "code_checks": [
                {
                    "check": "Prompt Injection Filter",
                    "status": "BLOCKED",
                    "code": "BLOCKED",
                    "detail": f"Matched pattern(s): {', '.join(patterns)}",
                }
            ],
            "widgets": [],
            "staged_action": None,
        }

    # ── SAFETY 2: Unauthorized Pre-Approved / Bypass Claim ────────────────────
    if _PRE_APPROVED_BYPASS_RE.search(cleaned):
        _audit(
            "CHAT_APPROVAL_BYPASS_REFUSED",
            "ASK_AQUAGRID",
            "REJECTED_UNAUTHORIZED_APPROVAL_CLAIM",
            f"Refused unauthorized approval/bypass claim in chat: {cleaned[:200]}",
            actor="CODE_GUARD",
            db_path=db_path,
        )
        return {
            "role": "assistant",
            "kind": "refusal",
            "content": (
                "Request refused (`REJECTED_UNAUTHORIZED_APPROVAL_CLAIM`): typed messages claiming "
                "`pre_approved=true` or asking to bypass operator approval are never honored. "
                "Ask AquaGrid can only stage changes as `STAGED_PENDING_APPROVAL` for you to Approve or Reject via buttons."
            ),
            "provider": "CODE_GUARD (Write Gateway)",
            "tools_called": [],
            "code_checks": [
                {
                    "check": "Prompt Injection Filter",
                    "status": "PASS",
                    "code": "ALLOWED",
                    "detail": "No injection syntax.",
                },
                {
                    "check": "Approval Authorization Guard",
                    "status": "BLOCKED",
                    "code": "REJECTED_UNAUTHORIZED_APPROVAL_CLAIM",
                    "detail": "Typed pre-approval / bypass claims are always refused.",
                },
            ],
            "widgets": [{"type": "approvals"}],
            "staged_action": None,
        }

    # ── SAFETY 3: Typed "approve" NEVER approves anything ─────────────────────
    if _TYPED_APPROVAL_RE.match(cleaned):
        _audit(
            "CHAT_TYPED_APPROVAL_REFUSED",
            "PENDING_ACTIONS",
            "REFUSED",
            f"Refused typed approval text in chat ('{cleaned}'); button click required.",
            actor="CODE_GUARD",
            db_path=db_path,
        )
        return {
            "role": "assistant",
            "kind": "refusal",
            "content": (
                "Typed text like **\"approve\"** cannot approve or execute physical actions (`REFUSED`). "
                "Only clicking the **Approve** button on an approval card below can execute an action "
                "(with the just-in-time safety re-check and single-use approval token)."
            ),
            "provider": "CODE_GUARD (HITL Button Guard)",
            "tools_called": [],
            "code_checks": [
                {
                    "check": "HITL Button-Only Approval Guard",
                    "status": "BLOCKED",
                    "code": "REFUSED",
                    "detail": "Typed approval ignored; physical actuation requires clicking the Approve button.",
                }
            ],
            "widgets": [{"type": "approvals"}],
            "staged_action": None,
        }

    # ── SCENARIO COMMAND: "Run scenario 2" / "Run U-5.1" ──────────────────────
    sc_id = parse_scenario_command(cleaned)
    if sc_id:
        return run_scenario_for_chat(sc_id, current_hour=current_hour, db_path=db_path, status_callback=status_callback)

    # ── ALWAYS PRE-FETCH LIVE SCADA SNAPSHOT BEFORE CALLING THE LLM ───────────
    _emit_status(status_callback, "Reading live plant data...")
    live_snapshot = build_live_scada_snapshot(current_hour=current_hour, db_path=db_path)

    # ── PRODUCTION PLANNER FLOW: "make a 5-hour plan", "plan next N hours", "plan for tomorrow"
    plan_hours = parse_planning_horizon_hours(cleaned)
    if plan_hours is not None:
        return run_production_planner_for_chat(
            user_message=cleaned,
            horizon_hours=plan_hours,
            snapshot=live_snapshot,
            current_hour=current_hour,
            db_path=db_path,
            status_callback=status_callback,
        )

    # ── LIVE TOOL EXECUTOR (read-only + display + single gated stage tool) ────
    widgets: list[dict] = []
    tools_called_summary: list[dict] = []
    code_checks: list[dict] = [
        {
            "check": "Prompt Injection Filter",
            "status": "PASS",
            "code": "ALLOWED",
            "detail": "No prompt-injection syntax detected.",
        }
    ]
    staged_action_holder: list[dict] = []
    map_results_holder: list[dict] = []
    twin_3d_results_holder: list[dict] = []

    def _record_check(name: str, passed: bool, code: str, detail: str) -> None:
        for existing in code_checks:
            if existing["check"] == name:
                existing["status"] = "PASS" if passed else "BLOCKED"
                existing["code"] = code
                existing["detail"] = detail
                return
        code_checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "BLOCKED",
                "code": code,
                "detail": detail,
            }
        )

    def _execute_tool(tool_name: str, args: dict) -> dict:
        args = args or {}
        out: dict = {}

        if tool_name == "read_network_telemetry":
            h = int(args.get("current_hour", current_hour))
            out = read_network_telemetry(current_hour=h, db_path=db_path, log_audit=True)
            tf = out.get("telemetry_freshness") or {}
            _record_check(
                "Telemetry Freshness",
                not bool(out.get("telemetry_loss")),
                tf.get("status", "FRESH"),
                tf.get("banner") or f"Readings fresh ({tf.get('age_seconds', 0)}s old).",
            )

        elif tool_name == "read_tariff_and_quality_feeds":
            h = int(args.get("current_hour", current_hour))
            horizon = int(args.get("horizon_hours", 6))
            out = read_tariff_and_quality_feeds(horizon_hours=horizon, current_hour=h, db_path=db_path, log_audit=True)
            res = out.get("reservoir") or {}
            res_pct = float(res.get("current_level_percent", 48.0))
            eff_floor = float(res.get("safety_floor_percent", 35.0))
            _record_check(
                "Reservoir Safety Floor",
                res_pct >= eff_floor,
                "VALID" if res_pct >= eff_floor else "REJECTED_BY_SAFETY_GUARD",
                f"RES-01 {res_pct:.1f}% vs {eff_floor:.1f}% floor.",
            )
            excl = out.get("excluded_sectors") or []
            _record_check(
                "Sector Water Quality",
                len(excl) == 0,
                "COMPLIANT" if not excl else "NON_COMPLIANT",
                "All 6 sectors >= 0.20 mg/L chlorine." if not excl else f"NON_COMPLIANT: {', '.join(excl)}",
            )

        elif tool_name == "read_valves_and_pending_actions":
            out = _read_valves_and_pending_snapshot(db_path=db_path)

        elif tool_name == "read_audit_log":
            entries = _read_audit_entries(
                limit=int(args.get("limit", 20)),
                event_filter=str(args.get("event_filter", "")),
                db_path=db_path,
            )
            out = {"count": len(entries), "entries": entries}

        elif tool_name == "show_3d":
            _emit_status(status_callback, "Drawing the 3D digital twin...")
            focus_arg = str(args.get("focus") or "all").strip()
            out = build_3d_render_summary(
                focus=focus_arg,
                current_hour=current_hour,
                db_path=db_path,
            )
            twin_3d_results_holder.append(out)
            _append_unique_widget(
                widgets,
                {
                    "type": "twin_3d",
                    "focus": out["focus"],
                },
            )

        elif tool_name == "show_map":
            _emit_status(status_callback, "Drawing the map...")
            sector = str(args.get("sector") or "all").strip()
            leak_id = str(args.get("leak_id") or "").strip() or None
            show_b = bool(args.get("show_boundaries", True))
            out = build_map_render_summary(
                sector=sector,
                leak_id=leak_id,
                show_boundaries=show_b,
                db_path=db_path,
            )
            map_results_holder.append(out)
            _append_unique_widget(
                widgets,
                {
                    "type": "map",
                    "sector": out["highlight_sector"] or "all",
                    "leak_id": out["leak_id"],
                    "show_boundaries": out["show_boundaries"],
                },
            )

        elif tool_name == "show_chart":
            metric = str(args.get("metric") or "reservoir").lower()
            if metric not in ("pressure", "flow", "reservoir", "reservoir_trend", "tariff"):
                metric = "reservoir"
            asset = str(args.get("asset") or ("J-14" if metric in ("pressure", "flow") else "RES-01")).strip()
            default_hrs = 2 if metric in ("pressure", "flow") else 24
            hours = max(1, min(int(args.get("hours", default_hrs)), 48))
            _append_unique_widget(widgets, {"type": "chart", "metric": metric, "asset": asset, "hours": hours})
            summary_txt = f"{metric} over {hours}h for {asset}"
            if metric in ("pressure", "flow"):
                conn = get_db_connection(db_path)
                try:
                    prefix = "P-%" if metric == "pressure" else "F-%"
                    rows = conn.execute(
                        "SELECT sensor_id, junction_id, sector, value, unit FROM telemetry WHERE (junction_id = ? OR sensor_id = ? OR sector = ?) AND sensor_id LIKE ? ORDER BY hour DESC LIMIT 6;",
                        (asset, asset, asset, prefix),
                    ).fetchall()
                finally:
                    conn.close()
                if rows:
                    vals = [float(r["value"]) for r in rows]
                    summary_txt = f"{asset} {metric}: latest {vals[0]:.1f} {rows[0]['unit']} (min {min(vals):.1f}, max {max(vals):.1f})"
            out = {"displayed": True, "metric": metric, "asset": asset, "hours": hours, "summary": summary_txt}

        elif tool_name == "show_table":
            what = str(args.get("what") or "leaks").lower()
            if what not in ("leaks", "sensors", "schedule", "audit", "valves"):
                what = "leaks"
            _append_unique_widget(widgets, {"type": "table", "what": what})
            conn = get_db_connection(db_path)
            try:
                tbl_map = {
                    "leaks": "SELECT COUNT(*) AS cnt FROM leaks;",
                    "sensors": "SELECT COUNT(*) AS cnt FROM sensors;",
                    "schedule": "SELECT COUNT(*) AS cnt FROM tariffs;",
                    "audit": "SELECT COUNT(*) AS cnt FROM audit_log;",
                    "valves": "SELECT COUNT(*) AS cnt FROM valves;",
                }
                cnt = conn.execute(tbl_map[what]).fetchone()["cnt"]
            finally:
                conn.close()
            out = {"displayed": True, "what": what, "row_count": cnt}

        elif tool_name == "show_schedule_validation":
            _emit_status(status_callback, "Validating the plan hour by hour...")
            plan = str(args.get("plan") or "").strip()
            h = int(args.get("current_hour", current_hour))
            feeds_now = read_tariff_and_quality_feeds(horizon_hours=6, current_hour=h, db_path=db_path, log_audit=False)
            res = feeds_now.get("reservoir") or {}
            wq = feeds_now.get("water_quality") or {}
            demand_list = [float(d["demand_mld"]) for d in feeds_now.get("demand_forecast", []) if "demand_mld" in d]
            val = validate_production_schedule(
                proposed_schedule=plan,
                current_volume_ml=float(res.get("current_volume_ml", 240.0)),
                demand_by_hour=feeds_now.get("hourly_demand_24h", {}),
                fallback_demand_mld=demand_list or None,
                bloom_severity=str(wq.get("bloom_severity", "NONE")),
                turbidity_ntu=float(wq.get("intake_turbidity_ntu", 1.2)),
                chlorine_ppm=float(wq.get("intake_chlorine_ppm", 1.8)),
                min_floor_percent=float(res.get("safety_floor_percent", 35.0)) / 100.0,
                current_hour=h,
                current_plant_output_mld=float(feeds_now.get("current_plant_output_mld", 100.0)),
                excluded_sectors=feeds_now.get("excluded_sectors", []),
            )
            _append_unique_widget(widgets, {"type": "schedule_validation", "plan": plan, "validation": val})
            _record_check(
                "24h Schedule Validator",
                val.get("status") == "VALID",
                val.get("status", "UNKNOWN"),
                val.get("summary", ""),
            )
            out = val

        elif tool_name == "show_pending_approvals":
            snap = _read_valves_and_pending_snapshot(db_path=db_path)
            _append_unique_widget(widgets, {"type": "approvals"})
            out = {"displayed": True, "pending_count": len(snap["pending_actions"]), "pending_actions": snap["pending_actions"]}

        elif tool_name == "propose_gated_action":
            _emit_status(status_callback, "Running safety checks...")
            a_type = str(args.get("action_type") or "").upper()
            target = str(args.get("target_id") or "").upper()
            try:
                p_val = float(args.get("proposed_value", 0.0))
            except (TypeError, ValueError):
                p_val = 0.0
            rationale = str(args.get("rationale") or f"Proposed via Ask AquaGrid chat: {cleaned[:140]}")
            out = stage_gated_valve_or_setpoint_action(
                action_type=a_type,
                target_id=target,
                proposed_value=p_val,
                rationale=rationale,
                db_path=db_path,
                provenance="AGENT_ANALYSIS",
                provenance_ref="ASK_AQUAGRID_CHAT",
            )
            staged_action_holder.append(out)
            gw_st = str(out.get("status", "UNKNOWN"))
            _record_check(
                "Write Gateway & Safety Guards",
                gw_st == "STAGED_PENDING_APPROVAL",
                gw_st,
                out.get("reason") or out.get("message") or "",
            )
            if gw_st in ("STAGED_PENDING_APPROVAL", "REJECTED_DUPLICATE_PENDING"):
                _append_unique_widget(widgets, {"type": "approvals"})
        else:
            out = {"error": f"Unknown tool '{tool_name}'."}

        summary = _summarize_tool_output(tool_name, out)
        tools_called_summary.append({"tool": tool_name, "args": args, "summary": summary})
        _audit(
            "CHAT_TOOL_CALL",
            tool_name,
            str(out.get("status", "ALLOWED")) if isinstance(out, dict) else "ALLOWED",
            f"args={json.dumps(args, default=str)[:180]} -> {summary[:220]}",
            actor="LLM_AGENT",
            db_path=db_path,
        )
        return out

    # Seed code checks & snapshot summary from the pre-fetched live snapshot
    tf_snap = live_snapshot["telemetry_freshness"]
    res_snap = live_snapshot["reservoir"]
    excl_snap = live_snapshot["excluded_sectors"]
    _record_check(
        "Telemetry Freshness",
        not bool(tf_snap.get("is_telemetry_loss")),
        tf_snap.get("status", "FRESH"),
        tf_snap.get("banner", "Telemetry fresh."),
    )
    _record_check(
        "Reservoir Safety Floor",
        float(res_snap["level_percent"]) >= float(res_snap["safety_floor_percent"]),
        "VALID" if float(res_snap["level_percent"]) >= float(res_snap["safety_floor_percent"]) else "REJECTED_BY_SAFETY_GUARD",
        f"RES-01 {res_snap['level_percent']:.1f}% vs {res_snap['safety_floor_percent']:.1f}% floor.",
    )
    _record_check(
        "Sector Water Quality",
        len(excl_snap) == 0,
        "COMPLIANT" if not excl_snap else "NON_COMPLIANT",
        "All 6 sectors >= 0.20 mg/L chlorine." if not excl_snap else f"NON_COMPLIANT: {', '.join(excl_snap)}",
    )
    snapshot_json = _compact_snapshot_json(live_snapshot)
    tools_called_summary.append(
        {
            "tool": "live_scada_snapshot",
            "args": {"current_hour": current_hour},
            "summary": (
                f"RES-01 {res_snap['level_percent']:.1f}% (floor {res_snap['safety_floor_percent']:.1f}%), "
                f"DESAL-01 {live_snapshot['desal_01']['current_output_mld']:.0f} MLD (cap {live_snapshot['desal_01']['algae_cap_mld']:.0f} MLD), "
                f"{len(live_snapshot['active_leaks'])} leak(s), {len(live_snapshot['pending_actions'])} pending"
            ),
        }
    )

    # Build multi-turn message list for the LLM WITH live SCADA snapshot always pre-loaded!
    llm_messages: list[dict] = [
        {"role": "system", "content": ASK_AQUAGRID_SYSTEM_PROMPT},
        {"role": "system", "content": "LIVE SCADA SNAPSHOT (current real-time values — do not ask to enable tools or run queries):\n" + snapshot_json},
    ]
    for msg in prior[-6:]:
        role = msg.get("role")
        content = str(msg.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            trimmed = content if len(content) <= 450 else (content[:447] + "...")
            llm_messages.append({"role": role, "content": trimmed})
    llm_messages.append({"role": "user", "content": f"[Active SCADA Hour: {current_hour:02d}:00]\n{cleaned}"})

    def _build_live_fallback_context() -> str:
        return snapshot_json

    # Ensure llm_client has chat_with_tools even after a live Streamlit hot-reload
    active_llm_client = llm_client
    if not hasattr(active_llm_client, "chat_with_tools"):
        import agent.llm_factory as _lf_mod

        _lf_mod = importlib.reload(_lf_mod)
        active_llm_client = _lf_mod.llm_client

    providers_tried: list[dict] = []
    try:
        chat_fn = active_llm_client.chat_with_tools
        accepts_ctx, accepts_status = _accepted_chat_optional_kwargs(chat_fn)

        chat_kwargs: dict[str, Any] = {
            "messages": llm_messages,
            "tools": CHAT_TOOLS_SCHEMA,
            "tool_executor": _execute_tool,
            "max_tool_steps": MAX_CHAT_TOOL_STEPS,
        }
        if accepts_ctx:
            chat_kwargs["fallback_context_builder"] = _build_live_fallback_context
        if accepts_status:
            chat_kwargs["status_callback"] = status_callback

        answer_text, provider_used, _ = chat_fn(**chat_kwargs)

        providers_tried = list(getattr(active_llm_client, "last_chat_providers_tried", []) or [])
        # If the operator asked for a 3D focus target ("show Sector 4", "where is the leak", "show the reservoir", "show valve V-12")
        # and the LLM didn't call show_3d or show_map, invoke show_3d directly!
        lower_q = cleaned.lower()
        requested_3d_focus = _parse_3d_focus_request(cleaned)
        if requested_3d_focus is not None and not twin_3d_results_holder and not map_results_holder:
            _execute_tool("show_3d", {"focus": requested_3d_focus})

        # If the operator asks for the reservoir trend ("show reservoir trend" / "reservoir depletion"), attach the 6h+12h trend chart
        if ("reservoir" in lower_q or "res-01" in lower_q) and any(k in lower_q for k in ("trend", "deplet", "project", "floor in")):
            upgraded = False
            for w in widgets:
                if w.get("type") == "chart" and w.get("metric") in ("reservoir", "reservoir_trend"):
                    w["metric"] = "reservoir_trend"
                    w["hours"] = 18
                    upgraded = True
            if not upgraded:
                _execute_tool("show_chart", {"metric": "reservoir_trend", "asset": "RES-01", "hours": 18})
            if not twin_3d_results_holder:
                _execute_tool("show_3d", {"focus": "reservoir_inside"})

        # If the user asked for a 2D map or "show the sectors" and the LLM skipped calling show_map, invoke show_map directly
        asked_for_map = _is_map_request(cleaned)
        if asked_for_map and not map_results_holder and not twin_3d_results_holder:
            m_sec = re.search(r"sector\s*([1-6])", cleaned, re.IGNORECASE)
            sec_arg = f"Sector {m_sec.group(1)}" if m_sec else "all"
            m_lid = re.search(r"(LEAK-\d+)", cleaned, re.IGNORECASE)
            lid_arg = m_lid.group(1).upper() if m_lid else None
            show_b = not ("without boundar" in lower_q or "no boundar" in lower_q or "hide boundar" in lower_q)
            _execute_tool("show_map", {"sector": sec_arg, "leak_id": lid_arg or "", "show_boundaries": show_b})

        # Safety net: never return a response begging the operator to enable tools or run live queries!
        answer_text = "" if answer_text is None else str(answer_text)
        if not answer_text.strip() or _TOOL_BEGGING_RE.search(answer_text):
            if answer_text and _TOOL_BEGGING_RE.search(answer_text):
                logger.warning("Replaced tool-begging LLM response from %s with live-snapshot answer", provider_used)
            answer_text = _offline_chat_fallback(
                user_message=cleaned,
                history=prior,
                tool_executor=_execute_tool,
                current_hour=current_hour,
            )
            if not provider_used or provider_used == "Deterministic Offline Engine":
                provider_used = "Deterministic Offline Engine"

        # If the operator asked for 3D or 2D map and the LLM asked "which sector?", answer directly using the tool summary
        if twin_3d_results_holder and _ASKING_WHICH_SECTOR_RE.search(answer_text):
            answer_text = str(twin_3d_results_holder[-1].get("summary_text") or answer_text)
        if map_results_holder and _ASKING_WHICH_SECTOR_RE.search(answer_text):
            answer_text = str(map_results_holder[-1].get("summary_text") or answer_text)

        # Truthfulness check: if AI claims boundaries/borders were drawn when show_map did NOT draw them, replace with exact tool summary!
        if map_results_holder:
            last_map = map_results_holder[-1]
            if _BOUNDARY_CLAIM_RE.search(answer_text) and int(last_map.get("boundaries_drawn", 0)) == 0:
                logger.warning("Replaced untruthful map boundary description with exact show_map tool summary")
                answer_text = str(last_map.get("summary_text") or "Displayed live map.")
        elif _BOUNDARY_CLAIM_RE.search(answer_text) and asked_for_map:
            answer_text = "No map was rendered for this reply."
    except Exception as exc:
        logging.exception("Ask AquaGrid chat turn failed for message: %s", cleaned[:160])
        providers_tried = list(getattr(active_llm_client, "last_chat_providers_tried", []) or [])
        tb_frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
        last_frame = tb_frames[-1] if tb_frames else None
        location = f"{last_frame.filename}:{last_frame.lineno}" if last_frame else "unknown"
        err_summary = f"{type(exc).__name__}: {exc} ({location})"
        tried_names = [t.get("provider", "?") for t in providers_tried]
        tried_note = f" Providers tried: {', '.join(tried_names)}." if tried_names else ""
        _audit("CHAT_ERROR", "ASK_AQUAGRID", "ERROR", err_summary, actor="CODE_GUARD", db_path=db_path)
        return {
            "role": "assistant",
            "kind": "error",
            "content": (
                f"Sorry, all AI providers failed to answer ({type(exc).__name__}).{tried_note} "
                f"All SCADA safety guards remain active — please try asking again."
            ),
            "provider": "Error Handler",
            "tools_called": tools_called_summary,
            "code_checks": code_checks,
            "widgets": widgets,
            "staged_action": staged_action_holder[-1] if staged_action_holder else None,
            "providers_tried": providers_tried,
            "error_detail": err_summary,
            "error": True,
        }

    final_staged = staged_action_holder[-1] if staged_action_holder else None
    ans_status = str(final_staged.get("status", "ALLOWED")) if final_staged else "ALLOWED"

    # Compute twin_view update so the persistent top 3D digital twin follows chat answers automatically
    from digital_twin.scene_state import build_twin_view_command

    reply_twin_view = None
    if twin_3d_results_holder:
        reply_twin_view = twin_3d_results_holder[-1].get("twin_view")
    elif map_results_holder:
        m_last = map_results_holder[-1]
        m_focus = m_last.get("leak_id") or m_last.get("highlight_sector") or "all"
        reply_twin_view = build_twin_view_command(focus=m_focus, event_type="focus", note=m_last.get("zoom_target", ""))
    elif final_staged and final_staged.get("status") == "STAGED_PENDING_APPROVAL":
        from digital_twin.hydraulics import simulate_valve_change_impact

        a_t = str(final_staged.get("action_type", "")).upper()
        t_id = str(final_staged.get("target_id", "")).upper()
        if a_t in ("CLOSE_VALVE", "OPEN_VALVE"):
            imp = simulate_valve_change_impact(t_id, "CLOSED" if a_t == "CLOSE_VALVE" else "OPEN", db_path=db_path)
            reply_twin_view = build_twin_view_command(
                focus=t_id,
                event_type="staged_proposal",
                note=f"Staged {a_t} {t_id} (awaiting Approve)",
                highlight_pipe_ids=imp["affected_pipe_ids"],
            )
        elif a_t == "SET_PLANT_OUTPUT":
            reply_twin_view = build_twin_view_command(focus="reservoir_inside", event_type="staged_proposal", note=f"Staged DESAL-01 -> {final_staged.get('proposed_value')} MLD")
        elif a_t == "ISOLATE_AND_FLUSH" and t_id.startswith("SECTOR-"):
            sec_num = t_id.split("-")[-1].lstrip("0") or "1"
            reply_twin_view = build_twin_view_command(focus=f"Sector {sec_num}", event_type="staged_proposal", note=f"Staged ISOLATE_AND_FLUSH Sector {sec_num}")

    _audit(
        "CHAT_ANSWER",
        "ASK_AQUAGRID",
        ans_status,
        f"Provider={provider_used} | Tools={len(tools_called_summary)} | Answer={answer_text[:240]}",
        actor="LLM_AGENT",
        db_path=db_path,
    )

    return {
        "role": "assistant",
        "kind": "text",
        "content": answer_text,
        "provider": provider_used,
        "tools_called": tools_called_summary,
        "code_checks": code_checks,
        "widgets": widgets,
        "staged_action": final_staged,
        "providers_tried": providers_tried,
        "twin_view": reply_twin_view,
    }


def collect_system_alerts(
    telemetry_data: dict,
    feeds_data: dict,
    seen_alert_keys: set[str],
    live_tick_summary: Optional[dict] = None,
) -> tuple[list[dict], set[str]]:
    """
    Inspect live telemetry, feeds, and optional Live Simulation batch summary.
    Returns (new_alert_chat_messages, updated_seen_keys) so new alerts appear in chat automatically.
    """
    new_msgs: list[dict] = []
    updated_seen = set(seen_alert_keys)
    from digital_twin.scene_state import build_twin_view_command

    # 1. Telemetry Loss (ONLY when actually in TELEMETRY_LOSS, never when Live Sim is merely OFF)
    if telemetry_data.get("telemetry_loss"):
        tf = telemetry_data.get("telemetry_freshness") or {}
        banner = tf.get("banner") or "SCADA telemetry silent > 180s. Holding last safe state."
        key = f"TELEMETRY_LOSS:{tf.get('last_data_local', 'stale')}:{banner}"
        if key not in updated_seen:
            updated_seen.add(key)
            new_msgs.append(
                {
                    "role": "assistant",
                    "kind": "alert",
                    "alert_code": "TELEMETRY_LOSS",
                    "alert_title": "ALERT",
                    "alert_body": banner,
                    "content": f"ALERT (TELEMETRY_LOSS): {banner}",
                    "provider": "CODE_GUARD (Auto Alert)",
                    "tools_called": [],
                    "code_checks": [{"check": "Telemetry Freshness", "status": "BLOCKED", "code": "TELEMETRY_LOSS", "detail": banner}],
                    "widgets": [{"type": "table", "what": "sensors"}],
                    "staged_action": None,
                    "twin_view": build_twin_view_command(focus="all", event_type="telemetry_loss", note="TELEMETRY LOSS — holding last safe state"),
                }
            )

    # 2. Water Quality Non-Compliance
    wq_comp = feeds_data.get("sector_water_quality_compliance") or {}
    for wq_alert in wq_comp.get("alerts", []):
        key = f"WQ:{wq_alert}"
        if key not in updated_seen:
            updated_seen.add(key)
            new_msgs.append(
                {
                    "role": "assistant",
                    "kind": "alert",
                    "alert_code": "NON_COMPLIANT",
                    "alert_title": "ALERT",
                    "alert_body": wq_alert,
                    "content": f"ALERT (NON_COMPLIANT): {wq_alert}",
                    "provider": "CODE_GUARD (Auto Alert)",
                    "tools_called": [],
                    "code_checks": [{"check": "Sector Water Quality", "status": "BLOCKED", "code": "NON_COMPLIANT", "detail": wq_alert}],
                    "widgets": [{"type": "map", "sector": "Sector 2", "leak_id": None}],
                    "staged_action": None,
                    "twin_view": build_twin_view_command(focus="Sector 2", event_type="water_quality", note="Sector 2 NON_COMPLIANT"),
                }
            )

    # 3. Confirmed Pipe Leaks
    for lk in telemetry_data.get("leak_alerts", []):
        key = f"LEAK:{lk.get('junction_id')}:{lk.get('sector')}"
        if key not in updated_seen:
            updated_seen.add(key)
            body = f"{lk.get('junction_id')} ({lk.get('sector')}) — {lk.get('diagnostic', '')}"
            lk_focus = lk.get("leak_id") or lk.get("junction_id") or lk.get("sector") or "Sector 4"
            new_msgs.append(
                {
                    "role": "assistant",
                    "kind": "alert",
                    "alert_code": "CONFIRMED_PIPE_LEAK",
                    "alert_title": "ALERT",
                    "alert_body": body,
                    "content": f"ALERT (CONFIRMED_PIPE_LEAK): {body}",
                    "provider": "CODE_GUARD (Auto Alert)",
                    "tools_called": [],
                    "code_checks": [{"check": "Hydraulic Mass Balance", "status": "BLOCKED", "code": "CONFIRMED_PIPE_LEAK", "detail": lk.get("diagnostic", "")}],
                    "widgets": [
                        {"type": "map", "sector": lk.get("sector"), "leak_id": None},
                        {"type": "chart", "metric": "pressure", "asset": lk.get("junction_id", "J-14"), "hours": 2},
                    ],
                    "staged_action": None,
                    "twin_view": build_twin_view_command(focus=lk_focus, event_type="new_leak", note=f"New leak at {lk.get('junction_id', lk_focus)}"),
                }
            )

    # 4. Reservoir Floor Breach, Warning Band (within 5% of floor), or Fire Emergency Floor Raised
    res = feeds_data.get("reservoir") or {}
    res_pct = float(res.get("current_level_percent", 48.0))
    eff_floor = float(res.get("safety_floor_percent", RESERVOIR_FLOOR_PERCENT * 100.0))
    rate_str = str(res.get("rate_str") or "")
    eta_str = str(res.get("floor_eta_str") or "")
    trend_note = f" ({rate_str}, {eta_str})" if rate_str and eta_str else ""
    if res_pct <= eff_floor:
        key = f"FLOOR_BREACH:{round(res_pct, 1)}:{round(eff_floor, 1)}"
        if key not in updated_seen:
            updated_seen.add(key)
            body = f"RES-01 at {res_pct:.1f}%{trend_note} — at/below mandatory {eff_floor:.1f}% safety floor."
            new_msgs.append(
                {
                    "role": "assistant",
                    "kind": "alert",
                    "alert_code": "REJECTED_BY_SAFETY_GUARD",
                    "alert_title": "RESERVOIR FLOOR BREACH",
                    "alert_body": body,
                    "content": f"ALERT (RESERVOIR FLOOR BREACH): {body}",
                    "provider": "CODE_GUARD (Auto Alert)",
                    "tools_called": [],
                    "code_checks": [{"check": "Reservoir Safety Floor", "status": "BLOCKED", "code": "REJECTED_BY_SAFETY_GUARD", "detail": f"{res_pct:.1f}% <= {eff_floor:.1f}%"}],
                    "widgets": [{"type": "chart", "metric": "reservoir_trend", "asset": "RES-01", "hours": 18}],
                    "staged_action": None,
                    "twin_view": build_twin_view_command(focus="reservoir_inside", event_type="reservoir_alert", note=f"RES-01 at/below floor ({res_pct:.1f}%)"),
                }
            )
    elif res_pct <= eff_floor + 5.0:
        key = f"FLOOR_WARN:{round(res_pct, 1)}:{round(eff_floor, 1)}"
        if key not in updated_seen:
            updated_seen.add(key)
            body = f"RES-01 is at {res_pct:.1f}%{trend_note} — within 5% of the {eff_floor:.1f}% safety floor."
            new_msgs.append(
                {
                    "role": "assistant",
                    "kind": "alert",
                    "alert_code": "RESERVOIR_WARNING",
                    "alert_title": "RESERVOIR LOW WARNING",
                    "alert_body": body,
                    "content": f"ALERT (RESERVOIR LOW WARNING): {body}",
                    "provider": "CODE_GUARD (Auto Alert)",
                    "tools_called": [],
                    "code_checks": [{"check": "Reservoir Warning Margin", "status": "BLOCKED", "code": "RESERVOIR_WARNING", "detail": f"{res_pct:.1f}% within 5% of {eff_floor:.1f}% floor"}],
                    "widgets": [{"type": "chart", "metric": "reservoir_trend", "asset": "RES-01", "hours": 18}],
                    "staged_action": None,
                    "twin_view": build_twin_view_command(focus="reservoir_inside", event_type="reservoir_alert", note=f"RES-01 near floor ({res_pct:.1f}%)"),
                }
            )
    elif res.get("has_active_emergency"):
        banner = res.get("floor_banner", f"Reservoir floor raised to {eff_floor:.1f}%")
        key = f"EMERGENCY_FLOOR:{banner}"
        if key not in updated_seen:
            updated_seen.add(key)
            new_msgs.append(
                {
                    "role": "assistant",
                    "kind": "alert",
                    "alert_code": "EMERGENCY_RESERVE",
                    "alert_title": "EMERGENCY RESERVE",
                    "alert_body": banner,
                    "content": f"ALERT (EMERGENCY RESERVE): {banner}",
                    "provider": "CODE_GUARD (Auto Alert)",
                    "tools_called": [],
                    "code_checks": [{"check": "Dynamic Emergency Floor", "status": "PASS", "code": "VALID", "detail": banner}],
                    "widgets": [{"type": "chart", "metric": "reservoir", "asset": "RES-01", "hours": 24}],
                    "staged_action": None,
                    "twin_view": build_twin_view_command(focus="Sector 6", event_type="fire_emergency", note=f"Sector 6 fire — floor {eff_floor:.1f}%"),
                }
            )

    # 5. Live Simulation Batch Results (post concise live check updates when a new batch completes)
    if live_tick_summary and live_tick_summary.get("tick"):
        tick_id = live_tick_summary["tick"]
        key = f"LIVE_TICK:{tick_id}"
        if key not in updated_seen:
            updated_seen.add(key)
            raised = live_tick_summary.get("raised_alerts") or []
            ai_run = live_tick_summary.get("ai_run") or {}
            clk = live_tick_summary.get("clock_display", "")
            if raised:
                body = f"Live Batch #{tick_id} ({clk}): " + " | ".join(raised[:3])
                new_lk_id = live_tick_summary.get("new_leak_id")
                live_tv = (
                    build_twin_view_command(focus=new_lk_id, event_type="new_leak", note=f"New live leak {new_lk_id}")
                    if new_lk_id
                    else build_twin_view_command(focus="all", event_type="live_alert", note=raised[0][:60])
                )
                new_msgs.append(
                    {
                        "role": "assistant",
                        "kind": "alert",
                        "alert_code": "LIVE_ALERT",
                        "alert_title": "ALERT",
                        "alert_body": body,
                        "content": f"ALERT: {body}",
                        "provider": "Live Simulator + CODE_GUARD",
                        "tools_called": [],
                        "code_checks": [{"check": "Live SCADA Batch Guard", "status": "BLOCKED", "code": "ALERT ACTIVE", "detail": "; ".join(raised[:3])}],
                        "widgets": [{"type": "map", "sector": None, "leak_id": new_lk_id}],
                        "staged_action": None,
                        "twin_view": live_tv,
                    }
                )
            elif ai_run.get("run_id"):
                new_msgs.append(
                    {
                        "role": "assistant",
                        "kind": "live_check",
                        "content": (
                            f"**Live Check (Batch #{tick_id} · {clk}):** RES-01 **{live_tick_summary.get('reservoir_level_percent', 0):.1f}%**, "
                            f"Demand **{live_tick_summary.get('demand_mld', 0):.0f} MLD**, Tariff **${live_tick_summary.get('tariff_rate_per_kwh', 0):.2f}/kWh** "
                            f"({live_tick_summary.get('tariff_tier', '')}). AI Check #{ai_run['run_id']}: `{ai_run.get('status', 'OK')}`."
                        ),
                        "provider": ai_run.get("llm_provider", "Live Analyzer"),
                        "tools_called": [],
                        "code_checks": [{"check": "Live SCADA Batch Guard", "status": "PASS", "code": "NOMINAL", "detail": f"Batch #{tick_id} healthy"}],
                        "widgets": [],
                        "staged_action": None,
                    }
                )

    return new_msgs, updated_seen
