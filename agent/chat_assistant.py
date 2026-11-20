"""Ask AquaGrid — Multi-turn real LLM chat assistant with live SCADA read tools, display tools, and stage-gated proposals.

HARD ARCHITECTURAL INVARIANTS:
- Uses the existing LLM fallback chain (Groq 120b -> Groq 20b -> Gemini -> Ollama -> offline engine).
- Read-only data tools & display tools only; physical changes go ONLY through stage_gated_valve_or_setpoint_action().
- NEVER mutates valves or plant_setpoints directly and NEVER approves or executes pending actions.
- Typed text like 'approve' or 'pre_approved=true' NEVER approves anything.
"""
from datetime import datetime, timezone
import json
import re
from typing import Any, Optional

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
1. ALWAYS call the live read tools before stating any number (reservoir %, MLD output, tariff rate, pressures, flows, chlorine residual, leaks, valve states, or audit history). NEVER invent or guess numbers.
2. Answer ANY operator question clearly and concisely: plant status, leaks, quarantined sensors, reservoir & dynamic emergency floor, electricity tariffs, what-if schedules, past actions in the Audit Log, why a plan was rejected, what any technical status code means, and recommended next steps.
3. Use prior conversation messages naturally for follow-up questions (e.g. "Why?", "What about Sector 2?", "And if demand rises?").
4. Use DISPLAY TOOLS to render helpful visuals inside your reply whenever relevant:
   - show_map(sector, leak_id): when asked about map locations, leaks, or a sector.
   - show_chart(metric, asset, hours): metric in ["pressure", "flow", "reservoir", "tariff"].
   - show_table(what): what in ["leaks", "sensors", "schedule", "audit", "valves"].
   - show_schedule_validation(plan): whenever validating or planning a 24h production schedule.
   - show_pending_approvals(): whenever showing pending actions awaiting operator signoff.
5. If the operator asks for a physical change (e.g. "reduce output to 70 MLD", "close V-12", "isolate and flush Sector 2"), you may ONLY call propose_gated_action. All Python safety guards run automatically inside the gateway. State the exact gateway status code (STAGED_PENDING_APPROVAL or REJECTED_* + reason) and remind the operator to click the Approve or Reject button on the approval card.
6. You can NEVER execute, auto-approve, or bypass operator approval. Keep exact technical status codes unchanged (STAGED_PENDING_APPROVAL, REJECTED_BY_SAFETY_GUARD, REJECTED_WATER_QUALITY, CANCELLED_BY_SAFETY_RECHECK, REJECTED_UNAUTHORIZED_APPROVAL_CLAIM, REJECTED_APPROVAL_MISMATCH, REJECTED_APPROVAL_ALREADY_USED, REJECTED_APPROVAL_EXPIRED, TELEMETRY_LOSS, REJECTED_STALE_TELEMETRY, CANCELLED_STALE_TELEMETRY, NON_COMPLIANT, UNTRUSTED_TEXT, VALID, GAP).
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
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_valves_and_pending_actions",
            "description": "Read live isolation valve states (V-01..V-20), DESAL-01 plant setpoint, active emergency commitments, pending/resolved actions, and tracked map leaks.",
            "parameters": {"type": "object", "properties": {}},
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
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_map",
            "description": "Display the interactive Leaflet pipe & leak map inside this chat reply, optionally focused on a sector or leak_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {"type": "string", "description": "Optional sector name like 'Sector 4'."},
                    "leak_id": {"type": "string", "description": "Optional leak ID like 'LEAK-001'."},
                },
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
            "description": "Validate a 24h production schedule using the deterministic Python physics guard and display the hour-by-hour check card in chat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": "Time-blocked schedule string, e.g. '00:00-06:00 at 115 MLD, 06:00-14:00 at 95 MLD, 14:00-18:00 at 60 MLD, 18:00-24:00 at 95 MLD'.",
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
            "parameters": {"type": "object", "properties": {}},
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
) -> dict:
    """Execute a Round 1 or Round 2 scenario and return a rich chat message dict with result card & widgets."""
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

    reset_database(db_path)
    info = entry["injector"](db_path) if db_path else entry["injector"]()

    scenario_num = entry.get("scenario_num")
    agent_scenario_id = scenario_num if scenario_num is not None else entry["id"]
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

    # Attach helpful contextual visuals right in the chat!
    if entry["id"] in ("R1-S1", "R1-S3"):
        widgets.append({"type": "map", "sector": "Sector 4", "leak_id": None})
    elif entry["id"] in ("R1-S2", "U-5.2"):
        widgets.append({"type": "chart", "metric": "reservoir", "asset": "RES-01", "hours": 24})
        if result.get("schedule_validation"):
            widgets.append(
                {
                    "type": "schedule_validation",
                    "plan": result.get("proposed_schedule", ""),
                    "validation": result["schedule_validation"],
                }
            )

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
    if tool_name == "show_map":
        return f"Displayed map (sector={result.get('sector') or 'All'}, leaks={result.get('matched_leaks', 0)})"
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
    sig = (widget.get("type"), widget.get("metric"), widget.get("asset"), widget.get("sector"), widget.get("leak_id"), widget.get("what"))
    for w in widgets:
        w_sig = (w.get("type"), w.get("metric"), w.get("asset"), w.get("sector"), w.get("leak_id"), w.get("what"))
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

    # 2. Telemetry loss notice
    if is_loss and not any(k in lower for k in ("audit", "why", "log", "map", "table")):
        return (
            f"**{(tel.get('telemetry_freshness') or {}).get('banner', 'TELEMETRY LOSS')}** (`TELEMETRY_LOSS`). "
            f"Planning and new proposals are paused while holding the last safe state (RES-01 {res_pct:.1f}%, DESAL-01 {plant_mld:.0f} MLD). "
            f"Use **Restore feed** in the sidebar to re-validate readings."
        )

    # 3. Map display requests ("Show Sector 4 on the map", "Show map", "Where are the leaks?")
    if "map" in lower or ("where" in lower and "leak" in lower):
        m_sec = re.search(r"sector\s*([1-6])", text, re.IGNORECASE)
        sec_arg = f"Sector {m_sec.group(1)}" if m_sec else ""
        m_lid = re.search(r"(LEAK-\d+)", text, re.IGNORECASE)
        lid_arg = m_lid.group(1).upper() if m_lid else ""
        mp = tool_executor("show_map", {"sector": sec_arg, "leak_id": lid_arg})
        scope = sec_arg or lid_arg or "Dubai network"
        return f"Here is the live pipe & leak map for **{scope}** ({mp.get('matched_leaks', 0)} tracked leak(s), {len(quarantined)} quarantined sensor(s))."

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

    # 10. Pending approvals question
    if "pending" in lower or "approval" in lower or "queue" in lower:
        ap = tool_executor("show_pending_approvals", {})
        return f"There are currently **{ap.get('pending_count', 0)}** pending action(s) in the approval queue."

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
) -> dict:
    """
    Process one operator chat message with injection filtering, multi-turn history, live read & display tools,
    and stage-gated write proposals.
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
        return run_scenario_for_chat(sc_id, current_hour=current_hour, db_path=db_path)

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

        elif tool_name == "show_map":
            sector = str(args.get("sector") or "").strip()
            leak_id = str(args.get("leak_id") or "").strip()
            _append_unique_widget(widgets, {"type": "map", "sector": sector or None, "leak_id": leak_id or None})
            snap = _read_valves_and_pending_snapshot(db_path=db_path)
            leaks_all = snap.get("tracked_leaks", [])
            matched = [
                l for l in leaks_all
                if (not sector or l["sector"].lower() == sector.lower())
                and (not leak_id or l["leak_id"].upper() == leak_id.upper())
            ]
            out = {"displayed": True, "sector": sector or "All", "leak_id": leak_id or None, "matched_leaks": len(matched), "leaks": matched[:5]}

        elif tool_name == "show_chart":
            metric = str(args.get("metric") or "reservoir").lower()
            if metric not in ("pressure", "flow", "reservoir", "tariff"):
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

    # Build multi-turn message list for the LLM (preserving prior user/assistant exchanges!)
    llm_messages: list[dict] = [{"role": "system", "content": ASK_AQUAGRID_SYSTEM_PROMPT}]
    for msg in prior[-12:]:
        role = msg.get("role")
        content = str(msg.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            llm_messages.append({"role": role, "content": content})
    llm_messages.append({"role": "user", "content": f"[Active SCADA Hour: {current_hour:02d}:00]\n{cleaned}"})

    try:
        answer_text, provider_used, _ = llm_client.chat_with_tools(
            messages=llm_messages,
            tools=CHAT_TOOLS_SCHEMA,
            tool_executor=_execute_tool,
            max_tool_steps=MAX_CHAT_TOOL_STEPS,
        )
        if not answer_text:
            answer_text = _offline_chat_fallback(
                user_message=cleaned,
                history=prior,
                tool_executor=_execute_tool,
                current_hour=current_hour,
            )
            provider_used = "Deterministic Offline Engine"
    except Exception as exc:
        _audit("CHAT_ERROR", "ASK_AQUAGRID", "ERROR", f"{type(exc).__name__}: {exc}", actor="CODE_GUARD", db_path=db_path)
        return {
            "role": "assistant",
            "kind": "error",
            "content": f"Sorry, the AI service encountered a temporary error ({type(exc).__name__}). The SCADA guards remain active — please try asking again.",
            "provider": "Error Handler",
            "tools_called": tools_called_summary,
            "code_checks": code_checks,
            "widgets": widgets,
            "staged_action": staged_action_holder[-1] if staged_action_holder else None,
            "error": True,
        }

    final_staged = staged_action_holder[-1] if staged_action_holder else None
    ans_status = str(final_staged.get("status", "ALLOWED")) if final_staged else "ALLOWED"
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

    # 1. Telemetry Loss
    if telemetry_data.get("telemetry_loss"):
        tf = telemetry_data.get("telemetry_freshness") or {}
        key = f"TELEMETRY_LOSS:{tf.get('last_timestamp_local', 'stale')}"
        if key not in updated_seen:
            updated_seen.add(key)
            new_msgs.append(
                {
                    "role": "assistant",
                    "kind": "alert",
                    "alert_code": "TELEMETRY_LOSS",
                    "content": f"**ALERT (`TELEMETRY_LOSS`):** {tf.get('banner', 'SCADA telemetry silent > 180s. Holding last safe state.')}",
                    "provider": "CODE_GUARD (Auto Alert)",
                    "tools_called": [],
                    "code_checks": [{"check": "Telemetry Freshness", "status": "BLOCKED", "code": "TELEMETRY_LOSS", "detail": tf.get("banner", "")}],
                    "widgets": [{"type": "table", "what": "sensors"}],
                    "staged_action": None,
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
                    "content": f"**ALERT (`NON_COMPLIANT`):** {wq_alert}",
                    "provider": "CODE_GUARD (Auto Alert)",
                    "tools_called": [],
                    "code_checks": [{"check": "Sector Water Quality", "status": "BLOCKED", "code": "NON_COMPLIANT", "detail": wq_alert}],
                    "widgets": [{"type": "map", "sector": "Sector 2", "leak_id": None}],
                    "staged_action": None,
                }
            )

    # 3. Confirmed Pipe Leaks
    for lk in telemetry_data.get("leak_alerts", []):
        key = f"LEAK:{lk.get('junction_id')}:{lk.get('sector')}"
        if key not in updated_seen:
            updated_seen.add(key)
            new_msgs.append(
                {
                    "role": "assistant",
                    "kind": "alert",
                    "alert_code": "CONFIRMED_PIPE_LEAK",
                    "content": f"**ALERT (`CONFIRMED_PIPE_LEAK`):** {lk.get('junction_id')} ({lk.get('sector')}) — {lk.get('diagnostic', '')}",
                    "provider": "CODE_GUARD (Auto Alert)",
                    "tools_called": [],
                    "code_checks": [{"check": "Hydraulic Mass Balance", "status": "BLOCKED", "code": "CONFIRMED_PIPE_LEAK", "detail": lk.get("diagnostic", "")}],
                    "widgets": [
                        {"type": "map", "sector": lk.get("sector"), "leak_id": None},
                        {"type": "chart", "metric": "pressure", "asset": lk.get("junction_id", "J-14"), "hours": 2},
                    ],
                    "staged_action": None,
                }
            )

    # 4. Reservoir Floor Breach or Fire Emergency Floor Raised
    res = feeds_data.get("reservoir") or {}
    res_pct = float(res.get("current_level_percent", 48.0))
    eff_floor = float(res.get("safety_floor_percent", RESERVOIR_FLOOR_PERCENT * 100.0))
    if res_pct < eff_floor:
        key = f"FLOOR_BREACH:{round(res_pct, 1)}:{round(eff_floor, 1)}"
        if key not in updated_seen:
            updated_seen.add(key)
            new_msgs.append(
                {
                    "role": "assistant",
                    "kind": "alert",
                    "alert_code": "REJECTED_BY_SAFETY_GUARD",
                    "content": f"**ALERT (RESERVOIR FLOOR BREACH):** RES-01 dropped to **{res_pct:.1f}%** (below mandatory **{eff_floor:.1f}%** floor).",
                    "provider": "CODE_GUARD (Auto Alert)",
                    "tools_called": [],
                    "code_checks": [{"check": "Reservoir Safety Floor", "status": "BLOCKED", "code": "REJECTED_BY_SAFETY_GUARD", "detail": f"{res_pct:.1f}% < {eff_floor:.1f}%"}],
                    "widgets": [{"type": "chart", "metric": "reservoir", "asset": "RES-01", "hours": 24}],
                    "staged_action": None,
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
                    "content": f"**ALERT (EMERGENCY RESERVE):** {banner}",
                    "provider": "CODE_GUARD (Auto Alert)",
                    "tools_called": [],
                    "code_checks": [{"check": "Dynamic Emergency Floor", "status": "PASS", "code": "VALID", "detail": banner}],
                    "widgets": [{"type": "chart", "metric": "reservoir", "asset": "RES-01", "hours": 24}],
                    "staged_action": None,
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
                new_msgs.append(
                    {
                        "role": "assistant",
                        "kind": "alert",
                        "alert_code": "LIVE_ALERT",
                        "content": (
                            f"**ALERT (Live Batch #{tick_id} · {clk}):** " + " | ".join(raised[:3])
                        ),
                        "provider": "Live Simulator + CODE_GUARD",
                        "tools_called": [],
                        "code_checks": [{"check": "Live SCADA Batch Guard", "status": "BLOCKED", "code": "ALERT ACTIVE", "detail": "; ".join(raised[:3])}],
                        "widgets": [{"type": "map", "sector": None, "leak_id": live_tick_summary.get("new_leak_id")}],
                        "staged_action": None,
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
