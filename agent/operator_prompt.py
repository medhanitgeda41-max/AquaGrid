"""Operator AI Prompt runner: step-by-step suggest-only analysis through all CODE guards.

HARD ARCHITECTURAL INVARIANTS:
- Read-only telemetry & feed tools only.
- All proposals pass through pure-Python safety guards and the stage-gated write gateway.
- NEVER writes directly to valves or plant_setpoints, NEVER auto-approves, NEVER issues tokens.
"""
import json
import re
from typing import Optional

from agent.llm_factory import llm_client
from agent.prompts import SYSTEM_PROMPT
from config.constants import DEFAULT_SCADA_HOUR
from database.connection import transaction_scope
from guards.injection_filter import detect_injection, inspect_untrusted_text
from guards.schedule_validator import validate_production_schedule
from tools.read_feeds import read_tariff_and_quality_feeds
from tools.read_telemetry import read_network_telemetry
from tools.write_gateway import stage_gated_valve_or_setpoint_action


OPERATOR_PROMPT_SYSTEM = (
    SYSTEM_PROMPT
    + "\n\nADDITIONAL OPERATOR REQUEST RULE:\n"
    + "- You are answering a direct operator prompt alongside live SCADA telemetry and feeds.\n"
    + "- You can ONLY suggest actions via action_proposal; every physical action still requires operator approval and passes deterministic Python safety guards.\n"
    + "- If the operator only asks a question or requests a status check, set action_proposal.action_type to 'NONE' and target_id to 'NONE'.\n"
    + "- If the operator requests a specific valve, plant setpoint, or sector flush, formulate the matching action_proposal so the Python safety gateway can validate and stage (or reject) it.\n"
)

_BYPASS_APPROVAL_RE = re.compile(
    r"(?:pre[\s_-]*approved|without\s+(?:human\s+|operator\s+)?approval|skip\s+approval|"
    r"bypass\s+(?:approval|gateway|hitl)|auto[\s_-]*approve|force\s+execute|direct\s+write)",
    re.IGNORECASE,
)


def _deterministic_operator_prompt_response(prompt_text: str, user_context: dict) -> dict:
    """Deterministic fallback parser/analyzer for operator requests when external LLMs are offline."""
    text = (prompt_text or "").strip()
    lower = text.lower()

    res = user_context.get("reservoir_status", {})
    res_pct = float(res.get("current_level_percent", 48.0))
    eff_floor = float(res.get("safety_floor_percent", 35.0))
    tariff_data = user_context.get("tariff_data", {})
    plant_mld = float(tariff_data.get("current_plant_output_mld", 100.0))
    wq = tariff_data.get("water_quality", {})
    max_allowable = float(wq.get("max_allowable_output_mld", 120.0))
    leaks = user_context.get("leak_alerts", [])
    quarantined = user_context.get("quarantined_sensors", [])
    excluded_sectors = user_context.get("excluded_sectors_from_tariff_scheduling", [])

    # 1. Explicit valve command: close / open V-XX
    m_valve = re.search(r"\b(close|shut|isolate|open)\b[^\n]{0,40}?\b(v-\d{1,2})\b", text, re.IGNORECASE)
    if not m_valve:
        m_valve_rev = re.search(r"\b(v-\d{1,2})\b[^\n]{0,40}?\b(close|shut|isolate|open)\b", text, re.IGNORECASE)
        if m_valve_rev:
            verb = m_valve_rev.group(2).lower()
            valve_id = m_valve_rev.group(1).upper()
        else:
            verb, valve_id = None, None
    else:
        verb = m_valve.group(1).lower()
        valve_id = m_valve.group(2).upper()

    if verb and valve_id:
        # Normalize single-digit valve like V-5 -> V-05
        parts = valve_id.split("-")
        if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 1:
            valve_id = f"V-0{parts[1]}"
        is_open = verb == "open"
        action_type = "OPEN_VALVE" if is_open else "CLOSE_VALVE"
        val = 1.0 if is_open else 0.0
        verb_label = "open" if is_open else "close"
        return {
            "operator_brief": (
                f"Prepared a gated proposal to {verb_label} valve {valve_id} per operator request. "
                f"Reservoir is at {res_pct:.1f}% (floor {eff_floor:.1f}%); awaiting operator approval before actuation."
            ),
            "maintenance_advisory": f"After operator approval, {verb_label} {valve_id} and verify downstream sector pressures.",
            "proposed_schedule": f"00:00-24:00 at {min(plant_mld, max_allowable):.0f} MLD (hold safe baseline during valve operation).",
            "action_proposal": {
                "action_type": action_type,
                "target_id": valve_id,
                "proposed_value": val,
                "rationale": f"Operator prompt requested: '{text[:140]}' (requires human approval).",
            },
        }

    # 2. Explicit sector isolate & flush command
    m_flush = re.search(r"\b(?:flush|isolate\s+and\s+flush|chlorine)\b[^\n]{0,50}?\bsector[\s-]*0?([1-6])\b", text, re.IGNORECASE)
    if not m_flush:
        m_flush = re.search(r"\bsector[\s-]*0?([1-6])\b[^\n]{0,50}?\b(?:flush|isolate)\b", text, re.IGNORECASE)
    if m_flush:
        sec_num = int(m_flush.group(1))
        target_sec = f"SECTOR-0{sec_num}"
        return {
            "operator_brief": (
                f"Prepared ISOLATE_AND_FLUSH proposal for Sector {sec_num} ({target_sec}) per operator request. "
                f"Water quality actions outrank tariff optimization and require operator approval."
            ),
            "maintenance_advisory": f"Dispatch water quality crew to Sector {sec_num}; after operator approval, execute ISOLATE_AND_FLUSH on {target_sec}.",
            "proposed_schedule": f"00:00-24:00 at {min(plant_mld, max_allowable):.0f} MLD (Sector {sec_num} excluded from tariff cuts).",
            "action_proposal": {
                "action_type": "ISOLATE_AND_FLUSH",
                "target_id": target_sec,
                "proposed_value": 0.0,
                "rationale": f"Operator requested isolate and flush for Sector {sec_num} ({target_sec}).",
            },
        }

    # 3. Explicit plant output / throttle / MLD command
    m_mld = re.search(r"(\d+(?:\.\d+)?)\s*mld\b", text, re.IGNORECASE)
    if m_mld and any(k in lower for k in ("set", "throttle", "cut", "reduce", "increase", "plant", "desal", "output", "run", "change")):
        target_mld = float(m_mld.group(1))
        return {
            "operator_brief": (
                f"Submitted DESAL-01 setpoint proposal to {target_mld:.1f} MLD for deterministic safety-floor "
                f"({eff_floor:.1f}%) and water-quality ({max_allowable:.0f} MLD cap) verification."
            ),
            "maintenance_advisory": "Monitor RES-01 storage level and intake turbidity during any desalination ramp.",
            "proposed_schedule": (
                f"00:00-06:00 at {min(115.0, max_allowable):.0f} MLD, "
                f"06:00-14:00 at {min(95.0, max_allowable):.0f} MLD, "
                f"14:00-18:00 at {target_mld:.0f} MLD, "
                f"18:00-24:00 at {min(95.0, max_allowable):.0f} MLD."
            ),
            "action_proposal": {
                "action_type": "SET_PLANT_OUTPUT",
                "target_id": "DESAL-01",
                "proposed_value": target_mld,
                "rationale": f"Operator prompt requested DESAL-01 output at {target_mld:.1f} MLD (subject to {eff_floor:.1f}% floor).",
            },
        }

    # 4. Optimize / tariff / schedule request
    if any(k in lower for k in ("optimiz", "optimis", "tariff", "schedule", "save energy", "peak")):
        if excluded_sectors:
            sec_name = excluded_sectors[0]
            sec_digits = "".join(ch for ch in sec_name if ch.isdigit()) or "2"
            target_sec = f"SECTOR-0{int(sec_digits)}"
            return {
                "operator_brief": (
                    f"Water quality outranks tariff savings: {', '.join(excluded_sectors)} is NON_COMPLIANT "
                    f"and excluded from tariff cuts. Proposed ISOLATE_AND_FLUSH on {target_sec}."
                ),
                "maintenance_advisory": f"After operator approval, isolate and flush {target_sec} and verify chlorine >= 0.20 mg/L.",
                "proposed_schedule": f"00:00-24:00 at {min(100.0, max_allowable):.0f} MLD ({', '.join(excluded_sectors)} excluded).",
                "action_proposal": {
                    "action_type": "ISOLATE_AND_FLUSH",
                    "target_id": target_sec,
                    "proposed_value": 0.0,
                    "rationale": f"Water quality non-compliance in {sec_name} outranks peak-tariff throttling.",
                },
            }
        safe_peak_mld = min(65.0 if res.get("has_active_emergency") else 60.0, max_allowable)
        return {
            "operator_brief": (
                f"Suggested peak-tariff adjustment to {safe_peak_mld:.0f} MLD (current {plant_mld:.0f} MLD, "
                f"reservoir {res_pct:.1f}%, floor {eff_floor:.1f}%). Submitted for code safety checks and operator approval."
            ),
            "maintenance_advisory": "Confirm desalination ramp stays within intake water-quality cap.",
            "proposed_schedule": (
                f"00:00-06:00 at {min(115.0, max_allowable):.0f} MLD, "
                f"06:00-14:00 at {min(95.0, max_allowable):.0f} MLD, "
                f"14:00-18:00 at {safe_peak_mld:.0f} MLD, "
                f"18:00-24:00 at {min(95.0, max_allowable):.0f} MLD."
            ),
            "action_proposal": {
                "action_type": "SET_PLANT_OUTPUT",
                "target_id": "DESAL-01",
                "proposed_value": safe_peak_mld,
                "rationale": f"Tariff-aware optimization holding reservoir above {eff_floor:.1f}% safety floor.",
            },
        }

    # 5. Leak / isolate request without explicit valve ID
    if any(k in lower for k in ("leak", "isolate")) and leaks:
        lk = leaks[0]
        return {
            "operator_brief": (
                f"Active leak confirmed at {lk.get('junction_id')} ({lk.get('sector')}). "
                f"Proposed closing V-12 for operator approval; {len(quarantined)} sensor(s) quarantined."
            ),
            "maintenance_advisory": "After operator approval, close V-12 and dispatch repair crew to Sector 4.",
            "proposed_schedule": f"00:00-24:00 at {min(plant_mld, max_allowable):.0f} MLD.",
            "action_proposal": {
                "action_type": "CLOSE_VALVE",
                "target_id": "V-12",
                "proposed_value": 0.0,
                "rationale": f"Isolate confirmed leak at {lk.get('junction_id')} ({lk.get('sector')}).",
            },
        }

    # 6. General status / question — advisory only (no physical actuation)
    leak_txt = f"{len(leaks)} confirmed leak(s)" if leaks else "no active hydraulic leaks"
    quar_txt = f"{len(quarantined)} quarantined sensor(s)" if quarantined else "all sensors healthy"
    wq_txt = f"NON_COMPLIANT sectors: {', '.join(excluded_sectors)}" if excluded_sectors else "all sectors chlorine compliant"
    return {
        "operator_brief": (
            f"Grid status at {user_context.get('current_hour', 14):02d}:00: Reservoir RES-01 at {res_pct:.1f}% "
            f"(floor {eff_floor:.1f}%), DESAL-01 at {plant_mld:.1f} MLD, {leak_txt}, {quar_txt}, {wq_txt}."
        ),
        "maintenance_advisory": "No immediate actuator change requested; continue normal SCADA monitoring.",
        "proposed_schedule": f"00:00-24:00 at {min(plant_mld, max_allowable):.0f} MLD (current baseline).",
        "action_proposal": {
            "action_type": "NONE",
            "target_id": "NONE",
            "proposed_value": 0.0,
            "rationale": "Advisory response only — no physical action proposed.",
        },
    }


def _log_operator_prompt_audit(
    target_id: str,
    status: str,
    details: str,
    actor: str = "LLM_AGENT",
    db_path: Optional[str] = None,
) -> None:
    with transaction_scope(db_path) as conn:
        conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            ("OPERATOR_AI_PROMPT", target_id, details[:500], actor, status),
        )


def run_operator_prompt(
    prompt_text: str,
    current_hour: int = DEFAULT_SCADA_HOUR,
    db_path: Optional[str] = None,
) -> dict:
    """
    Execute an operator prompt through read tools -> pure-Python guards -> LLM suggestion -> write gateway.

    Returns a step-by-step dict with:
      - prompt: original operator prompt text
      - step_read: summary of telemetry & feeds read
      - step_checks: list of deterministic code guard checks (name, status, detail)
      - step_proposal: what the AI proposed and what the write gateway returned
      - final_result: {status, severity ('ok'|'pending'|'danger'), headline, summary}
      - operator_brief, maintenance_advisory, proposed_schedule, schedule_validation, gateway_response, llm_provider
    """
    cleaned_prompt = (prompt_text or "").strip()
    if not cleaned_prompt:
        return {
            "prompt": "",
            "step_read": {},
            "step_checks": [],
            "step_proposal": {"action": None, "gateway_response": None},
            "final_result": {
                "status": "EMPTY_PROMPT",
                "severity": "pending",
                "headline": "Enter a request first",
                "summary": "Type a question or action request in the prompt box and press Run.",
            },
            "operator_brief": "",
            "maintenance_advisory": "",
            "proposed_schedule": "",
            "schedule_validation": None,
            "gateway_response": None,
            "llm_provider": "None",
        }

    # ── STEP 1: Read live SCADA telemetry & feeds (read-only tools) ───────────
    telemetry_data = read_network_telemetry(current_hour=current_hour, db_path=db_path, log_audit=True)
    feeds_data = read_tariff_and_quality_feeds(horizon_hours=6, current_hour=current_hour, db_path=db_path, log_audit=True)

    res = feeds_data.get("reservoir", {})
    res_pct = float(res.get("current_level_percent", 48.0))
    eff_floor = float(res.get("safety_floor_percent", 35.0))
    plant_mld = float(feeds_data.get("current_plant_output_mld", 100.0))
    tariffs = feeds_data.get("tariffs", [])
    curr_tariff = float(tariffs[0]["rate_per_kwh"]) if tariffs else 0.48
    is_peak = bool(feeds_data.get("is_current_peak_tariff", False))
    leaks = telemetry_data.get("leak_alerts", [])
    quarantined = telemetry_data.get("quarantined_sensors", [])
    sanitized_alerts = feeds_data.get("sanitized_alerts", [])
    field_tickets = feeds_data.get("field_tickets", [])
    wq_comp = feeds_data.get("sector_water_quality_compliance", {})
    excluded_sectors = feeds_data.get("excluded_sectors", [])
    wq = feeds_data.get("water_quality", {})
    freshness = telemetry_data.get("telemetry_freshness") or {}
    is_telemetry_loss = bool(telemetry_data.get("telemetry_loss", False))

    step_read = {
        "hour": current_hour,
        "reservoir_pct": res_pct,
        "safety_floor_pct": eff_floor,
        "plant_mld": plant_mld,
        "tariff_rate": curr_tariff,
        "is_peak": is_peak,
        "verified_sensors": len(telemetry_data.get("verified_telemetry", [])),
        "quarantined_sensors": len(quarantined),
        "active_leaks": len(leaks),
        "excluded_sectors": list(excluded_sectors),
        "alerts_count": len(sanitized_alerts),
        "tickets_count": len(field_tickets),
        "telemetry_status": freshness.get("status", "FRESH"),
    }

    # ── STEP 2: Pure-Python Safety & Security Checks ──────────────────────────
    step_checks: list[dict] = []

    # 2a. Prompt injection check on typed prompt + incoming alerts
    prompt_is_inj, prompt_patterns = detect_injection(cleaned_prompt)
    prompt_inspect = inspect_untrusted_text(cleaned_prompt)
    bypass_claim = bool(_BYPASS_APPROVAL_RE.search(cleaned_prompt)) or bool(prompt_inspect.get("approval_claim_detected"))
    alert_injections = [a for a in sanitized_alerts if a.get("injection_detected")]

    if prompt_is_inj:
        step_checks.append({
            "check": "Prompt Security Filter",
            "status": "BLOCKED",
            "detail": f"Blocked injection pattern(s) in prompt: {', '.join(prompt_patterns)}",
        })
    elif alert_injections:
        step_checks.append({
            "check": "Prompt Security Filter",
            "status": "BLOCKED",
            "detail": f"{len(alert_injections)} quarantined injection alert(s) active in SCADA feed.",
        })
    else:
        step_checks.append({
            "check": "Prompt Security Filter",
            "status": "ALLOWED",
            "detail": "No prompt injection syntax detected.",
        })

    # 2b. Telemetry Freshness check
    if is_telemetry_loss:
        step_checks.append({
            "check": "Telemetry Freshness",
            "status": "TELEMETRY_LOSS",
            "detail": freshness.get("banner") or "Telemetry stale (> 180s). Holding last safe state.",
        })
    else:
        step_checks.append({
            "check": "Telemetry Freshness",
            "status": "FRESH",
            "detail": f"Latest readings within freshness limit ({freshness.get('age_seconds', 0)}s old).",
        })

    # 2c. Sector Water Quality check
    if excluded_sectors:
        step_checks.append({
            "check": "Water Quality (Chlorine >= 0.20 mg/L)",
            "status": "NON_COMPLIANT",
            "detail": "; ".join(wq_comp.get("alerts", [])) or f"Excluded sectors: {', '.join(excluded_sectors)}",
        })
    else:
        step_checks.append({
            "check": "Water Quality (Chlorine >= 0.20 mg/L)",
            "status": "COMPLIANT",
            "detail": "All 6 sectors compliant; intake cap "
            f"{float(wq.get('max_allowable_output_mld', 120.0)):.0f} MLD.",
        })

    # 2d. Reservoir Floor check
    floor_ok = res_pct >= eff_floor
    step_checks.append({
        "check": "Reservoir Safety Floor",
        "status": "VALID" if floor_ok else "REJECTED_BY_SAFETY_GUARD",
        "detail": (
            f"RES-01 at {res_pct:.1f}% vs {eff_floor:.1f}% floor"
            + (" (emergency reserve active)" if res.get("has_active_emergency") else "")
            + "."
        ),
    })

    # 2e. Halt immediately if prompt injection detected
    if prompt_is_inj:
        _log_operator_prompt_audit(
            target_id="AI_PROMPT",
            status="BLOCKED",
            details=f"Injection detected in operator prompt: {prompt_patterns}",
            actor="CODE_GUARD",
            db_path=db_path,
        )
        return {
            "prompt": cleaned_prompt,
            "step_read": step_read,
            "step_checks": step_checks,
            "step_proposal": {"action": None, "gateway_response": None},
            "final_result": {
                "status": "BLOCKED",
                "severity": "danger",
                "headline": "Request blocked by security guard",
                "summary": "Prompt injection syntax was detected and blocked before reaching the AI. No actuators were changed.",
            },
            "operator_brief": "Security filter blocked prompt injection in the submitted request. No action staged.",
            "maintenance_advisory": "None.",
            "proposed_schedule": "",
            "schedule_validation": None,
            "gateway_response": None,
            "llm_provider": "CODE_GUARD (Injection Filter)",
        }

    # 2f. Halt immediately if TELEMETRY_LOSS
    if is_telemetry_loss:
        _log_operator_prompt_audit(
            target_id="SCADA_FEED",
            status="TELEMETRY_LOSS",
            details=f"Operator prompt halted during TELEMETRY_LOSS: '{cleaned_prompt[:120]}'",
            actor="CODE_GUARD",
            db_path=db_path,
        )
        return {
            "prompt": cleaned_prompt,
            "step_read": step_read,
            "step_checks": step_checks,
            "step_proposal": {"action": None, "gateway_response": None},
            "final_result": {
                "status": "TELEMETRY_LOSS",
                "severity": "danger",
                "headline": "Paused — telemetry loss active",
                "summary": (
                    f"{freshness.get('banner', 'TELEMETRY LOSS')} "
                    "AI planning and new proposals are blocked until telemetry is restored."
                ),
            },
            "operator_brief": f"{freshness.get('banner', 'TELEMETRY LOSS')} Holding last safe state.",
            "maintenance_advisory": "Restore SCADA telemetry feed and verify readings before resuming AI proposals.",
            "proposed_schedule": "PAUSED — TELEMETRY_LOSS",
            "schedule_validation": None,
            "gateway_response": None,
            "llm_provider": "CODE_GUARD (TELEMETRY_LOSS)",
        }

    # ── STEP 3: Ask AI to analyze & suggest (never direct write) ──────────────
    user_context = {
        "operator_request": cleaned_prompt,
        "current_hour": current_hour,
        "reservoir_status": res,
        "leak_alerts": leaks,
        "quarantined_sensors": quarantined,
        "sanitized_alerts": sanitized_alerts,
        "sector_water_quality_compliance": wq_comp,
        "excluded_sectors_from_tariff_scheduling": excluded_sectors,
        "tariff_data": {
            "current_hour": current_hour,
            "is_current_peak_tariff": is_peak,
            "tariffs": tariffs,
            "demand_forecast": feeds_data.get("demand_forecast", []),
            "current_plant_output_mld": plant_mld,
            "water_quality": wq,
        },
    }
    user_prompt_json = (
        f"Operator Request: {cleaned_prompt}\n"
        f"Live SCADA Context:\n{json.dumps(user_context, separators=(',', ':'))}"
    )

    analysis, provider_used = llm_client.query(OPERATOR_PROMPT_SYSTEM, user_prompt_json)
    if provider_used == "Deterministic Offline Engine":
        analysis = _deterministic_operator_prompt_response(cleaned_prompt, user_context)

    operator_brief = str(analysis.get("operator_brief", "Analysis complete."))
    maintenance_advisory = str(analysis.get("maintenance_advisory", ""))
    proposed_schedule = str(analysis.get("proposed_schedule", ""))
    action_prop = analysis.get("action_proposal") or {}
    if not isinstance(action_prop, dict) or str(action_prop.get("action_type", "NONE")).upper() == "NONE":
        action_prop = None

    # Validate schedule if one contains time blocks
    schedule_validation = None
    if proposed_schedule and re.search(r"\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}", proposed_schedule):
        demand_list = [
            float(d["demand_mld"])
            for d in feeds_data.get("demand_forecast", [])
            if "demand_mld" in d
        ]
        schedule_validation = validate_production_schedule(
            proposed_schedule=proposed_schedule,
            current_volume_ml=float(res.get("current_volume_ml", 240.0)),
            demand_by_hour=feeds_data.get("hourly_demand_24h", {}),
            fallback_demand_mld=demand_list or None,
            bloom_severity=str(wq.get("bloom_severity", "NONE")),
            turbidity_ntu=float(wq.get("intake_turbidity_ntu", 1.2)),
            chlorine_ppm=float(wq.get("intake_chlorine_ppm", 1.8)),
            min_floor_percent=eff_floor / 100.0,
            current_hour=current_hour,
            current_plant_output_mld=plant_mld,
            excluded_sectors=excluded_sectors,
        )
        step_checks.append({
            "check": "24h Schedule Validation",
            "status": schedule_validation.get("status", "UNKNOWN"),
            "detail": schedule_validation.get("summary", ""),
        })

    # Submit any proposed action strictly through the stage-gated write gateway
    gateway_response = None
    if action_prop:
        raw_val = action_prop.get("proposed_value", 0.0)
        try:
            proposed_val = float(raw_val) if raw_val is not None else 0.0
        except (TypeError, ValueError):
            proposed_val = 0.0

        gw_kwargs = {}
        if bypass_claim:
            # Let the gateway explicitly reject any approval-bypass claim with REJECTED_UNAUTHORIZED_APPROVAL_CLAIM
            gw_kwargs["pre_approved"] = True

        gateway_response = stage_gated_valve_or_setpoint_action(
            action_type=str(action_prop.get("action_type") or ""),
            target_id=str(action_prop.get("target_id") or ""),
            proposed_value=proposed_val,
            rationale=str(action_prop.get("rationale") or f"Operator prompt: {cleaned_prompt[:140]}"),
            db_path=db_path,
            provenance="AGENT_ANALYSIS",
            provenance_ref="OPERATOR_PROMPT",
            **gw_kwargs,
        )
        step_checks.append({
            "check": "Write Gateway & Safety Guards",
            "status": gateway_response.get("status", "UNKNOWN"),
            "detail": gateway_response.get("reason") or gateway_response.get("message") or "",
        })
    elif bypass_claim:
        # Even if no target was parsed, refuse any explicit approval-bypass request cleanly
        gateway_response = stage_gated_valve_or_setpoint_action(
            action_type="CLOSE_VALVE",
            target_id="V-12",
            proposed_value=0.0,
            rationale=f"Refused approval-bypass claim in prompt: {cleaned_prompt[:120]}",
            db_path=db_path,
            pre_approved=True,
        )
        step_checks.append({
            "check": "Write Gateway & Safety Guards",
            "status": gateway_response.get("status", "REJECTED_UNAUTHORIZED_APPROVAL_CLAIM"),
            "detail": gateway_response.get("reason", ""),
        })

    # Build final result summary
    if gateway_response:
        gw_status = str(gateway_response.get("status", "UNKNOWN"))
        if gw_status == "STAGED_PENDING_APPROVAL":
            final_result = {
                "status": gw_status,
                "severity": "pending",
                "headline": (
                    f"Action #{gateway_response.get('action_id')} staged — waiting for your approval"
                ),
                "summary": (
                    f"{gateway_response.get('action_type')} {gateway_response.get('target_id')} "
                    f"passed all code checks and is PENDING. Use Approve or Reject above to decide."
                ),
            }
        elif gw_status == "REJECTED_DUPLICATE_PENDING":
            final_result = {
                "status": gw_status,
                "severity": "pending",
                "headline": "Already waiting for approval",
                "summary": gateway_response.get("reason") or "An identical action is already PENDING.",
            }
        else:
            final_result = {
                "status": gw_status,
                "severity": "danger",
                "headline": f"Proposal blocked ({gw_status})",
                "summary": gateway_response.get("reason") or "Blocked by deterministic code safety guard.",
            }
        _log_operator_prompt_audit(
            target_id=str(gateway_response.get("target_id") or "AI_PROMPT"),
            status=gw_status,
            details=f"Prompt: '{cleaned_prompt[:120]}' -> {gw_status}",
            db_path=db_path,
        )
    else:
        final_result = {
            "status": "ADVISORY_ONLY",
            "severity": "ok",
            "headline": "Advisory answer — no actuator changes proposed",
            "summary": operator_brief,
        }
        _log_operator_prompt_audit(
            target_id="AI_PROMPT",
            status="ALLOWED",
            details=f"Advisory prompt answered: '{cleaned_prompt[:120]}'",
            db_path=db_path,
        )

    return {
        "prompt": cleaned_prompt,
        "step_read": step_read,
        "step_checks": step_checks,
        "step_proposal": {
            "action": action_prop,
            "gateway_response": gateway_response,
        },
        "final_result": final_result,
        "operator_brief": operator_brief,
        "maintenance_advisory": maintenance_advisory,
        "proposed_schedule": proposed_schedule,
        "schedule_validation": schedule_validation,
        "gateway_response": gateway_response,
        "llm_provider": provider_used,
    }
