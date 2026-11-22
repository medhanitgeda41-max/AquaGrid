"""Pure-Python valve request validation: rationale provenance, sector-match check,
hydraulic impact assessment, supply-continuity protection, and technical justification check.

Read-only analysis — never mutates valves, plant setpoints, or leaks.
"""
import re
from typing import Optional

from database.connection import get_db_connection
from database.dubai_network import DUBAI_PIPE_SEGMENTS, VALVE_SECTOR_MAP
from database.leak_registry import get_active_leaks
from digital_twin.hydraulics import simulate_valve_change_impact
from guards.water_quality import CHLORINE_MIN_MG_L, target_id_to_sector_name

_SECTOR_MENTION_RE = re.compile(r"\bsector[\s-]*0?([1-6])\b", re.IGNORECASE)

_MAINTENANCE_REASON_RE = re.compile(
    r"\b(?:planned\s+maintenance|scheduled\s+maintenance|maintenance|inspection|repair|"
    r"pressure\s+test|valve\s+test|seal\s+test|flush|flushing|emergency|rupture|burst)\b",
    re.IGNORECASE,
)

_COMMAND_BOILERPLATE_RE = re.compile(
    r"\b(?:please|can\s+you|could\s+you|would\s+you|kindly|go\s+ahead\s+and|"
    r"close|shut|shut\s+off|turn\s+off|isolate|open|turn\s+on|"
    r"the|this|that|valve|v-?0?\d{1,2}|now|asap|immediately|right\s+now|today)\b",
    re.IGNORECASE,
)

_OPERATOR_VALVE_CMD_RE = re.compile(
    r"\b(close|shut|isolate|open)\b[^\n]{0,35}?\b(v-?0?\d{1,2})\b",
    re.IGNORECASE,
)
_OPERATOR_VALVE_CMD_REV_RE = re.compile(
    r"\b(v-?0?\d{1,2})\b[^\n]{0,35}?\b(close|shut|isolate|open)\b",
    re.IGNORECASE,
)


def parse_operator_valve_request(user_message: str) -> Optional[tuple[str, str]]:
    """
    Detect an explicit operator valve command (e.g. 'close V-16', 'open V-12').
    Returns (action_type, normalized_valve_id) or None.
    """
    text = (user_message or "").strip()
    if not text:
        return None
    m = _OPERATOR_VALVE_CMD_RE.search(text)
    if m:
        verb, raw_vid = m.group(1).lower(), m.group(2).upper()
    else:
        m_rev = _OPERATOR_VALVE_CMD_REV_RE.search(text)
        if not m_rev:
            return None
        raw_vid, verb = m_rev.group(1).upper(), m_rev.group(2).lower()
    digits = re.sub(r"\D", "", raw_vid)
    if not digits:
        return None
    vid = f"V-{int(digits):02d}"
    action_type = "OPEN_VALVE" if verb == "open" else "CLOSE_VALVE"
    return action_type, vid


def extract_operator_reason(user_message: str, target_id: str = "") -> str:
    """
    Extract any explicit operational reason the operator gave in their request.
    Returns empty string '' when the operator gave no reason (e.g. 'close V-16' or 'please close valve V-16').
    """
    raw = (user_message or "").strip()
    if not raw:
        return ""
    stripped = raw
    if target_id:
        stripped = re.sub(re.escape(target_id), " ", stripped, flags=re.IGNORECASE)
    stripped = _COMMAND_BOILERPLATE_RE.sub(" ", stripped)
    stripped = re.sub(r"[\s.,;:!?/\\\-–—'\"()\[\]]+", " ", stripped).strip()
    # Ignore trivial 1-3 char leftovers or bare connector words
    if len(stripped) < 4 or stripped.lower() in ("for", "to", "it", "now", "pls", "thx", "thanks"):
        return ""
    return raw


def format_action_rationale(
    source: str,
    operator_message: str = "",
    ai_reason: str = "",
    target_id: str = "",
) -> str:
    """
    Format a truthful, provenance-tagged rationale that NEVER invents operator reasons:
    - Operator asked with no reason -> 'Operator request — no reason given'
    - Operator asked with a reason  -> 'Operator request: <operator's exact words>'
    - AI proposed                   -> 'AI proposal: <data-backed reason>'
    """
    src = str(source or "ai").strip().lower()
    if src == "operator":
        op_reason = extract_operator_reason(operator_message, target_id=target_id)
        if not op_reason:
            return "Operator request — no reason given"
        return f"Operator request: {operator_message.strip()}"
    clean_ai = str(ai_reason or "").strip()
    if not clean_ai:
        return "AI proposal: data-backed operational recommendation"
    if clean_ai.lower().startswith("ai proposal:"):
        return clean_ai
    return f"AI proposal: {clean_ai}"


def validate_rationale_sector_match(
    action_type: str,
    target_id: str,
    rationale: str,
) -> dict:
    """
    Reject any rationale that mentions a different sector than the target valve/sector's real sector.
    Example: V-16 belongs to Sector 5; a rationale mentioning 'Sector 6' is rejected with REJECTED_WRONG_SECTOR.
    """
    a_type = str(action_type or "").strip().upper()
    t_id = str(target_id or "").strip().upper()
    rat = str(rationale or "")

    expected_sector: Optional[str] = None
    if a_type in ("CLOSE_VALVE", "OPEN_VALVE"):
        expected_sector = VALVE_SECTOR_MAP.get(t_id)
    elif a_type == "ISOLATE_AND_FLUSH":
        expected_sector = target_id_to_sector_name(t_id)

    if not expected_sector:
        return {"valid": True, "expected_sector": None, "mentioned_sectors": []}

    mentioned = sorted({f"Sector {int(m)}" for m in _SECTOR_MENTION_RE.findall(rat)})
    wrong = [sec for sec in mentioned if sec.lower() != expected_sector.lower()]
    if wrong:
        return {
            "valid": False,
            "status": "REJECTED_WRONG_SECTOR",
            "expected_sector": expected_sector,
            "mentioned_sectors": mentioned,
            "wrong_sectors": wrong,
            "reason": (
                f"REJECTED_WRONG_SECTOR: Rationale for {t_id} mentions {', '.join(wrong)}, "
                f"but {t_id} is in {expected_sector}."
            ),
        }
    return {
        "valid": True,
        "expected_sector": expected_sector,
        "mentioned_sectors": mentioned,
        "wrong_sectors": [],
    }


def assess_valve_change_impact(
    valve_id: str,
    action_type: str = "CLOSE_VALVE",
    db_path: Optional[str] = None,
) -> dict:
    """
    Pure-Python hydraulic impact assessment BEFORE staging a valve change.
    Computes:
    - pipes & junctions losing flow, exact before -> after L/s changes
    - affected sector(s), street area, approximate customers losing supply
    - active leaks isolated by closing this valve (from single leaks registry)
    - whether closing this valve leaves ANY sector with zero open supply paths (cutoff_sectors)
    """
    vid = str(valve_id or "").strip().upper()
    a_type = str(action_type or "CLOSE_VALVE").strip().upper()
    new_status = "OPEN" if a_type == "OPEN_VALVE" else "CLOSED"
    valve_sector = VALVE_SECTOR_MAP.get(vid, "Unknown Sector")

    conn = get_db_connection(db_path)
    try:
        seg_rows = conn.execute("SELECT * FROM pipe_segments ORDER BY sector, segment_id;").fetchall()
        segments = [dict(r) for r in seg_rows] if seg_rows else list(DUBAI_PIPE_SEGMENTS)
        v_rows = conn.execute("SELECT valve_id, status FROM valves;").fetchall()
        valve_statuses = {r["valve_id"]: str(r["status"]).upper() for r in v_rows}
        p_row = conn.execute("SELECT production_mld FROM plant_setpoints WHERE plant_id = 'DESAL-01';").fetchone()
        plant_output_mld = float(p_row["production_mld"]) if p_row else 100.0
    finally:
        conn.close()

    active_leaks = get_active_leaks(db_path=db_path)

    hyd_impact = simulate_valve_change_impact(
        valve_id=vid,
        new_status=new_status,
        segments=segments,
        valve_statuses=valve_statuses,
        plant_output_mld=plant_output_mld,
        leaks=active_leaks,
        db_path=db_path,
    )

    seg_by_id = {s["segment_id"]: s for s in segments}
    before_flows = hyd_impact["before_flows_lps"]
    after_flows = hyd_impact["after_flows_lps"]

    affected_pipes: list[dict] = []
    pipes_losing_flow: list[dict] = []
    affected_junctions_set: set[str] = set()
    affected_sectors_set: set[str] = set()
    street_names: list[str] = []
    approx_customers = 0

    for item in hyd_impact["affected_pipes"]:
        sid = item["segment_id"]
        s_meta = seg_by_id.get(sid, {})
        rel_raw = str(s_meta.get("related_sensor_ids", "") or "")
        juncs = sorted(
            {
                tok.strip().replace("P-", "J-").replace("F-", "J-")
                for tok in rel_raw.split(",")
                if tok.strip().startswith(("P-", "F-", "J-"))
            }
        )
        est_cust = int(s_meta.get("est_customers_affected", 350) or 350)
        area_lbl = str(s_meta.get("street_area") or item.get("name") or sid)
        enriched = {
            **item,
            "street_area": area_lbl,
            "est_customers_affected": est_cust,
            "junctions": juncs,
        }
        affected_pipes.append(enriched)
        if item["before_lps"] > 0.05 and item["after_lps"] <= 0.05:
            pipes_losing_flow.append(enriched)
            approx_customers += est_cust
            affected_junctions_set.update(juncs)
            if item.get("sector"):
                affected_sectors_set.add(item["sector"])
            if item.get("name") and item["name"] not in street_names:
                street_names.append(item["name"])

    if not affected_sectors_set and valve_sector != "Unknown Sector":
        affected_sectors_set.add(valve_sector)

    affected_sectors = sorted(affected_sectors_set)
    if street_names:
        area_desc = f"{', '.join(affected_sectors)} ({', '.join(street_names[:2])})"
    else:
        area_desc = ", ".join(affected_sectors) if affected_sectors else valve_sector

    # Identify active leaks isolated by closing this valve
    losing_seg_ids = {p["segment_id"] for p in pipes_losing_flow}
    iso_after_ids = set(hyd_impact.get("isolated_leaks_after") or [])
    iso_before_ids = set(hyd_impact.get("isolated_leaks_before") or [])
    isolated_active_leaks: list[dict] = []
    if new_status == "CLOSED":
        for lk in active_leaks:
            lid = lk["leak_id"]
            if lk.get("is_isolated"):
                continue
            if (
                (lid in iso_after_ids and lid not in iso_before_ids)
                or lk.get("pipe_segment_id") in losing_seg_ids
                or str(lk.get("isolating_valve_id", "")).upper() == vid
            ):
                isolated_active_leaks.append(
                    {
                        "leak_id": lid,
                        "junction_id": lk.get("junction_id", ""),
                        "sector": lk.get("sector", ""),
                        "status": lk.get("status", ""),
                        "loss_rate_m3h": float(lk.get("loss_rate_m3h", 0.0)),
                    }
                )

    # Supply Continuity Check: does ANY sector lose ALL open supply paths?
    cutoff_sectors: list[str] = []
    if new_status == "CLOSED":
        segs_by_sec: dict[str, list[str]] = {}
        for s in segments:
            segs_by_sec.setdefault(s["sector"], []).append(s["segment_id"])
        for sec_name, sec_sids in sorted(segs_by_sec.items()):
            had_supply_before = any(float(before_flows.get(sid, 0.0)) > 0.05 for sid in sec_sids)
            has_supply_after = any(float(after_flows.get(sid, 0.0)) > 0.05 for sid in sec_sids)
            if had_supply_before and not has_supply_after:
                cutoff_sectors.append(sec_name)

    return {
        "valve_id": vid,
        "valve_sector": valve_sector,
        "action_type": a_type,
        "new_status": new_status,
        "affected_pipes": affected_pipes,
        "pipes_losing_flow": pipes_losing_flow,
        "affected_pipe_ids": list(hyd_impact["affected_pipe_ids"]),
        "affected_junctions": sorted(affected_junctions_set),
        "affected_sectors": affected_sectors,
        "affected_area": area_desc,
        "approx_customers_affected": approx_customers,
        "summary_lines": list(hyd_impact["summary_lines"]),
        "isolated_active_leaks": isolated_active_leaks,
        "isolates_active_leak": len(isolated_active_leaks) > 0,
        "cutoff_sectors": cutoff_sectors,
        "cuts_entire_sector": len(cutoff_sectors) > 0,
    }


def evaluate_valve_justification(
    valve_id: str,
    action_type: str,
    rationale: str,
    impact: dict,
    db_path: Optional[str] = None,
) -> dict:
    """
    Pure-Python justification check for CLOSE_VALVE proposals:
    - JUSTIFIED if:
      1) Closing the valve isolates an active leak, OR
      2) The valve's sector has a water-quality NON_COMPLIANT alert (< 0.20 mg/L chlorine), OR
      3) Planned maintenance / inspection / repair is explicitly stated or documented.
    - NOT_JUSTIFIED otherwise -> requires operator typed reason (>= 10 chars) + supply-loss confirmation.
    """
    a_type = str(action_type or "").strip().upper()
    vid = str(valve_id or "").strip().upper()
    rat = str(rationale or "").strip()
    valve_sector = impact.get("valve_sector") or VALVE_SECTOR_MAP.get(vid, "Unknown Sector")

    if a_type != "CLOSE_VALVE":
        return {
            "justification_status": "JUSTIFIED",
            "is_justified": True,
            "requires_override_reason": False,
            "justification_basis": "Restoring valve supply / non-cutoff operation.",
            "warning": "",
        }

    # 1. Active leak isolated by this valve
    iso_leaks = impact.get("isolated_active_leaks") or []
    if iso_leaks:
        leak_desc = ", ".join(f"{l['leak_id']} at {l['junction_id']} ({l['status']})" for l in iso_leaks)
        return {
            "justification_status": "JUSTIFIED",
            "is_justified": True,
            "requires_override_reason": False,
            "justification_basis": f"Isolates active leak(s): {leak_desc}.",
            "warning": "",
        }

    # 2. Water-quality NON_COMPLIANT in this valve's sector
    conn = get_db_connection(db_path)
    try:
        wq_row = conn.execute(
            """
            SELECT chlorine_mg_l FROM sector_water_quality
            WHERE sector = ?
            ORDER BY id DESC LIMIT 1;
            """,
            (valve_sector,),
        ).fetchone()
    finally:
        conn.close()
    if wq_row is not None and float(wq_row["chlorine_mg_l"]) < CHLORINE_MIN_MG_L:
        cl_val = float(wq_row["chlorine_mg_l"])
        return {
            "justification_status": "JUSTIFIED",
            "is_justified": True,
            "requires_override_reason": False,
            "justification_basis": f"{valve_sector} chlorine is NON_COMPLIANT ({cl_val:.2f} mg/L < {CHLORINE_MIN_MG_L:.2f} mg/L).",
            "warning": "",
        }

    # 3. Planned maintenance / inspection / repair explicitly stated (never when 'no reason given')
    if "no reason given" not in rat.lower() and _MAINTENANCE_REASON_RE.search(rat):
        return {
            "justification_status": "JUSTIFIED",
            "is_justified": True,
            "requires_override_reason": False,
            "justification_basis": "Planned maintenance / inspection / repair stated.",
            "warning": "",
        }

    # Otherwise: NOT JUSTIFIED -> stage with explicit warning & mandatory operator override reason
    area = impact.get("affected_area") or valve_sector
    warning = f"NO TECHNICAL JUSTIFICATION — this will cut supply to {area} with no leak or quality issue"
    return {
        "justification_status": "NOT_JUSTIFIED",
        "is_justified": False,
        "requires_override_reason": True,
        "justification_basis": "No active leak, no water-quality issue, and no planned maintenance found.",
        "warning": warning,
    }
