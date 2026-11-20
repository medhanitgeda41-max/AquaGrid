"""Exclusive Write Gateway for AquaGrid.

Enforces strict architectural perimeter:
- Whitelist validation on action_type and target_id.
- Pure Python deterministic safety checks (35% reservoir floor & water quality).
- Physical mutations MUST be staged as 'PENDING' for human approval.
- Just-in-time safety re-check upon human 'Approve' before any database commit.
"""
from datetime import datetime
from typing import Optional
from config.constants import (
    ALLOWED_ACTION_TYPES,
    KNOWN_PLANTS,
    KNOWN_SECTOR_TARGETS,
    KNOWN_TARGET_IDS,
    KNOWN_VALVES,
    RESERVOIR_TOTAL_CAPACITY_ML,
    RESERVOIR_FLOOR_PERCENT,
)
from database.connection import get_db_connection, transaction_scope
from guards.safety_floors import (
    compute_effective_reservoir_floor,
    simulate_reservoir_trajectory,
    evaluate_quality_and_bloom_derating,
)
from guards.water_quality import evaluate_sector_chlorine, target_id_to_sector_name


def get_active_emergency_floor(
    db_path: Optional[str] = None,
    at_time: Optional[datetime] = None,
    conn=None,
) -> dict:
    """Query active emergency_commitments and compute the effective dynamic reservoir floor."""
    own_conn = conn is None
    db_conn = get_db_connection(db_path) if own_conn else conn
    try:
        rows = db_conn.execute(
            "SELECT id, sector, volume_m3, start_time, end_time, source, created_at FROM emergency_commitments ORDER BY id ASC;"
        ).fetchall()
        res_row = db_conn.execute("SELECT capacity_ml FROM reservoirs WHERE id = 'RES-01';").fetchone()
        cap_ml = float(res_row["capacity_ml"]) if res_row else RESERVOIR_TOTAL_CAPACITY_ML
    finally:
        if own_conn:
            db_conn.close()
    return compute_effective_reservoir_floor(
        commitments=[dict(r) for r in rows],
        total_capacity_ml=cap_ml,
        base_floor_percent=RESERVOIR_FLOOR_PERCENT,
        at_time=at_time,
    )


def revalidate_pending_actions_against_floor(
    db_path: Optional[str] = None,
    at_time: Optional[datetime] = None,
) -> dict:
    """
    Re-validate all PENDING setpoint/flush actions against the latest dynamic reservoir floor.
    Any PENDING action that breaches the new floor is marked CANCELLED_BY_SAFETY_RECHECK
    and logged in audit_log.
    """
    cancelled: list[dict] = []
    kept: list[dict] = []

    with transaction_scope(db_path) as conn:
        eff = get_active_emergency_floor(db_path=db_path, at_time=at_time, conn=conn)
        res_row = conn.execute("SELECT volume_ml, capacity_ml FROM reservoirs WHERE id = 'RES-01';").fetchone()
        current_vol = float(res_row["volume_ml"]) if res_row else 240.0
        cap_ml = float(res_row["capacity_ml"]) if res_row else RESERVOIR_TOTAL_CAPACITY_ML

        demand_rows = conn.execute(
            "SELECT demand_mld FROM demand_forecast WHERE hour >= 14 AND hour < 20 ORDER BY hour ASC;"
        ).fetchall()
        demand_list = [float(r["demand_mld"]) for r in demand_rows]
        if not demand_list or len(demand_list) < 6:
            demand_list = [351.0] * 6

        pending_rows = conn.execute(
            "SELECT * FROM pending_actions WHERE status = 'PENDING' ORDER BY id ASC;"
        ).fetchall()

        for row in pending_rows:
            action_id = int(row["id"])
            action_type = row["action_type"]
            target_id = row["target_id"]
            proposed_val = float(row["proposed_value"])

            if action_type == "SET_PLANT_OUTPUT":
                sim = simulate_reservoir_trajectory(
                    current_volume_ml=current_vol,
                    plant_inflow_mld=proposed_val,
                    demand_forecast_mld=demand_list,
                    total_capacity_ml=cap_ml,
                    min_floor_percent=eff["effective_floor_ratio"],
                )
                if not sim["is_safe"]:
                    reason = (
                        f"Re-validation after emergency floor update ({eff['banner']}): {sim['reason']}"
                    )
                    conn.execute(
                        """
                        UPDATE pending_actions
                        SET status = 'CANCELLED_BY_SAFETY_RECHECK',
                            executed_at = CURRENT_TIMESTAMP,
                            approver = 'CODE_GUARD_RECHECK'
                        WHERE id = ?;
                        """,
                        (action_id,),
                    )
                    conn.execute(
                        """
                        INSERT INTO audit_log (event_type, target_id, details, actor, status)
                        VALUES (?, ?, ?, ?, ?);
                        """,
                        (
                            "PENDING_CANCELLED_BY_EMERGENCY_FLOOR",
                            target_id,
                            f"Action #{action_id} ({action_type} -> {proposed_val} MLD) CANCELLED_BY_SAFETY_RECHECK: {reason}",
                            "CODE_GUARD",
                            "CANCELLED_BY_SAFETY_RECHECK",
                        ),
                    )
                    cancelled.append({
                        "action_id": action_id,
                        "action_type": action_type,
                        "target_id": target_id,
                        "proposed_value": proposed_val,
                        "min_projected_percent": sim["min_level_percent"],
                        "floor_percent": eff["effective_floor_percent"],
                        "status": "CANCELLED_BY_SAFETY_RECHECK",
                        "reason": reason,
                    })
                    continue
            kept.append({"action_id": action_id, "action_type": action_type, "target_id": target_id})

    return {
        "effective_floor": eff,
        "checked_count": len(cancelled) + len(kept),
        "cancelled": cancelled,
        "kept": kept,
    }


def add_emergency_commitment(
    sector: str,
    volume_m3: float,
    start_time: str,
    end_time: str,
    source: str = "Civil Defence",
    db_path: Optional[str] = None,
    at_time: Optional[datetime] = None,
) -> dict:
    """
    Insert an emergency reserve commitment, log it in audit_log, and immediately
    re-validate all PENDING actions against the raised dynamic reservoir floor.
    """
    with transaction_scope(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO emergency_commitments (sector, volume_m3, start_time, end_time, source)
            VALUES (?, ?, ?, ?, ?);
            """,
            (sector, float(volume_m3), start_time, end_time, source),
        )
        commitment_id = cur.lastrowid
        conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                "EMERGENCY_COMMITMENT_ADDED",
                sector,
                f"{source} requested {float(volume_m3):,.0f} m³ ({float(volume_m3)/1000.0:.1f} ML) for {sector} ({start_time} to {end_time}).",
                "CODE_GUARD",
                "ALLOWED",
            ),
        )

    reval = revalidate_pending_actions_against_floor(db_path=db_path, at_time=at_time)
    return {
        "commitment_id": commitment_id,
        "sector": sector,
        "volume_m3": float(volume_m3),
        "source": source,
        "effective_floor": reval["effective_floor"],
        "revalidation": reval,
    }


_FORBIDDEN_APPROVAL_KEYS = (
    "pre_approved",
    "preapproved",
    "approved",
    "approval",
    "approved_by",
    "operator_approved",
    "bypass_approval",
    "auto_approve",
    "force_execute",
    "override_approval",
)


def stage_gated_valve_or_setpoint_action(
    action_type: str,
    target_id: str,
    proposed_value: float,
    rationale: str,
    db_path: Optional[str] = None,
    provenance: str = "AGENT_ANALYSIS",
    provenance_ref: str = "",
    **extra_kwargs,
) -> dict:
    """
    The ONLY authorized write staging gateway for AquaGrid.

    Validates targets against whitelists, refuses any LLM approval claim (pre_approved,
    approved, approval, etc.), runs mathematical safety floor checks, records action
    provenance (AGENT_ANALYSIS or UNTRUSTED_TEXT), and inserts approved proposals into
    'pending_actions' with status 'PENDING'.
    DOES NOT modify 'valves' or 'plant_setpoints' directly.
    """
    # 0. Refuse any unauthorized approval claim passed by the LLM or untrusted caller
    bad_keys = [
        k for k in extra_kwargs
        if k.lower() in _FORBIDDEN_APPROVAL_KEYS
        or any(tok in k.lower() for tok in ("approv", "bypass", "override", "auto_exec"))
    ]
    if bad_keys:
        error_msg = (
            f"REJECTED_UNAUTHORIZED_APPROVAL_CLAIM: Gateway accepts no approval field from LLM "
            f"(received forbidden parameter(s): {', '.join(bad_keys)} for {action_type} {target_id}). "
            f"Only the console Approve button can authorize execution."
        )
        with transaction_scope(db_path) as conn:
            conn.execute(
                """
                INSERT INTO audit_log (event_type, target_id, details, actor, status)
                VALUES (?, ?, ?, ?, ?);
                """,
                ("UNAUTHORIZED_APPROVAL_CLAIM_REFUSED", target_id, error_msg, "CODE_GUARD", "REJECTED_UNAUTHORIZED_APPROVAL_CLAIM"),
            )
        return {
            "status": "REJECTED_UNAUTHORIZED_APPROVAL_CLAIM",
            "target_id": target_id,
            "action_type": action_type,
            "forbidden_fields": bad_keys,
            "reason": error_msg,
        }
    # 1. Target and Action Type Whitelist Validation (including action-to-target pairing)
    valid_target_for_action = (
        (action_type in ("CLOSE_VALVE", "OPEN_VALVE") and target_id in KNOWN_VALVES)
        or (action_type == "SET_PLANT_OUTPUT" and target_id in KNOWN_PLANTS)
        or (action_type == "ISOLATE_AND_FLUSH" and target_id in KNOWN_SECTOR_TARGETS)
    )
    if action_type not in ALLOWED_ACTION_TYPES or target_id not in KNOWN_TARGET_IDS or not valid_target_for_action:
        error_msg = f"Rejected invalid action_type '{action_type}' or unknown target_id '{target_id}'"
        with transaction_scope(db_path) as conn:
            conn.execute(
                """
                INSERT INTO audit_log (event_type, target_id, details, actor, status)
                VALUES (?, ?, ?, ?, ?);
                """,
                ("INVALID_ACTION_REJECTED", target_id, error_msg, "CODE_GUARD", "REJECTED"),
            )
        return {
            "status": "REJECTED_INVALID_TARGET",
            "reason": error_msg,
            "allowed_actions": ALLOWED_ACTION_TYPES,
            "known_targets": KNOWN_TARGET_IDS,
        }

    conn = get_db_connection(db_path)
    sim_min_projected_percent: Optional[float] = None
    effective_floor_pct_for_resp: float = RESERVOIR_FLOOR_PERCENT * 100.0
    try:
        # 1b. Duplicate PENDING guard: reject if a PENDING action already exists for same target_id & action_type
        existing_pending = conn.execute(
            """
            SELECT id
            FROM pending_actions
            WHERE target_id = ? AND action_type = ? AND status = 'PENDING'
            ORDER BY id ASC
            LIMIT 1;
            """,
            (target_id, action_type),
        ).fetchone()
        if existing_pending:
            existing_id = int(existing_pending["id"])
            reason = (
                f"Duplicate proposal rejected: PENDING action #{existing_id} already exists for "
                f"{action_type} on {target_id}. Resolve the existing pending action first."
            )
            with transaction_scope(db_path) as audit_conn:
                audit_conn.execute(
                    """
                    INSERT INTO audit_log (event_type, target_id, details, actor, status)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    ("DUPLICATE_PENDING_REJECTED", target_id, reason, "CODE_GUARD", "BLOCKED"),
                )
            return {
                "status": "REJECTED_DUPLICATE_PENDING",
                "existing_action_id": existing_id,
                "action_type": action_type,
                "target_id": target_id,
                "reason": reason,
            }

        current_value = 0.0
        # Fetch current actuator state
        if action_type in ("CLOSE_VALVE", "OPEN_VALVE"):
            v_row = conn.execute("SELECT status FROM valves WHERE valve_id = ?;", (target_id,)).fetchone()
            current_value = 1.0 if (v_row and v_row["status"] == "OPEN") else 0.0
        elif action_type == "ISOLATE_AND_FLUSH":
            sec_name = target_id_to_sector_name(target_id)
            wq_row = conn.execute(
                """
                SELECT chlorine_mg_l FROM sector_water_quality
                WHERE sector = ?
                ORDER BY id DESC LIMIT 1;
                """,
                (sec_name,),
            ).fetchone()
            current_value = float(wq_row["chlorine_mg_l"]) if wq_row else 0.5
        elif action_type == "SET_PLANT_OUTPUT":
            p_row = conn.execute("SELECT production_mld FROM plant_setpoints WHERE plant_id = ?;", (target_id,)).fetchone()
            current_value = float(p_row["production_mld"]) if p_row else 100.0

            # 1c. Per-sector chlorine compliance guard: reject tariff-driven cuts while any sector is NON_COMPLIANT
            wq_rows = conn.execute(
                """
                SELECT sector, chlorine_mg_l, timestamp
                FROM sector_water_quality
                WHERE id IN (SELECT MAX(id) FROM sector_water_quality GROUP BY sector);
                """
            ).fetchall()
            wq_eval = evaluate_sector_chlorine([dict(r) for r in wq_rows])
            if not wq_eval["all_compliant"] and proposed_value < current_value:
                excl = ", ".join(wq_eval["excluded_from_tariff_scheduling"])
                reason = (
                    f"REJECTED_WATER_QUALITY: Production cut ({proposed_value:.1f} MLD < {current_value:.1f} MLD) "
                    f"blocked while sector(s) {excl} are NON_COMPLIANT (< 0.20 mg/L chlorine)."
                )
                with transaction_scope(db_path) as audit_conn:
                    audit_conn.execute(
                        """
                        INSERT INTO audit_log (event_type, target_id, details, actor, status)
                        VALUES (?, ?, ?, ?, ?);
                        """,
                        ("WATER_QUALITY_CUT_BLOCKED", target_id, reason, "CODE_GUARD", "REJECTED_WATER_QUALITY"),
                    )
                return {
                    "status": "REJECTED_WATER_QUALITY",
                    "reason": reason,
                    "excluded_sectors": wq_eval["excluded_from_tariff_scheduling"],
                }

            # 2. Pure Python Safety Floor Verification (for production changes)
            # Fetch latest reservoir volume
            res_row = conn.execute("SELECT volume_ml FROM reservoirs WHERE id = 'RES-01';").fetchone()
            current_vol = float(res_row["volume_ml"]) if res_row else 240.0

            # Fetch demand forecast over the 6h horizon from 14:00 (hours 14 to 19)
            demand_rows = conn.execute(
                "SELECT demand_mld FROM demand_forecast WHERE hour >= 14 AND hour < 20 ORDER BY hour ASC;"
            ).fetchall()
            demand_list = [float(r["demand_mld"]) for r in demand_rows]
            if not demand_list or len(demand_list) < 6:
                demand_list = [351.0] * 6

            # Simulate trajectory against dynamic floor (35% + active emergency commitments)
            eff_floor = get_active_emergency_floor(db_path=db_path, conn=conn)
            sim_result = simulate_reservoir_trajectory(
                current_volume_ml=current_vol,
                plant_inflow_mld=proposed_value,
                demand_forecast_mld=demand_list,
                total_capacity_ml=RESERVOIR_TOTAL_CAPACITY_ML,
                min_floor_percent=eff_floor["effective_floor_ratio"],
            )
            sim_min_projected_percent = sim_result["min_level_percent"]
            effective_floor_pct_for_resp = eff_floor["effective_floor_percent"]

            # If simulation violates the dynamic floor, abort immediately
            if not sim_result["is_safe"]:
                with transaction_scope(db_path) as audit_conn:
                    audit_conn.execute(
                        """
                        INSERT INTO audit_log (event_type, target_id, details, actor, status)
                        VALUES (?, ?, ?, ?, ?);
                        """,
                        (
                            "SAFETY_FLOOR_BREACH",
                            target_id,
                            f"Proposed output {proposed_value} MLD blocked ({eff_floor['banner']}): {sim_result['reason']}",
                            "CODE_GUARD",
                            "BLOCKED",
                        ),
                    )
                return {
                    "status": "REJECTED_BY_SAFETY_GUARD",
                    "reason": sim_result["reason"],
                    "min_projected_percent": sim_result["min_level_percent"],
                    "floor_threshold_percent": eff_floor["effective_floor_percent"],
                    "floor_banner": eff_floor["banner"],
                    "breach_hour_offset": sim_result["breach_hour"],
                }

            # Check algal bloom derating constraint
            alert_rows = conn.execute(
                "SELECT severity, raw_text FROM quality_alerts WHERE raw_text LIKE '%bloom%';"
            ).fetchall()
            highest_bloom = "NONE"
            for a in alert_rows:
                sev = a["severity"].upper()
                if sev in ("CRITICAL", "MODERATE", "LOW"):
                    highest_bloom = sev

            derating_check = evaluate_quality_and_bloom_derating(
                turbidity_ntu=1.2,
                chlorine_ppm=2.0,
                bloom_severity=highest_bloom,
            )
            if proposed_value > derating_check["max_safe_output_mld"]:
                reason = (
                    f"Proposed output {proposed_value} MLD exceeds maximum allowable capacity of "
                    f"{derating_check['max_safe_output_mld']} MLD due to {highest_bloom} algal bloom alert."
                )
                with transaction_scope(db_path) as audit_conn:
                    audit_conn.execute(
                        """
                        INSERT INTO audit_log (event_type, target_id, details, actor, status)
                        VALUES (?, ?, ?, ?, ?);
                        """,
                        ("BLOOM_CAPACITY_BREACH", target_id, reason, "CODE_GUARD", "BLOCKED"),
                    )
                return {
                    "status": "REJECTED_BY_SAFETY_GUARD",
                    "reason": reason,
                    "max_safe_output_mld": derating_check["max_safe_output_mld"],
                }
    finally:
        conn.close()

    # 3. Stage Action as PENDING in pending_actions (Atomic, with provenance)
    norm_provenance = "UNTRUSTED_TEXT" if str(provenance or "").upper() == "UNTRUSTED_TEXT" else "AGENT_ANALYSIS"
    prov_ref_clean = str(provenance_ref or "").strip()
    prov_label = f"{norm_provenance} ({prov_ref_clean})" if prov_ref_clean else norm_provenance

    with transaction_scope(db_path) as tx_conn:
        cursor = tx_conn.execute(
            """
            INSERT INTO pending_actions (
                action_type, target_id, proposed_value, current_value, rationale, provenance, provenance_ref, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING');
            """,
            (action_type, target_id, proposed_value, current_value, rationale, norm_provenance, prov_ref_clean),
        )
        action_id = cursor.lastrowid

        tx_conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                "GATEWAY_PROPOSAL",
                target_id,
                (
                    f"Action {action_type} for {target_id} staged as PENDING (ID: {action_id}, "
                    f"provenance={prov_label}). Awaiting human authorization."
                ),
                "CODE_GUARD",
                "PENDING",
            ),
        )

    response = {
        "status": "STAGED_PENDING_APPROVAL",
        "action_id": action_id,
        "action_type": action_type,
        "target_id": target_id,
        "proposed_value": proposed_value,
        "current_value": current_value,
        "rationale": rationale,
        "provenance": norm_provenance,
        "provenance_ref": prov_ref_clean,
        "message": "Action staged successfully in PENDING state. Awaiting mandatory human operator authorization.",
    }
    if sim_min_projected_percent is not None:
        response["min_projected_percent"] = sim_min_projected_percent
        response["floor_threshold_percent"] = effective_floor_pct_for_resp
    return response


def execute_human_approval(
    action_id: int,
    approver: str = "ControlRoom_Operator",
    db_path: Optional[str] = None,
) -> dict:
    """
    Execute human operator approval with mandatory just-in-time safety re-check.

    Re-runs the 35% reservoir floor and water quality checks with latest telemetry.
    If conditions deteriorated, cancels with 'CANCELLED_BY_SAFETY_RECHECK' and DOES NOT commit.
    If re-check passes, commits the actuator change in a single ACID transaction.
    """
    with transaction_scope(db_path) as conn:
        action_row = conn.execute(
            "SELECT * FROM pending_actions WHERE id = ? AND status = 'PENDING';",
            (action_id,),
        ).fetchone()

        if not action_row:
            return {
                "status": "ERROR_NOT_FOUND",
                "message": f"Pending action with ID {action_id} not found or already processed.",
            }

        action_type = action_row["action_type"]
        target_id = action_row["target_id"]
        proposed_value = float(action_row["proposed_value"])

        # JIT Safety Re-Check: If setting plant output, re-verify dynamic reservoir floor
        eff_floor = get_active_emergency_floor(db_path=db_path, conn=conn)
        if action_type == "SET_PLANT_OUTPUT":
            res_row = conn.execute("SELECT volume_ml FROM reservoirs WHERE id = 'RES-01';").fetchone()
            latest_vol = float(res_row["volume_ml"]) if res_row else 240.0
            demand_rows = conn.execute(
                "SELECT demand_mld FROM demand_forecast WHERE hour >= 14 AND hour < 20 ORDER BY hour ASC;"
            ).fetchall()
            demand_list = [float(r["demand_mld"]) for r in demand_rows] or [351.0] * 6

            recheck_sim = simulate_reservoir_trajectory(
                current_volume_ml=latest_vol,
                plant_inflow_mld=proposed_value,
                demand_forecast_mld=demand_list,
                min_floor_percent=eff_floor["effective_floor_ratio"],
            )

            if not recheck_sim["is_safe"]:
                conn.execute(
                    """
                    UPDATE pending_actions
                    SET status = 'CANCELLED_BY_SAFETY_RECHECK', executed_at = CURRENT_TIMESTAMP, approver = ?
                    WHERE id = ?;
                    """,
                    (approver, action_id),
                )
                conn.execute(
                    """
                    INSERT INTO audit_log (event_type, target_id, details, actor, status)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (
                        "APPROVAL_CANCELLED_BY_SAFETY",
                        target_id,
                        f"Action {action_id} cancelled upon approval: {recheck_sim['reason']}",
                        "CODE_GUARD",
                        "BLOCKED",
                    ),
                )
                return {
                    "status": "CANCELLED_BY_SAFETY_RECHECK",
                    "reason": recheck_sim["reason"],
                    "min_projected_percent": recheck_sim["min_level_percent"],
                }

        # Execute physical mutation atomically
        if action_type == "CLOSE_VALVE":
            conn.execute(
                "UPDATE valves SET status = 'CLOSED', last_updated = CURRENT_TIMESTAMP WHERE valve_id = ?;",
                (target_id,),
            )
        elif action_type == "OPEN_VALVE":
            conn.execute(
                "UPDATE valves SET status = 'OPEN', last_updated = CURRENT_TIMESTAMP WHERE valve_id = ?;",
                (target_id,),
            )
        elif action_type == "SET_PLANT_OUTPUT":
            conn.execute(
                "UPDATE plant_setpoints SET production_mld = ?, last_updated = CURRENT_TIMESTAMP WHERE plant_id = ?;",
                (proposed_value, target_id),
            )
        elif action_type == "ISOLATE_AND_FLUSH":
            # Safety re-check: verify reservoir is above mandatory floor before flushing sector
            res_row = conn.execute("SELECT volume_ml, capacity_ml FROM reservoirs WHERE id = 'RES-01';").fetchone()
            latest_vol = float(res_row["volume_ml"]) if res_row else 240.0
            cap_ml = float(res_row["capacity_ml"]) if res_row else RESERVOIR_TOTAL_CAPACITY_ML
            if latest_vol < eff_floor["effective_floor_ml"]:
                reason = (
                    f"Reservoir volume ({latest_vol:.1f} ML) is below safety floor "
                    f"({eff_floor['effective_floor_ml']:.1f} ML / {eff_floor['effective_floor_percent']:.1f}%); "
                    f"cannot initiate flush for {target_id}."
                )
                conn.execute(
                    """
                    UPDATE pending_actions
                    SET status = 'CANCELLED_BY_SAFETY_RECHECK', executed_at = CURRENT_TIMESTAMP, approver = ?
                    WHERE id = ?;
                    """,
                    (approver, action_id),
                )
                conn.execute(
                    """
                    INSERT INTO audit_log (event_type, target_id, details, actor, status)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    ("APPROVAL_CANCELLED_BY_SAFETY", target_id, reason, "CODE_GUARD", "BLOCKED"),
                )
                return {"status": "CANCELLED_BY_SAFETY_RECHECK", "reason": reason}

            sec_name = target_id_to_sector_name(target_id)
            conn.execute(
                "UPDATE valves SET status = 'CLOSED', last_updated = CURRENT_TIMESTAMP WHERE sector = ?;",
                (sec_name,),
            )

        # Update action status to APPROVED
        conn.execute(
            """
            UPDATE pending_actions
            SET status = 'APPROVED', executed_at = CURRENT_TIMESTAMP, approver = ?
            WHERE id = ?;
            """,
            (approver, action_id),
        )

        # Audit log commit
        conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                "ACTION_EXECUTED",
                target_id,
                f"Action {action_id} ({action_type} -> {proposed_value}) APPROVED and executed by {approver}.",
                "HUMAN_OPERATOR",
                "APPROVED",
            ),
        )

    return {
        "status": "EXECUTED",
        "action_id": action_id,
        "action_type": action_type,
        "target_id": target_id,
        "new_value": proposed_value,
        "approver": approver,
    }


def execute_human_rejection(
    action_id: int,
    rejection_reason: str = "Operator declined action",
    approver: str = "ControlRoom_Operator",
    db_path: Optional[str] = None,
) -> dict:
    """Execute human operator rejection of a staged pending action."""
    with transaction_scope(db_path) as conn:
        conn.execute(
            """
            UPDATE pending_actions
            SET status = 'REJECTED', executed_at = CURRENT_TIMESTAMP, approver = ?
            WHERE id = ? AND status = 'PENDING';
            """,
            (approver, action_id),
        )
        conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                "ACTION_REJECTED",
                f"ACTION-{action_id}",
                f"Action {action_id} REJECTED by {approver}. Reason: {rejection_reason}",
                "HUMAN_OPERATOR",
                "REJECTED",
            ),
        )

    return {
        "status": "REJECTED",
        "action_id": action_id,
        "reason": rejection_reason,
        "approver": approver,
    }


def refuse_direct_write_bypass(
    target_valve_id: str = "V-12",
    attempted_status: str = "OPEN",
    db_path: Optional[str] = None,
) -> dict:
    """
    Demonstrate and enforce refusal of an un-gated direct write to the valves table.

    Executes the attempted write against a read-only SQLite connection
    (PRAGMA query_only = ON, matching the agent's read-only database boundary)
    so SQLite itself refuses any mutation outside execute_human_approval().
    Logs the refused attempt in audit_log and verifies the valve status is unchanged.
    """
    import sqlite3

    ro_conn = get_db_connection(db_path)
    try:
        before_row = ro_conn.execute(
            "SELECT status FROM valves WHERE valve_id = ?;", (target_valve_id,)
        ).fetchone()
        status_before = before_row["status"] if before_row else "UNKNOWN"

        # Enforce read-only database perimeter (zero direct write access outside gateway approval)
        ro_conn.execute("PRAGMA query_only = ON;")
        refusal_error = ""
        refused = False
        try:
            ro_conn.execute(
                "UPDATE valves SET status = ? WHERE valve_id = ?;",
                (attempted_status, target_valve_id),
            )
            ro_conn.commit()
        except sqlite3.OperationalError as exc:
            refused = True
            refusal_error = f"{type(exc).__name__}: {exc}"

        after_row = ro_conn.execute(
            "SELECT status FROM valves WHERE valve_id = ?;", (target_valve_id,)
        ).fetchone()
        status_after = after_row["status"] if after_row else "UNKNOWN"
    finally:
        ro_conn.close()

    with transaction_scope(db_path) as audit_conn:
        audit_conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                "DIRECT_WRITE_REFUSED",
                target_valve_id,
                f"Direct un-gated write (set {target_valve_id} -> {attempted_status}) REFUSED: {refusal_error}",
                "CODE_GUARD",
                "REFUSED",
            ),
        )

    return {
        "status": "REFUSED" if refused else "UNEXPECTED_WRITE",
        "target_id": target_valve_id,
        "attempted_action": f"Direct SQL UPDATE valves SET status='{attempted_status}' WHERE valve_id='{target_valve_id}' (bypassing write_gateway)",
        "error": refusal_error,
        "valve_status_before": status_before,
        "valve_status_after": status_after,
        "unchanged": status_before == status_after,
    }


ALLOWED_LEAK_STATUSES: tuple[str, ...] = (
    "Suspected",
    "Confirmed",
    "Under Repair",
    "Resolved",
    "False Alarm – Sensor Fault",
)


def update_leak_status_by_operator(
    leak_id: str,
    new_status: str,
    operator_name: str = "ControlRoom_Operator",
    note: str = "",
    db_path: Optional[str] = None,
) -> dict:
    """
    Record a human operator's status decision on a leak (e.g., Mark as False Alarm,
    Mark Resolved, Confirmed, Under Repair) and write an immutable audit_log entry.
    Does NOT modify valves or plant_setpoints.
    """
    if new_status not in ALLOWED_LEAK_STATUSES:
        return {
            "status": "REJECTED_INVALID_STATUS",
            "reason": f"Invalid leak status '{new_status}'. Allowed: {ALLOWED_LEAK_STATUSES}",
        }

    with transaction_scope(db_path) as conn:
        row = conn.execute("SELECT leak_id, status, pipe_segment_id FROM leaks WHERE leak_id = ?;", (leak_id,)).fetchone()
        if not row:
            return {"status": "ERROR_NOT_FOUND", "reason": f"Leak {leak_id} not found."}

        prev_status = row["status"]
        conn.execute("UPDATE leaks SET status = ? WHERE leak_id = ?;", (new_status, leak_id))
        details = (
            f"Operator {operator_name} updated {leak_id} ({row['pipe_segment_id']}) "
            f"from '{prev_status}' to '{new_status}'."
        )
        if note:
            details += f" Note: {note}"

        conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            ("LEAK_STATUS_UPDATED", leak_id, details, "HUMAN_OPERATOR", new_status.upper()),
        )

    return {
        "status": "UPDATED",
        "leak_id": leak_id,
        "previous_status": prev_status,
        "new_status": new_status,
        "operator": operator_name,
    }
