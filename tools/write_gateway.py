"""Exclusive Write Gateway for AquaGrid.

Enforces strict architectural perimeter:
- Whitelist validation on action_type and target_id.
- Pure Python deterministic safety checks (35% reservoir floor & water quality).
- Physical mutations MUST be staged as 'PENDING' for human approval.
- Just-in-time safety re-check upon human 'Approve' before any database commit.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from typing import Optional
from config.constants import (
    ALLOWED_ACTION_TYPES,
    APPROVAL_TOKEN_TTL_MINUTES,
    KNOWN_PLANTS,
    KNOWN_SECTOR_TARGETS,
    KNOWN_TARGET_IDS,
    KNOWN_VALVES,
    RESERVOIR_TOTAL_CAPACITY_ML,
    RESERVOIR_FLOOR_PERCENT,
)
from database.connection import get_db_connection, transaction_scope
from guards.safety_floors import (
    _parse_utc_timestamp,
    compute_effective_reservoir_floor,
    simulate_reservoir_trajectory,
    evaluate_quality_and_bloom_derating,
)
from guards.telemetry_freshness import check_db_telemetry_freshness
from guards.water_quality import evaluate_sector_chlorine, target_id_to_sector_name


def compute_action_params_hash(action_type: str, target_id: str, proposed_value: float) -> str:
    """Deterministic SHA-256 hash binding an approval token to exact action parameters."""
    canonical = f"{str(action_type).strip().upper()}|{str(target_id).strip().upper()}|{float(proposed_value):.4f}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def issue_approval_token(
    action_id: int,
    operator: str = "ControlRoom_Operator",
    issued_at: Optional[datetime] = None,
    db_path: Optional[str] = None,
) -> dict:
    """
    Issue a single-use cryptographic approval token bound to a specific PENDING action.
    Only callable from the operator approval console (never exposed to agent/).
    """
    now_utc = _parse_utc_timestamp(issued_at) or datetime.now(timezone.utc)
    expires_utc = now_utc + timedelta(minutes=APPROVAL_TOKEN_TTL_MINUTES)
    issued_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")
    expires_str = expires_utc.strftime("%Y-%m-%d %H:%M:%S")

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
        params_hash = compute_action_params_hash(action_type, target_id, proposed_value)
        token_id = f"TOK-{secrets.token_hex(10).upper()}"

        conn.execute(
            """
            INSERT INTO approval_tokens (
                token_id, action_id, action_type, target, params_hash, operator, issued_at, expires_at, used_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL);
            """,
            (token_id, action_id, action_type, target_id, params_hash, operator, issued_str, expires_str),
        )
        conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                "APPROVAL_TOKEN_ISSUED",
                target_id,
                f"Issued single-use token {token_id} for Action #{action_id} ({action_type} {target_id}) by {operator}, expires {expires_str} UTC.",
                "HUMAN_OPERATOR",
                "ALLOWED",
            ),
        )

    return {
        "status": "ISSUED",
        "token_id": token_id,
        "action_id": action_id,
        "action_type": action_type,
        "target": target_id,
        "params_hash": params_hash,
        "operator": operator,
        "issued_at": issued_str,
        "expires_at": expires_str,
        "used_at": None,
    }


def _execute_bound_action_in_tx(
    conn,
    token_row,
    action_type: str,
    target_id: str,
    proposed_value: float,
    now_utc: datetime,
    db_path: Optional[str] = None,
) -> dict:
    """Validate single-use token and execute physical mutation + mark used_at in ONE atomic transaction."""
    token_id = token_row["token_id"]
    action_id = int(token_row["action_id"])
    approver = token_row["operator"]
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")

    expected_hash = compute_action_params_hash(action_type, target_id, proposed_value)
    expires_dt = _parse_utc_timestamp(token_row["expires_at"])
    is_expired = bool(expires_dt is not None and now_utc > expires_dt)
    is_used = token_row["used_at"] is not None

    # 1. Exact binding check (action_type, target, params_hash)
    if (
        token_row["action_type"] != action_type
        or token_row["target"] != target_id
        or token_row["params_hash"] != expected_hash
    ):
        extra_notes = []
        if is_expired:
            extra_notes.append(f"also expired at {token_row['expires_at']} UTC")
        if is_used:
            extra_notes.append(f"already used at {token_row['used_at']} UTC")
        suffix = f" ({'; '.join(extra_notes)})" if extra_notes else ""
        reason = (
            f"REJECTED_APPROVAL_MISMATCH: Token {token_id} is bound to {token_row['action_type']} {token_row['target']} "
            f"(hash={token_row['params_hash'][:12]}...), cannot be used for {action_type} {target_id}{suffix}."
        )
        conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            ("APPROVAL_TOKEN_REJECTED", target_id, reason, "CODE_GUARD", "REJECTED_APPROVAL_MISMATCH"),
        )
        return {
            "status": "REJECTED_APPROVAL_MISMATCH",
            "token_id": token_id,
            "target_id": target_id,
            "bound_target": token_row["target"],
            "also_expired": is_expired,
            "also_used": is_used,
            "reason": reason,
        }

    # 2. Single-use check (already consumed)
    if is_used:
        exp_note = f" (also expired at {token_row['expires_at']} UTC)" if is_expired else ""
        reason = (
            f"REJECTED_APPROVAL_ALREADY_USED: Single-use token {token_id} for {action_type} {target_id} "
            f"was already consumed at {token_row['used_at']} UTC{exp_note}."
        )
        conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            ("APPROVAL_TOKEN_REJECTED", target_id, reason, "CODE_GUARD", "REJECTED_APPROVAL_ALREADY_USED"),
        )
        return {
            "status": "REJECTED_APPROVAL_ALREADY_USED",
            "token_id": token_id,
            "target_id": target_id,
            "used_at": token_row["used_at"],
            "also_expired": is_expired,
            "reason": reason,
        }

    # 3. Expiration TTL check
    if is_expired:
        reason = (
            f"REJECTED_APPROVAL_EXPIRED: Token {token_id} for {action_type} {target_id} "
            f"expired at {token_row['expires_at']} UTC (TTL {APPROVAL_TOKEN_TTL_MINUTES} min)."
        )
        conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            ("APPROVAL_TOKEN_REJECTED", target_id, reason, "CODE_GUARD", "REJECTED_APPROVAL_EXPIRED"),
        )
        return {
            "status": "REJECTED_APPROVAL_EXPIRED",
            "token_id": token_id,
            "target_id": target_id,
            "expires_at": token_row["expires_at"],
            "reason": reason,
        }

    # 3b. Telemetry freshness check at Approve (refuse execution if telemetry is STALE)
    freshness_at_approve = check_db_telemetry_freshness(db_path=db_path, now_utc=now_utc, conn=conn)
    if freshness_at_approve["is_telemetry_loss"]:
        reason = (
            f"CANCELLED_STALE_TELEMETRY: {freshness_at_approve['banner']} "
            f"Refused Approve on Action #{action_id} ({action_type} {target_id}). Last safe state held."
        )
        conn.execute(
            """
            UPDATE pending_actions
            SET status = 'CANCELLED_STALE_TELEMETRY', executed_at = ?, approver = ?
            WHERE id = ?;
            """,
            (now_str, approver, action_id),
        )
        conn.execute("UPDATE approval_tokens SET used_at = ? WHERE token_id = ?;", (now_str, token_id))
        conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            ("APPROVAL_CANCELLED_STALE_TELEMETRY", target_id, reason, "CODE_GUARD", "CANCELLED_STALE_TELEMETRY"),
        )
        return {
            "status": "CANCELLED_STALE_TELEMETRY",
            "token_id": token_id,
            "action_id": action_id,
            "telemetry_freshness": freshness_at_approve,
            "reason": reason,
        }

    # 4. JIT Safety Re-Check against dynamic floor
    eff_floor = get_active_emergency_floor(db_path=db_path, at_time=now_utc, conn=conn)
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
                SET status = 'CANCELLED_BY_SAFETY_RECHECK', executed_at = ?, approver = ?
                WHERE id = ?;
                """,
                (now_str, approver, action_id),
            )
            conn.execute(
                "UPDATE approval_tokens SET used_at = ? WHERE token_id = ?;",
                (now_str, token_id),
            )
            conn.execute(
                """
                INSERT INTO audit_log (event_type, target_id, details, actor, status)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    "APPROVAL_CANCELLED_BY_SAFETY",
                    target_id,
                    f"Action {action_id} (token {token_id}) cancelled upon approval: {recheck_sim['reason']}",
                    "CODE_GUARD",
                    "BLOCKED",
                ),
            )
            return {
                "status": "CANCELLED_BY_SAFETY_RECHECK",
                "token_id": token_id,
                "reason": recheck_sim["reason"],
                "min_projected_percent": recheck_sim["min_level_percent"],
            }

    # 5. Execute physical mutation + consume token (set used_at) in the SAME ACID transaction
    if action_type == "CLOSE_VALVE":
        conn.execute(
            "UPDATE valves SET status = 'CLOSED', last_updated = ? WHERE valve_id = ?;",
            (now_str, target_id),
        )
    elif action_type == "OPEN_VALVE":
        conn.execute(
            "UPDATE valves SET status = 'OPEN', last_updated = ? WHERE valve_id = ?;",
            (now_str, target_id),
        )
    elif action_type == "SET_PLANT_OUTPUT":
        conn.execute(
            "UPDATE plant_setpoints SET production_mld = ?, last_updated = ? WHERE plant_id = ?;",
            (proposed_value, now_str, target_id),
        )
    elif action_type == "ISOLATE_AND_FLUSH":
        res_row = conn.execute("SELECT volume_ml, capacity_ml FROM reservoirs WHERE id = 'RES-01';").fetchone()
        latest_vol = float(res_row["volume_ml"]) if res_row else 240.0
        if latest_vol < eff_floor["effective_floor_ml"]:
            reason = (
                f"Reservoir volume ({latest_vol:.1f} ML) is below safety floor "
                f"({eff_floor['effective_floor_ml']:.1f} ML / {eff_floor['effective_floor_percent']:.1f}%); "
                f"cannot initiate flush for {target_id}."
            )
            conn.execute(
                """
                UPDATE pending_actions
                SET status = 'CANCELLED_BY_SAFETY_RECHECK', executed_at = ?, approver = ?
                WHERE id = ?;
                """,
                (now_str, approver, action_id),
            )
            conn.execute("UPDATE approval_tokens SET used_at = ? WHERE token_id = ?;", (now_str, token_id))
            conn.execute(
                """
                INSERT INTO audit_log (event_type, target_id, details, actor, status)
                VALUES (?, ?, ?, ?, ?);
                """,
                ("APPROVAL_CANCELLED_BY_SAFETY", target_id, reason, "CODE_GUARD", "BLOCKED"),
            )
            return {"status": "CANCELLED_BY_SAFETY_RECHECK", "token_id": token_id, "reason": reason}

        sec_name = target_id_to_sector_name(target_id)
        conn.execute(
            "UPDATE valves SET status = 'CLOSED', last_updated = ? WHERE sector = ?;",
            (now_str, sec_name),
        )

    conn.execute(
        """
        UPDATE pending_actions
        SET status = 'APPROVED', executed_at = ?, approver = ?
        WHERE id = ?;
        """,
        (now_str, approver, action_id),
    )
    conn.execute(
        "UPDATE approval_tokens SET used_at = ? WHERE token_id = ?;",
        (now_str, token_id),
    )
    conn.execute(
        """
        INSERT INTO audit_log (event_type, target_id, details, actor, status)
        VALUES (?, ?, ?, ?, ?);
        """,
        (
            "ACTION_EXECUTED",
            target_id,
            f"Action {action_id} ({action_type} -> {proposed_value}) APPROVED via single-use token {token_id} and executed by {approver}.",
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
        "token_id": token_id,
        "token_issued_at": token_row["issued_at"],
        "token_expires_at": token_row["expires_at"],
        "token_used_at": now_str,
        "params_hash": token_row["params_hash"],
    }


def execute_action_with_approval_token(
    token_id: str,
    action_type: str,
    target_id: str,
    proposed_value: float,
    at_time: Optional[datetime] = None,
    db_path: Optional[str] = None,
) -> dict:
    """
    Attempt to execute an actuator change by presenting an approval token.
    Enforces: token exists, matches action_type/target/params_hash, not already used, not expired.
    Marks used_at and commits the mutation in the same transaction on success.
    """
    now_utc = _parse_utc_timestamp(at_time) or datetime.now(timezone.utc)
    with transaction_scope(db_path) as conn:
        token_row = conn.execute(
            "SELECT * FROM approval_tokens WHERE token_id = ?;",
            (token_id,),
        ).fetchone()
        if not token_row:
            reason = f"REJECTED_APPROVAL_MISMATCH: Approval token '{token_id}' does not exist."
            conn.execute(
                """
                INSERT INTO audit_log (event_type, target_id, details, actor, status)
                VALUES (?, ?, ?, ?, ?);
                """,
                ("APPROVAL_TOKEN_REJECTED", target_id, reason, "CODE_GUARD", "REJECTED_APPROVAL_MISMATCH"),
            )
            return {
                "status": "REJECTED_APPROVAL_MISMATCH",
                "token_id": token_id,
                "target_id": target_id,
                "reason": reason,
            }
        return _execute_bound_action_in_tx(
            conn=conn,
            token_row=token_row,
            action_type=action_type,
            target_id=target_id,
            proposed_value=proposed_value,
            now_utc=now_utc,
            db_path=db_path,
        )


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
    # 0b. Telemetry freshness guard: refuse new proposals during TELEMETRY_LOSS
    freshness = check_db_telemetry_freshness(db_path=db_path)
    if freshness["is_telemetry_loss"]:
        reason = (
            f"REJECTED_STALE_TELEMETRY: {freshness['banner']} "
            f"Refusing new {action_type} proposal for {target_id} until telemetry returns and is re-validated."
        )
        with transaction_scope(db_path) as conn:
            conn.execute(
                """
                INSERT INTO audit_log (event_type, target_id, details, actor, status)
                VALUES (?, ?, ?, ?, ?);
                """,
                ("STALE_TELEMETRY_REFUSED", target_id, reason, "CODE_GUARD", "REJECTED_STALE_TELEMETRY"),
            )
        return {
            "status": "REJECTED_STALE_TELEMETRY",
            "target_id": target_id,
            "action_type": action_type,
            "telemetry_freshness": freshness,
            "reason": reason,
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
    approved_at: Optional[datetime] = None,
) -> dict:
    """
    Execute human operator approval by issuing a single-use approval token bound to
    the exact PENDING action, running the just-in-time safety re-check, and consuming
    the token (used_at) in the same ACID transaction as the physical mutation.
    """
    now_utc = _parse_utc_timestamp(approved_at) or datetime.now(timezone.utc)
    issued = issue_approval_token(
        action_id=action_id,
        operator=approver,
        issued_at=now_utc,
        db_path=db_path,
    )
    if issued.get("status") != "ISSUED":
        return issued

    conn_val = get_db_connection(db_path)
    try:
        row = conn_val.execute("SELECT action_type, target_id, proposed_value FROM pending_actions WHERE id = ?;", (action_id,)).fetchone()
    finally:
        conn_val.close()
    if not row:
        return {"status": "ERROR_NOT_FOUND", "message": f"Pending action {action_id} not found."}

    return execute_action_with_approval_token(
        token_id=issued["token_id"],
        action_type=row["action_type"],
        target_id=row["target_id"],
        proposed_value=float(row["proposed_value"]),
        at_time=now_utc,
        db_path=db_path,
    )


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
    "SUSPECTED",
    "CONFIRMED",
    "UNDER_REPAIR",
    "RESOLVED",
    "FALSE_ALARM",
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
    Record a human operator's status decision on a leak (e.g., FALSE_ALARM, RESOLVED,
    CONFIRMED, UNDER_REPAIR, SUSPECTED) and write an immutable audit_log entry.
    Does NOT modify valves or plant_setpoints.
    """
    if new_status not in ALLOWED_LEAK_STATUSES:
        return {
            "status": "REJECTED_INVALID_STATUS",
            "reason": f"Invalid leak status '{new_status}'. Allowed: {ALLOWED_LEAK_STATUSES}",
        }

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with transaction_scope(db_path) as conn:
        row = conn.execute("SELECT leak_id, status, pipe_segment_id FROM leaks WHERE leak_id = ?;", (leak_id,)).fetchone()
        if not row:
            return {"status": "ERROR_NOT_FOUND", "reason": f"Leak {leak_id} not found."}

        prev_status = row["status"]
        conn.execute(
            "UPDATE leaks SET status = ?, updated_at = ? WHERE leak_id = ?;",
            (new_status, now_utc, leak_id),
        )
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
