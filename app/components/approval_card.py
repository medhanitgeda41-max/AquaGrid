"""Human-in-the-Loop (HITL) Approval Card with Just-in-Time Safety Re-Check."""
import html
import pandas as pd
import streamlit as st
from database.connection import get_db_connection
from tools.write_gateway import execute_human_approval, execute_human_rejection
from components.badges import format_dubai_time, section_header, status_badge


def _format_actuator_value(action_type: str, value: float) -> str:
    """Format actuator values cleanly for control-room display."""
    if action_type in ("CLOSE_VALVE", "OPEN_VALVE"):
        return "OPEN (1.0)" if float(value) >= 0.5 else "CLOSED (0.0)"
    if action_type == "ISOLATE_AND_FLUSH":
        return f"{float(value):.2f} mg/L" if float(value) > 0 else "ISOLATE_AND_FLUSH"
    return f"{float(value):.1f} MLD"


def _render_feedback_banner() -> None:
    """Show the outcome of the most recent Approve / Reject / Reset action."""
    fb = st.session_state.get("last_hitl_feedback")
    if not fb:
        return
    kind = fb.get("kind", "info")
    msg = fb.get("message", "")
    if kind == "success":
        st.success(msg)
    elif kind == "error":
        st.error(msg)
    elif kind == "warning":
        st.warning(msg)
    else:
        st.info(msg)


def render_approval_card(key_prefix: str = "main", show_history: bool = True) -> None:
    """Render all PENDING actions from pending_actions with [Approve] and [Reject] buttons."""
    section_header(
        "Human-in-the-Loop (HITL) Approval Queue",
        ["HUMAN", "CODE"],
    )
    _render_feedback_banner()

    conn = get_db_connection()
    try:
        pending_rows = conn.execute(
            "SELECT * FROM pending_actions WHERE status = 'PENDING' ORDER BY id DESC;"
        ).fetchall()
        resolved_rows = conn.execute(
            "SELECT * FROM pending_actions WHERE status != 'PENDING' ORDER BY id DESC LIMIT 15;"
        ).fetchall()
    finally:
        conn.close()

    if not pending_rows:
        st.markdown(
            """
            <div class="scada-card-ok">
                <strong>No PENDING physical actuator actions.</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        operator_name = st.text_input(
            "Control Room Operator Signature",
            value=st.session_state.get("operator_signature", "Lead_Operator_SCADA"),
            key=f"{key_prefix}_operator_name",
        )
        st.session_state["operator_signature"] = operator_name

        for row in pending_rows:
            action_id = int(row["id"])
            action_type = row["action_type"]
            target_id = row["target_id"]
            curr_fmt = _format_actuator_value(action_type, float(row["current_value"]))
            prop_fmt = _format_actuator_value(action_type, float(row["proposed_value"]))
            rationale = row["rationale"]
            created_at = format_dubai_time(row["created_at"])
            prov = row["provenance"] if "provenance" in row.keys() and row["provenance"] else "AGENT_ANALYSIS"
            prov_ref = row["provenance_ref"] if "provenance_ref" in row.keys() and row["provenance_ref"] else ""
            prov_text = f"{prov} ({prov_ref})" if prov_ref else prov
            prov_badge_cls = "ag-badge-warn" if prov == "UNTRUSTED_TEXT" else "ag-badge-neutral"

            st.markdown(
                f"""
                <div class="scada-card-pending">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                        <div>
                            {status_badge("PENDING")}
                            <span class="ag-badge {prov_badge_cls}">SOURCE: {html.escape(prov_text)}</span>
                            <strong>Action #{action_id}:</strong>
                            <span class="ag-mono">{html.escape(action_type)} &rarr; {html.escape(target_id)}</span>
                        </div>
                        <div style="font-size:0.8rem; color:#CBD5E1;">Staged: {html.escape(str(created_at))}</div>
                    </div>
                    <div style="margin-top:0.5rem; font-size:0.9rem;">
                        <strong>Proposed Change:</strong>
                        <span class="ag-mono">{html.escape(curr_fmt)} &rarr; {html.escape(prop_fmt)}</span>
                    </div>
                    <div style="margin-top:0.35rem; font-size:0.87rem;">
                        <strong>Rationale:</strong> {html.escape(str(rationale))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c_reason, c_approve, c_reject = st.columns([2.4, 1.0, 1.0])
            with c_reason:
                reject_reason = st.text_input(
                    "Rejection reason (optional)",
                    value="Operator declined physical actuation",
                    key=f"{key_prefix}_rej_reason_{action_id}",
                    label_visibility="collapsed",
                    placeholder="Optional rejection reason...",
                )
            with c_approve:
                if st.button(
                    f"Approve #{action_id}",
                    key=f"{key_prefix}_approve_{action_id}",
                    type="primary",
                    width="stretch",
                ):
                    res = execute_human_approval(
                        action_id=action_id,
                        approver=operator_name.strip() or "ControlRoom_Operator",
                    )
                    res_status = res.get("status")
                    if res_status == "EXECUTED":
                        detail = res.get("details") or res.get("message") or "Executed."
                        st.session_state["last_hitl_feedback"] = {
                            "kind": "success",
                            "message": (
                                f"Action #{action_id} ({action_type} {target_id}) EXECUTED "
                                f"by {res.get('approver')}: {detail}"
                            ),
                        }
                    elif res_status in ("CANCELLED_BY_SAFETY_RECHECK", "CANCELLED_STALE_TELEMETRY"):
                        st.session_state["last_hitl_feedback"] = {
                            "kind": "error",
                            "message": (
                                f"Action #{action_id} -> {res_status}. "
                                f"Reason: {res.get('reason')}"
                            ),
                        }
                    else:
                        st.session_state["last_hitl_feedback"] = {
                            "kind": "warning",
                            "message": f"Action #{action_id} returned {res_status}: {res.get('message', res.get('reason', ''))}",
                        }
                    st.rerun()
            with c_reject:
                if st.button(
                    f"Reject #{action_id}",
                    key=f"{key_prefix}_reject_{action_id}",
                    width="stretch",
                ):
                    res = execute_human_rejection(
                        action_id=action_id,
                        rejection_reason=reject_reason.strip() or "Operator declined action",
                        approver=operator_name.strip() or "ControlRoom_Operator",
                    )
                    st.session_state["last_hitl_feedback"] = {
                        "kind": "warning",
                        "message": (
                            f"Action #{action_id} ({action_type} {target_id}) REJECTED "
                            f"by {res.get('approver')}. Reason: {res.get('reason')}"
                        ),
                    }
                    st.rerun()

    if show_history and resolved_rows:
        st.markdown("#### Recently Resolved Actions")
        hist_df = pd.DataFrame(
            [
                {
                    "ID": r["id"],
                    "Action": r["action_type"],
                    "Target": r["target_id"],
                    "Current": _format_actuator_value(r["action_type"], float(r["current_value"])),
                    "Proposed": _format_actuator_value(r["action_type"], float(r["proposed_value"])),
                    "Status": r["status"],
                    "Approver": r["approver"] or "-",
                    "Executed At (Asia/Dubai)": format_dubai_time(r["executed_at"]),
                }
                for r in resolved_rows
            ]
        )
        st.dataframe(hist_df, width="stretch", hide_index=True)
