"""Clean Pending Approvals panel with 1-click Approve / Reject and collapsed details."""
import html
import pandas as pd
import streamlit as st
from database.connection import get_db_connection
from tools.write_gateway import execute_human_approval, execute_human_rejection
from components.badges import format_dubai_time, section_header, status_badge


def _format_actuator_value(action_type: str, value: float) -> str:
    """Format actuator values cleanly for control-room display."""
    if action_type in ("CLOSE_VALVE", "OPEN_VALVE"):
        return "OPEN" if float(value) >= 0.5 else "CLOSED"
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
    """Render PENDING actions with Approve / Reject buttons; hide history & extra inputs in Show details."""
    section_header("Pending Approvals")
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
            '<div class="scada-card-ok"><strong>No pending approvals.</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        operator_name = st.session_state.get("operator_signature", "Lead_Operator_SCADA")
        with st.expander("Show details — Operator Signature", expanded=False):
            operator_name = st.text_input(
                "Operator Signature",
                value=operator_name,
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
            prov = row["provenance"] if "provenance" in row.keys() and row["provenance"] else "AGENT_ANALYSIS"
            prov_ref = row["provenance_ref"] if "provenance_ref" in row.keys() and row["provenance_ref"] else ""
            prov_text = f"{prov} ({prov_ref})" if prov_ref else prov
            prov_badge_cls = "ag-badge-warn" if prov == "UNTRUSTED_TEXT" else "ag-badge-neutral"

            st.markdown(
                f"""
                <div class="scada-card-pending">
                    <div>
                        {status_badge("PENDING")}
                        <span class="ag-badge {prov_badge_cls}">{html.escape(prov_text)}</span>
                        <strong>#{action_id}</strong>
                        <span class="ag-mono">{html.escape(action_type)} {html.escape(target_id)}: {html.escape(curr_fmt)} &rarr; {html.escape(prop_fmt)}</span>
                    </div>
                    <div style="margin-top:0.35rem; font-size:0.86rem; color:#CBD5E1;">
                        {html.escape(str(rationale))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c_approve, c_reject, c_spacer = st.columns([1.0, 1.0, 2.5])
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
                            "message": f"Action #{action_id} ({action_type} {target_id}) EXECUTED: {detail}",
                        }
                    elif res_status in ("CANCELLED_BY_SAFETY_RECHECK", "CANCELLED_STALE_TELEMETRY"):
                        st.session_state["last_hitl_feedback"] = {
                            "kind": "error",
                            "message": f"Action #{action_id} -> {res_status}: {res.get('reason')}",
                        }
                    else:
                        st.session_state["last_hitl_feedback"] = {
                            "kind": "warning",
                            "message": f"Action #{action_id} -> {res_status}: {res.get('message', res.get('reason', ''))}",
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
                        rejection_reason="Operator declined physical actuation",
                        approver=operator_name.strip() or "ControlRoom_Operator",
                    )
                    st.session_state["last_hitl_feedback"] = {
                        "kind": "warning",
                        "message": f"Action #{action_id} ({action_type} {target_id}) REJECTED by {res.get('approver')}.",
                    }
                    st.rerun()

    if show_history and resolved_rows:
        with st.expander(f"Show details — Resolved Actions ({len(resolved_rows)})", expanded=False):
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
                        "Time": format_dubai_time(r["executed_at"]),
                    }
                    for r in resolved_rows
                ]
            )
            st.dataframe(hist_df, width="stretch", hide_index=True)
