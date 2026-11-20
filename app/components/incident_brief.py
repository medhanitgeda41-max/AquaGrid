"""Clean Scenario Result summary with full guard breakdown tucked inside Show details."""
import html
import json
import streamlit as st
from database.connection import get_db_connection
from guards.injection_filter import detect_injection
from components.badges import format_dubai_time, section_header, status_badge


def _render_final_valve_v12_status() -> None:
    """Read and render live Valve V-12 status directly from SQLite."""
    conn = get_db_connection()
    try:
        v12 = conn.execute("SELECT valve_id, sector, status, last_updated FROM valves WHERE valve_id = 'V-12';").fetchone()
    finally:
        conn.close()

    if v12:
        card_cls = "scada-card-ok" if v12["status"] == "OPEN" else "scada-card-pending"
        st.markdown(
            f"""
            <div class="{card_cls}">
                <strong>{html.escape(v12['valve_id'])} ({html.escape(v12['sector'])})</strong>:
                {status_badge(v12['status'])}
                <span style="color:#94A3B8; font-size:0.82rem; margin-left:0.5rem;">
                    Updated: {html.escape(format_dubai_time(v12['last_updated']))}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_incident_brief() -> None:
    """Render concise Scenario Result up front; tuck verbose explanations & tables into Show details."""
    result = st.session_state.get("last_agent_result")
    scenario_id = st.session_state.get("active_scenario")
    scenario_name = st.session_state.get("active_scenario_name", "Scenario Result")
    bypass_result = st.session_state.get("last_bypass_result")
    pre_approved_refusal = st.session_state.get("last_pre_approved_refusal")
    scenario_info = st.session_state.get("last_scenario_info") or {}

    if not result:
        return

    gw_resp = result.get("gateway_response") or {}
    wq_comp = result.get("sector_water_quality_compliance") or {}
    res_st = result.get("reservoir_status") or {}
    tf = result.get("telemetry_freshness") or {}
    u55_refusal = scenario_info.get("setpoint_refusal")
    u54_token = scenario_info.get("token")
    flagged_alerts = [a for a in result.get("sanitized_alerts", []) if a.get("injection_detected")]
    flagged_tickets = [t for t in result.get("field_tickets", []) if t.get("command_or_approval_claim_detected")]

    # Determine headline status badge & card color (only red=danger, orange=pending, green=OK)
    if result.get("telemetry_loss"):
        top_status = "TELEMETRY_LOSS"
        top_cls = "scada-card-alert"
    elif flagged_alerts or bypass_result:
        top_status = "BLOCKED"
        top_cls = "scada-card-alert"
    elif pre_approved_refusal:
        top_status = pre_approved_refusal.get("status", "REJECTED_UNAUTHORIZED_APPROVAL_CLAIM")
        top_cls = "scada-card-pending"
    elif u54_token:
        top_status = "REJECTED_APPROVAL_MISMATCH"
        top_cls = "scada-card-alert"
    elif gw_resp.get("status") == "STAGED_PENDING_APPROVAL":
        top_status = "STAGED_PENDING_APPROVAL"
        top_cls = "scada-card-pending"
    elif wq_comp.get("alerts"):
        top_status = "NON_COMPLIANT"
        top_cls = "scada-card-alert"
    else:
        top_status = result.get("status", "COMPLETED")
        top_cls = "scada-card-ok"

    section_header(f"Result — {scenario_name}", extra_badge=status_badge(top_status))

    # ── Concise Operator Outcome Card ─────────────────────────────────────────
    st.markdown(
        f'<div class="{top_cls}">{html.escape(str(result.get("operator_brief", "")))}</div>',
        unsafe_allow_html=True,
    )

    # ── Key Guard Verdicts (compact, scannable, exact technical status codes) ─
    old_plan = scenario_info.get("old_plan")
    if res_st.get("has_active_emergency") and old_plan:
        st.markdown(
            f"""
            <div class="scada-card-pending">
                <div><strong>Old Plan:</strong>
                    <span class="ag-mono">DESAL-01 &rarr; {old_plan.get('proposed_mld')} MLD</span>
                    (min {old_plan.get('min_projected_percent')}%)
                    {status_badge(old_plan.get('recheck_status', 'CANCELLED_BY_SAFETY_RECHECK'))}
                </div>
                <div style="margin-top:0.3rem;"><strong>New Plan:</strong>
                    <span class="ag-mono">DESAL-01 &rarr; {gw_resp.get('proposed_value', 65.0)} MLD</span>
                    (min {gw_resp.get('min_projected_percent')}%)
                    {status_badge(gw_resp.get('status', 'STAGED_PENDING_APPROVAL'))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if u55_refusal:
        st.markdown(
            f"""
            <div class="scada-card-alert">
                <strong>Setpoint Request During Silence:</strong>
                <span class="ag-mono">SET_PLANT_OUTPUT DESAL-01 &rarr; 60.0 MLD</span>
                {status_badge(u55_refusal.get('status', 'REJECTED_STALE_TELEMETRY'))}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if pre_approved_refusal:
        conn_v17 = get_db_connection()
        try:
            v17 = conn_v17.execute("SELECT status FROM valves WHERE valve_id = 'V-17';").fetchone()
            v17_status = v17["status"] if v17 else "OPEN"
        finally:
            conn_v17.close()
        st.markdown(
            f"""
            <div class="scada-card-alert">
                <strong>Pre-Approved Call Refused:</strong>
                <span class="ag-mono">CLOSE_VALVE V-17 (pre_approved=True)</span>
                {status_badge(pre_approved_refusal.get('status', 'REJECTED_UNAUTHORIZED_APPROVAL_CLAIM'))}
                &nbsp;|&nbsp; V-17 {status_badge(v17_status)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if u54_token:
        replay_v18 = scenario_info.get("replay_v18") or {}
        replay_v17 = scenario_info.get("replay_v17") or {}
        conn_v = get_db_connection()
        try:
            v17_st = conn_v.execute("SELECT status FROM valves WHERE valve_id = 'V-17';").fetchone()["status"]
            v18_st = conn_v.execute("SELECT status FROM valves WHERE valve_id = 'V-18';").fetchone()["status"]
        finally:
            conn_v.close()
        st.markdown(
            f"""
            <div class="scada-card-alert">
                <div><strong>14:05 Executed:</strong> V-17 {status_badge(v17_st)} &nbsp;|&nbsp; V-18 {status_badge(v18_st)}</div>
                <div style="margin-top:0.3rem;"><strong>14:40 Replay V-18:</strong> {status_badge(replay_v18.get('status', 'REJECTED_APPROVAL_MISMATCH'))}</div>
                <div style="margin-top:0.25rem;"><strong>14:40 Replay V-17:</strong> {status_badge(replay_v17.get('status', 'REJECTED_APPROVAL_ALREADY_USED'))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if bypass_result:
        st.markdown(
            f"""
            <div class="scada-card-alert">
                <strong>Direct Write Bypass:</strong>
                <span class="ag-mono">{html.escape(str(bypass_result.get('attempted_action')))}</span>
                {status_badge(bypass_result.get('status', 'REFUSED'))}
                &nbsp;|&nbsp; V-12 {status_badge(str(bypass_result.get('valve_status_after', 'OPEN')))}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Detailed Breakdown (collapsed by default) ─────────────────────────────
    with st.expander("Show details", expanded=False):
        if result.get("maintenance_advisory"):
            st.markdown(f"**Maintenance Advisory:** {result['maintenance_advisory']}")

        if result.get("proposed_schedule"):
            st.markdown(f"**Proposed Schedule:** `{result['proposed_schedule']}`")

        sched_val = result.get("schedule_validation")
        if sched_val:
            sv_status = sched_val.get("status", "UNKNOWN")
            st.markdown(f"**Schedule Validation:** {status_badge(sv_status)}", unsafe_allow_html=True)
            card_cls = "scada-card-ok" if sv_status == "VALID" else "scada-card-alert"
            lines_html = [f"<div><strong>{html.escape(str(sched_val.get('summary', '')))}</strong></div>"]
            if sched_val.get("first_breach_hour"):
                lines_html.append(
                    f'<div style="margin-top:0.25rem; color:#FCA5A5;">'
                    f'<strong>First Floor Breach:</strong> '
                    f'<span class="ag-mono">{html.escape(str(sched_val.get("first_breach_hour")))} '
                    f'({sched_val.get("first_breach_percent")}%)</span></div>'
                )
            blocks = sched_val.get("chronological_blocks") or sched_val.get("blocks", [])
            for blk in blocks:
                blk_badge = status_badge(blk.get("status", "UNKNOWN"))
                win = html.escape(str(blk.get("window")))
                mld = blk.get("proposed_mld")
                start_pct = blk.get("start_projected_percent")
                min_pct = blk.get("min_projected_percent")
                end_pct = blk.get("end_projected_percent")
                reasons_html = "".join(
                    f'<div style="color:#FCA5A5; margin-left:1.25rem; font-size:0.84rem;">&rarr; {html.escape(str(r))}</div>'
                    for r in blk.get("reasons", [])
                )
                gap_html = (
                    f' <span style="color:#FCD34D; font-size:0.82rem;">[{html.escape(str(blk.get("gap_note")))}]</span>'
                    if blk.get("is_gap")
                    else ""
                )
                lines_html.append(
                    f'<div style="margin-top:0.35rem;">{blk_badge} '
                    f'<span class="ag-mono">{win} @ {mld} MLD</span>{gap_html} '
                    f'<span style="color:#94A3B8;">({start_pct}% &rarr; min {min_pct}% &rarr; {end_pct}%)</span>'
                    f'{reasons_html}</div>'
                )
            st.markdown(f'<div class="{card_cls}">{"".join(lines_html)}</div>', unsafe_allow_html=True)

        proposal_history = result.get("proposal_history", [])
        if proposal_history:
            st.markdown("**Proposal & Gateway History**")
            for entry in proposal_history:
                attempt = entry.get("attempt")
                prop = entry.get("action_proposal") or {}
                gw = entry.get("gateway_response") or {}
                gw_status = gw.get("status", "UNKNOWN")
                min_pct = gw.get("min_projected_percent")
                pct_html = f' | Min reservoir: <span class="ag-mono">{min_pct}%</span>' if min_pct is not None else ""
                guard_msg = gw.get("reason") or gw.get("message") or ""
                card_cls = "scada-card-pending" if gw_status == "STAGED_PENDING_APPROVAL" else "scada-card-alert"
                st.markdown(
                    f"""
                    <div class="{card_cls}">
                        <div>
                            <strong>Attempt #{attempt}:</strong>
                            <span class="ag-mono">{html.escape(str(prop.get('action_type')))}
                            {html.escape(str(prop.get('target_id')))} &rarr; {html.escape(str(prop.get('proposed_value')))}</span>
                            &nbsp;{status_badge(gw_status)}{pct_html}
                        </div>
                        <div style="margin-top:0.25rem; font-size:0.84rem; color:#CBD5E1;">{html.escape(str(guard_msg))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if scenario_id == 4 or flagged_alerts:
            st.markdown("**Injection Shield**")
            conn = get_db_connection()
            try:
                raw_rows = conn.execute(
                    "SELECT id, raw_text FROM quality_alerts WHERE is_injection = 1 ORDER BY id DESC;"
                ).fetchall()
                raw_by_id = {r["id"]: r["raw_text"] for r in raw_rows}
            finally:
                conn.close()
            last_prompt = result.get("last_user_prompt", "")
            for alert in flagged_alerts:
                a_id = alert.get("alert_id")
                raw_text = raw_by_id.get(a_id, "")
                _, matched_patterns = detect_injection(raw_text)
                raw_reached_llm = bool(raw_text) and (raw_text in last_prompt)
                confirm_text = "YES (BREACH!)" if raw_reached_llm else "NO — quarantined before model context"
                st.markdown(
                    f"""
                    <div class="scada-card-alert">
                        <div><strong>Raw Alert:</strong> <span class="ag-mono" style="color:#FCA5A5;">{html.escape(raw_text)}</span></div>
                        <div style="margin-top:0.25rem;"><strong>Patterns:</strong> <span class="ag-mono">{html.escape(str(matched_patterns))}</span></div>
                        <div style="margin-top:0.25rem;"><strong>Sanitized:</strong> <span class="ag-mono">{html.escape(json.dumps(alert))}</span></div>
                        <div style="margin-top:0.25rem;"><strong>Reached Model?</strong> <span style="color:#6EE7B7;">{html.escape(confirm_text)}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if flagged_tickets:
            st.markdown("**Untrusted Field Tickets**")
            for ft in flagged_tickets:
                st.markdown(
                    f"""
                    <div class="scada-card-pending">
                        {status_badge("UNTRUSTED_TEXT")} <strong>{html.escape(str(ft.get('ticket_ref')))} ({html.escape(str(ft.get('sector')))})</strong>
                        <div style="margin-top:0.25rem;" class="ag-mono">{html.escape(str(ft.get('quoted_data')))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if u54_token:
            st.markdown("**Approval Token Metadata**")
            st.markdown(
                f"""
                <div class="scada-card">
                    <div class="ag-mono">
                        token_id={html.escape(str(u54_token['token_id']))} |
                        action=#{u54_token['action_id']} ({html.escape(str(u54_token['action_type']))} {html.escape(str(u54_token['target']))}) |
                        operator={html.escape(str(u54_token['operator']))}<br/>
                        issued={html.escape(format_dubai_time(u54_token['issued_at']))} |
                        expires={html.escape(format_dubai_time(u54_token['expires_at']))} |
                        used={html.escape(format_dubai_time(u54_token['used_at']))} |
                        hash={html.escape(str(u54_token['params_hash'])[:16])}...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if tf.get("banner"):
            st.markdown(f"**Telemetry Freshness:** `{tf.get('status')}` — {tf.get('banner')}")

        _render_final_valve_v12_status()
