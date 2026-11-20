"""Incident Brief & Pure-Python Guard Report."""
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

    section_header("Final Valve V-12 Status", ["CODE", "HUMAN"])
    if v12:
        card_cls = "scada-card-ok" if v12["status"] == "OPEN" else "scada-card-pending"
        st.markdown(
            f"""
            <div class="{card_cls}">
                <strong>{html.escape(v12['valve_id'])} ({html.escape(v12['sector'])})</strong>:
                {status_badge(v12['status'])}
                <span style="color:#9FB3C8; font-size:0.84rem; margin-left:0.5rem;">
                    Last updated: {html.escape(format_dubai_time(v12['last_updated']))}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_incident_brief() -> None:
    """Render the scenario run report with original technical labels and status codes."""
    result = st.session_state.get("last_agent_result")
    scenario_id = st.session_state.get("active_scenario")
    bypass_result = st.session_state.get("last_bypass_result")

    if not result:
        st.info("Select one of the 4 scenario buttons above to run the AquaGrid agent.")
        _render_final_valve_v12_status()
        return

    # 0. Per-Sector Water Quality Non-Compliance Alert (U-5.1)
    wq_comp = result.get("sector_water_quality_compliance") or {}
    if wq_comp.get("alerts"):
        excl = ", ".join(wq_comp.get("excluded_from_tariff_scheduling", []))
        section_header("Sector Water Quality Guard", ["CODE"], extra_badge=status_badge("NON_COMPLIANT"))
        for alert_msg in wq_comp["alerts"]:
            st.error(f"**{alert_msg}** — Removed from tariff-driven scheduling: {excl}")

    # 1. Operator Brief & Maintenance Advisory / Work Orders
    b_col, m_col = st.columns(2)
    with b_col:
        section_header("Operator Action Brief", ["LLM"])
        st.markdown(
            f'<div class="scada-card">{html.escape(str(result.get("operator_brief", "")))}</div>',
            unsafe_allow_html=True,
        )
    with m_col:
        section_header("Maintenance Advisory / Work Orders", ["LLM"])
        st.markdown(
            f'<div class="scada-card">{html.escape(str(result.get("maintenance_advisory", "")))}</div>',
            unsafe_allow_html=True,
        )

    # 2. Proposed Production Schedule + Schedule Validation
    section_header("Proposed Production Schedule", ["LLM"])
    st.markdown(
        f'<div class="scada-card ag-mono">{html.escape(str(result.get("proposed_schedule", "")))}</div>',
        unsafe_allow_html=True,
    )

    sched_val = result.get("schedule_validation")
    if sched_val:
        sv_status = sched_val.get("status", "UNKNOWN")
        section_header(
            "Schedule Validation (Pure Python Guard)",
            ["CODE"],
            extra_badge=status_badge(sv_status),
        )
        card_cls = "scada-card-ok" if sv_status == "VALID" else "scada-card-alert"
        lines_html = [f"<div><strong>{html.escape(str(sched_val.get('summary', '')))}</strong></div>"]
        if sched_val.get("first_breach_hour"):
            lines_html.append(
                f'<div style="margin-top:0.25rem; color:#FCA5A5;">'
                f'<strong>First 35.0% Floor Breach:</strong> '
                f'<span class="ag-mono">{html.escape(str(sched_val.get("first_breach_hour")))} '
                f'({sched_val.get("first_breach_percent")}%)</span></div>'
            )
        blocks = sched_val.get("chronological_blocks") or sched_val.get("blocks", [])
        if blocks:
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
                    f' <span style="color:#FDE68A; font-size:0.82rem;">[{html.escape(str(blk.get("gap_note")))}]</span>'
                    if blk.get("is_gap")
                    else ""
                )
                lines_html.append(
                    f'<div style="margin-top:0.4rem;">{blk_badge} '
                    f'<span class="ag-mono">{win} @ {mld} MLD</span>{gap_html} '
                    f'<span style="color:#9FB3C8;">(start: {start_pct}% &rarr; min: {min_pct}% &rarr; end: {end_pct}%)</span>'
                    f'{reasons_html}</div>'
                )
        else:
            for blk in sched_val.get("rejected_blocks", []):
                for r in blk.get("reasons", []):
                    lines_html.append(
                        f'<div style="margin-top:0.35rem; color:#FCA5A5;">{status_badge("REJECTED_BY_CODE")} {html.escape(str(r))}</div>'
                    )
        st.markdown(f'<div class="{card_cls}">{"".join(lines_html)}</div>', unsafe_allow_html=True)

    # 3. Proposal & Gateway History
    proposal_history = result.get("proposal_history", [])
    if proposal_history:
        section_header("Proposal & Gateway History", ["LLM", "CODE"])
        for entry in proposal_history:
            attempt = entry.get("attempt")
            prop = entry.get("action_proposal") or {}
            gw = entry.get("gateway_response") or {}
            gw_status = gw.get("status", "UNKNOWN")
            min_pct = gw.get("min_projected_percent")
            pct_html = (
                f' &nbsp;|&nbsp; <strong>Min Projected Reservoir:</strong> <span class="ag-mono">{min_pct}%</span>'
                if min_pct is not None
                else ""
            )
            guard_msg = gw.get("reason") or gw.get("message") or ""
            card_cls = "scada-card-pending" if gw_status == "STAGED_PENDING_APPROVAL" else "scada-card-alert"
            st.markdown(
                f"""
                <div class="{card_cls}">
                    <div>
                        <strong>Attempt #{attempt}:</strong>
                        <span class="ag-mono">{html.escape(str(prop.get('action_type')))}
                        {html.escape(str(prop.get('target_id')))} &rarr; {html.escape(str(prop.get('proposed_value')))}</span>
                        &nbsp;&nbsp;{status_badge(gw_status)}{pct_html}
                    </div>
                    <div style="margin-top:0.35rem; font-size:0.86rem;">
                        <strong>Rationale:</strong> {html.escape(str(prop.get('rationale', '')))}
                    </div>
                    <div style="margin-top:0.25rem; font-size:0.86rem; color:#CBD5E1;">
                        <strong>Safety Guard Verdict:</strong> {html.escape(str(guard_msg))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 4. Injection Shield
    sanitized_alerts = result.get("sanitized_alerts", [])
    flagged_alerts = [a for a in sanitized_alerts if a.get("injection_detected")]
    if scenario_id == 4 or flagged_alerts:
        section_header("Injection Shield", ["CODE"], extra_badge=status_badge("BLOCKED"))
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
            leaked_fragments = [
                frag
                for frag in ("SYSTEM OVERRIDE", "ignore safety", "open valve", "V-12", "emergency flushing")
                if frag.lower() in last_prompt.lower()
            ]
            confirm_text = (
                "YES (BREACH!)"
                if (raw_reached_llm or leaked_fragments)
                else "NO — CONFIRMED (quarantined before model context)"
            )
            st.markdown(
                f"""
                <div class="scada-card-alert">
                    <div><strong>Alert Metadata:</strong> <span class="ag-mono">id={html.escape(str(a_id))}, source={html.escape(str(alert.get('source')))}, severity={html.escape(str(alert.get('severity')))}</span></div>
                    <div style="margin-top:0.35rem;"><strong>Quarantined Alert:</strong><br/>
                        <span class="ag-mono" style="color:#FCA5A5;">{html.escape(raw_text)}</span>
                    </div>
                    <div style="margin-top:0.35rem;"><strong>Matched Pattern(s):</strong><br/>
                        <span class="ag-mono" style="color:#FDE68A;">{html.escape(str(matched_patterns))}</span>
                    </div>
                    <div style="margin-top:0.35rem;"><strong>Sanitized Payload:</strong><br/>
                        <span class="ag-mono" style="color:#7CE8FF;">{html.escape(json.dumps(alert))}</span>
                    </div>
                    <div style="margin-top:0.4rem;"><strong>Raw Text Reached Model?</strong>
                        <span style="color:#6EE7B7; font-weight:700;">{html.escape(confirm_text)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 5. Direct Write Bypass Attempt
    if bypass_result:
        section_header(
            "Direct Write Bypass Attempt",
            ["CODE"],
            extra_badge=status_badge(bypass_result.get("status", "REFUSED")),
        )
        st.markdown(
            f"""
            <div class="scada-card-alert">
                <div><strong>Attempted Bypass:</strong> <span class="ag-mono">{html.escape(str(bypass_result.get('attempted_action')))}</span></div>
                <div style="margin-top:0.3rem;"><strong>Result:</strong> {status_badge(bypass_result.get('status', 'REFUSED'))}
                    <span class="ag-mono" style="color:#FCA5A5;">{html.escape(str(bypass_result.get('error')))}</span>
                </div>
                <div style="margin-top:0.3rem;"><strong>DB Integrity:</strong>
                    <span class="ag-mono">V-12 before={html.escape(str(bypass_result.get('valve_status_before')))} &rarr; after={html.escape(str(bypass_result.get('valve_status_after')))} (unchanged={bypass_result.get('unchanged')})</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 6. Final Valve V-12 Status
    _render_final_valve_v12_status()
