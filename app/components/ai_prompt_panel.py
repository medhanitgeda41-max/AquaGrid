"""AI Prompt Area: operator request input, step-by-step trace, and suggest-only gated result."""
import html
import streamlit as st

from agent.operator_prompt import run_operator_prompt
from config.constants import DEFAULT_SCADA_HOUR
from components.badges import section_header, status_badge
from simulator import get_simulator


def _format_proposed_action(action: dict | None) -> str:
    if not action:
        return "NONE (advisory only)"
    a_type = str(action.get("action_type", "NONE"))
    target = str(action.get("target_id", "NONE"))
    val = action.get("proposed_value")
    if a_type in ("CLOSE_VALVE", "OPEN_VALVE"):
        val_txt = "OPEN" if float(val or 0.0) >= 0.5 else "CLOSED"
        return f"{a_type} {target} ({val_txt})"
    if a_type == "ISOLATE_AND_FLUSH":
        return f"{a_type} {target}"
    return f"{a_type} {target} -> {val} MLD"


def render_ai_prompt_panel(display_hour: int = DEFAULT_SCADA_HOUR) -> None:
    """Render the operator AI prompt box, Run button, step-by-step trace, and final result."""
    section_header("AI Prompt")

    with st.form("ai_prompt_form", clear_on_submit=False):
        in_col, btn_col = st.columns([5.2, 1.0])
        with in_col:
            prompt_text = st.text_input(
                "Ask AI",
                placeholder="Ask a question or suggest an action (e.g. Check grid status, Optimize for peak tariff, Close V-12)...",
                key="ai_prompt_input",
                label_visibility="collapsed",
            )
        with btn_col:
            submitted = st.form_submit_button("Run", type="primary", width="stretch")

    if submitted:
        if not (prompt_text or "").strip():
            st.warning("Type a request in the box above and press Run.")
        else:
            sim = get_simulator()
            with st.spinner("Reading SCADA data, running code checks, asking AI..."):
                with sim.step_lock:
                    prompt_res = run_operator_prompt(
                        prompt_text=prompt_text,
                        current_hour=display_hour,
                    )
            st.session_state["last_ai_prompt_result"] = prompt_res
            st.rerun()

    res = st.session_state.get("last_ai_prompt_result")
    if not res:
        return

    final = res.get("final_result") or {}
    sev = final.get("severity", "ok")
    card_cls = {
        "ok": "scada-card-ok",
        "pending": "scada-card-pending",
        "danger": "scada-card-alert",
    }.get(sev, "scada-card")

    # ── Clear Final Result ────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="{card_cls}">
            <div>
                {status_badge(final.get('status', 'COMPLETED'))}
                <strong>{html.escape(str(final.get('headline', 'Done')))}</strong>
            </div>
            <div style="margin-top:0.35rem; font-size:0.9rem; color:#F8FAFC;">
                {html.escape(str(final.get('summary', '')))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Step-by-Step Trace ────────────────────────────────────────────────────
    step_read = res.get("step_read") or {}
    step_checks = res.get("step_checks") or []
    step_prop = res.get("step_proposal") or {}

    if step_read:
        peak_lbl = "PEAK" if step_read.get("is_peak") else "OFF/MID-PEAK"
        excl = step_read.get("excluded_sectors") or []
        excl_txt = f" | Non-compliant: {', '.join(excl)}" if excl else ""
        read_summary = (
            f"Reservoir {step_read.get('reservoir_pct', 0):.1f}% (floor {step_read.get('safety_floor_pct', 35.0):.1f}%) · "
            f"Plant {step_read.get('plant_mld', 0):.0f} MLD · "
            f"Tariff ${step_read.get('tariff_rate', 0):.2f}/kWh ({peak_lbl}) · "
            f"Leaks: {step_read.get('active_leaks', 0)} · "
            f"Quarantined: {step_read.get('quarantined_sensors', 0)} · "
            f"Telemetry: {step_read.get('telemetry_status', 'FRESH')}"
            f"{excl_txt}"
        )

        checks_html = "".join(
            f'<div style="margin-top:0.25rem;">'
            f'{status_badge(chk.get("status", "UNKNOWN"))} '
            f'<strong>{html.escape(str(chk.get("check", "")))}:</strong> '
            f'{html.escape(str(chk.get("detail", "")))}'
            f'</div>'
            for chk in step_checks
        )

        action_obj = step_prop.get("action")
        gw = step_prop.get("gateway_response") or {}
        prop_txt = _format_proposed_action(action_obj)
        gw_badge = status_badge(gw["status"]) if gw.get("status") else status_badge("ADVISORY_ONLY")
        gw_note = (
            "Waiting for operator Approve / Reject — no physical change made."
            if gw.get("status") == "STAGED_PENDING_APPROVAL"
            else str(gw.get("reason") or gw.get("message") or "No physical change proposed.")
        )

        st.markdown(
            f"""
            <div class="scada-step">
                <div class="scada-step-title">1. What AI Read</div>
                <div>{html.escape(read_summary)}</div>
            </div>
            <div class="scada-step">
                <div class="scada-step-title">2. What Code Checked</div>
                {checks_html}
            </div>
            <div class="scada-step">
                <div class="scada-step-title">3. What AI Proposed</div>
                <div>
                    {gw_badge}
                    <span class="ag-mono">{html.escape(prop_txt)}</span>
                </div>
                <div style="margin-top:0.25rem; font-size:0.84rem; color:#CBD5E1;">
                    {html.escape(gw_note)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Extra Details (collapsed by default) ──────────────────────────────────
    with st.expander("Show details", expanded=False):
        st.markdown(f"**Prompt:** `{res.get('prompt', '')}`")
        if res.get("operator_brief"):
            st.markdown(f"**AI Brief:** {res['operator_brief']}")
        if res.get("maintenance_advisory"):
            st.markdown(f"**Maintenance Advisory:** {res['maintenance_advisory']}")
        if res.get("proposed_schedule"):
            st.markdown(f"**Proposed Schedule:** `{res['proposed_schedule']}`")
        sv = res.get("schedule_validation")
        if sv:
            st.markdown(f"**Schedule Validation:** `{sv.get('status')}` — {sv.get('summary', '')}")
        st.caption(f"Provider: {res.get('llm_provider', '-')}")
