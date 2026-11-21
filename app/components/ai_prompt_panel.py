"""Ask AquaGrid — Full-width multi-turn AI Chat Assistant with inline visual widgets & 'How I got this'."""
import html
import streamlit as st

from agent.chat_assistant import EXAMPLE_QUESTIONS, run_chat_turn
from config.constants import DEFAULT_SCADA_HOUR
from config.settings import settings
from simulator import get_simulator, simulator_step_lock
from components.badges import status_badge
from components.chat_widgets import render_chat_widget


def _handle_user_prompt(prompt_text: str, display_hour: int) -> None:
    """Append operator message, run multi-turn chat assistant through live tools & guards, append reply."""
    messages = st.session_state.setdefault("ask_aquagrid_messages", [])
    prior_history = [{"role": m["role"], "content": m.get("content", "")} for m in messages if m.get("role") in ("user", "assistant")]

    messages.append({"role": "user", "content": prompt_text})

    sim = get_simulator()
    with simulator_step_lock(sim):
        reply = run_chat_turn(
            user_message=prompt_text,
            history=prior_history,
            current_hour=display_hour,
        )

    # Sync scenario session state if the user typed "Run scenario X"
    meta = reply.get("scenario_meta")
    if meta:
        st.session_state["active_scenario"] = meta.get("scenario_num")
        st.session_state["active_scenario_id"] = meta.get("id")
        st.session_state["active_scenario_name"] = meta.get("name")
        st.session_state["last_agent_result"] = meta.get("result")
        st.session_state["last_scenario_info"] = meta.get("info")
        st.session_state["last_bypass_result"] = meta.get("bypass_result")
        st.session_state["last_pre_approved_refusal"] = meta.get("pre_approved_refusal")

    messages.append(reply)


def build_how_i_got_this_summary(msg: dict) -> dict:
    """Build guaranteed non-empty bullets and fallback summary for the 'How I got this' expander."""
    tool_bullets: list[str] = []
    for tc in (msg.get("tools_called") or []):
        if not isinstance(tc, dict):
            continue
        t_name = str(tc.get("tool") or "").strip()
        t_sum = str(tc.get("summary") or "").strip()
        if not t_name and not t_sum:
            continue
        if not t_name:
            t_name = "tool"
        if not t_sum:
            t_sum = "Executed successfully"
        tool_bullets.append(f"- `{t_name}` — {t_sum}")

    provider = str(msg.get("provider") or "System").strip() or "System"
    providers_tried = msg.get("providers_tried") or []
    fallback_notes: list[str] = []
    failed_providers = [
        pt for pt in providers_tried
        if isinstance(pt, dict) and str(pt.get("status", "")).upper().startswith("FAILED")
    ]
    retry_providers = [
        pt for pt in providers_tried
        if isinstance(pt, dict) and "retried" in str(pt.get("status", "")).lower() and pt.get("error")
    ]
    for fp in failed_providers:
        fp_name = str(fp.get("provider") or "Primary model").strip()
        fp_err = str(fp.get("error") or "unavailable").strip()
        fallback_notes.append(f"- `{fp_name}` failed ({fp_err}) -> fell back to `{provider}`")
    for rp in retry_providers:
        rp_name = str(rp.get("provider") or provider).strip()
        rp_err = str(rp.get("error") or "retried without tools").strip()
        fallback_notes.append(f"- `{rp_name}`: {rp_err}")

    return {
        "tool_bullets": tool_bullets,
        "fallback_notes": fallback_notes,
        "provider": provider,
    }


def render_ask_aquagrid_chat(display_hour: int = DEFAULT_SCADA_HOUR, snapshot: dict | None = None) -> None:
    """Render full-width Ask AquaGrid chat interface with inline maps, charts, tables, and approval cards."""
    snapshot = snapshot or {}
    show_tech = bool(st.session_state.get("show_tech_details", False))
    messages = st.session_state.setdefault("ask_aquagrid_messages", [])

    # Welcome prompt & quick-ask buttons when conversation is empty
    if not messages:
        st.markdown(
            """
            <div class="scada-card" style="text-align:center; padding:1.25rem;">
                <div style="font-size:1.15rem; font-weight:700; color:#F8FAFC;">Ask AquaGrid</div>
                <div style="font-size:0.86rem; color:#94A3B8; margin-top:0.3rem;">
                    Ask about live plant status, leaks, pressures, tariffs, schedules, past rejections, or propose a change.
                    Physical actions are always checked by Python safety guards and staged for your button approval.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Compact example question buttons
    ex_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for idx, q_text in enumerate(EXAMPLE_QUESTIONS):
        with ex_cols[idx]:
            if st.button(q_text, key=f"btn_example_q_{idx}", width="stretch"):
                _handle_user_prompt(q_text, display_hour)
                st.rerun()

    # Render conversation history
    for msg_idx, msg in enumerate(messages):
        role = msg.get("role", "assistant")
        kind = msg.get("kind", "text")
        avatar = "🚨" if kind == "alert" else None

        with st.chat_message(role, avatar=avatar):
            if kind == "alert":
                alert_code = msg.get("alert_code", "ALERT")
                alert_title = msg.get("alert_title") or "ALERT"
                raw_body = msg.get("alert_body") or msg.get("content", "")
                # Strip any stray markdown asterisks/backticks so raw syntax never appears in HTML alert cards
                clean_body = str(raw_body).replace("**", "").replace("`", "")
                st.markdown(
                    f'<div class="scada-card-alert">'
                    f'{status_badge(alert_code)} '
                    f'<strong>{html.escape(str(alert_title))}:</strong> '
                    f'{html.escape(clean_body)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg.get("content", ""))

            # Render any inline visual components attached to this message
            for w_idx, widget in enumerate(msg.get("widgets") or []):
                render_chat_widget(widget, widget_key=f"m{msg_idx}_w{w_idx}", snapshot=snapshot)

            if role == "assistant":
                provider = msg.get("provider", "System")
                st.markdown(
                    f'<div style="font-size:0.75rem; color:#64748B; margin-top:0.25rem;">Provider: {html.escape(str(provider))}</div>',
                    unsafe_allow_html=True,
                )

                hgt = build_how_i_got_this_summary(msg)
                tool_bullets = hgt["tool_bullets"]
                fallback_notes = hgt["fallback_notes"]
                code_checks = msg.get("code_checks") or []
                staged = msg.get("staged_action")
                providers_tried = msg.get("providers_tried") or []
                error_detail = msg.get("error_detail") or ""

                if tool_bullets or code_checks or staged or error_detail or fallback_notes or show_tech:
                    with st.expander("How I got this", expanded=show_tech):
                        if tool_bullets:
                            st.markdown("**Tools called:**\n" + "\n".join(tool_bullets))
                        if code_checks:
                            checks_lines = []
                            for chk in code_checks:
                                st_tag = chk.get("status", "PASS")
                                badge = status_badge("VALID" if st_tag == "PASS" else "BLOCKED").replace("VALID", "PASS")
                                code_val = str(chk.get("code") or "").strip()
                                code_html = f' <span class="ag-mono">({html.escape(code_val)})</span>' if code_val else ""
                                checks_lines.append(
                                    f'<div style="margin:0.2rem 0;">{badge} '
                                    f'<strong>{html.escape(str(chk.get("check", "")))}</strong>{code_html}: '
                                    f'{html.escape(str(chk.get("detail", "")))}</div>'
                                )
                            st.markdown("**Code checks:**", unsafe_allow_html=False)
                            st.markdown("".join(checks_lines), unsafe_allow_html=True)
                        if staged:
                            st.markdown(
                                f"**Gateway result:** `{staged.get('status')}` — "
                                f"{staged.get('action_type', '')} {staged.get('target_id', '')} "
                                f"({staged.get('reason') or staged.get('details') or staged.get('message') or ''})"
                            )
                        if fallback_notes:
                            st.markdown("**Model fallback:**\n" + "\n".join(fallback_notes))
                        if show_tech and error_detail:
                            st.error(f"**Technical Error:** `{error_detail}`")
                        if show_tech and providers_tried:
                            all_pt = [
                                f"- `{pt.get('provider')}`: `{pt.get('status')}`" + (f" — {pt['error']}" if pt.get("error") else "")
                                for pt in providers_tried
                                if isinstance(pt, dict) and pt.get("provider")
                            ]
                            if all_pt:
                                st.markdown("**All providers tried:**\n" + "\n".join(all_pt))
                        st.caption(f"Answered by: {provider} · LLM_MODE: {settings.LLM_MODE}")

    # Chat input bar at bottom
    user_typed = st.chat_input(
        "Ask AquaGrid anything (e.g. 'Show Sector 4 on the map', 'Close V-12', 'Run scenario 2')...",
        key="ask_aquagrid_chat_input",
    )
    if user_typed and user_typed.strip():
        _handle_user_prompt(user_typed.strip(), display_hour)
        st.rerun()


# Backward-compatible alias
render_ai_prompt_panel = render_ask_aquagrid_chat
