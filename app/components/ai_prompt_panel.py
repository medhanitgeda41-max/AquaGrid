"""Ask AquaGrid — Full-width multi-turn AI Chat Assistant with live Thinking status, streaming, inline widgets & 'How I got this'."""
import html
import importlib
import inspect
import logging
import re
import time
import traceback
import streamlit as st

import agent.chat_assistant as _chat_assistant_mod
from agent.chat_assistant import EXAMPLE_QUESTIONS, run_chat_turn, run_scenario_for_chat
from agent.llm_factory import provider_supports_streaming
from config.constants import DEFAULT_SCADA_HOUR
from config.settings import settings
from simulator import get_simulator, simulator_step_lock
from components.badges import status_badge
from components.chat_widgets import render_chat_widget

logger = logging.getLogger("aquagrid.ui.chat")


def _call_with_optional_status(fn, status_callback, **kwargs):
    """
    Invoke run_chat_turn / run_scenario_for_chat safely even if a Streamlit hot-reload left an older
    signature cached in sys.modules or a unit test mocked the function without status_callback.
    """
    target = getattr(fn, "side_effect", None)
    check_fn = target if callable(target) else fn
    try:
        params = inspect.signature(check_fn).parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        accepts_status = ("status_callback" in params) or has_var_kw
    except (TypeError, ValueError):
        accepts_status = False

    # If the real module function is stale after hot-reload, reload agent.chat_assistant once
    if not accepts_status and target is None and getattr(fn, "__module__", "") == "agent.chat_assistant":
        reloaded = importlib.reload(_chat_assistant_mod)
        fresh_fn = getattr(reloaded, getattr(fn, "__name__", ""), fn)
        fn = fresh_fn
        try:
            params = inspect.signature(fn).parameters
            accepts_status = "status_callback" in params
        except (TypeError, ValueError):
            accepts_status = False

    if accepts_status and status_callback is not None:
        kwargs["status_callback"] = status_callback
    return fn(**kwargs)


def is_chat_generating(session_state: dict) -> bool:
    """Return True while an answer or scenario is currently being generated."""
    return bool(session_state.get("_chat_generating", False))


def queue_chat_request(session_state: dict, prompt_text: str, display_hour: int = DEFAULT_SCADA_HOUR) -> bool:
    """
    Append the operator's message to conversation history BEFORE calling the LLM and mark generation active.
    Returns False (ignoring double submit) if a reply is already being generated.
    """
    cleaned = (prompt_text or "").strip()
    if not cleaned:
        return False
    if session_state.get("_chat_generating", False):
        session_state["_chat_busy_notice"] = "Please wait for the current answer"
        return False

    messages = session_state.setdefault("ask_aquagrid_messages", [])
    prior_history = [
        {"role": m["role"], "content": m.get("content", "")}
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]
    # Append user message BEFORE the LLM call
    messages.append({"role": "user", "content": cleaned})
    session_state["_chat_generating"] = True
    session_state["_pending_chat_request"] = {
        "kind": "chat",
        "text": cleaned,
        "display_hour": display_hour,
        "prior_history": prior_history,
    }
    session_state.pop("_chat_busy_notice", None)
    return True


def queue_scenario_request(
    session_state: dict,
    scenario_id: str,
    title: str,
    display_hour: int = DEFAULT_SCADA_HOUR,
) -> bool:
    """
    Append a scenario run user message BEFORE executing the scenario and mark generation active.
    Returns False (ignoring double submit) if a reply is already being generated.
    """
    if session_state.get("_chat_generating", False):
        session_state["_chat_busy_notice"] = "Please wait for the current answer"
        return False

    messages = session_state.setdefault("ask_aquagrid_messages", [])
    messages.append({"role": "user", "content": f"Run {scenario_id}: {title}"})
    session_state["_chat_generating"] = True
    session_state["_pending_chat_request"] = {
        "kind": "scenario",
        "scenario_id": scenario_id,
        "title": title,
        "display_hour": display_hour,
    }
    session_state.pop("_chat_busy_notice", None)
    return True


def word_stream_generator(text: str, max_total_seconds: float = 0.6):
    """Yield text word-by-word (preserving markdown spacing/newlines) as a pure generator for st.write_stream."""
    safe_text = "" if text is None else (text if isinstance(text, str) else str(text))
    if not safe_text:
        return
    tokens = re.findall(r"\S+\s*|\n+", safe_text)
    delay = min(0.015, max_total_seconds / max(1, len(tokens))) if max_total_seconds > 0 else 0.0
    for tok in tokens:
        yield tok
        if delay > 0:
            time.sleep(delay)


def _render_text_response(content, provider: str, kind: str, stream_text: bool) -> None:
    """Stream text word-by-word via st.write_stream when supported; otherwise render with st.markdown."""
    safe_text = "" if content is None else (content if isinstance(content, str) else str(content))
    if (
        stream_text
        and kind == "text"
        and safe_text.strip()
        and provider_supports_streaming(str(provider or ""))
        and hasattr(st, "write_stream")
    ):
        try:
            st.write_stream(word_stream_generator(safe_text))
            return
        except Exception:
            logging.exception("st.write_stream failed; falling back to st.markdown")
    st.markdown(safe_text)


def _sync_scenario_state(reply: dict) -> None:
    """Sync scenario metadata into st.session_state when a scenario completes."""
    meta = reply.get("scenario_meta")
    if meta:
        st.session_state["active_scenario"] = meta.get("scenario_num")
        st.session_state["active_scenario_id"] = meta.get("id")
        st.session_state["active_scenario_name"] = meta.get("name")
        st.session_state["last_agent_result"] = meta.get("result")
        st.session_state["last_scenario_info"] = meta.get("info")
        st.session_state["last_bypass_result"] = meta.get("bypass_result")
        st.session_state["last_pre_approved_refusal"] = meta.get("pre_approved_refusal")


def _handle_user_prompt(prompt_text: str, display_hour: int) -> bool:
    """Backward-compatible helper: queue user prompt (before LLM), run chat turn, append reply. Ignores double submits."""
    if not queue_chat_request(st.session_state, prompt_text, display_hour):
        return False
    pending = st.session_state.get("_pending_chat_request")
    try:
        sim = get_simulator()
        with simulator_step_lock(sim):
            reply = run_chat_turn(
                user_message=pending["text"],
                history=pending.get("prior_history") or [],
                current_hour=display_hour,
            )
    finally:
        st.session_state["_pending_chat_request"] = None
        st.session_state["_chat_generating"] = False

    _sync_scenario_state(reply)
    st.session_state.setdefault("ask_aquagrid_messages", []).append(reply)
    return True


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


def _render_assistant_message_contents(
    msg: dict,
    msg_idx: int,
    snapshot: dict,
    show_tech: bool,
    stream_text: bool = False,
) -> None:
    """Render an assistant message's text (streamed when supported), inline widgets, provider line, and 'How I got this'."""
    kind = msg.get("kind", "text")
    raw_content = msg.get("content")
    content = "" if raw_content is None else (raw_content if isinstance(raw_content, str) else str(raw_content))
    provider = str(msg.get("provider") or "System")

    if kind == "alert":
        alert_code = str(msg.get("alert_code") or "ALERT")
        alert_title = str(msg.get("alert_title") or "ALERT")
        raw_body = msg.get("alert_body") if msg.get("alert_body") is not None else content
        clean_body = str(raw_body).replace("**", "").replace("`", "")
        st.markdown(
            f'<div class="scada-card-alert">'
            f'{status_badge(alert_code)} '
            f'<strong>{html.escape(alert_title)}:</strong> '
            f'{html.escape(clean_body)}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        _render_text_response(content=content, provider=provider, kind=kind, stream_text=stream_text)

    # Render any inline visual components attached to this message
    for w_idx, widget in enumerate(msg.get("widgets") or []):
        render_chat_widget(widget, widget_key=f"m{msg_idx}_w{w_idx}", snapshot=snapshot)

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


def _execute_pending_request(pending: dict, snapshot: dict, show_tech: bool) -> bool:
    """Run a queued chat or scenario request inside an assistant message with live st.status('Thinking...') feedback."""
    steps_logged: list[str] = []

    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=True) as status_box:
            def _on_status(step_msg: str) -> None:
                if step_msg:
                    steps_logged.append(step_msg)
                    status_box.write(step_msg)
                    status_box.update(label=step_msg, state="running")

            try:
                sim = get_simulator()
                if pending.get("kind") == "scenario":
                    sc_id = pending["scenario_id"]
                    if sc_id != "U-5.5" and sim is not None and hasattr(sim, "resume_feed"):
                        sim.resume_feed()
                    with simulator_step_lock(sim):
                        reply = _call_with_optional_status(
                            run_scenario_for_chat,
                            _on_status,
                            scenario_key=sc_id,
                            current_hour=pending.get("display_hour", DEFAULT_SCADA_HOUR),
                        )
                else:
                    with simulator_step_lock(sim):
                        reply = _call_with_optional_status(
                            run_chat_turn,
                            _on_status,
                            user_message=pending["text"],
                            history=pending.get("prior_history") or [],
                            current_hour=pending.get("display_hour", DEFAULT_SCADA_HOUR),
                        )

                if not isinstance(reply, dict):
                    reply = {
                        "role": "assistant",
                        "kind": "text",
                        "content": "" if reply is None else str(reply),
                        "provider": "System",
                        "tools_called": [],
                        "code_checks": [],
                        "widgets": [],
                        "staged_action": None,
                    }

                if reply.get("error") or reply.get("kind") == "error":
                    err_line = str(reply.get("error_detail") or reply.get("content") or "Request failed")
                    status_box.write(f"Error: {err_line}")
                    status_box.update(label="Error", state="error", expanded=False)
                else:
                    status_box.update(label="Done", state="complete", expanded=False)
            except Exception as exc:
                logging.exception("Live chat request failed unexpectedly")
                tb_frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
                last_frame = tb_frames[-1] if tb_frames else None
                location = f"{last_frame.filename}:{last_frame.lineno}" if last_frame else "unknown"
                err_detail = f"{type(exc).__name__}: {exc} ({location})"
                status_box.write(f"Error: {type(exc).__name__}: {exc}")
                status_box.update(label="Error", state="error", expanded=False)
                reply = {
                    "role": "assistant",
                    "kind": "error",
                    "content": f"Sorry, an unexpected error occurred ({type(exc).__name__}). All safety guards remain active.",
                    "provider": "Error Handler",
                    "tools_called": [],
                    "code_checks": [],
                    "widgets": [],
                    "staged_action": None,
                    "error_detail": err_detail,
                    "error": True,
                }
            finally:
                st.session_state["_pending_chat_request"] = None
                st.session_state["_chat_generating"] = False

        reply["thinking_steps"] = steps_logged
        _sync_scenario_state(reply)

        messages = st.session_state.setdefault("ask_aquagrid_messages", [])
        new_msg_idx = len(messages)
        messages.append(reply)

        _render_assistant_message_contents(
            msg=reply,
            msg_idx=new_msg_idx,
            snapshot=snapshot,
            show_tech=show_tech,
            stream_text=True,
        )

    return True


def render_ask_aquagrid_chat(display_hour: int = DEFAULT_SCADA_HOUR, snapshot: dict | None = None) -> bool:
    """
    Render full-width Ask AquaGrid chat interface with immediate user messages, live Thinking status,
    word-by-word streaming, inline maps/charts/tables, and button-only approvals.
    Returns True if a new chat/scenario turn was executed during this render pass.
    """
    snapshot = snapshot or {}
    show_tech = bool(st.session_state.get("show_tech_details", False))
    messages = st.session_state.setdefault("ask_aquagrid_messages", [])
    executed_turn = False

    # Surface busy notice if operator clicked while another reply was still generating
    busy_notice = st.session_state.pop("_chat_busy_notice", None)
    if busy_notice:
        st.warning(busy_notice)
    elif is_chat_generating(st.session_state) and not st.session_state.get("_pending_chat_request"):
        st.warning("Please wait for the current answer")

    # Welcome prompt when conversation is empty
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

    # Compact example question buttons (ignored when an answer is already generating)
    ex_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for idx, q_text in enumerate(EXAMPLE_QUESTIONS):
        with ex_cols[idx]:
            if st.button(q_text, key=f"btn_example_q_{idx}", width="stretch"):
                if not queue_chat_request(st.session_state, q_text, display_hour):
                    st.warning("Please wait for the current answer")

    # Render full conversation history (includes any newly queued user message BEFORE calling the LLM!)
    for msg_idx, msg in enumerate(messages):
        role = msg.get("role", "assistant")
        kind = msg.get("kind", "text")
        avatar = "🚨" if kind == "alert" else None

        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                _render_assistant_message_contents(
                    msg=msg,
                    msg_idx=msg_idx,
                    snapshot=snapshot,
                    show_tech=show_tech,
                    stream_text=False,
                )
            else:
                st.markdown(msg.get("content", ""))

    # Container right below conversation history for the active turn (live user bubble + live Thinking assistant)
    live_turn_container = st.container()

    # If a sidebar Scenario Run or quick-question button queued a pending request, run it now with live status!
    pending_top = st.session_state.get("_pending_chat_request")
    if pending_top:
        with live_turn_container:
            executed_turn = _execute_pending_request(pending_top, snapshot, show_tech)

    # Chat input bar at bottom
    user_typed = st.chat_input(
        "Ask AquaGrid anything (e.g. 'Show Sector 4 on the map', 'Close V-12', 'Run scenario 2')...",
        key="ask_aquagrid_chat_input",
    )
    if user_typed and user_typed.strip():
        cleaned_typed = user_typed.strip()
        if queue_chat_request(st.session_state, cleaned_typed, display_hour):
            with live_turn_container:
                # Immediately show user message BEFORE calling the LLM
                with st.chat_message("user"):
                    st.markdown(cleaned_typed)
                executed_turn = _execute_pending_request(
                    st.session_state["_pending_chat_request"],
                    snapshot,
                    show_tech,
                )
        else:
            with live_turn_container:
                st.warning("Please wait for the current answer")

    return executed_turn


# Backward-compatible alias
render_ai_prompt_panel = render_ask_aquagrid_chat
