"""Tests for Part 2: immediate user message in history, live Thinking status steps, slow mocked LLM, streaming, and double-submit guard."""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from agent.chat_assistant import run_chat_turn, run_scenario_for_chat
from agent.llm_factory import LLMClient, provider_supports_streaming
from database.scenarios import reset_database


@pytest.fixture(autouse=True)
def clean_db():
    reset_database()
    yield
    reset_database()


def test_user_message_added_to_history_before_llm_call():
    """User message must already be in session_state['ask_aquagrid_messages'] BEFORE the LLM call runs."""
    import components.ai_prompt_panel as panel

    fake_state: dict = {}
    observed_history_during_llm: list[dict] = []

    def _spy_run_chat_turn(user_message, history=None, current_hour=14, db_path=None, status_callback=None):
        # Capture session_state snapshot WHILE the LLM call is in progress
        observed_history_during_llm.extend(list(fake_state.get("ask_aquagrid_messages", [])))
        assert fake_state.get("_chat_generating") is True
        return {
            "role": "assistant",
            "kind": "text",
            "content": "RES-01 is at 48.0%.",
            "provider": "Groq (openai/gpt-oss-120b)",
            "tools_called": [],
            "code_checks": [],
            "widgets": [],
            "staged_action": None,
        }

    with patch.object(panel.st, "session_state", fake_state), patch(
        "components.ai_prompt_panel.run_chat_turn", side_effect=_spy_run_chat_turn
    ):
        ok = panel._handle_user_prompt("What is the reservoir level?", display_hour=14)

    assert ok is True
    assert len(observed_history_during_llm) == 1
    assert observed_history_during_llm[0] == {"role": "user", "content": "What is the reservoir level?"}
    assert len(fake_state["ask_aquagrid_messages"]) == 2
    assert fake_state["ask_aquagrid_messages"][1]["role"] == "assistant"
    assert fake_state["_chat_generating"] is False


def test_slow_mocked_llm_shows_live_status_and_completes_done():
    """A slow mocked LLM still drives st.status('Thinking...') step by step and transitions to 'Done' (collapsed)."""
    import components.ai_prompt_panel as panel

    fake_state: dict = {}
    assert panel.queue_chat_request(fake_state, "Show Sector 4 on the map", display_hour=14) is True
    # User message already present before execution
    assert fake_state["ask_aquagrid_messages"][-1] == {"role": "user", "content": "Show Sector 4 on the map"}

    status_events: list[tuple] = []
    streamed_chunks: list[str] = []

    class _FakeStatusBox:
        def __enter__(self):
            status_events.append(("enter", "Thinking..."))
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, msg):
            status_events.append(("write", msg))

        def update(self, label=None, state=None, expanded=None):
            status_events.append(("update", label, state, expanded))

    class _FakeChatMsg:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _slow_llm(messages, tools, tool_executor, max_tool_steps=5, fallback_context_builder=None, status_callback=None):
        if status_callback:
            status_callback("Asking Groq (openai/gpt-oss-120b)...")
        time.sleep(0.12)
        tool_executor("show_map", {"sector": "Sector 4", "show_boundaries": True})
        return "Map zoomed to Sector 4.", "Groq (openai/gpt-oss-120b)", []

    def _capture_stream(gen):
        chunks = list(gen)
        streamed_chunks.extend(chunks)
        return "".join(chunks)

    with (
        patch.object(panel.st, "session_state", fake_state),
        patch.object(panel.st, "chat_message", return_value=_FakeChatMsg()),
        patch.object(panel.st, "status", return_value=_FakeStatusBox()),
        patch.object(panel.st, "write_stream", side_effect=_capture_stream),
        patch.object(panel.st, "markdown"),
        patch.object(panel.st, "expander", return_value=_FakeChatMsg()),
        patch.object(panel.st, "caption"),
        patch("components.ai_prompt_panel.render_chat_widget"),
        patch("agent.chat_assistant.llm_client.chat_with_tools", side_effect=_slow_llm),
    ):
        ran = panel._execute_pending_request(fake_state["_pending_chat_request"], snapshot={}, show_tech=False)

    assert ran is True
    written = [e[1] for e in status_events if e[0] == "write"]
    assert "Checking your request (injection filter)..." in written
    assert "Reading live plant data..." in written
    assert "Asking Groq (openai/gpt-oss-120b)..." in written
    assert "Drawing the map..." in written
    # Final status update sets label="Done", state="complete", expanded=False
    updates = [e for e in status_events if e[0] == "update"]
    assert updates[-1] == ("update", "Done", "complete", False)
    # Word-by-word streaming ran for Groq
    assert len(streamed_chunks) >= 3
    assert "".join(streamed_chunks).strip() == "Map zoomed to Sector 4."
    assert fake_state["_chat_generating"] is False
    assert fake_state["_pending_chat_request"] is None


def test_double_submit_is_ignored_while_generating():
    """While an answer is being generated, new chat inputs and sidebar scenario clicks are ignored with a wait notice."""
    import components.ai_prompt_panel as panel

    fake_state: dict = {}
    assert panel.queue_chat_request(fake_state, "First question", display_hour=14) is True
    assert panel.is_chat_generating(fake_state) is True
    assert len(fake_state["ask_aquagrid_messages"]) == 1

    # Second chat submit while first is still generating -> ignored!
    assert panel.queue_chat_request(fake_state, "Second question while busy", display_hour=14) is False
    assert fake_state["_chat_busy_notice"] == "Please wait for the current answer"
    assert len(fake_state["ask_aquagrid_messages"]) == 1
    assert fake_state["_pending_chat_request"]["text"] == "First question"

    # Sidebar scenario click while busy -> also ignored!
    assert panel.queue_scenario_request(fake_state, "R1-S1", "Pipe Leak") is False
    assert fake_state["_chat_busy_notice"] == "Please wait for the current answer"
    assert len(fake_state["ask_aquagrid_messages"]) == 1


def test_provider_failover_and_planner_and_scenario_emit_expected_status_steps():
    """Provider failure emits '<Provider> busy, trying backup model...'; planner & scenario runs emit live steps."""
    # 1. Provider failover status
    steps: list[str] = []
    client = LLMClient(mode="demo")
    fake_providers = [
        {"name": "Groq (openai/gpt-oss-120b)", "base_url": "https://api.groq.com/openai/v1", "api_key": "gsk_real_key_1", "model": "openai/gpt-oss-120b"},
        {"name": "Groq (openai/gpt-oss-20b)", "base_url": "https://api.groq.com/openai/v1", "api_key": "gsk_real_key_2", "model": "openai/gpt-oss-20b"},
    ]

    ok_choice = MagicMock()
    ok_choice.message.tool_calls = None
    ok_choice.message.content = "Backup model answered with live values."
    ok_resp = MagicMock()
    ok_resp.choices = [ok_choice]

    call_idx = {"n": 0}

    def _fake_create(**kwargs):
        call_idx["n"] += 1
        if call_idx["n"] <= 2:  # fail tool call + no-tools retry on 120b
            raise RuntimeError("429 rate limit")
        return ok_resp

    fake_openai = MagicMock()
    fake_openai.chat.completions.create.side_effect = _fake_create

    with patch.object(client, "_get_chat_provider_chain", return_value=fake_providers), patch(
        "agent.llm_factory.OpenAI", return_value=fake_openai
    ):
        ans, prov, _ = client.chat_with_tools(
            messages=[{"role": "user", "content": "status"}],
            tools=[],
            tool_executor=lambda n, a: {},
            status_callback=steps.append,
        )

    assert prov == "Groq (openai/gpt-oss-20b)"
    assert ans == "Backup model answered with live values."
    assert "Asking Groq (openai/gpt-oss-120b)..." in steps
    assert "Groq busy, trying backup model..." in steps
    assert "Asking Groq (openai/gpt-oss-20b)..." in steps

    # 2. Planner live status steps
    plan_steps: list[str] = []
    with patch("agent.chat_assistant.llm_client.chat_with_tools", return_value=(None, "Deterministic Offline Engine", [])):
        run_chat_turn("make a 5-hour production plan", history=[], current_hour=14, status_callback=plan_steps.append)
    assert "Checking your request (injection filter)..." in plan_steps
    assert "Reading live plant data..." in plan_steps
    assert "Validating the plan hour by hour..." in plan_steps
    assert "Running safety checks..." in plan_steps

    # 3. Scenario live status steps (used by sidebar Run button)
    sc_steps: list[str] = []
    with patch("agent.llm_factory.llm_client.query", return_value=({"operator_brief": "ok", "maintenance_advisory": "ok", "proposed_schedule": "", "action_proposal": {"action_type": "NONE", "target_id": "NONE", "proposed_value": 0.0, "rationale": "ok"}}, "Deterministic Offline Engine")):
        run_scenario_for_chat("R1-S1", current_hour=14, status_callback=sc_steps.append)
    assert "Checking your request (injection filter)..." in sc_steps
    assert "Reading live plant data..." in sc_steps
    assert "Running safety checks..." in sc_steps
    assert "Drawing the map..." in sc_steps

    # 4. Streaming capability helper
    assert provider_supports_streaming("Groq (openai/gpt-oss-120b)") is True
    assert provider_supports_streaming("Gemini (gemini-3.8-flash)") is True
    assert provider_supports_streaming("Ollama (qwen3:8b)") is True
    assert provider_supports_streaming("Deterministic Offline Engine") is False
    assert provider_supports_streaming("CODE_GUARD") is False
