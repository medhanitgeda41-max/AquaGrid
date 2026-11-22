"""Tests for valve request rationale provenance, hydraulic impact assessment,
supply-continuity protection, and technical justification checks.
"""
from pathlib import Path
import sys
from unittest.mock import patch
import pytest

APP_DIR = str(Path(__file__).resolve().parent.parent / "app")
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

from agent.chat_assistant import run_chat_turn
from database.connection import get_db_connection
from database.scenarios import reset_database
from guards.valve_guard import (
    assess_valve_change_impact,
    extract_operator_reason,
    format_action_rationale,
    validate_rationale_sector_match,
)
from simulator import get_simulator
from tools.write_gateway import (
    execute_human_approval,
    stage_gated_valve_or_setpoint_action,
)


@pytest.fixture(autouse=True)
def clean_db():
    get_simulator().resume_feed()
    reset_database()
    yield
    get_simulator().resume_feed()
    reset_database()


def test_close_v16_no_reason_gets_exact_rationale_warning_and_requires_typed_reason():
    """'close V-16' with no reason -> rationale 'Operator request — no reason given', NO TECHNICAL JUSTIFICATION, approval needs typed reason + confirmation."""
    # Even if a noisy LLM invents a bogus Sector 6 rationale, Python must NEVER allow an invented reason!
    def _mock_noisy_llm(messages, tools, tool_executor, max_tool_steps=5, **kwargs):
        res = tool_executor(
            "propose_gated_action",
            {
                "action_type": "CLOSE_VALVE",
                "target_id": "V-16",
                "proposed_value": 0.0,
                "rationale": "Reduce flow to Sector 6 for upcoming high demand while keeping the reservoir above its safety floor",
            },
        )
        return (
            f"Staged CLOSE_VALVE V-16 ({res['status']}).",
            "MockNoisyLLM",
            [],
        )

    with patch("agent.chat_assistant.llm_client.chat_with_tools", side_effect=_mock_noisy_llm):
        turn = run_chat_turn("close V-16", history=[])

    staged = turn["staged_action"]
    assert staged is not None
    assert staged["status"] == "STAGED_PENDING_APPROVAL"
    assert staged["rationale"] == "Operator request — no reason given"
    assert staged["justification_status"] == "NOT_JUSTIFIED"
    assert staged["requires_override_reason"] is True
    assert "NO TECHNICAL JUSTIFICATION" in staged["warning"]
    assert "Sector 5" in staged["warning"]

    # Impact computed via hydraulic graph & affected pipes flagged for red highlight in 3D twin
    imp = staged["impact"]
    assert imp["valve_sector"] == "Sector 5"
    assert "SEG-S5-04" in imp["affected_pipe_ids"]
    assert imp["approx_customers_affected"] > 0
    assert imp["isolates_active_leak"] is False

    tv = turn.get("twin_view")
    assert tv is not None
    assert tv["focus"] == "V-16"
    assert tv["event_type"] == "staged_proposal"
    assert "SEG-S5-04" in tv["highlight_pipe_ids"]

    # Chat message must warn operator and ask if they want to continue
    low_msg = turn["content"].lower()
    assert "no leak or issue on" in low_msg and "v-16" in low_msg
    assert "do you want to continue" in low_msg

    action_id = int(staged["action_id"])

    # Approval WITHOUT typed reason / checkbox is REFUSED (V-16 stays OPEN)
    rej_no_reason = execute_human_approval(action_id=action_id, approver="Op_One")
    assert rej_no_reason["status"] == "REJECTED_REASON_REQUIRED"

    rej_short = execute_human_approval(
        action_id=action_id,
        approver="Op_One",
        operator_reason="short",
        confirmed_supply_cutoff=True,
    )
    assert rej_short["status"] == "REJECTED_REASON_REQUIRED"

    rej_unchecked = execute_human_approval(
        action_id=action_id,
        approver="Op_One",
        operator_reason="Emergency acoustic survey on Al Barsha line",
        confirmed_supply_cutoff=False,
    )
    assert rej_unchecked["status"] == "REJECTED_REASON_REQUIRED"

    conn = get_db_connection()
    try:
        assert conn.execute("SELECT status FROM valves WHERE valve_id = 'V-16';").fetchone()["status"] == "OPEN"
    finally:
        conn.close()

    # Approval WITH valid typed reason (>= 10 chars) + checkbox SUCCEEDS and saves reason to Audit Log
    typed_reason = "Emergency acoustic pressure survey on Sector 5 Al Barsha main"
    ok = execute_human_approval(
        action_id=action_id,
        approver="Op_One",
        operator_reason=typed_reason,
        confirmed_supply_cutoff=True,
    )
    assert ok["status"] == "EXECUTED"
    assert ok["operator_reason"] == typed_reason

    conn = get_db_connection()
    try:
        assert conn.execute("SELECT status FROM valves WHERE valve_id = 'V-16';").fetchone()["status"] == "CLOSED"
        audit_row = conn.execute(
            "SELECT details FROM audit_log WHERE event_type = 'ACTION_EXECUTED' AND target_id = 'V-16' ORDER BY id DESC LIMIT 1;"
        ).fetchone()
        assert audit_row is not None
        assert typed_reason in audit_row["details"]
    finally:
        conn.close()


def test_close_v12_isolating_active_j14_leak_is_justified_normal_card():
    """Closing V-12 while LEAK-001 at J-14 is active -> JUSTIFIED, normal approval card, executes cleanly."""
    with patch("agent.chat_assistant.llm_client.chat_with_tools", return_value=(None, "Deterministic Offline Engine", [])):
        turn = run_chat_turn("close V-12", history=[])

    staged = turn["staged_action"]
    assert staged is not None
    assert staged["status"] == "STAGED_PENDING_APPROVAL"
    assert staged["rationale"] == "Operator request — no reason given"
    assert staged["justification_status"] == "JUSTIFIED"
    assert staged["requires_override_reason"] is False
    assert staged["warning"] == ""

    imp = staged["impact"]
    assert imp["valve_sector"] == "Sector 4"
    assert imp["isolates_active_leak"] is True
    assert any(l["leak_id"] == "LEAK-001" for l in imp["isolated_active_leaks"])
    assert "SEG-S4-01" in imp["affected_pipe_ids"]
    assert any("-> 0 L/s" in line for line in imp["summary_lines"])

    # Normal approval succeeds without extra override reason box
    exec_res = execute_human_approval(action_id=int(staged["action_id"]), approver="Lead_Operator")
    assert exec_res["status"] == "EXECUTED"

    conn = get_db_connection()
    try:
        assert conn.execute("SELECT status FROM valves WHERE valve_id = 'V-12';").fetchone()["status"] == "CLOSED"
    finally:
        conn.close()


def test_closing_valve_that_cuts_whole_sector_returns_rejected_supply_continuity():
    """Closing V-20 (sole feeder to Sector 6) or V-07 (sole feeder to Sector 3) -> REJECTED_SUPPLY_CONTINUITY, not staged, shown in 3D."""
    # Direct gateway check on V-20 (cuts all of Sector 6)
    res_v20 = stage_gated_valve_or_setpoint_action(
        action_type="CLOSE_VALVE",
        target_id="V-20",
        proposed_value=0.0,
        rationale="Operator request — no reason given",
    )
    assert res_v20["status"] == "REJECTED_SUPPLY_CONTINUITY"
    assert "Sector 6" in res_v20["cutoff_sectors"]

    # Chat turn on "close V-20" must explain refusal and focus 3D twin on cut-off Sector 6
    with patch("agent.chat_assistant.llm_client.chat_with_tools", return_value=(None, "Deterministic Offline Engine", [])):
        turn = run_chat_turn("close V-20", history=[])

    assert turn["staged_action"]["status"] == "REJECTED_SUPPLY_CONTINUITY"
    assert "REJECTED_SUPPLY_CONTINUITY" in turn["content"]
    assert "Sector 6" in turn["content"]
    tv = turn.get("twin_view")
    assert tv is not None
    assert tv["focus"] == "Sector 6"
    assert tv["event_type"] == "supply_continuity_refused"
    assert len(tv["highlight_pipe_ids"]) >= 4

    # Must NOT be staged in pending_actions, and audit_log must record the refusal
    conn = get_db_connection()
    try:
        pending_cnt = conn.execute(
            "SELECT COUNT(*) AS cnt FROM pending_actions WHERE target_id = 'V-20' AND status = 'PENDING';"
        ).fetchone()["cnt"]
        assert pending_cnt == 0
        aud = conn.execute(
            "SELECT status FROM audit_log WHERE event_type = 'SUPPLY_CONTINUITY_BLOCKED' AND target_id = 'V-20' ORDER BY id DESC LIMIT 1;"
        ).fetchone()
        assert aud is not None and aud["status"] == "REJECTED_SUPPLY_CONTINUITY"
    finally:
        conn.close()


def test_rationale_naming_wrong_sector_is_rejected():
    """A rationale naming Sector 6 for V-16 (which belongs to Sector 5) is rejected with REJECTED_WRONG_SECTOR."""
    chk = validate_rationale_sector_match(
        action_type="CLOSE_VALVE",
        target_id="V-16",
        rationale="AI proposal: Reduce flow to Sector 6 for upcoming high demand",
    )
    assert chk["valid"] is False
    assert chk["status"] == "REJECTED_WRONG_SECTOR"
    assert chk["expected_sector"] == "Sector 5"
    assert "Sector 6" in chk["wrong_sectors"]

    res = stage_gated_valve_or_setpoint_action(
        action_type="CLOSE_VALVE",
        target_id="V-16",
        proposed_value=0.0,
        rationale="AI proposal: Reduce flow to Sector 6 for upcoming high demand",
    )
    assert res["status"] == "REJECTED_WRONG_SECTOR"
    assert "Sector 5" in res["reason"]
    assert "Sector 6" in res["reason"]

    conn = get_db_connection()
    try:
        pending_cnt = conn.execute(
            "SELECT COUNT(*) AS cnt FROM pending_actions WHERE target_id = 'V-16' AND status = 'PENDING';"
        ).fetchone()["cnt"]
        assert pending_cnt == 0
        aud = conn.execute(
            "SELECT status FROM audit_log WHERE event_type = 'RATIONALE_SECTOR_MISMATCH' AND target_id = 'V-16' ORDER BY id DESC LIMIT 1;"
        ).fetchone()
        assert aud is not None and aud["status"] == "REJECTED_WRONG_SECTOR"
    finally:
        conn.close()


def test_rationale_formatting_preserves_operator_words_and_ai_prefix():
    """Verify exact rationale provenance formatting rules."""
    assert extract_operator_reason("close V-16", "V-16") == ""
    assert format_action_rationale("operator", operator_message="close V-16", target_id="V-16") == "Operator request — no reason given"
    assert (
        format_action_rationale("operator", operator_message="close V-16 for planned valve seal maintenance", target_id="V-16")
        == "Operator request: close V-16 for planned valve seal maintenance"
    )
    assert (
        format_action_rationale("ai", ai_reason="Isolate confirmed leak LEAK-001 at J-14 (64.5 m³/hr)")
        == "AI proposal: Isolate confirmed leak LEAK-001 at J-14 (64.5 m³/hr)"
    )


def test_llm_fallback_after_propose_gated_action_does_not_duplicate_and_renders_clean_html():
    """If LLM stages CLOSE_VALVE V-12 via tool call then returns empty text, fallback MUST NOT re-propose a duplicate and card HTML has no code-block indentation."""
    import components.chat_widgets as cw

    def _mock_tool_then_empty(messages, tools, tool_executor, max_tool_steps=5, **kwargs):
        # Simulate Provider 1 staging V-12, failing mid-loop, and Provider 2 calling propose_gated_action again!
        r1 = tool_executor(
            "propose_gated_action",
            {
                "action_type": "CLOSE_VALVE",
                "target_id": "V-12",
                "proposed_value": 0.0,
                "rationale": "",
            },
        )
        r2 = tool_executor(
            "propose_gated_action",
            {
                "action_type": "CLOSE_VALVE",
                "target_id": "V-12",
                "proposed_value": 0.0,
                "rationale": "",
            },
        )
        assert r1["status"] == "STAGED_PENDING_APPROVAL"
        assert r2["status"] == "STAGED_PENDING_APPROVAL"
        assert r2["action_id"] == r1["action_id"]
        return None, "Deterministic Offline Engine", []

    with patch("agent.chat_assistant.llm_client.chat_with_tools", side_effect=_mock_tool_then_empty):
        turn = run_chat_turn("close V-12", history=[])

    assert turn["staged_action"]["status"] == "STAGED_PENDING_APPROVAL"
    assert "REJECTED_DUPLICATE_PENDING" not in turn["content"]
    assert "LEAK-001" in turn["content"]
    tv = turn.get("twin_view")
    assert tv is not None
    assert tv["focus"] == "V-12"
    assert tv["event_type"] == "staged_proposal"
    assert "SEG-S4-01" in tv["highlight_pipe_ids"]

    # Render approval card and ensure HTML has zero blank lines or 4-space indented <div> tags
    rendered_html: list[str] = []

    class _Col:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with (
        patch.object(cw.st, "session_state", {"operator_signature": "Test_Op", "ask_aquagrid_messages": []}),
        patch.object(cw.st, "markdown", side_effect=lambda txt, **kw: rendered_html.append(str(txt))),
        patch.object(cw.st, "columns", return_value=(_Col(), _Col(), _Col())),
        patch.object(cw.st, "button", return_value=False),
    ):
        cw.render_inline_approvals("clean_html_chk")

    card_html = "\n".join(rendered_html)
    assert "Computed Flow Change:" in card_html
    assert "LEAK-001 at J-14" in card_html
    assert "\n    <div" not in card_html

