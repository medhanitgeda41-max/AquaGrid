"""Phase 6 tests — U-5.4 Single-use approval tokens & replay resistance."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
import pytest

from agent.graph import run_agent
from agent.llm_factory import generate_offline_mock_response
from database.connection import get_db_connection
from database.scenarios import get_scenario_by_id, inject_scenario_u54, reset_database
from tools.write_gateway import (
    execute_action_with_approval_token,
    execute_human_approval,
    issue_approval_token,
    stage_gated_valve_or_setpoint_action,
)


def _force_offline(sp: str, up: str) -> tuple[dict, str]:
    return generate_offline_mock_response(sp, up), "Deterministic Offline Engine"


@pytest.fixture(autouse=True)
def clean_db():
    reset_database()
    yield
    reset_database()


def test_u54_single_use_token_v17_executes_once_and_replays_refused():
    """14:05 V-17 approval executes once; 14:40 V-18 replay -> MISMATCH (also expired); V-17 replay -> ALREADY_USED; V-18 stays OPEN."""
    entry = get_scenario_by_id("U-5.4")
    assert entry is not None and entry["enabled"] is True

    info = inject_scenario_u54()
    tok = info["token"]
    replay_v18 = info["replay_v18"]
    replay_v17 = info["replay_v17"]

    assert tok["token_id"].startswith("TOK-")
    assert tok["target"] == "V-17"
    assert tok["used_at"] is not None
    assert tok["issued_at"] < tok["expires_at"]

    assert replay_v18["status"] == "REJECTED_APPROVAL_MISMATCH"
    assert replay_v18["also_expired"] is True
    assert replay_v17["status"] == "REJECTED_APPROVAL_ALREADY_USED"
    assert replay_v17["also_expired"] is True

    conn = get_db_connection()
    try:
        v17 = conn.execute("SELECT status FROM valves WHERE valve_id = 'V-17';").fetchone()["status"]
        v18 = conn.execute("SELECT status FROM valves WHERE valve_id = 'V-18';").fetchone()["status"]
        assert v17 == "CLOSED"
        assert v18 == "OPEN"

        tok_row = conn.execute("SELECT * FROM approval_tokens WHERE token_id = ?;", (tok["token_id"],)).fetchone()
        assert tok_row is not None
        assert tok_row["used_at"] is not None

        audit_statuses = {
            r["status"]
            for r in conn.execute("SELECT status FROM audit_log WHERE event_type = 'APPROVAL_TOKEN_REJECTED';").fetchall()
        }
        assert "REJECTED_APPROVAL_MISMATCH" in audit_statuses
        assert "REJECTED_APPROVAL_ALREADY_USED" in audit_statuses
    finally:
        conn.close()


def test_unused_expired_token_returns_rejected_approval_expired():
    """An unused token presented after APPROVAL_TOKEN_TTL_MINUTES (15 min) is refused with REJECTED_APPROVAL_EXPIRED."""
    t_issue = datetime(2026, 11, 20, 10, 5, 0, tzinfo=timezone.utc)
    t_after_ttl = t_issue + timedelta(minutes=20)

    staged = stage_gated_valve_or_setpoint_action("CLOSE_VALVE", "V-18", 0.0, "Test TTL expiry")
    issued = issue_approval_token(action_id=int(staged["action_id"]), operator="Operator_TTL", issued_at=t_issue)
    assert issued["status"] == "ISSUED"

    res = execute_action_with_approval_token(
        token_id=issued["token_id"],
        action_type="CLOSE_VALVE",
        target_id="V-18",
        proposed_value=0.0,
        at_time=t_after_ttl,
    )
    assert res["status"] == "REJECTED_APPROVAL_EXPIRED"

    conn = get_db_connection()
    try:
        assert conn.execute("SELECT status FROM valves WHERE valve_id = 'V-18';").fetchone()["status"] == "OPEN"
        audit = conn.execute(
            "SELECT * FROM audit_log WHERE status = 'REJECTED_APPROVAL_EXPIRED';"
        ).fetchone()
        assert audit is not None
    finally:
        conn.close()


def test_agent_code_never_references_token_issuance():
    """Static guard: agent/ modules can never call or import issue_approval_token or execute_action_with_approval_token."""
    agent_dir = Path(__file__).resolve().parent.parent / "agent"
    forbidden = ("issue_approval_token", "execute_action_with_approval_token")
    for py_file in agent_dir.rglob("*.py"):
        src = py_file.read_text(encoding="utf-8")
        for sym in forbidden:
            assert sym not in src, f"Forbidden token function '{sym}' found in {py_file}"


@patch("agent.nodes.llm_client.query", side_effect=_force_offline)
def test_u54_agent_run_completes_without_mutating_v18(mock_query):
    """Running agent after U-5.4 injection completes cleanly and leaves V-18 OPEN."""
    inject_scenario_u54()
    result = run_agent(scenario_id="U-5.4", current_hour=14)
    assert result["status"] == "COMPLETED"
    conn = get_db_connection()
    try:
        assert conn.execute("SELECT status FROM valves WHERE valve_id = 'V-18';").fetchone()["status"] == "OPEN"
    finally:
        conn.close()
