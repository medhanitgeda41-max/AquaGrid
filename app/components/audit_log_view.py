"""Immutable SCADA & Agent Audit Log viewer with opt-in Technical Details toggle."""
import html
import pandas as pd
import streamlit as st
from config.settings import settings
from database.connection import get_db_connection
from components.badges import actor_badge, format_dubai_time, section_header


def render_audit_log() -> None:
    """Render live SQLite audit_log table with actor filtering and hidden technical details toggle."""
    section_header("Immutable SCADA Audit Log", ["CODE", "LLM", "HUMAN"], show_actors=True)

    show_tech = st.toggle(
        "Show technical details",
        value=False,
        key="show_tech_details",
    )

    if show_tech:
        last_result = st.session_state.get("last_agent_result") or {}
        provider = last_result.get("llm_provider", "Ready (no scenario run yet)")
        agent_status = last_result.get("status", "IDLE")
        steps = last_result.get("agent_step_count", 0)
        replans = last_result.get("replan_count", 0)

        conn_meta = get_db_connection()
        try:
            latest_ai = conn_meta.execute(
                "SELECT run_id, llm_provider, status FROM ai_run_log ORDER BY run_id DESC LIMIT 1;"
            ).fetchone()
        finally:
            conn_meta.close()

        latest_live_provider = latest_ai["llm_provider"] if latest_ai else "-"
        latest_live_status = latest_ai["status"] if latest_ai else "-"

        st.markdown(
            f"""
            <div class="scada-card">
                <div style="display:flex; flex-wrap:wrap; gap:1.25rem; font-size:0.86rem;">
                    <div><strong>LLM_MODE:</strong> <span class="ag-mono">{html.escape(settings.LLM_MODE)}</span></div>
                    <div><strong>Model Provider Used:</strong> <span class="ag-mono">{html.escape(str(provider))}</span></div>
                    <div><strong>Agent Status:</strong> <span class="ag-mono">{html.escape(str(agent_status))}</span></div>
                    <div><strong>Steps:</strong> <span class="ag-mono">{steps}</span></div>
                    <div><strong>Replans:</strong> <span class="ag-mono">{replans}</span></div>
                    <div><strong>Live Analyzer:</strong> <span class="ag-mono">{html.escape(str(latest_live_provider))} ({html.escape(str(latest_live_status))})</span></div>
                </div>
                <div style="margin-top:0.5rem; font-size:0.78rem; color:#9FB3C8;">
                    Responsibility Legend:&nbsp;
                    {actor_badge("CODE")} Deterministic Python Safety Guards &amp; SQLite Gateway &nbsp;&nbsp;
                    {actor_badge("LLM")} Cognitive Analysis, Briefs &amp; Proposals &nbsp;&nbsp;
                    {actor_badge("HUMAN")} Mandatory Approve / Reject Signoff
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    actor_filter = st.selectbox(
        "Filter by Actor",
        options=["ALL", "CODE_GUARD", "HUMAN_OPERATOR", "LLM_AGENT"],
        index=0,
        key="audit_actor_filter",
    )

    conn = get_db_connection()
    try:
        if actor_filter == "ALL":
            rows = conn.execute(
                "SELECT id, timestamp, event_type, target_id, actor, status, details FROM audit_log ORDER BY id DESC LIMIT 100;"
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, timestamp, event_type, target_id, actor, status, details
                FROM audit_log
                WHERE actor = ?
                ORDER BY id DESC
                LIMIT 100;
                """,
                (actor_filter,),
            ).fetchall()
    finally:
        conn.close()

    if not rows:
        st.info("No audit log entries match the selected filter.")
        return

    df = pd.DataFrame(
        [
            {
                "ID": r["id"],
                "Timestamp (Asia/Dubai)": format_dubai_time(r["timestamp"]),
                "Actor": r["actor"],
                "Event": r["event_type"],
                "Target": r["target_id"] or "-",
                "Status": r["status"],
                "Details": r["details"],
            }
            for r in rows
        ]
    )
    st.dataframe(df, width="stretch", hide_index=True, height=420)
