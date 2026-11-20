"""Audit Log & Technical Details viewer (raw tables & metadata hidden inside Show details)."""
import html
import pandas as pd
import streamlit as st
from config.settings import settings
from database.connection import get_db_connection
from components.badges import actor_badge, format_dubai_time, section_header


def render_audit_log() -> None:
    """Render Audit Log & Technical Details inside collapsed Show details sections."""
    section_header("Audit Log")

    with st.expander("Show details — Technical Metadata", expanded=False):
        show_tech = st.toggle(
            "Show actor badges on headers",
            value=False,
            key="show_tech_details",
        )
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
                    <div><strong>Provider:</strong> <span class="ag-mono">{html.escape(str(provider))}</span></div>
                    <div><strong>Agent Status:</strong> <span class="ag-mono">{html.escape(str(agent_status))}</span></div>
                    <div><strong>Steps:</strong> <span class="ag-mono">{steps}</span></div>
                    <div><strong>Replans:</strong> <span class="ag-mono">{replans}</span></div>
                    <div><strong>Live Analyzer:</strong> <span class="ag-mono">{html.escape(str(latest_live_provider))} ({html.escape(str(latest_live_status))})</span></div>
                </div>
                <div style="margin-top:0.5rem; font-size:0.78rem; color:#94A3B8;">
                    {actor_badge("CODE")} Python Safety Guards &nbsp;
                    {actor_badge("LLM")} AI Analysis &nbsp;
                    {actor_badge("HUMAN")} Operator Approval
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    conn = get_db_connection()
    try:
        total_cnt = conn.execute("SELECT COUNT(*) AS cnt FROM audit_log;").fetchone()["cnt"]
    finally:
        conn.close()

    with st.expander(f"Show details — Immutable Audit Log ({total_cnt} events)", expanded=False):
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
        st.dataframe(df, width="stretch", hide_index=True, height=400)
