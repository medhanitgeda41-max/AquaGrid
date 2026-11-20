"""Grid-Wide AI Suggestions Panel & AI Run Log Table."""
import html
import json
import pandas as pd
import streamlit as st

from database.connection import get_db_connection
from components.badges import format_dubai_time, section_header


def render_ai_suggestions_and_run_log() -> None:
    """Render Grid-Wide AI Suggestions and the chronological AI Run Log table from SQLite."""
    show_tech = bool(st.session_state.get("show_tech_details", False))
    conn = get_db_connection()
    try:
        run_rows = conn.execute(
            "SELECT * FROM ai_run_log ORDER BY run_id DESC LIMIT 30;"
        ).fetchall()
    finally:
        conn.close()

    section_header("Grid-Wide AI Suggestions", ["LLM", "CODE"])
    if not run_rows:
        st.info("No AI analysis runs logged yet.")
        return

    latest_run = dict(run_rows[0])
    if latest_run.get("error_message"):
        st.error(f"**AI Run Warning (Run #{latest_run['run_id']}):** {latest_run['error_message']}")

    try:
        grid_suggs = json.loads(latest_run.get("grid_suggestions_json") or "[]")
    except json.JSONDecodeError:
        grid_suggs = []

    if grid_suggs:
        cols = st.columns(min(3, len(grid_suggs)))
        for idx, s in enumerate(grid_suggs):
            with cols[idx % len(cols)]:
                conf = float(s.get("confidence", 0.85))
                gen_at = format_dubai_time(s.get("generated_at_utc") or latest_run.get("run_at"))
                st.markdown(
                    f"""
                    <div class="scada-card">
                        <div style="margin-bottom:0.35rem;">
                            <span class="ag-badge ag-badge-ok">{html.escape(str(s.get('category', 'Grid')))}</span>
                            <span class="ag-badge ag-badge-neutral">Conf: {conf:.0%}</span>
                        </div>
                        <div style="font-size:0.88rem; color:#E6EDF7;">
                            {html.escape(str(s.get('suggestion', '')))}
                        </div>
                        <div style="margin-top:0.45rem; font-size:0.76rem; color:#9FB3C8;">
                            <strong>Generated:</strong> {html.escape(gen_at)}<br/>
                            <strong>Based on:</strong> {html.escape(str(s.get('based_on', '')))}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No grid-wide suggestions recorded in the latest run.")

    with st.expander(f"Show details — AI Run Log ({len(run_rows)} runs)", expanded=False):
        log_records = []
        for r in run_rows:
            try:
                findings = json.loads(r["findings_json"] or "[]")
                findings_txt = "; ".join(
                    f"{f.get('classification', f.get('type', 'Finding'))}: {f.get('target')} ({float(f.get('confidence', 0)):.0%})"
                    for f in findings
                )
            except json.JSONDecodeError:
                findings_txt = str(r["findings_json"])

            try:
                suggs = json.loads(r["suggestions_json"] or "[]")
                suggs_txt = " | ".join(str(x) for x in suggs[:3])
            except json.JSONDecodeError:
                suggs_txt = str(r["suggestions_json"])

            rec = {
                "Run ID": r["run_id"],
                "Timestamp (Asia/Dubai)": format_dubai_time(r["run_at"]),
                "Trigger": r["trigger_source"],
                "Status": r["status"],
                "Inputs Summary": r["inputs_summary"],
                "Findings": findings_txt,
                "Suggestions": suggs_txt,
                "Error": r["error_message"] or "-",
            }
            if show_tech:
                rec["Provider"] = r["llm_provider"]
            log_records.append(rec)

        st.dataframe(pd.DataFrame(log_records), width="stretch", hide_index=True, height=340)
