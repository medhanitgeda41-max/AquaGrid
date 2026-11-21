"""Clean SCADA styling and meaningful status badges (red = danger, orange = pending, green = OK)."""
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import html
import sys
import threading
from zoneinfo import ZoneInfo
import streamlit as st

_FALLBACK_STEP_LOCK = threading.Lock()


def get_sim_step_lock(sim=None):
    """Return the simulator step lock when running, else a no-op context manager.

    Works safely even if Live Simulation is OFF, sim is None, or an older simulator object
    without step_lock is still cached across a hot-reload.
    """
    if sim is None or not getattr(sim, "is_running", False):
        return nullcontext()
    lock = getattr(sim, "step_lock", None)
    if lock is None:
        sim_mod = sys.modules.get("simulator")
        lock = getattr(sim_mod, "STEP_LOCK", None) or _FALLBACK_STEP_LOCK
        try:
            setattr(sim, "step_lock", lock)
        except Exception:
            pass
    return lock

try:
    DUBAI_TZ = ZoneInfo("Asia/Dubai")
except Exception:
    # Asia/Dubai is fixed UTC+4 year-round (no DST)
    DUBAI_TZ = timezone(timedelta(hours=4))


def format_dubai_time(ts: str | None) -> str:
    """Convert SQLite UTC timestamp ('YYYY-MM-DD HH:MM:SS') to Asia/Dubai local time (UTC+4)."""
    if not ts:
        return "-"
    raw = str(ts).strip()
    if not raw or raw == "-":
        return "-"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            dt_utc = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            dt_local = dt_utc.astimezone(DUBAI_TZ)
            return dt_local.strftime("%Y-%m-%d %H:%M:%S GST")
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(DUBAI_TZ).strftime("%Y-%m-%d %H:%M:%S GST")
    except ValueError:
        return raw


SCADA_CSS = """
<style>
    /* Keep Streamlit header & sidebar open/close button (») visible; hide only menu, toolbar actions, decoration, and footer */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    #MainMenu,
    footer,
    [data-testid="stMainMenu"],
    [data-testid="stToolbarActions"],
    [data-testid="stActionButton"],
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"],
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        opacity: 1 !important;
    }
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }
    .scada-banner {
        background: #111827;
        border: 1px solid #263144;
        border-radius: 10px;
        padding: 0.95rem 1.25rem;
        margin-bottom: 1rem;
    }
    .scada-title {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        color: #F8FAFC;
        margin: 0;
    }
    .scada-subtitle {
        font-size: 0.85rem;
        color: #94A3B8;
        margin: 0.2rem 0 0 0;
    }
    .scada-card {
        background: #111827;
        border: 1px solid #263144;
        border-radius: 8px;
        padding: 0.9rem 1.05rem;
        margin-bottom: 0.8rem;
    }
    .scada-card-alert {
        background: #1F1215;
        border: 1px solid #7F1D1D;
        border-left: 4px solid #EF4444;
        border-radius: 8px;
        padding: 0.9rem 1.05rem;
        margin-bottom: 0.8rem;
    }
    .scada-card-pending {
        background: #1E1910;
        border: 1px solid #784B12;
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 0.95rem 1.1rem;
        margin-bottom: 0.85rem;
    }
    .scada-card-ok {
        background: #0F1F1A;
        border: 1px solid #14532D;
        border-left: 4px solid #10B981;
        border-radius: 8px;
        padding: 0.9rem 1.05rem;
        margin-bottom: 0.8rem;
    }
    .scada-section-hdr {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #E2E8F0;
        margin: 0.35rem 0 0.5rem 0;
    }
    .scada-step {
        background: #111827;
        border: 1px solid #263144;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.55rem;
        font-size: 0.87rem;
        color: #E2E8F0;
    }
    .scada-step-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #94A3B8;
        margin-bottom: 0.3rem;
    }
    .ag-badge {
        display: inline-block;
        padding: 0.14rem 0.52rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        vertical-align: middle;
        margin-right: 0.25rem;
    }
    .ag-badge-llm, .ag-badge-code, .ag-badge-human, .ag-badge-neutral {
        background: #1E293B;
        color: #CBD5E1;
        border: 1px solid #475569;
    }
    .ag-badge-ok {
        background: #064E3B;
        color: #6EE7B7;
        border: 1px solid #10B981;
    }
    .ag-badge-warn {
        background: #451A03;
        color: #FCD34D;
        border: 1px solid #F59E0B;
    }
    .ag-badge-danger {
        background: #450A0A;
        color: #FCA5A5;
        border: 1px solid #EF4444;
    }
    .ag-mono {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.83rem;
    }
    .scada-table {
        width: 100%;
        border-collapse: collapse;
        background: #111827;
        border: 1px solid #263144;
        border-radius: 8px;
        overflow: hidden;
        font-size: 0.84rem;
    }
    .scada-table th {
        background: #1E293B;
        color: #CBD5E1;
        text-align: left;
        padding: 0.5rem 0.65rem;
        border-bottom: 1px solid #334155;
        white-space: nowrap;
    }
    .scada-table td {
        padding: 0.5rem 0.65rem;
        border-bottom: 1px solid #1E293B;
        color: #E2E8F0;
        vertical-align: top;
        white-space: normal;
        word-wrap: break-word;
        overflow-wrap: anywhere;
    }
    .scada-round-hdr {
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: #CBD5E1;
        margin: 0.7rem 0 0.45rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid #263144;
    }
    .scada-id-badge, .scada-id-badge-r2 {
        display: inline-block;
        padding: 0.12rem 0.48rem;
        border-radius: 6px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.74rem;
        font-weight: 700;
        background: #1E293B;
        color: #E2E8F0;
        border: 1px solid #475569;
        margin-right: 0.35rem;
    }
    .scada-card-title {
        font-size: 0.93rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0.35rem 0 0.4rem 0;
        line-height: 1.3;
        white-space: normal;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .scada-card-field {
        font-size: 0.8rem;
        color: #CBD5E1;
        line-height: 1.4;
        margin-bottom: 0.35rem;
        white-space: normal;
        word-wrap: break-word;
    }
    .scada-card-field strong {
        color: #E2E8F0;
    }
    .scada-card-watch strong {
        color: #E2E8F0;
    }
</style>
"""


def inject_scada_css() -> None:
    """Inject clean dark-mode SCADA styles into Streamlit page."""
    st.markdown(SCADA_CSS, unsafe_allow_html=True)


def actor_badge(actor: str) -> str:
    """Return neutral HTML badge showing whether a step was done by LLM, CODE, or HUMAN."""
    norm = (actor or "").strip().upper()
    if norm in ("LLM", "LLM_AGENT"):
        return '<span class="ag-badge ag-badge-llm">LLM</span>'
    if norm in ("CODE", "CODE_GUARD"):
        return '<span class="ag-badge ag-badge-code">CODE</span>'
    if norm in ("HUMAN", "HUMAN_OPERATOR"):
        return '<span class="ag-badge ag-badge-human">HUMAN</span>'
    return f'<span class="ag-badge ag-badge-neutral">{html.escape(norm or "SYSTEM")}</span>'


def status_badge(status: str) -> str:
    """Return meaningful color-coded HTML status badge (green=OK, orange=pending, red=danger)."""
    s = (status or "UNKNOWN").strip().upper()
    if s in (
        "VALID",
        "EXECUTED",
        "APPROVED",
        "ALLOWED",
        "COMPLETED",
        "ADVISORY_ONLY",
        "OPEN",
        "ONLINE",
        "NORMAL",
        "RESOLVED",
        "NOMINAL",
        "COMPLIANT",
        "FRESH",
        "TELEMETRY_RESTORED",
    ):
        cls = "ag-badge-ok"
    elif s in (
        "PENDING",
        "STAGED_PENDING_APPROVAL",
        "AWAITING_HUMAN_APPROVAL",
        "THROTTLED",
        "REJECTED_DUPLICATE_PENDING",
        "GAP",
        "SUSPECTED",
        "UNDER REPAIR",
        "MEDIUM",
        "LOW",
        "ELEVATED WATCH",
        "UNTRUSTED_TEXT",
        "EMPTY_PROMPT",
    ):
        cls = "ag-badge-warn"
    elif s in (
        "REJECTED_BY_CODE",
        "REJECTED_BY_SAFETY_GUARD",
        "REJECTED_WATER_QUALITY",
        "REJECTED_UNAUTHORIZED_APPROVAL_CLAIM",
        "REJECTED_APPROVAL_MISMATCH",
        "REJECTED_APPROVAL_ALREADY_USED",
        "REJECTED_APPROVAL_EXPIRED",
        "REJECTED_STALE_TELEMETRY",
        "CANCELLED_STALE_TELEMETRY",
        "TELEMETRY_LOSS",
        "STALE",
        "REJECTED_INVALID_TARGET",
        "CANCELLED_BY_SAFETY_RECHECK",
        "ABORTED_BY_SAFETY",
        "NON_COMPLIANT",
        "REJECTED",
        "REFUSED",
        "BLOCKED",
        "QUARANTINED",
        "CONFIRMED_PIPE_LEAK",
        "CONFIRMED",
        "CRITICAL",
        "HIGH",
        "CLOSED",
        "ALERT ACTIVE",
    ):
        cls = "ag-badge-danger"
    else:
        cls = "ag-badge-neutral"
    return f'<span class="ag-badge {cls}">{html.escape(s)}</span>'


def section_header(title: str, actors: list[str] | None = None, extra_badge: str = "", show_actors: bool = False) -> None:
    """Render a clean section header (actor tags hidden unless technical details are enabled)."""
    show_tech = bool(show_actors or st.session_state.get("show_tech_details", False))
    badges_html = "".join(actor_badge(a) for a in (actors or [])) if show_tech else ""
    st.markdown(
        f'<div class="scada-section-hdr">{badges_html}<span>{html.escape(title)}</span>{extra_badge}</div>',
        unsafe_allow_html=True,
    )
