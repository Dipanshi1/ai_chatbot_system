"""
Streamlit Web Application for University Student Support (Phase 5C).

A thin, restrained presentation layer for student support questions.
Delegates all intent classification and response gating to src.ml.answer().
"""
import html
import logging
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import streamlit as st

from src.ml.artifacts import verify_artifacts
from src.ml.config import MAX_MESSAGE_CHARS
from src.ml.pipeline import ChatPipeline, get_pipeline
from src.ml.schemas import ChatResult
from src.ui.helpers import (
    ABOUT_TEXT,
    CUSTOM_CSS,
    DISCLAIMER_TEXT,
    ESCALATION_CONSENT_LABEL,
    ESCALATION_PRIVACY_NOTE,
    EXAMPLE_QUESTIONS,
    OPT_OUT_ACTIVE_TEXT,
    OPT_OUT_HELP,
    OPT_OUT_LABEL,
    PAGE_SUBTITLE,
    PAGE_TITLE,
    PRIVACY_NOTICE,
    REDACTION_NOTICE,
    SUGGESTED_TILES,
    TONE_SIGNAL_LABEL,
    KB_RECORD_SETS,
    KB_RUNTIME_LABEL,
    extract_diagnostics,
    format_outcome_badge,
    get_initial_messages,
    load_evaluation_knowledge_base,
    load_runtime_knowledge_base,
    search_knowledge_base,
    validate_user_message,
)
from src.support.admin_auth import admin_is_configured, attempt_login, is_locked_out
from src.support.analytics import get_admin_analytics
from src.support.export import export_feedback_csv, export_tickets_csv
from src.support.config import (
    ESCALATION_OFFER_MESSAGE,
    HUMAN_REQUEST_MESSAGE,
    MAX_COMMENT_CHARS,
    MAX_SUMMARY_CHARS,
    OPT_OUT_MESSAGE,
    TICKET_FOLLOW_UP_TEXT,
)
from src.support.redaction import redact
from src.support.sentiment import analyze_sentiment
from src.support.session import (
    close_escalation,
    init_support_state,
    open_escalation,
    set_opt_out,
    submit_escalation,
    submit_feedback,
)
from src.support.store import (
    TICKET_STATUSES,
    SupportStore,
    SupportStoreError,
    TicketLimitError,
)
from src.support.triage import (
    Signal,
    crisis_guidance,
    detect_signals,
    should_offer_escalation,
)

logger = logging.getLogger(__name__)

# Configure page settings
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Verifying artifacts and initializing assistant...")
def load_cached_pipeline() -> ChatPipeline:
    """
    Verify canonical artifacts and load the singleton inference pipeline.

    Cached across runs to prevent reloading pickling artifacts on every rerun.
    """
    verify_artifacts()
    return get_pipeline()


def render_assistant_message(
    content: str, result: ChatResult = None, error: str = None, sentiment: str = None
) -> None:
    """Render an assistant response with outcome indicators and optional diagnostics.

    Internal error details are never rendered; they are logged server-side only.
    """
    st.markdown(content)

    if error:
        return

    if result is None:
        return

    # Outcome explanation & visual indicator pill
    badge = format_outcome_badge(result.outcome)
    badge_class = badge.get("badge_class", "answered")
    badge_icon = badge.get("icon", "✓")

    badge_html = (
        f'<div class="outcome-badge-container outcome-badge-{badge_class}">'
        f'<span class="outcome-badge-icon">{badge_icon}</span>'
        f'<span class="outcome-badge-title">{badge["title"]}</span>'
        f'<span class="outcome-badge-sep">—</span>'
        f'<span class="outcome-badge-desc">{badge["explanation"]}</span>'
        f'</div>'
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    # Technical diagnostics expander with clean tabular layout
    with st.expander("🔍 View system diagnostics", expanded=False):
        diagnostics = extract_diagnostics(result)
        rows = []
        for label, val in diagnostics.items():
            if isinstance(val, list):
                val_str = f"`{', '.join(str(x) for x in val)}`"
            else:
                val_str = f"`{val}`"
            rows.append(f"| **{label}** | {val_str} |")
        if sentiment:
            rows.append(f"| **{TONE_SIGNAL_LABEL}** | `{sentiment}` |")
        st.markdown(
            "| Parameter / Signal | Diagnostic Evidence |\n"
            "| :--- | :--- |\n" + "\n".join(rows)
        )


def process_user_query(query_text: str, pipeline: ChatPipeline) -> None:
    """Validate, redact and triage a query, then invoke the pipeline if appropriate.

    Only redacted text goes past this point: raw input never reaches the pipeline,
    session state or the support store.
    """
    is_valid, error_msg = validate_user_message(query_text, max_chars=MAX_MESSAGE_CHARS)
    if not is_valid:
        st.warning(error_msg)
        return

    state = st.session_state
    init_support_state(state)
    if state.escalation_reason == "auto_offer":
        close_escalation(state)

    redaction = redact(query_text)
    message = redaction.text
    state.messages.append(
        {"role": "user", "content": message, "result": None, "redacted": redaction.found}
    )

    # Fixed replies run before ML processing. Crisis guidance always comes first.
    signals = detect_signals(message)
    guidance = crisis_guidance(signals)
    if guidance:
        close_escalation(state)
        state.messages.append({"role": "assistant", "content": guidance, "result": None})
        st.rerun()
    if Signal.HUMAN_REQUEST in signals:
        open_escalation(state, "manual", summary=message)
        state.messages.append(
            {"role": "assistant", "content": HUMAN_REQUEST_MESSAGE, "result": None}
        )
        st.rerun()
    if Signal.OPT_OUT in signals:
        state.messages.append({"role": "assistant", "content": OPT_OUT_MESSAGE, "result": None})
        st.rerun()

    # Call ML inference pipeline
    try:
        res = pipeline.answer(message)
        # Runs on the finished result: sentiment cannot influence the answer.
        sentiment = analyze_sentiment(message).label
        state.messages.append(
            {
                "role": "assistant",
                "content": res.answer,
                "result": res,
                "sentiment": sentiment,
            }
        )
        if should_offer_escalation(res.abstained, sentiment, signals):
            open_escalation(
                state,
                "auto_offer",
                summary=message,
                intent=res.intent or res.predicted_intent,
                sentiment=sentiment,
            )
    except Exception:
        logger.exception("Pipeline failed while answering a query")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "An unexpected error occurred while processing your question. "
                    "Please try again or contact campus support."
                ),
                "result": None,
                "error": True,
            }
        )

    st.rerun()


def render_escalation_panel(store: SupportStore) -> None:
    """Consent-based request form, shown after a manual request or an automatic offer."""
    state = st.session_state
    if state.escalation_reason == "auto_offer":
        st.info(ESCALATION_OFFER_MESSAGE)

    with st.form("escalation_form"):
        st.markdown("##### Request help from student support")
        summary = st.text_area(
            "What do you need help with?",
            value=state.escalation_summary,
            max_chars=MAX_SUMMARY_CHARS,
            key=f"escalation_summary_{len(state.messages)}",
        )
        st.caption(ESCALATION_PRIVACY_NOTE)
        consent = st.checkbox(ESCALATION_CONSENT_LABEL, key="escalation_consent")
        submitted = st.form_submit_button("Send request")

    if st.button("Not now", key="btn_escalation_dismiss"):
        close_escalation(state)
        st.rerun()

    if not submitted:
        return
    try:
        submit_escalation(state, store, summary, consent)
    except ValueError as exc:
        st.warning(str(exc))
    except TicketLimitError:
        st.warning("You have too many open requests. Please wait for one to be resolved.")
    except SupportStoreError:
        st.error("Your request could not be saved. Please contact the student support office.")
    else:
        st.rerun()


def render_privacy_controls(store: SupportStore) -> None:
    """Sidebar privacy notice, opt-out, request status and optional feedback."""
    state = st.session_state
    st.markdown('<div class="sidebar-section-title">Privacy</div>', unsafe_allow_html=True)
    st.caption(PRIVACY_NOTICE)
    opted_out = st.toggle(
        OPT_OUT_LABEL, value=state.opted_out, key="toggle_opt_out", help=OPT_OUT_HELP
    )
    if opted_out != state.opted_out:
        try:
            set_opt_out(state, store, opted_out)
        except SupportStoreError:
            st.error(
                "Opt-out is on, but stored data could not be deleted. "
                "Turn it off and on again to retry."
            )
    if state.opted_out:
        st.caption(OPT_OUT_ACTIVE_TEXT)

    if state.last_ticket_reference:
        st.markdown(
            '<div class="sidebar-section-title">Your Requests</div>', unsafe_allow_html=True
        )
        try:
            for ticket in store.session_tickets(state.session_id):
                st.text(f"{ticket.reference}: {ticket.status.replace('_', ' ')}")
        except SupportStoreError:
            st.caption("Request status is unavailable right now.")

    if state.opted_out:
        return
    with st.expander("⭐ Share feedback (optional)", expanded=False):
        if state.feedback_sent:
            st.caption("Thanks for your feedback.")
            return
        stars = st.feedback("stars", key="session_rating")
        comment = st.text_input(
            "Anything to add? (optional)", max_chars=MAX_COMMENT_CHARS, key="session_comment"
        )
        if st.button("Send feedback", key="btn_send_feedback"):
            if stars is None:
                st.warning("Please choose a rating first.")
                return
            try:
                submit_feedback(state, store, stars + 1, comment)
            except (ValueError, SupportStoreError):
                st.error("Your feedback could not be saved.")
            else:
                st.rerun()


def render_ticket(ticket, store: SupportStore) -> None:
    """Show one ticket. Student text uses st.text so it is never rendered as markup."""
    with st.container(border=True):
        st.text(f"{ticket.reference} | {ticket.status.replace('_', ' ')} | {ticket.created_at}")
        st.text(
            f"Source: {ticket.reason.replace('_', ' ')} | "
            f"Intent hint: {ticket.intent or 'none'} | "
            f"{TONE_SIGNAL_LABEL}: {ticket.sentiment or 'none'}"
        )
        st.text(ticket.summary)
        for status in TICKET_STATUSES:
            if status != ticket.status and st.button(
                f"Mark {status.replace('_', ' ')}", key=f"ticket_{ticket.reference}_{status}"
            ):
                try:
                    store.update_status(ticket.reference, status)
                except SupportStoreError:
                    st.error("The ticket could not be updated.")
                else:
                    st.rerun()


def render_admin_ticket_queue(store: SupportStore) -> None:
    """Administrator ticket queue and recent feedback view."""
    selected = st.selectbox(
        "Status", ("open", "in_progress", "resolved", "all"), key="admin_status_filter"
    )
    try:
        tickets = store.list_tickets(None if selected == "all" else selected)
        feedback = store.recent_feedback()
    except SupportStoreError:
        st.error("The ticket queue is unavailable.")
        return

    st.caption(f"{len(tickets)} ticket(s). Student contact details are not stored.")
    if not tickets:
        st.info("No tickets match this filter.")
    for ticket in tickets:
        render_ticket(ticket, store)

    st.markdown("#### Recent Feedback")
    if not feedback:
        st.caption("No feedback yet.")
        return
    average = sum(entry.rating for entry in feedback) / len(feedback)
    st.metric("Average rating (recent, optional responses)", f"{average:.1f} / 5")
    for entry in feedback:
        if entry.comment:
            st.text(f"{entry.rating}/5: {entry.comment}")


def render_admin_analytics(store: SupportStore) -> None:
    """Administrator analytics dashboard based on real stored records."""
    st.markdown("#### Operational Support Analytics")
    st.caption(
        "Telemetry is computed strictly from stored support tickets and consented student feedback "
        "within the 30-day retention window. Unescalated student chat interactions are processed "
        "ephemerally and never logged."
    )
    try:
        data = get_admin_analytics(store, st.session_state)
    except (SupportStoreError, PermissionError):
        st.error("Analytics data is currently unavailable.")
        return

    counts = data["ticket_counts"]
    fb = data["feedback_metrics"]
    sentiments = data["sentiment_distribution"]
    trends = data["creation_trends"]

    # 1. Ticket Counts
    st.markdown("##### Support Tickets Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tickets", counts["total"])
    c2.metric("Open", counts["open"])
    c3.metric("In Progress", counts["in_progress"])
    c4.metric("Resolved", counts["resolved"])

    if counts["total"] == 0:
        st.info("No support tickets are currently recorded in the 30-day store.")
    else:
        status_df = pd.DataFrame(
            [
                {"Status": "Open", "Tickets": counts["open"]},
                {"Status": "In Progress", "Tickets": counts["in_progress"]},
                {"Status": "Resolved", "Tickets": counts["resolved"]},
            ]
        )
        st.bar_chart(status_df, x="Status", y="Tickets", color="#2563eb")

    # 2. Feedback Metrics
    st.markdown("##### Feedback Submissions & Rating Distribution")
    if fb["total"] == 0:
        st.info("No student feedback has been submitted.")
    else:
        fc1, fc2 = st.columns(2)
        fc1.metric("Total Submissions", fb["total"])
        fc2.metric("Average Rating", f"{fb['average']:.1f} / 5")

        dist_rows = []
        for star in range(5, 0, -1):
            count = fb["distribution"].get(star, 0)
            pct = (count / fb["total"] * 100) if fb["total"] > 0 else 0
            dist_rows.append(f"| {'⭐' * star} ({star} Star) | {count} | {pct:.1f}% |")
        st.markdown(
            "| Star Rating | Submissions | Percentage |\n"
            "| :--- | :--- | :--- |\n" + "\n".join(dist_rows)
        )

    # 3. Sentiment Distribution
    st.markdown("##### Sentiment Distribution (From Escalated Tickets)")
    if not sentiments:
        st.info("Insufficient data: No stored escalation tickets currently contain tone or sentiment labels.")
    else:
        total_sentiment_tickets = sum(sentiments.values())
        sent_rows = []
        for s_label, count in sorted(sentiments.items()):
            pct = (count / total_sentiment_tickets * 100) if total_sentiment_tickets > 0 else 0
            sent_rows.append(f"| `{s_label.capitalize()}` | {count} | {pct:.1f}% |")
        st.markdown(
            "| Tone Signal | Ticket Count | Percentage |\n"
            "| :--- | :--- | :--- |\n" + "\n".join(sent_rows)
        )
        st.caption("Derived exclusively from escalated tickets where tone signal was recorded.")

    # 4. Creation Trends Over Time
    st.markdown("##### Ticket Creation Trends (By Day)")
    if not trends:
        st.info("Insufficient data to display creation trends.")
    else:
        trend_df = pd.DataFrame(trends, columns=["Date", "Tickets Created"])
        st.bar_chart(trend_df, x="Date", y="Tickets Created", color="#0284c7")
        trend_rows = [f"| {day} | {count} |" for day, count in trends]
        st.markdown(
            "| Date (UTC) | Tickets Created |\n"
            "| :--- | :--- |\n" + "\n".join(trend_rows)
        )


def render_knowledge_base_viewer() -> None:
    """Administrator searchable read-only viewer for the runtime FAQ records."""
    record_set = st.selectbox(
        "Record set", list(KB_RECORD_SETS), key="kb_record_set"
    )
    st.markdown(f"#### {record_set}")
    st.caption(KB_RECORD_SETS[record_set])

    try:
        kb_df = (
            load_runtime_knowledge_base()
            if record_set == KB_RUNTIME_LABEL
            else load_evaluation_knowledge_base()
        )
    except Exception:
        st.error("Knowledge base could not be loaded.")
        return

    intents = sorted(list(set(kb_df["Intent"].dropna().unique())))

    col_search, col_filter = st.columns([2, 1])
    with col_search:
        search_query = st.text_input(
            "🔍 Search Knowledge Base",
            key="kb_search_input",
            placeholder="Search by question, answer, topic...",
        )
    with col_filter:
        intent_filter = st.selectbox(
            "Filter by Intent", ["All"] + intents, key="kb_intent_filter"
        )

    filtered = search_knowledge_base(kb_df, search_query, intent_filter)
    st.caption(f"Displaying {len(filtered)} of {len(kb_df)} records.")

    if filtered.empty:
        st.info("No knowledge base records match your search criteria.")
    else:
        cols_to_show = ["User Message", "Bot Response", "Intent"]
        has_topic = "Topic" in filtered.columns
        if has_topic:
            cols_to_show.append("Topic")
        display_df = filtered[cols_to_show].copy()
        display_df.columns = [
            "FAQ Question",
            "Answer",
            "Intent",
            *(["Topic"] if has_topic else []),
        ]

        # Render readable, text-wrapped table within a contained scrollable container
        rows_html = []
        for _, row in display_df.iterrows():
            q_esc = html.escape(str(row["FAQ Question"]))
            a_esc = html.escape(str(row["Answer"]))
            i_esc = html.escape(str(row["Intent"]))
            topic_cell = (
                f'<td class="kb-cell-topic">{html.escape(str(row["Topic"]))}</td>'
                if has_topic
                else ""
            )
            rows_html.append(
                f'<tr>'
                f'<td class="kb-cell-question"><strong>{q_esc}</strong></td>'
                f'<td class="kb-cell-answer">{a_esc}</td>'
                f'<td class="kb-cell-intent"><span class="kb-intent-badge">{i_esc}</span></td>'
                f'{topic_cell}'
                f'</tr>'
            )

        topic_th = '<th class="kb-th-topic">Topic</th>' if has_topic else ""
        table_html = (
            f'<div class="kb-viewer-container">'
            f'  <table class="kb-viewer-table">'
            f'    <thead>'
            f'      <tr>'
            f'        <th class="kb-th-question">FAQ Question</th>'
            f'        <th class="kb-th-answer">Answer</th>'
            f'        <th class="kb-th-intent">Intent</th>'
            f'        {topic_th}'
            f'      </tr>'
            f'    </thead>'
            f'    <tbody>'
            f'      {"".join(rows_html)}'
            f'    </tbody>'
            f'  </table>'
            f'</div>'
        )
        st.markdown(table_html, unsafe_allow_html=True)


def render_admin_export(store: SupportStore) -> None:
    """Administrator CSV export view with PII redaction and formula neutralization."""
    st.markdown("#### Export Support Data")
    st.caption(
        "Download sanitized UTF-8 CSV exports for offline reporting and compliance auditing. "
        "All exports enforce PII redaction, neutralise formula injection, and omit internal IDs. "
        "Records older than 30 days are automatically purged."
    )

    try:
        tickets_csv = export_tickets_csv(store, st.session_state)
        feedback_csv = export_feedback_csv(store, st.session_state)
    except PermissionError:
        st.error("Administrator authorization required for data export.")
        return
    except SupportStoreError:
        st.error("Data export is temporarily unavailable.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 📥 Support Tickets CSV")
        st.caption(
            "Contains active ticket references, UTC timestamps, statuses, reasons, "
            "intents, sentiments, and redacted summaries."
        )
        st.download_button(
            label="Download Tickets CSV",
            data=tickets_csv,
            file_name="support_tickets.csv",
            mime="text/csv",
            key="btn_download_tickets_csv",
            use_container_width=True,
        )

    with col2:
        st.markdown("##### 📥 Student Feedback CSV")
        st.caption("Contains submission timestamps, star ratings (1-5), and redacted comments.")
        st.download_button(
            label="Download Feedback CSV",
            data=feedback_csv,
            file_name="student_feedback.csv",
            mime="text/csv",
            key="btn_download_feedback_csv",
            use_container_width=True,
        )


def render_admin_portal(store: SupportStore) -> None:
    """Administrator portal with Ticket Queue, Analytics, Knowledge Base, and Data Export."""
    state = st.session_state
    st.markdown("### Staff Support Portal")

    if not admin_is_configured():
        st.warning("The staff portal has not been configured on this server.")
        return

    if not state.admin_authenticated:
        if is_locked_out(state):
            st.error("Too many failed attempts. Start a new session to try again.")
            return
        with st.form("admin_login"):
            password = st.text_input("Administrator password", type="password")
            signed_in = st.form_submit_button("Sign in")
        if signed_in:
            if attempt_login(state, password) or is_locked_out(state):
                st.rerun()
            st.error("Incorrect password.")
        return

    if st.button("Sign out", key="btn_admin_sign_out"):
        state.admin_authenticated = False
        st.rerun()

    tab_queue, tab_analytics, tab_kb, tab_export = st.tabs(
        ["📋 Ticket Queue", "📊 Analytics Dashboard", "📖 Knowledge Base", "💾 Data Export"]
    )

    with tab_queue:
        render_admin_ticket_queue(store)

    with tab_analytics:
        render_admin_analytics(store)

    with tab_kb:
        render_knowledge_base_viewer()

    with tab_export:
        render_admin_export(store)


def main() -> None:
    # 0. Inject SaaS Design System Styles
    st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

    # 1. Startup & artifact loading with safe error boundary
    try:
        pipeline = load_cached_pipeline()
    except Exception:
        logger.exception("Artifact verification or pipeline loading failed at startup")
        st.error(
            "Application Startup Failure: Artifact verification or pipeline loading failed. "
            "Check the server log, or run "
            "`python -c \"from src.ml.artifacts import verify_artifacts; verify_artifacts()\"` "
            "from the project root."
        )
        st.stop()

    # 2. Session state initialization (guaranteed before widget interactions)
    if "messages" not in st.session_state:
        st.session_state.messages = get_initial_messages()
    init_support_state(st.session_state)
    # Holds no connection and touches no disk until something is stored.
    store = SupportStore()
    staff_view = st.session_state.portal_view == "admin"

    # 3. Main product header (compact identity strip)
    st.markdown(
        '<header class="app-header product-header">'
        '  <div class="product-identity-row">'
        '    <div class="product-brand">'
        '      <div class="brand-badge">🎓</div>'
        '      <div class="brand-text">'
        f'        <h1 class="brand-title">{PAGE_TITLE}</h1>'
        f'        <span class="brand-meta">{PAGE_SUBTITLE}</span>'
        '      </div>'
        '    </div>'
        '    <div class="product-status-tag">'
        '      <span class="status-indicator-dot"></span>'
        '      <span>Classical ML Baseline · Single-Turn</span>'
        '    </div>'
        '  </div>'
        '  <div class="product-header-subbar">'
        f'    <span class="subbar-disclaimer">ℹ️ {DISCLAIMER_TEXT}</span>'
        '  </div>'
        '</header>',
        unsafe_allow_html=True,
    )

    # 4. Sidebar setup
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand-header">'
            '  <span class="sidebar-brand-icon">🎓</span>'
            '  <span class="sidebar-brand-title">Campus AI Assistant</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        if not staff_view:
            st.markdown('<div class="sidebar-section-title">Sample Questions</div>', unsafe_allow_html=True)
            for example_query in EXAMPLE_QUESTIONS:
                if st.button(example_query, key=f"btn_{example_query}", use_container_width=True):
                    process_user_query(example_query, pipeline)

        with st.expander("ℹ️ System Information", expanded=False):
            st.markdown(
                '<div class="sidebar-meta-list">'
                f'  <div class="meta-item"><span class="meta-label">Classifier</span><span class="meta-val">LinearSVC + TF-IDF</span></div>'
                f'  <div class="meta-item"><span class="meta-label">Model Hash</span><span class="meta-val" title="{pipeline.model_version}">{pipeline.model_version[:14]}…</span></div>'
                f'  <div class="meta-item"><span class="meta-label">Knowledge Base</span><span class="meta-val">{pipeline.kb_version} ({len(pipeline.retriever.df)} rows)</span></div>'
                f'  <div class="meta-item"><span class="meta-label">Safety Policy</span><span class="meta-val">{pipeline.policy_version}</span></div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown("##### Architecture & Safeguards")
            st.markdown(ABOUT_TEXT)

        st.markdown('<div class="sidebar-section-title">Session</div>', unsafe_allow_html=True)
        if not staff_view:
            if st.button("🗑️ Clear Conversation", key="btn_clear_conversation", use_container_width=True):
                st.session_state.messages = get_initial_messages()
                close_escalation(st.session_state)
                st.rerun()
            if st.button("🙋 Talk to a Person", key="btn_talk_to_person", use_container_width=True):
                open_escalation(st.session_state, "manual")
                st.rerun()

        portal_label = "← Back to Student Chat" if staff_view else "🛡️ Staff Portal"
        if st.button(portal_label, key="btn_portal_switch", use_container_width=True):
            st.session_state.portal_view = "student" if staff_view else "admin"
            st.rerun()

        if not staff_view:
            render_privacy_controls(store)

        st.markdown(
            f'<div class="sidebar-disclaimer">ℹ️ {DISCLAIMER_TEXT}</div>',
            unsafe_allow_html=True,
        )

    if staff_view:
        render_admin_portal(store)
        return

    # 5. Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_message(
                    msg["content"],
                    result=msg.get("result"),
                    error=msg.get("error"),
                    sentiment=msg.get("sentiment"),
                )
            else:
                st.markdown(msg["content"])
                if msg.get("redacted"):
                    st.caption(REDACTION_NOTICE)

    # 6. Interactive Suggested Questions (Welcome state only)
    if len(st.session_state.messages) == 1:
        st.markdown(
            '<div class="welcome-container">'
            '  <div class="welcome-section-header">'
            '    <div class="welcome-section-title">Suggested Inquiries</div>'
            '    <div class="welcome-section-subtitle">Select a topic below for instant guidance, or enter your question in the composer:</div>'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            for icon, title, query in SUGGESTED_TILES[:2]:
                if st.button(
                    f"**{title}**\n\n{query}",
                    icon=icon,
                    key=f"tile_{query}",
                    use_container_width=True,
                ):
                    process_user_query(query, pipeline)
        with col2:
            for icon, title, query in SUGGESTED_TILES[2:]:
                if st.button(
                    f"**{title}**\n\n{query}",
                    icon=icon,
                    key=f"tile_{query}",
                    use_container_width=True,
                ):
                    process_user_query(query, pipeline)

        st.markdown(
            '<div class="composer-hint-bridge">'
            '  <span class="bridge-dash"></span>'
            '  <span class="bridge-label">or ask any custom question below</span>'
            '  <span class="bridge-dash"></span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # 7. Support request confirmation and form
    if st.session_state.last_ticket_reference:
        st.success(
            f"Request sent. Your reference is {st.session_state.last_ticket_reference}. "
            f"{TICKET_FOLLOW_UP_TEXT}"
        )
    if st.session_state.escalation_open:
        render_escalation_panel(store)

    # 8. Chat input for new student query
    user_input = st.chat_input("Ask a question about financial aid, registration, Wi-Fi, campus services...")
    if user_input:
        process_user_query(user_input, pipeline)


if __name__ == "__main__":
    main()
