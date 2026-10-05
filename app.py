"""
Streamlit Web Application for University Student Support (Phase 5C).

A thin, restrained presentation layer for student support questions.
Delegates all intent classification and response gating to src.ml.answer().
"""
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st

from src.ml.artifacts import verify_artifacts
from src.ml.config import MAX_MESSAGE_CHARS
from src.ml.pipeline import ChatPipeline, get_pipeline
from src.ml.schemas import ChatResult
from src.ui.helpers import (
    ABOUT_TEXT,
    DISCLAIMER_TEXT,
    EXAMPLE_QUESTIONS,
    PAGE_SUBTITLE,
    PAGE_TITLE,
    extract_diagnostics,
    format_outcome_badge,
    get_initial_messages,
    validate_user_message,
)

# Configure page settings
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🎓",
    layout="centered",
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


def render_assistant_message(content: str, result: ChatResult = None, error: str = None) -> None:
    """Render an assistant response with outcome indicators and optional diagnostics."""
    st.markdown(content)

    if error:
        with st.expander("Technical details"):
            st.code(error)
        return

    if result is None:
        return

    # Outcome explanation
    badge = format_outcome_badge(result.outcome)
    st.caption(f"ℹ️ {badge['explanation']}")

    # Technical diagnostics expander
    with st.expander("🔍 View system diagnostics"):
        diagnostics = extract_diagnostics(result)
        for label, val in diagnostics.items():
            if isinstance(val, list):
                st.text(f"{label}: {', '.join(str(x) for x in val)}")
            else:
                st.text(f"{label}: {val}")


def process_user_query(query_text: str, pipeline: ChatPipeline) -> None:
    """Validate query, invoke pipeline, and append messages to session state."""
    is_valid, error_msg = validate_user_message(query_text, max_chars=MAX_MESSAGE_CHARS)
    if not is_valid:
        st.warning(error_msg)
        return

    # Append user message
    st.session_state.messages.append({"role": "user", "content": query_text, "result": None})

    # Call ML inference pipeline
    try:
        res = pipeline.answer(query_text)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": res.answer,
                "result": res,
            }
        )
    except Exception as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "An unexpected error occurred while processing your question. "
                    "Please try again or contact campus support."
                ),
                "result": None,
                "error": str(exc),
            }
        )

    st.rerun()


def main() -> None:
    # 1. Startup & artifact loading with safe error boundary
    try:
        pipeline = load_cached_pipeline()
    except Exception as exc:
        st.error(
            "Application Startup Failure: Artifact verification or pipeline loading failed.\n\n"
            f"Details: {exc}"
        )
        st.stop()

    # 2. Main title and header
    st.title(f"🎓 {PAGE_TITLE}")
    st.markdown(f"**{PAGE_SUBTITLE}**")
    st.caption(f"*{DISCLAIMER_TEXT}*")
    st.divider()

    # 3. Sidebar setup
    with st.sidebar:
        st.header("About")
        st.markdown(ABOUT_TEXT)
        st.info(f"💡 **Note**: {DISCLAIMER_TEXT}")

        st.header("Example Questions")
        st.markdown("Click any example to submit it directly:")
        for example_query in EXAMPLE_QUESTIONS:
            if st.button(example_query, key=f"btn_{example_query}", use_container_width=True):
                process_user_query(example_query, pipeline)

        st.header("System Information")
        st.text(f"Model: {pipeline.model_version}")
        st.text(f"KB:    {pipeline.kb_version}")
        st.text(f"Policy: {pipeline.policy_version}")

        st.header("Conversation")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = get_initial_messages()
            st.rerun()

    # 4. Session state initialization
    if "messages" not in st.session_state:
        st.session_state.messages = get_initial_messages()

    # 5. Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_message(
                    msg["content"],
                    result=msg.get("result"),
                    error=msg.get("error"),
                )
            else:
                st.markdown(msg["content"])

    # 6. Chat input for new student query
    user_input = st.chat_input("Ask a university support question...")
    if user_input:
        process_user_query(user_input, pipeline)


if __name__ == "__main__":
    main()
