"""
Presentation helpers and UI-specific state utilities for Streamlit (Phase 5C).

Maintains a strict presentation-only boundary: performs no inference logic,
modifies no thresholds, and delegates all answering behavior to the ML core.
"""
from typing import Any, Dict, List, Optional, Tuple

from src.ml.config import MAX_MESSAGE_CHARS
from src.ml.schemas import ChatResult, Outcome

PAGE_TITLE = "University Student Support"
PAGE_SUBTITLE = "AI-powered assistant for common student support questions."
DISCLAIMER_TEXT = "Each question is evaluated independently. This assistant does not have conversational memory."

WELCOME_MESSAGE = (
    "Hi! I can help with common university questions about academics, "
    "registration, financial aid, campus services, careers, and more."
)

EXAMPLE_QUESTIONS = (
    "How do I apply for financial aid?",
    "How can I connect to campus Wi-Fi?",
    "When is course registration?",
    "Where can I find student services?",
)

ABOUT_TEXT = (
    "This is an AI-powered student support assistant prototype.\n\n"
    "- **Underlying Technology**: Supervised intent classification (Linear SVM with TF-IDF) "
    "combined with lexical analysis and paraphrase retrieval.\n"
    "- **Curated Knowledge Base**: Responses are retrieved from a verified repository of "
    "university support questions and canned answers.\n"
    "- **Independent Evaluation**: Each question is scored independently; previous queries "
    "do not influence subsequent responses.\n"
    "- **Safety Policy**: When the system cannot find a reliable answer with sufficient confidence "
    "or lexical evidence, it safely abstains or offers general area guidance rather than inventing answers."
)


def validate_user_message(
    message: Optional[str],
    max_chars: int = MAX_MESSAGE_CHARS,
) -> Tuple[bool, Optional[str]]:
    """
    Validate user input before sending to the inference pipeline.

    Enforces the configured maximum character length (500) and guards against
    empty/whitespace-only submissions.

    Parameters
    ----------
    message : str or None
        Raw input string from the user.
    max_chars : int, optional
        Maximum allowed character length (default: MAX_MESSAGE_CHARS = 500).

    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_valid, error_message)
    """
    if message is None or not message.strip():
        return False, "Please enter a question so I can assist you."

    if len(message) > max_chars:
        return (
            False,
            f"Message exceeds the maximum limit of {max_chars} characters "
            f"({len(message)} characters). Please shorten your question.",
        )

    return True, None


def get_initial_messages() -> List[Dict[str, Any]]:
    """Construct the initial message history with the standard assistant welcome."""
    return [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
            "result": None,
        }
    ]


def format_outcome_badge(outcome: Outcome) -> Dict[str, str]:
    """
    Provide user-friendly presentation labels and explanations for the three outcomes.

    Does NOT inspect margins, Dice scores, or cosine similarity.
    """
    if outcome == Outcome.ANSWERED:
        return {
            "status": "ANSWERED",
            "title": "Verified Knowledge Base Answer",
            "explanation": "This response matched a verified university knowledge base question.",
        }
    elif outcome == Outcome.INTENT_FALLBACK:
        return {
            "status": "INTENT_FALLBACK",
            "title": "General Topic Guidance",
            "explanation": "The assistant identified your general inquiry area, but could not find a sufficiently specific match in the knowledge base.",
        }
    else:  # Outcome.ABSTAINED
        return {
            "status": "ABSTAINED",
            "title": "Clarification Requested",
            "explanation": "The assistant could not reliably determine an answer and is asking for rephrasing or more detail.",
        }


def extract_diagnostics(result: ChatResult) -> Dict[str, Any]:
    """
    Extract populated diagnostic fields from ChatResult for optional technical inspection.

    Ensures the LinearSVC margin is explicitly labeled as an uncalibrated margin
    and never converted into a fake probability or confidence percentage.
    """
    raw_fields: List[Tuple[str, Any]] = [
        ("Outcome", result.outcome.value if result.outcome else None),
        ("Committed Intent", result.intent),
        ("Predicted Intent", result.predicted_intent),
        (
            "Classification margin (uncalibrated)",
            f"{result.margin:.4f}" if result.margin is not None else None,
        ),
        (
            "Retrieval Cosine Similarity",
            f"{result.similarity:.4f}" if result.similarity is not None else None,
        ),
        (
            "Content-Token Overlap (Dice)",
            f"{result.overlap:.4f}" if result.overlap is not None else None,
        ),
        ("Matched Knowledge-Base Question", result.matched_question),
        ("Matched Knowledge-Base Row ID", result.matched_row_id),
        (
            "Shared Content Tokens",
            list(result.shared_content_tokens) if result.shared_content_tokens else None,
        ),
        ("Abstention Reason", result.abstain_reason.value if result.abstain_reason else None),
        ("Answer Source", result.answer_source),
        ("Model Version", result.model_version),
        ("Knowledge Base Version", result.kb_version),
        ("Policy Version", result.policy_version),
    ]

    return {label: val for label, val in raw_fields if val is not None}
