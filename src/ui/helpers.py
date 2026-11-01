"""
Presentation helpers and UI-specific state utilities for Streamlit (Phase 5C).

Maintains a strict presentation-only boundary: performs no inference logic,
modifies no thresholds, and delegates all answering behavior to the ML core.
"""
from typing import Any, Dict, List, Optional, Tuple

from src.ml.config import MAX_MESSAGE_CHARS
from src.ml.schemas import ChatResult, Outcome
from src.support.config import RETENTION_DAYS

PAGE_TITLE = "University Student Support"
PAGE_SUBTITLE = "AI-powered assistant for common student support questions."
DISCLAIMER_TEXT = "Each question is evaluated independently. This assistant does not have conversational memory."

WELCOME_MESSAGE = (
    "Hi! I can help with common university questions about academics, "
    "registration, financial aid, campus services, careers, and more."
)

EXAMPLE_QUESTIONS = (
    "How do I add a course to my schedule?",
    "When will my financial aid be disbursed?",
    "When is the library open on weekends?",
    "Who do I contact about accessibility services?",
)

SUGGESTED_TILES = (
    ("📝", "Course Registration", "How do I add a course to my schedule?"),
    ("💳", "Financial Aid", "When will my financial aid be disbursed?"),
    ("📚", "Library Hours", "When is the library open on weekends?"),
    ("♿", "Student Services", "Who do I contact about accessibility services?"),
)

ABOUT_TEXT = (
    "This is an AI-powered student support assistant prototype.\n\n"
    "- **Underlying Technology**: Supervised intent classification (Linear SVM with TF-IDF) "
    "combined with lexical analysis and paraphrase retrieval.\n"
    "- **Curated Knowledge Base**: Responses are retrieved from a stored set of "
    "university support questions and canned answers.\n"
    "- **Independent Evaluation**: Each question is scored independently; previous queries "
    "do not influence subsequent responses.\n"
    "- **Safety Policy**: When the system cannot find a reliable answer with sufficient confidence "
    "or lexical evidence, it safely abstains or offers general area guidance rather than inventing answers."
)


KB_RUNTIME_LABEL = "Runtime FAQ Knowledge Base (200 Records)"
KB_EVALUATION_LABEL = "Evaluation Subset (160 Records)"
KB_RECORD_SETS = {
    KB_RUNTIME_LABEL: "Read-only reference of every question-answer pair the live assistant can return.",
    KB_EVALUATION_LABEL: (
        "Read-only view of the training-split rows used for full-pipeline evaluation metrics. "
        "The live assistant searches all 200 records, not only this subset."
    ),
}

PRIVACY_NOTICE = (
    "This chat is not monitored. Personal details such as email addresses and phone numbers "
    "are removed automatically when detected, but detection is not perfect, so please avoid "
    "sharing them. Your conversation is held temporarily in the active app session and is not "
    "written to the support database. Only a support request or feedback you "
    f"choose to send is stored, for up to {RETENTION_DAYS} days."
)
OPT_OUT_LABEL = "Don't keep my data"
OPT_OUT_HELP = (
    "Deletes any support request or feedback stored from this session, including open "
    "requests, and turns off feedback. It applies to this browser session only."
)
OPT_OUT_ACTIVE_TEXT = (
    "Opted out. Nothing is stored unless you send a support request yourself."
)
REDACTION_NOTICE = "Personal details were removed from this message before it was processed."
TONE_SIGNAL_LABEL = "Tone Signal (heuristic, not validated)"
ESCALATION_CONSENT_LABEL = (
    "I agree to this summary being stored and shared with student support staff."
)
ESCALATION_PRIVACY_NOTE = (
    "Detected personal details are removed. Please don't add contact details: they aren't "
    "stored, and staff won't be able to reply directly."
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
    Provide user-friendly presentation labels, icons, and styling for the three outcomes.

    Does NOT inspect margins, Dice scores, or cosine similarity.
    """
    if outcome == Outcome.ANSWERED:
        return {
            "status": "ANSWERED",
            "title": "Direct Knowledge Base Match",
            "explanation": "This response is the stored answer to a closely matching knowledge base question.",
            "icon": "✓",
            "badge_class": "answered",
        }
    elif outcome == Outcome.INTENT_FALLBACK:
        return {
            "status": "INTENT_FALLBACK",
            "title": "General Topic Guidance",
            "explanation": "The assistant identified your general inquiry area, but could not find a sufficiently specific match in the knowledge base.",
            "icon": "ℹ",
            "badge_class": "fallback",
        }
    else:  # Outcome.ABSTAINED
        return {
            "status": "ABSTAINED",
            "title": "Clarification Requested",
            "explanation": "The assistant could not reliably determine an answer and is asking for rephrasing or more detail.",
            "icon": "?",
            "badge_class": "abstained",
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


CUSTOM_CSS = """
/* ==========================================================================
   University AI Assistant - Minimal SaaS Design System
   ========================================================================== */

/* 1. Global Reset & Warm-Neutral Foundation */
.stApp {
    background-color: #FAF9F6 !important;
    color: #0F172A !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Constrained, centered reading column with balanced vertical rhythm */
.main .block-container,
section[data-testid="stMain"] .block-container {
    max-width: 860px !important;
    padding-top: 1rem !important;
    padding-bottom: 4.85rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    margin: 0 auto !important;
}

/* 2. Compact Product Identity Header (Non-boxed, minimal top strip) */
.product-header, .app-header {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid #E8E8E3 !important;
    border-radius: 0 !important;
    padding: 0 0 0.75rem 0 !important;
    margin-bottom: 1rem !important;
    box-shadow: none !important;
}

.product-identity-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.65rem;
}

.product-brand {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}

.brand-badge {
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: #EEF2FF;
    border: 1px solid #C7D2FE;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}

.brand-text {
    display: flex;
    align-items: baseline;
    gap: 0.45rem;
    flex-wrap: wrap;
}

.brand-title {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    letter-spacing: -0.02em !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}

.brand-meta {
    font-size: 0.785rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
}

.product-status-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.2rem 0.55rem;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 500;
    color: #334155;
    letter-spacing: 0.01em;
}

.status-indicator-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #10B981;
}

.product-header-subbar {
    margin-top: 0.35rem;
    display: flex;
    align-items: center;
}

.subbar-disclaimer {
    font-size: 0.74rem;
    color: #64748B;
    line-height: 1.35;
}

/* 3. Welcome Screen Stage & Suggestion Prompt Cards */
.welcome-container {
    margin: 0.5rem 0 0.75rem 0;
}

.welcome-section-header {
    margin-bottom: 0.65rem;
}

.welcome-section-title {
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #475569 !important;
}

.welcome-section-subtitle {
    font-size: 0.835rem !important;
    color: #64748B !important;
    margin-top: 0.15rem !important;
    line-height: 1.4 !important;
}

/* Responsive 2x2 Suggestion Grid (Prompt Cards) */
div[data-testid="stColumn"] div.stButton,
div[data-testid="stColumn"] .stButton,
div[data-testid="column"] div.stButton,
div[data-testid="column"] .stButton {
    margin-bottom: 0.65rem !important;
    width: 100% !important;
}

div[data-testid="stColumn"] .stButton > button,
div[data-testid="stColumn"] .stButton button,
div[data-testid="column"] .stButton > button,
div[data-testid="column"] .stButton button {
    position: relative !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
    text-align: left !important;
    padding: 0.95rem 1.15rem !important;
    padding-right: 2.5rem !important;
    min-height: 84px !important;
    border-radius: 12px !important;
    border: 1px solid #E2E8F0 !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02) !important;
    transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
    width: 100% !important;
    cursor: pointer !important;
    overflow: visible !important;
}

/* Override Streamlit button internal flex wrappers */
div[data-testid="stColumn"] .stButton button > div,
div[data-testid="column"] .stButton button > div {
    display: flex !important;
    flex-direction: row !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
    width: 100% !important;
    text-align: left !important;
}

div[data-testid="stColumn"] .stButton button > div > span,
div[data-testid="column"] .stButton button > div > span {
    display: flex !important;
    flex-direction: row !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
    gap: 0.75rem !important;
    width: 100% !important;
    text-align: left !important;
}

div[data-testid="stColumn"] .stButton button > div > span > span,
div[data-testid="column"] .stButton button > div > span > span {
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
    flex: 1 1 auto !important;
    width: 100% !important;
    text-align: left !important;
}

/* Distinct icon badge in prompt card */
div[data-testid="stColumn"] .stButton button [data-testid="stIconEmoji"],
div[data-testid="stColumn"] .stButton button [data-testid="stIconMaterial"],
div[data-testid="stColumn"] .stButton button span[class*="stIcon"],
div[data-testid="column"] .stButton button [data-testid="stIconEmoji"],
div[data-testid="column"] .stButton button [data-testid="stIconMaterial"],
div[data-testid="column"] .stButton button span[class*="stIcon"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 34px !important;
    height: 34px !important;
    min-width: 34px !important;
    min-height: 34px !important;
    font-size: 1.15rem !important;
    background-color: #EEF2FF !important;
    border: 1px solid #C7D2FE !important;
    border-radius: 9px !important;
    box-shadow: 0 1px 2px rgba(79, 70, 229, 0.05) !important;
    flex-shrink: 0 !important;
    margin-top: 0.1rem !important;
    transition: all 0.15s ease-in-out !important;
}

div[data-testid="stColumn"] .stButton button:hover [data-testid="stIconEmoji"],
div[data-testid="stColumn"] .stButton button:hover [data-testid="stIconMaterial"],
div[data-testid="column"] .stButton button:hover [data-testid="stIconEmoji"],
div[data-testid="column"] .stButton button:hover [data-testid="stIconMaterial"] {
    background-color: #E0E7FF !important;
    border-color: #A5B4FC !important;
}

/* Subtle top-right action indicator arrow (↗) */
div[data-testid="stColumn"] .stButton > button::after,
div[data-testid="stColumn"] .stButton button::after,
div[data-testid="column"] .stButton > button::after,
div[data-testid="column"] .stButton button::after {
    content: "↗" !important;
    position: absolute !important;
    top: 0.95rem !important;
    right: 1.05rem !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
    line-height: 1 !important;
    transition: all 0.18s ease-in-out !important;
    pointer-events: none !important;
    z-index: 2 !important;
}

div[data-testid="stColumn"] .stButton > button:hover::after,
div[data-testid="stColumn"] .stButton button:hover::after,
div[data-testid="column"] .stButton > button:hover::after,
div[data-testid="column"] .stButton button:hover::after {
    color: #4F46E5 !important;
    transform: translate(2px, -2px) !important;
}

div[data-testid="stColumn"] .stButton > button:hover,
div[data-testid="stColumn"] .stButton button:hover,
div[data-testid="column"] .stButton > button:hover,
div[data-testid="column"] .stButton button:hover {
    border-color: #4F46E5 !important;
    background-color: #FAFAFF !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.09), 0 2px 4px rgba(15, 23, 42, 0.03) !important;
    transform: translateY(-2px) !important;
}

div[data-testid="stColumn"] .stButton > button:focus-visible,
div[data-testid="stColumn"] .stButton button:focus-visible,
div[data-testid="column"] .stButton > button:focus-visible,
div[data-testid="column"] .stButton button:focus-visible {
    outline: none !important;
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.18) !important;
}

div[data-testid="stColumn"] .stButton > button:active,
div[data-testid="stColumn"] .stButton button:active,
div[data-testid="column"] .stButton > button:active,
div[data-testid="column"] .stButton button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
}

/* Prompt Card Text Structure */
div[data-testid="stColumn"] .stButton button [data-testid="stCaptionContainer"],
div[data-testid="stColumn"] .stButton button [data-testid="stMarkdownContainer"],
div[data-testid="column"] .stButton button [data-testid="stCaptionContainer"],
div[data-testid="column"] .stButton button [data-testid="stMarkdownContainer"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
    width: 100% !important;
    text-align: left !important;
    white-space: normal !important;
    gap: 0.25rem !important;
}

div[data-testid="stColumn"] .stButton button p,
div[data-testid="column"] .stButton button p {
    margin: 0 !important;
    padding: 0 !important;
    text-align: left !important;
    width: 100% !important;
}

/* Category & Title Header */
div[data-testid="stColumn"] .stButton button p:first-child,
div[data-testid="stColumn"] .stButton button p:first-of-type,
div[data-testid="column"] .stButton button p:first-child,
div[data-testid="column"] .stButton button p:first-of-type {
    font-size: 0.885rem !important;
    font-weight: 650 !important;
    color: #0F172A !important;
    line-height: 1.35 !important;
    letter-spacing: -0.01em !important;
}

div[data-testid="stColumn"] .stButton button p:first-child strong,
div[data-testid="column"] .stButton button p:first-child strong {
    font-weight: 700 !important;
    color: #0F172A !important;
}

/* Question Body */
div[data-testid="stColumn"] .stButton button p:last-child,
div[data-testid="stColumn"] .stButton button p:last-of-type,
div[data-testid="column"] .stButton button p:last-child,
div[data-testid="column"] .stButton button p:last-of-type {
    font-size: 0.815rem !important;
    font-weight: 400 !important;
    color: #64748B !important;
    line-height: 1.4 !important;
}

div[data-testid="stColumn"] .stButton button:hover p:last-child,
div[data-testid="column"] .stButton button:hover p:last-child {
    color: #334155 !important;
}

/* Central Column to Bottom Composer Connector */
.composer-hint-bridge {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    margin: 0.45rem auto 0.15rem auto;
    max-width: 360px;
}

.bridge-dash {
    flex: 1;
    height: 1px;
    background: #E8E8E3;
}

.bridge-label {
    font-size: 0.74rem;
    color: #64748B;
    font-weight: 500;
    letter-spacing: 0.01em;
    white-space: nowrap;
}

/* 4. Chat Messages & Hierarchy */
div[data-testid="stChatMessage"] {
    background-color: transparent !important;
    padding: 0.35rem 0 !important;
}

/* User Message Bubble */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    background-color: #EEF2FF !important;
    border: 1px solid #C7D2FE !important;
    border-radius: 14px 14px 4px 14px !important;
    padding: 0.85rem 1.15rem !important;
    margin-bottom: 0.85rem !important;
    margin-left: auto !important;
    max-width: 86% !important;
    color: #1E1B4B !important;
    box-shadow: 0 1px 2px rgba(79, 70, 229, 0.04) !important;
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) p {
    font-size: 0.935rem !important;
    font-weight: 500 !important;
    line-height: 1.5 !important;
    color: #1E1B4B !important;
    margin: 0 !important;
}

/* Assistant Message Bubble */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px 14px 14px 4px !important;
    padding: 1.1rem 1.35rem !important;
    margin-bottom: 1rem !important;
    margin-right: auto !important;
    max-width: 96% !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02) !important;
    color: #0F172A !important;
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) p {
    font-size: 0.95rem !important;
    font-weight: 450 !important;
    line-height: 1.6 !important;
    color: #0F172A !important;
    margin-bottom: 0.55rem !important;
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) p:last-child {
    margin-bottom: 0 !important;
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) ul,
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) ol {
    margin-top: 0.35rem !important;
    margin-bottom: 0.65rem !important;
    padding-left: 1.25rem !important;
    font-size: 0.935rem !important;
    line-height: 1.55 !important;
    color: #1E293B !important;
}

/* Avatars */
div[data-testid="stChatMessageAvatarUser"],
div[data-testid="stChatMessageAvatarAssistant"] {
    border-radius: 8px !important;
    font-size: 1rem !important;
}

div[data-testid="stChatMessageAvatarUser"] {
    background-color: #E0E7FF !important;
}

div[data-testid="stChatMessageAvatarAssistant"] {
    background-color: #EEF2FF !important;
}

/* Accessible & Restrained Outcome Status Badges */
.outcome-badge-container {
    display: inline-flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
    padding: 0.22rem 0.55rem;
    border-radius: 6px;
    font-size: 0.74rem;
    line-height: 1.35;
    margin-top: 0.55rem;
    margin-bottom: 0.2rem;
    border: 1px solid transparent;
}

.outcome-badge-answered {
    background-color: #F0FDF4;
    border-color: #DCFCE7;
    color: #15803D;
}

.outcome-badge-fallback {
    background-color: #EFF6FF;
    border-color: #DBEAFE;
    color: #1D4ED8;
}

.outcome-badge-abstained {
    background-color: #F8FAFC;
    border-color: #E2E8F0;
    color: #475569;
}

.outcome-badge-icon {
    font-size: 0.75rem;
    font-weight: 700;
}

.outcome-badge-title {
    font-weight: 600;
}

.outcome-badge-sep {
    opacity: 0.35;
}

.outcome-badge-desc {
    font-weight: 400;
    opacity: 0.9;
}

/* 5. Input Composer Integration */
div[data-testid="stBottom"] {
    background: linear-gradient(180deg, rgba(250, 249, 246, 0) 0%, rgba(250, 249, 246, 0.92) 20%, #FAF9F6 100%) !important;
    padding-top: 0.55rem !important;
    padding-bottom: 0.85rem !important;
}

div[data-testid="stBottom"] > div {
    max-width: 860px !important;
    margin: 0 auto !important;
}

div[data-testid="stChatInput"] {
    max-width: 860px !important;
    margin: 0 auto !important;
}

div[data-testid="stChatInput"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02) !important;
    padding: 0.3rem 0.45rem !important;
    transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out !important;
}

div[data-testid="stChatInput"] > div:hover {
    border-color: #9CA3AF !important;
}

div[data-testid="stChatInput"] > div:focus-within {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.14) !important;
}

div[data-testid="stChatInput"] textarea {
    font-size: 0.925rem !important;
    color: #0F172A !important;
    line-height: 1.45 !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #94A3B8 !important;
}

div[data-testid="stChatInput"] button {
    border-radius: 8px !important;
    color: #4F46E5 !important;
    transition: background-color 0.15s ease-in-out !important;
}

div[data-testid="stChatInput"] button:hover {
    background-color: #EEF2FF !important;
}

/* 6. Quieter Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F4F3EF !important;
    border-right: 1px solid #E8E8E3 !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Sidebar header collapse toggle button accessibility */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="baseButton-header"],
header[data-testid="stHeader"] button {
    color: #475569 !important;
}

button[data-testid="stSidebarCollapseButton"]:hover,
button[data-testid="baseButton-header"]:hover {
    color: #0F172A !important;
    background-color: #E8E8E3 !important;
}

.sidebar-brand-header {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding-bottom: 0.65rem;
    margin-bottom: 0.85rem;
    border-bottom: 1px solid #E8E8E3;
}

.sidebar-brand-icon {
    font-size: 1.15rem;
}

.sidebar-brand-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #0F172A;
    letter-spacing: -0.01em;
}

.sidebar-section-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-top: 0.85rem;
    margin-bottom: 0.35rem;
}

/* Sample inquiry buttons in sidebar: quiet secondary card style */
section[data-testid="stSidebar"] div.stButton > button {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #334155 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 0.65rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02) !important;
    min-height: auto !important;
    margin-bottom: 0.35rem !important;
    transition: all 0.15s ease-in-out !important;
    line-height: 1.35 !important;
}

section[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #EEF2FF !important;
    border-color: #C7D2FE !important;
    color: #4338CA !important;
    transform: none !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] div.stButton > button:focus-visible {
    outline: none !important;
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2) !important;
}

/* Clear Conversation Button in Sidebar */
section[data-testid="stSidebar"] div.stButton > button[key="btn_clear_conversation"],
section[data-testid="stSidebar"] div.stButton > button:has(p:contains("Clear")) {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    color: #64748B !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
}

section[data-testid="stSidebar"] div.stButton > button:hover:has(p:contains("Clear")),
section[data-testid="stSidebar"] div.stButton > button[key="btn_clear_conversation"]:hover {
    background-color: #FEF2F2 !important;
    border-color: #FECACA !important;
    color: #DC2626 !important;
}

.sidebar-disclaimer {
    font-size: 0.725rem;
    color: #64748B;
    line-height: 1.4;
    margin-top: 0.75rem;
}

.sidebar-meta-list {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.75rem;
}

.meta-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.25rem 0;
    font-size: 0.775rem;
    border-bottom: 1px solid #F1F5F9;
}

.meta-item:last-child {
    border-bottom: none;
}

.meta-label {
    color: #64748B;
    font-weight: 500;
}

.meta-val {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.725rem;
    color: #0F172A;
    background: #F8FAFC;
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
    border: 1px solid #E2E8F0;
}

/* 7. Secondary Technical Diagnostics */
div[data-testid="stExpander"] {
    background-color: #FAFAFA !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    margin-top: 0.5rem !important;
    overflow: hidden !important;
}

div[data-testid="stExpander"] details {
    border: none !important;
}

div[data-testid="stExpander"] details summary {
    font-size: 0.785rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    padding: 0.45rem 0.75rem !important;
    background-color: #FAFAFA !important;
    cursor: pointer !important;
    transition: color 0.15s ease-in-out !important;
}

div[data-testid="stExpander"] details summary:hover {
    color: #0F172A !important;
}

div[data-testid="stExpander"] details[open] summary {
    border-bottom: 1px solid #E2E8F0 !important;
    background-color: #F8FAFC !important;
}

div[data-testid="stExpander"] table {
    width: 100% !important;
    border-collapse: collapse !important;
    font-size: 0.785rem !important;
    margin: 0.4rem 0 !important;
    background: #FFFFFF !important;
    border-radius: 6px !important;
    border: 1px solid #E2E8F0 !important;
}

div[data-testid="stExpander"] th {
    background-color: #F1F5F9 !important;
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    padding: 0.45rem 0.65rem !important;
    border: 1px solid #E2E8F0 !important;
    text-align: left !important;
}

div[data-testid="stExpander"] td {
    padding: 0.4rem 0.65rem !important;
    border: 1px solid #E2E8F0 !important;
    color: #1E293B !important;
    line-height: 1.4 !important;
    vertical-align: top !important;
    word-break: break-word !important;
}

div[data-testid="stExpander"] tr:nth-child(even) td {
    background-color: #F8FAFC !important;
}

div[data-testid="stExpander"] code {
    background-color: #EEF2FF !important;
    color: #312E81 !important;
    font-size: 0.74rem !important;
    padding: 0.12rem 0.35rem !important;
    border-radius: 4px !important;
    border: 1px solid #E0E7FF !important;
    word-break: break-word !important;
    white-space: pre-wrap !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
}

/* 8. Responsive Overrides */
@media (max-width: 768px) {
    .main .block-container,
    section[data-testid="stMain"] .block-container {
        padding-top: 0.75rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-bottom: 5rem !important;
    }
    .product-identity-row {
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 0.35rem !important;
    }
    .brand-title {
        font-size: 1.05rem !important;
    }
    .brand-meta {
        font-size: 0.75rem !important;
    }
    div[data-testid="stColumn"],
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    div[data-testid="stColumn"] .stButton > button,
    div[data-testid="stColumn"] .stButton button,
    div[data-testid="column"] div.stButton > button {
        min-height: auto !important;
        padding: 0.8rem 1rem !important;
        padding-right: 2.2rem !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        max-width: 92% !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        max-width: 100% !important;
    }
    div[data-testid="stExpander"] table {
        font-size: 0.75rem !important;
        display: block !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
        white-space: nowrap !important;
    }
    div[data-testid="stExpander"] td {
        white-space: normal !important;
    }
    .composer-hint-bridge {
        margin: 0.65rem auto 0.2rem auto;
    }
}

/* 7. Knowledge Base Viewer - Text Wrapping & Contained Scroll */
.kb-viewer-container {
    width: 100%;
    max-height: 560px;
    overflow-x: auto;
    overflow-y: auto;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    background: #FFFFFF;
    margin-top: 0.75rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.kb-viewer-table {
    width: 100%;
    min-width: 720px;
    border-collapse: collapse;
    font-size: 0.875rem;
    text-align: left;
}

.kb-viewer-table thead th {
    position: sticky;
    top: 0;
    background: #F8FAFC;
    color: #334155;
    font-weight: 600;
    padding: 10px 14px;
    border-bottom: 2px solid #E2E8F0;
    z-index: 2;
}

.kb-th-question { width: 28%; }
.kb-th-answer { width: 48%; }
.kb-th-intent { width: 14%; }
.kb-th-topic { width: 10%; }

.kb-viewer-table tbody td {
    padding: 12px 14px;
    border-bottom: 1px solid #F1F5F9;
    vertical-align: top;
    white-space: normal !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    line-height: 1.5;
    color: #1E293B;
}

.kb-viewer-table tbody tr:hover {
    background-color: #F8FAFC;
}

.kb-cell-question {
    color: #0F172A;
}

.kb-cell-answer {
    color: #334155;
}

.kb-intent-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    background-color: #F1F5F9;
    border: 1px solid #E2E8F0;
    font-family: monospace;
    font-size: 0.78rem;
    color: #475569;
}
"""


def load_runtime_knowledge_base():
    """
    Load the 200-record workbook that the live retriever searches.
    Provides a read-only snapshot; never modifies canonical artifacts or datasets.
    """
    from src.ml.data import load_dataset

    return load_dataset().reset_index(drop=True)


def load_evaluation_knowledge_base():
    """
    Load the 160-record training-split subset used for full-pipeline evaluation.
    The live retriever does not use this subset.
    """
    from src.ml.data import load_dataset, split_dataset

    train_df, _ = split_dataset(load_dataset())
    return train_df.sort_index().copy()


def search_knowledge_base(
    df,
    query: str = "",
    intent_filter: Optional[str] = None,
):
    """
    Search and filter knowledge base records.
    Matches query substrings against question, response, intent, and topic.
    Punctuation-normalized matching ensures queries like 'wifi' match 'Wi-Fi'.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    filtered = df.copy()

    if intent_filter and intent_filter != "All":
        filtered = filtered[filtered["Intent"] == intent_filter]

    if query and query.strip():
        q = query.strip().lower()
        q_norm = q.replace("-", " ")
        q_compact = q.replace("-", "").replace(" ", "")

        def matches(series: pd.Series) -> pd.Series:
            s_lower = series.astype(str).str.lower()
            s_norm = s_lower.str.replace("-", " ")
            s_compact = s_lower.str.replace("-", "").str.replace(" ", "")
            return (
                s_lower.str.contains(q, regex=False)
                | s_norm.str.contains(q_norm, regex=False)
                | s_compact.str.contains(q_compact, regex=False)
            )

        q_mask = matches(filtered["User Message"]) | matches(filtered["Bot Response"]) | matches(filtered["Intent"])
        if "Topic" in filtered.columns:
            q_mask = q_mask | matches(filtered["Topic"])
        filtered = filtered[q_mask]

    return filtered
