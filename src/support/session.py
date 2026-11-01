"""
Per-session support state, kept in any mutable mapping (st.session_state in the app).

Opt-out lasts for the browser session only: without student login there is no
identity to attach a lasting preference to.
"""
import secrets
from typing import MutableMapping, Optional

from .store import SupportStore


def init_support_state(state: MutableMapping) -> None:
    """Set defaults without disturbing values that already exist."""
    defaults = {
        "opted_out": False,
        "escalation_open": False,
        "escalation_reason": "manual",
        "escalation_summary": "",
        "escalation_intent": None,
        "escalation_sentiment": None,
        "last_ticket_reference": None,
        "feedback_sent": False,
        "portal_view": "student",
        "admin_authenticated": False,
        "admin_failures": 0,
    }
    if "session_id" not in state:
        state["session_id"] = secrets.token_hex(16)
    for key, value in defaults.items():
        if key not in state:
            state[key] = value


def set_opt_out(state: MutableMapping, store: SupportStore, enabled: bool) -> None:
    """
    Apply the opt-out choice; enabling it erases this session's stored rows.

    The flag is set first, so a failed deletion still blocks further storage.
    """
    state["opted_out"] = enabled
    if enabled:
        state["feedback_sent"] = False
        state["last_ticket_reference"] = None
        store.delete_session(state["session_id"])


def open_escalation(
    state: MutableMapping,
    reason: str,
    summary: str = "",
    intent: Optional[str] = None,
    sentiment: Optional[str] = None,
) -> None:
    """Show the request form with context from the current turn."""
    state["escalation_open"] = True
    state["escalation_reason"] = reason
    state["escalation_summary"] = summary
    state["escalation_intent"] = intent
    state["escalation_sentiment"] = sentiment


def close_escalation(state: MutableMapping) -> None:
    state["escalation_open"] = False
    state["escalation_summary"] = ""


def submit_escalation(
    state: MutableMapping, store: SupportStore, summary: str, consent: bool
) -> str:
    """
    Create a ticket and close the form.

    Allowed after opt-out: the student's explicit consent for this one request
    is the only write an opted-out session can make.
    """
    reference = store.create_escalation(
        state["session_id"],
        summary,
        consent=consent,
        reason=state["escalation_reason"],
        intent=state["escalation_intent"],
        sentiment=state["escalation_sentiment"],
    )
    state["last_ticket_reference"] = reference
    close_escalation(state)
    return reference


def submit_feedback(
    state: MutableMapping, store: SupportStore, rating: int, comment: str = ""
) -> bool:
    """Store optional feedback. Returns False, storing nothing, for an opted-out session."""
    if state["opted_out"]:
        return False
    store.add_feedback(state["session_id"], rating, comment)
    state["feedback_sent"] = True
    return True
