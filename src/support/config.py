"""Configuration and fixed user-facing text for the support layer."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SUPPORT_DB_ENV = "SUPPORT_DB_PATH"
ADMIN_PASSWORD_HASH_ENV = "SUPPORT_ADMIN_PASSWORD_HASH"
DEFAULT_DB_PATH = BASE_DIR / "data" / "runtime" / "support.sqlite3"

# Prototype constants; not fitted; not optimal.
RETENTION_DAYS = 30
MAX_SUMMARY_CHARS = 500
MAX_COMMENT_CHARS = 300
MAX_OPEN_TICKETS = 3
MAX_ADMIN_LOGIN_ATTEMPTS = 5

# No telephone numbers here: the repository holds no verified emergency contacts.
SELF_HARM_MESSAGE = (
    "I'm really sorry you're feeling this way. You deserve support from a real person right now. "
    "**This chatbot is not monitored**, so nobody will see this message or respond to it.\n\n"
    "- If you might act on these thoughts, or you have already hurt yourself, call your local "
    "emergency number now or go to the nearest emergency department.\n"
    "- Contact a crisis helpline in your country or your university counselling service.\n"
    "- Tell someone you trust, and try not to stay alone right now."
)
PHYSICAL_DANGER_MESSAGE = (
    "**If you are in immediate danger, call your local emergency number or campus security now.** "
    "**This chatbot is not monitored**, so nobody will see this message and it cannot send help.\n\n"
    "- Move to a safe place if you can, and stay near other people.\n"
    "- If someone is hurt, stay on the line with emergency services and follow their instructions.\n"
    "- When you are safe, report what happened to campus security."
)

HUMAN_REQUEST_MESSAGE = (
    "You can send a request to the student support team using the form below. "
    "This chat is not monitored, so a request is the way to reach a person."
)
OPT_OUT_MESSAGE = (
    "To opt out, turn on **Don't keep my data** in the sidebar. "
    "It deletes anything stored from this session and turns off feedback."
)
ESCALATION_OFFER_MESSAGE = (
    "It sounds like this has been frustrating, and I couldn't give you a reliable answer. "
    "Would you like to send a request to the student support team?"
)
TICKET_FOLLOW_UP_TEXT = (
    "We don't store contact details, so staff cannot reach you directly. "
    "Keep this reference and quote it when you contact the student support office."
)


def get_db_path() -> Path:
    """Support database location; the environment override keeps tests away from real data."""
    return Path(os.environ.get(SUPPORT_DB_ENV) or DEFAULT_DB_PATH)
