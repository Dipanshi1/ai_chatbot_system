"""
Administrator data export utilities for support tickets and student feedback.

Produces sanitized UTF-8 CSV files with:
1. Application-level administrator authorization enforcement.
2. PII redaction defense-in-depth via redact().
3. Spreadsheet formula injection neutralization (CWE-1236).
4. Exclusion of opted-out sessions.
5. Exclusion of internal database keys, session IDs, and credentials.
"""
import csv
import io
from typing import Collection, MutableMapping, Optional

from .admin_auth import admin_is_configured
from .analytics import require_admin_auth
from .redaction import redact
from .store import SupportStore

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_csv_cell(value: object) -> str:
    """
    Neutralize spreadsheet formula injection (CWE-1236 / CSV Injection).

    If the string begins with any trigger character (=, +, -, @, tab, return),
    prefix with an apostrophe (') so spreadsheet software evaluates it as plain text.
    Also handles leading space padding.
    """
    if value is None:
        return ""
    text = str(value)
    if text and text[0] in _FORMULA_PREFIXES:
        return "'" + text
    stripped_spaces = text.lstrip(" ")
    if stripped_spaces and stripped_spaces[0] in _FORMULA_PREFIXES:
        return "'" + text
    return text


def export_tickets_csv(
    store: SupportStore,
    state: MutableMapping,
    exclude_session_ids: Optional[Collection[str]] = None,
) -> bytes:
    """
    Export unexpired support tickets as UTF-8 encoded CSV.

    Enforces admin authorization, redacts PII, neutralizes formula injection,
    and omits session IDs, database internals, and contact details.
    """
    require_admin_auth(state)
    excluded = set(exclude_session_ids or ())
    if state.get("opted_out") and "session_id" in state:
        excluded.add(state["session_id"])

    records = store.export_tickets_records(exclude_session_ids=excluded)

    fieldnames = [
        "Reference",
        "Created At (UTC)",
        "Status",
        "Reason",
        "Intent",
        "Sentiment",
        "Summary",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()

    for row in records:
        clean_row = {}
        for key in fieldnames:
            raw_val = row.get(key, "")
            # Apply defense-in-depth redaction to free-text summary
            if key == "Summary" and isinstance(raw_val, str):
                raw_val = redact(raw_val).text
            clean_row[key] = sanitize_csv_cell(raw_val)
        writer.writerow(clean_row)

    return output.getvalue().encode("utf-8")


def export_feedback_csv(
    store: SupportStore,
    state: MutableMapping,
    exclude_session_ids: Optional[Collection[str]] = None,
) -> bytes:
    """
    Export unexpired student feedback submissions as UTF-8 encoded CSV.

    Enforces admin authorization, redacts PII, neutralizes formula injection,
    and omits session IDs and internal identifiers.
    """
    require_admin_auth(state)
    excluded = set(exclude_session_ids or ())
    if state.get("opted_out") and "session_id" in state:
        excluded.add(state["session_id"])

    records = store.export_feedback_records(exclude_session_ids=excluded)

    fieldnames = [
        "Created At (UTC)",
        "Rating",
        "Comment",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()

    for row in records:
        clean_row = {}
        for key in fieldnames:
            raw_val = row.get(key, "")
            # Apply defense-in-depth redaction to free-text comments
            if key == "Comment" and isinstance(raw_val, str):
                raw_val = redact(raw_val).text
            clean_row[key] = sanitize_csv_cell(raw_val)
        writer.writerow(clean_row)

    return output.getvalue().encode("utf-8")
