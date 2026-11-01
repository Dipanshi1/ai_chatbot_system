"""
Local SQLite store for escalation tickets and session feedback.

Holds no contact details and no chat transcripts: a ticket is one redacted summary
written with explicit consent. Text is redacted here as well as by callers, so no
code path can persist it unredacted. Rows expire after RETENTION_DAYS.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import secrets
import sqlite3
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

from .config import (
    MAX_COMMENT_CHARS,
    MAX_OPEN_TICKETS,
    MAX_SUMMARY_CHARS,
    RETENTION_DAYS,
    get_db_path,
)
from .redaction import redact

logger = logging.getLogger(__name__)

TICKET_STATUSES = ("open", "in_progress", "resolved")
TICKET_REASONS = ("manual", "auto_offer")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    reference TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    intent TEXT,
    sentiment TEXT,
    summary TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT NOT NULL
);
"""


class SupportStoreError(Exception):
    """The store could not complete an operation. Carries no path or SQL detail."""


class TicketLimitError(SupportStoreError):
    """This session already has the maximum number of unresolved tickets."""


@dataclass(frozen=True)
class Ticket:
    reference: str
    created_at: str
    status: str
    reason: str
    intent: Optional[str]
    sentiment: Optional[str]
    summary: str


@dataclass(frozen=True)
class Feedback:
    created_at: str
    rating: int
    comment: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SupportStore:
    """Opens one connection per call, so instances are safe to share across threads."""

    def __init__(
        self,
        path: Optional[Union[Path, str]] = None,
        now: Callable[[], datetime] = _utc_now,
    ):
        self.path = Path(path) if path is not None else get_db_path()
        self._now = now

    def _timestamp(self, moment: Optional[datetime] = None) -> str:
        return (moment or self._now()).astimezone(timezone.utc).isoformat(timespec="seconds")

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.close(os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600))
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.executescript(_SCHEMA)
        return connection

    def _run(self, operation: Callable[[sqlite3.Connection], object]) -> object:
        """Run one write transaction, purging expired rows first."""
        connection = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            cutoff = self._timestamp(self._now() - timedelta(days=RETENTION_DAYS))
            connection.execute("DELETE FROM tickets WHERE created_at < ?", (cutoff,))
            connection.execute("DELETE FROM feedback WHERE created_at < ?", (cutoff,))
            result = operation(connection)
            connection.execute("COMMIT")
            return result
        except (sqlite3.Error, OSError) as exc:
            logger.error("Support store operation failed: %s", type(exc).__name__)
            raise SupportStoreError("The support store is unavailable.") from exc
        finally:
            if connection is not None:
                connection.close()

    def create_escalation(
        self,
        session_id: str,
        summary: str,
        consent: bool,
        reason: str,
        intent: Optional[str] = None,
        sentiment: Optional[str] = None,
    ) -> str:
        """Store a redacted ticket and return its reference. Requires explicit consent."""
        if consent is not True:
            raise ValueError("Explicit consent is required to create a ticket.")
        if reason not in TICKET_REASONS:
            raise ValueError("Unknown ticket reason.")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("A summary is required.")
        if len(summary) > MAX_SUMMARY_CHARS:
            raise ValueError("The summary is too long.")

        reference = f"ESC-{secrets.token_hex(4).upper()}"
        row = (
            reference, session_id, self._timestamp(), "open", reason,
            intent, sentiment, redact(summary.strip()).text,
        )

        def insert(connection: sqlite3.Connection) -> str:
            unresolved = connection.execute(
                "SELECT COUNT(*) FROM tickets WHERE session_id = ? AND status != 'resolved'",
                (session_id,),
            ).fetchone()[0]
            if unresolved >= MAX_OPEN_TICKETS:
                raise TicketLimitError("Too many open requests for this session.")
            connection.execute("INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?)", row)
            return reference

        return self._run(insert)

    def list_tickets(self, status: Optional[str] = None) -> List[Ticket]:
        """Return unexpired tickets, newest first."""
        if status is not None and status not in TICKET_STATUSES:
            raise ValueError("Unknown ticket status.")
        query = (
            "SELECT reference, created_at, status, reason, intent, sentiment, summary "
            "FROM tickets {} ORDER BY created_at DESC, rowid DESC"
        )
        if status is None:
            rows = self._run(lambda c: c.execute(query.format("")).fetchall())
        else:
            rows = self._run(
                lambda c: c.execute(query.format("WHERE status = ?"), (status,)).fetchall()
            )
        return [Ticket(*row) for row in rows]

    def session_tickets(self, session_id: str) -> List[Ticket]:
        """Return this session's tickets, newest first."""
        rows = self._run(
            lambda c: c.execute(
                "SELECT reference, created_at, status, reason, intent, sentiment, summary "
                "FROM tickets WHERE session_id = ? ORDER BY created_at DESC, rowid DESC",
                (session_id,),
            ).fetchall()
        )
        return [Ticket(*row) for row in rows]

    def update_status(self, reference: str, status: str) -> bool:
        """Set a ticket's status. Returns False if the reference does not exist."""
        if status not in TICKET_STATUSES:
            raise ValueError("Unknown ticket status.")
        return self._run(
            lambda c: c.execute(
                "UPDATE tickets SET status = ? WHERE reference = ?", (status, reference)
            ).rowcount
            > 0
        )

    def add_feedback(self, session_id: str, rating: int, comment: str = "") -> None:
        """Store this session's feedback, replacing any earlier submission."""
        if type(rating) is not int or not 1 <= rating <= 5:
            raise ValueError("Rating must be a whole number from 1 to 5.")
        if not isinstance(comment, str) or len(comment) > MAX_COMMENT_CHARS:
            raise ValueError("The comment is too long.")
        row = (session_id, self._timestamp(), rating, redact(comment.strip()).text)
        self._run(lambda c: c.execute("INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?)", row))

    def recent_feedback(self, limit: int = 20) -> List[Feedback]:
        """Return unexpired feedback, newest first. Session ids are not exposed."""
        rows = self._run(
            lambda c: c.execute(
                "SELECT created_at, rating, comment FROM feedback "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        )
        return [Feedback(*row) for row in rows]

    def delete_session(self, session_id: str) -> int:
        """Erase everything stored for a session. Returns the number of rows removed."""

        def delete(connection: sqlite3.Connection) -> int:
            removed = connection.execute(
                "DELETE FROM tickets WHERE session_id = ?", (session_id,)
            ).rowcount
            removed += connection.execute(
                "DELETE FROM feedback WHERE session_id = ?", (session_id,)
            ).rowcount
            return removed

        return self._run(delete)

    def ticket_counts(self) -> Dict[str, int]:
        """Return counts of unexpired tickets by status, plus total."""
        def query(connection: sqlite3.Connection) -> Dict[str, int]:
            rows = connection.execute(
                "SELECT status, COUNT(*) FROM tickets GROUP BY status"
            ).fetchall()
            counts = {status: 0 for status in TICKET_STATUSES}
            for status, count in rows:
                if status in counts:
                    counts[status] = count
            counts["total"] = sum(counts.values())
            return counts

        return self._run(query)

    def feedback_metrics(self) -> Dict[str, object]:
        """Aggregate ratings across all unexpired feedback."""
        def query(connection: sqlite3.Connection) -> Dict[str, object]:
            rows = connection.execute("SELECT rating FROM feedback").fetchall()
            ratings = [r[0] for r in rows]
            distribution = {star: 0 for star in range(1, 6)}
            for r in ratings:
                if 1 <= r <= 5:
                    distribution[r] += 1
            total = len(ratings)
            average = (sum(ratings) / total) if total > 0 else None
            return {
                "total": total,
                "average": average,
                "distribution": distribution,
            }

        return self._run(query)

    def sentiment_distribution(self) -> Dict[str, int]:
        """Return counts of unexpired tickets by sentiment label where available."""
        def query(connection: sqlite3.Connection) -> Dict[str, int]:
            rows = connection.execute(
                "SELECT sentiment, COUNT(*) FROM tickets "
                "WHERE sentiment IS NOT NULL AND TRIM(sentiment) != '' "
                "GROUP BY sentiment"
            ).fetchall()
            return {sentiment: count for sentiment, count in rows}

        return self._run(query)

    def ticket_creation_trends(self) -> List[Tuple[str, int]]:
        """Return ticket creation volume grouped by UTC date, sorted chronologically."""
        def query(connection: sqlite3.Connection) -> List[Tuple[str, int]]:
            rows = connection.execute(
                "SELECT SUBSTR(created_at, 1, 10) AS day, COUNT(*) "
                "FROM tickets GROUP BY day ORDER BY day ASC"
            ).fetchall()
            return [(str(day), int(count)) for day, count in rows]

        return self._run(query)

    def export_tickets_records(
        self, exclude_session_ids: Optional[Collection[str]] = None
    ) -> List[Dict[str, object]]:
        """
        Return unexpired tickets for CSV export, omitting session_id and internal fields.
        Excludes records for any session listed in exclude_session_ids.
        """
        def query(connection: sqlite3.Connection) -> List[Dict[str, object]]:
            rows = connection.execute(
                "SELECT reference, created_at, status, reason, intent, sentiment, summary, session_id "
                "FROM tickets ORDER BY created_at DESC, rowid DESC"
            ).fetchall()
            excluded = set(exclude_session_ids) if exclude_session_ids else set()
            records = []
            for ref, cat, stat, reas, inten, sent, summ, sess in rows:
                if sess in excluded:
                    continue
                records.append({
                    "Reference": ref,
                    "Created At (UTC)": cat,
                    "Status": stat,
                    "Reason": reas,
                    "Intent": inten or "",
                    "Sentiment": sent or "",
                    "Summary": summ,
                })
            return records

        return self._run(query)

    def export_feedback_records(
        self, exclude_session_ids: Optional[Collection[str]] = None
    ) -> List[Dict[str, object]]:
        """
        Return unexpired feedback for CSV export, omitting session_id and internal fields.
        Excludes records for any session listed in exclude_session_ids.
        """
        def query(connection: sqlite3.Connection) -> List[Dict[str, object]]:
            rows = connection.execute(
                "SELECT created_at, rating, comment, session_id "
                "FROM feedback ORDER BY created_at DESC, rowid DESC"
            ).fetchall()
            excluded = set(exclude_session_ids) if exclude_session_ids else set()
            records = []
            for cat, rat, comm, sess in rows:
                if sess in excluded:
                    continue
                records.append({
                    "Created At (UTC)": cat,
                    "Rating": rat,
                    "Comment": comm,
                })
            return records

        return self._run(query)
