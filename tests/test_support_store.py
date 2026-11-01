"""Unit and bounded concurrency tests for the support store (src/support/store.py)."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import inspect
import os
from pathlib import Path
import stat
import sqlite3
import sys
import tempfile
import unittest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.support.config import (
    MAX_COMMENT_CHARS,
    MAX_OPEN_TICKETS,
    MAX_SUMMARY_CHARS,
    RETENTION_DAYS,
)
from src.support.store import SupportStore, SupportStoreError, TicketLimitError


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "nested" / "support.sqlite3"
        self.clock = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
        self.store = SupportStore(self.path, now=lambda: self.clock)

    def create(self, session="s1", summary="Need help with my fees", **kwargs):
        kwargs.setdefault("consent", True)
        kwargs.setdefault("reason", "manual")
        return self.store.create_escalation(session, summary, **kwargs)


class TestTickets(StoreTestCase):
    def test_constructor_touches_no_disk(self):
        self.assertFalse(self.path.parent.exists())

    def test_create_and_list(self):
        reference = self.create(intent="financial_aid", sentiment="negative", reason="auto_offer")
        self.assertRegex(reference, r"^ESC-[0-9A-F]{8}$")
        ticket = self.store.list_tickets()[0]
        self.assertEqual(ticket.reference, reference)
        self.assertEqual(ticket.status, "open")
        self.assertEqual(ticket.reason, "auto_offer")
        self.assertEqual(ticket.intent, "financial_aid")
        self.assertEqual(ticket.sentiment, "negative")

    def test_consent_is_mandatory(self):
        for consent in (False, None, "yes", 1):
            with self.subTest(consent=consent):
                with self.assertRaises(ValueError):
                    self.create(consent=consent)
        self.assertEqual(self.store.list_tickets(), [])

    def test_invalid_summary_and_reason(self):
        for summary in ("", "   ", None, 5, "x" * (MAX_SUMMARY_CHARS + 1)):
            with self.subTest(summary=str(summary)[:10]):
                with self.assertRaises(ValueError):
                    self.create(summary=summary)
        with self.assertRaises(ValueError):
            self.create(reason="other")

    def test_summary_boundary_length_is_accepted(self):
        self.create(summary="x" * MAX_SUMMARY_CHARS)
        self.assertEqual(len(self.store.list_tickets()), 1)

    def test_pii_is_redacted_inside_the_store(self):
        self.create(summary="I am jane.doe@uni.edu, phone 9876543210")
        summary = self.store.list_tickets()[0].summary
        self.assertIn("[REDACTED_EMAIL]", summary)
        self.assertIn("[REDACTED_PHONE]", summary)
        raw = self.path.read_bytes()
        self.assertNotIn(b"jane.doe", raw)
        self.assertNotIn(b"9876543210", raw)

    def test_schema_has_nowhere_to_put_contact_details(self):
        columns = {
            row[1]
            for row in sqlite3.connect(self._created_path()).execute("PRAGMA table_info(tickets)")
        }
        self.assertFalse(columns & {"contact", "email", "phone", "name", "transcript"})
        parameters = inspect.signature(SupportStore.create_escalation).parameters
        self.assertNotIn("contact", parameters)

    def _created_path(self):
        self.create()
        return self.path

    def test_sql_metacharacters_are_stored_literally(self):
        text = "'); DROP TABLE tickets; --"
        self.create(summary=text)
        self.assertEqual(self.store.list_tickets()[0].summary, text)

    def test_open_ticket_cap_is_per_session(self):
        for _ in range(MAX_OPEN_TICKETS):
            self.create()
        with self.assertRaises(TicketLimitError):
            self.create()
        self.create(session="s2")

    def test_resolving_frees_the_cap(self):
        references = [self.create() for _ in range(MAX_OPEN_TICKETS)]
        self.assertTrue(self.store.update_status(references[0], "resolved"))
        self.create()

    def test_update_status(self):
        reference = self.create()
        self.assertTrue(self.store.update_status(reference, "in_progress"))
        self.assertEqual(self.store.list_tickets()[0].status, "in_progress")
        self.assertFalse(self.store.update_status("ESC-00000000", "resolved"))
        with self.assertRaises(ValueError):
            self.store.update_status(reference, "deleted")

    def test_list_filters_and_orders_newest_first(self):
        first = self.create()
        self.clock += timedelta(minutes=1)
        second = self.create()
        self.assertEqual([t.reference for t in self.store.list_tickets()], [second, first])
        self.store.update_status(first, "resolved")
        self.assertEqual([t.reference for t in self.store.list_tickets("open")], [second])

    def test_session_tickets_are_isolated(self):
        mine = self.create(session="s1")
        self.create(session="s2")
        self.assertEqual([t.reference for t in self.store.session_tickets("s1")], [mine])


class TestFeedback(StoreTestCase):
    def test_rating_and_redacted_comment(self):
        self.store.add_feedback("s1", 4, "write to jane.doe@uni.edu")
        row = self.store.recent_feedback()[0]
        self.assertEqual(row.rating, 4)
        self.assertEqual(row.comment, "write to [REDACTED_EMAIL]")

    def test_comment_is_optional(self):
        self.store.add_feedback("s1", 1)
        self.assertEqual(self.store.recent_feedback()[0].comment, "")

    def test_one_row_per_session(self):
        self.store.add_feedback("s1", 2)
        self.store.add_feedback("s1", 5)
        rows = self.store.recent_feedback()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].rating, 5)

    def test_invalid_input(self):
        for rating in (0, 6, "5", None, True, 2.5):
            with self.subTest(rating=rating):
                with self.assertRaises(ValueError):
                    self.store.add_feedback("s1", rating)
        with self.assertRaises(ValueError):
            self.store.add_feedback("s1", 3, "x" * (MAX_COMMENT_CHARS + 1))


class TestLifecycle(StoreTestCase):
    def test_delete_session_removes_only_that_session(self):
        self.create(session="s1")
        self.store.add_feedback("s1", 3)
        other = self.create(session="s2")
        self.assertEqual(self.store.delete_session("s1"), 2)
        self.assertEqual([t.reference for t in self.store.list_tickets()], [other])
        self.assertEqual(self.store.recent_feedback(), [])

    def test_retention_boundary(self):
        self.create()
        self.store.add_feedback("s1", 3)
        self.clock += timedelta(days=RETENTION_DAYS)
        self.assertEqual(len(self.store.list_tickets()), 1)
        self.clock += timedelta(seconds=1)
        self.assertEqual(self.store.list_tickets(), [])
        self.assertEqual(self.store.recent_feedback(), [])

    @unittest.skipIf(os.name == "nt", "POSIX permissions only")
    def test_database_file_is_private(self):
        self.create()
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_failure_is_wrapped_without_the_path(self):
        blocker = Path(self.tmp.name) / "file"
        blocker.write_text("x")
        store = SupportStore(blocker / "support.sqlite3")
        with self.assertRaises(SupportStoreError) as caught:
            store.create_escalation("s1", "help", consent=True, reason="manual")
        self.assertNotIn(self.tmp.name, str(caught.exception))

    def test_failure_log_holds_no_path_or_traceback(self):
        blocker = Path(self.tmp.name) / "file"
        blocker.write_text("x")
        store = SupportStore(blocker / "support.sqlite3")
        with self.assertLogs("src.support.store", level="ERROR") as logs:
            with self.assertRaises(SupportStoreError):
                store.list_tickets()
        self.assertNotIn(self.tmp.name, "\n".join(logs.output))
        self.assertTrue(all(record.exc_info is None for record in logs.records))


class TestConcurrency(StoreTestCase):
    def test_four_workers_cannot_exceed_the_cap_or_mix_sessions(self):
        def attempt(args):
            session, index = args
            try:
                return self.store.create_escalation(
                    session, f"{session} message {index}", consent=True, reason="manual"
                )
            except TicketLimitError:
                return None

        jobs = [(f"s{worker}", index) for worker in range(4) for index in range(MAX_OPEN_TICKETS + 2)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(attempt, jobs))

        references = [r for r in results if r]
        self.assertEqual(len(references), len(set(references)))
        self.assertEqual(len(references), 4 * MAX_OPEN_TICKETS)
        for worker in range(4):
            session = f"s{worker}"
            tickets = self.store.session_tickets(session)
            self.assertEqual(len(tickets), MAX_OPEN_TICKETS)
            self.assertTrue(all(t.summary.startswith(session) for t in tickets))


if __name__ == "__main__":
    unittest.main()
