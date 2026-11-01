"""Unit tests for session privacy and admin sign-in (session.py, admin_auth.py)."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.support import admin_auth
from src.support.config import ADMIN_PASSWORD_HASH_ENV, MAX_ADMIN_LOGIN_ATTEMPTS
from src.support.session import (
    init_support_state,
    open_escalation,
    set_opt_out,
    submit_escalation,
    submit_feedback,
)
from src.support.store import SupportStore, SupportStoreError


class SessionTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = SupportStore(Path(tmp.name) / "support.sqlite3")
        self.state = {}
        init_support_state(self.state)


class TestSessionState(SessionTestCase):
    def test_defaults_are_private_and_stable(self):
        self.assertFalse(self.state["opted_out"])
        self.assertFalse(self.state["escalation_open"])
        session_id = self.state["session_id"]
        self.assertEqual(len(session_id), 32)
        init_support_state(self.state)
        self.assertEqual(self.state["session_id"], session_id)

    def test_session_ids_differ(self):
        other = {}
        init_support_state(other)
        self.assertNotEqual(other["session_id"], self.state["session_id"])


class TestOptOut(SessionTestCase):
    def test_opt_out_erases_stored_rows(self):
        open_escalation(self.state, "manual")
        submit_escalation(self.state, self.store, "help me", consent=True)
        submit_feedback(self.state, self.store, 4, "fine")
        set_opt_out(self.state, self.store, True)
        self.assertTrue(self.state["opted_out"])
        self.assertEqual(self.store.list_tickets(), [])
        self.assertEqual(self.store.recent_feedback(), [])

    def test_opt_out_blocks_feedback(self):
        set_opt_out(self.state, self.store, True)
        self.assertFalse(submit_feedback(self.state, self.store, 5))
        self.assertEqual(self.store.recent_feedback(), [])

    def test_opting_back_in_allows_feedback(self):
        set_opt_out(self.state, self.store, True)
        set_opt_out(self.state, self.store, False)
        self.assertTrue(submit_feedback(self.state, self.store, 5))

    def test_failed_deletion_still_leaves_session_opted_out(self):
        with mock.patch.object(self.store, "delete_session", side_effect=SupportStoreError("x")):
            with self.assertRaises(SupportStoreError):
                set_opt_out(self.state, self.store, True)
        self.assertTrue(self.state["opted_out"])

    def test_consented_escalation_is_allowed_after_opt_out(self):
        set_opt_out(self.state, self.store, True)
        open_escalation(self.state, "manual")
        reference = submit_escalation(self.state, self.store, "please help", consent=True)
        self.assertEqual(self.store.list_tickets()[0].reference, reference)


class TestEscalationFlow(SessionTestCase):
    def test_context_is_passed_and_panel_closes(self):
        open_escalation(self.state, "auto_offer", summary="fees", intent="financial_aid",
                        sentiment="negative")
        self.assertTrue(self.state["escalation_open"])
        self.assertEqual(self.state["escalation_summary"], "fees")
        submit_escalation(self.state, self.store, "fees are wrong", consent=True)
        ticket = self.store.list_tickets()[0]
        self.assertEqual((ticket.reason, ticket.intent, ticket.sentiment),
                         ("auto_offer", "financial_aid", "negative"))
        self.assertFalse(self.state["escalation_open"])
        self.assertEqual(self.state["last_ticket_reference"], ticket.reference)

    def test_no_consent_keeps_panel_open_and_stores_nothing(self):
        open_escalation(self.state, "manual")
        with self.assertRaises(ValueError):
            submit_escalation(self.state, self.store, "help", consent=False)
        self.assertTrue(self.state["escalation_open"])
        self.assertEqual(self.store.list_tickets(), [])


class TestAdminAuth(unittest.TestCase):
    def setUp(self):
        self.encoded = admin_auth.hash_password("correct horse")
        self.state = {}
        init_support_state(self.state)

    def configured(self):
        return mock.patch.dict(os.environ, {ADMIN_PASSWORD_HASH_ENV: self.encoded})

    def test_hash_round_trip_and_salting(self):
        self.assertTrue(admin_auth.verify_password("correct horse", self.encoded))
        self.assertFalse(admin_auth.verify_password("wrong", self.encoded))
        self.assertNotEqual(self.encoded, admin_auth.hash_password("correct horse"))
        self.assertNotIn("correct", self.encoded)

    def test_short_password_cannot_be_hashed(self):
        with self.assertRaises(ValueError):
            admin_auth.hash_password("short")

    def test_malformed_hash_is_rejected(self):
        for encoded in ("", "x", "scrypt$zz$zz", "md5$00$00", "scrypt$00"):
            with self.subTest(encoded=encoded):
                self.assertFalse(admin_auth.verify_password("x", encoded))

    def test_unconfigured_admin_fails_closed(self):
        environment = {k: v for k, v in os.environ.items() if k != ADMIN_PASSWORD_HASH_ENV}
        with mock.patch.dict(os.environ, environment, clear=True):
            self.assertFalse(admin_auth.admin_is_configured())
            self.assertFalse(admin_auth.attempt_login(self.state, "anything"))
        self.assertFalse(self.state["admin_authenticated"])

    def test_login_success_and_failure(self):
        with self.configured():
            self.assertFalse(admin_auth.attempt_login(self.state, "wrong"))
            self.assertFalse(self.state["admin_authenticated"])
            self.assertTrue(admin_auth.attempt_login(self.state, "correct horse"))
            self.assertTrue(self.state["admin_authenticated"])
            self.assertEqual(self.state["admin_failures"], 0)

    def test_lockout_blocks_even_the_correct_password(self):
        with self.configured():
            for _ in range(MAX_ADMIN_LOGIN_ATTEMPTS):
                admin_auth.attempt_login(self.state, "wrong")
            self.assertTrue(admin_auth.is_locked_out(self.state))
            self.assertFalse(admin_auth.attempt_login(self.state, "correct horse"))
            self.assertFalse(self.state["admin_authenticated"])

    def test_hash_command_prints_a_usable_hash(self):
        with mock.patch("getpass.getpass", return_value="correct horse"):
            with mock.patch("builtins.print") as printed:
                admin_auth.main()
        self.assertTrue(admin_auth.verify_password("correct horse", printed.call_args.args[0]))

    def test_hash_command_rejects_a_mismatch(self):
        with mock.patch("getpass.getpass", side_effect=["correct horse", "different"]):
            with self.assertRaises(SystemExit):
                admin_auth.main()

    def test_non_string_password(self):
        with self.configured():
            self.assertFalse(admin_auth.attempt_login(self.state, None))


if __name__ == "__main__":
    unittest.main()
