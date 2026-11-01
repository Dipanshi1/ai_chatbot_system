"""End-to-end support journeys through the real app.py, real pipeline and real SQLite store."""
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from streamlit.testing.v1 import AppTest

from src.support.admin_auth import hash_password
from src.support.config import (
    ADMIN_PASSWORD_HASH_ENV,
    MAX_ADMIN_LOGIN_ATTEMPTS,
    PHYSICAL_DANGER_MESSAGE,
    SELF_HARM_MESSAGE,
    SUPPORT_DB_ENV,
)
from src.support.store import SupportStore

APP = str(BASE_DIR / "app.py")
PASSWORD = "correct horse battery"
PASSWORD_HASH = hash_password(PASSWORD)
RAW_EMAIL = "jane.doe@uni.edu"
RAW_PHONE = "9876543210"


def _rendered_text(at) -> str:
    parts = []
    for group in (at.markdown, at.caption, at.text, at.code, at.error, at.warning, at.info,
                  at.success):
        parts.extend(str(el.value) for el in group)
    return "\n".join(parts)


def _failing_store_script(repo):
    """A store that fails on every write must not produce a fabricated reference."""
    import sys

    sys.path.insert(0, repo)
    import streamlit as st

    import app
    from src.support.session import init_support_state, open_escalation
    from src.support.store import SupportStoreError

    class FailingStore:
        def create_escalation(self, *args, **kwargs):
            raise SupportStoreError("unavailable")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    init_support_state(st.session_state)
    open_escalation(st.session_state, "manual", summary="help")
    app.render_escalation_panel(FailingStore())


class SupportAppTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db_path = Path(tmp.name) / "support.sqlite3"
        patcher = mock.patch.dict(
            os.environ,
            {SUPPORT_DB_ENV: str(self.db_path), ADMIN_PASSWORD_HASH_ENV: PASSWORD_HASH},
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = SupportStore(self.db_path)

    def run_app(self) -> AppTest:
        return AppTest.from_file(APP, default_timeout=120).run()

    def send(self, at, text):
        at.chat_input[0].set_value(text).run()
        self.assertEqual(len(at.exception), 0)
        return at

    def submit_form(self, at, summary=None, consent=True):
        if summary is not None:
            at.text_area[0].set_value(summary)
        at.checkbox(key="escalation_consent").set_value(consent)
        next(b for b in at.button if b.label == "Send request").click().run()
        self.assertEqual(len(at.exception), 0)
        return at

    def sign_in(self, at, password=PASSWORD):
        at.button(key="btn_portal_switch").click().run()
        at.text_input[0].set_value(password)
        next(b for b in at.button if b.label == "Sign in").click().run()
        self.assertEqual(len(at.exception), 0)
        return at


class TestRedactionJourney(SupportAppTestCase):
    def test_pii_never_reaches_display_or_session_state(self):
        at = self.send(self.run_app(), f"my email is {RAW_EMAIL} and phone {RAW_PHONE}, financial aid")
        text = _rendered_text(at)
        self.assertNotIn(RAW_EMAIL, text)
        self.assertNotIn(RAW_PHONE, text)
        self.assertIn("[REDACTED_EMAIL]", text)
        self.assertIn("Personal details were removed", text)
        for message in at.session_state["messages"]:
            self.assertNotIn(RAW_PHONE, str(message))
            self.assertNotIn("jane", str(message))

    def test_message_without_pii_keeps_the_same_answer(self):
        at = self.send(self.run_app(), "When will my financial aid be disbursed?")
        text = _rendered_text(at)
        self.assertIn("Disbursement", text)
        self.assertNotIn("Personal details were removed", text)


class TestCrisisJourney(SupportAppTestCase):
    def test_self_harm_guidance_skips_the_pipeline(self):
        at = self.send(self.run_app(), "I want to kill myself")
        reply = at.session_state["messages"][-1]
        self.assertEqual(reply["content"], SELF_HARM_MESSAGE)
        self.assertIsNone(reply["result"])
        text = _rendered_text(at)
        self.assertIn("not monitored", text)
        self.assertNotIn("rephrase", text)

    def test_physical_danger_guidance_is_distinct(self):
        at = self.send(self.run_app(), "someone is following me on campus and I am scared")
        self.assertEqual(at.session_state["messages"][-1]["content"], PHYSICAL_DANGER_MESSAGE)

    def test_crisis_stores_nothing_and_offers_no_ticket(self):
        at = self.send(self.run_app(), "I feel suicidal and everything is terrible")
        self.assertFalse(at.session_state["escalation_open"])
        self.assertFalse(self.db_path.exists())


class TestEscalationJourney(SupportAppTestCase):
    def test_typed_human_request_creates_a_redacted_ticket_after_consent(self):
        at = self.send(self.run_app(), f"I want to talk to a human, I am {RAW_EMAIL}")
        self.assertIn("not monitored", _rendered_text(at))
        self.assertFalse(self.db_path.exists())
        at = self.submit_form(at)
        tickets = self.store.list_tickets()
        self.assertEqual(len(tickets), 1)
        self.assertEqual(tickets[0].reason, "manual")
        self.assertNotIn(RAW_EMAIL, tickets[0].summary)
        self.assertIn(tickets[0].reference, _rendered_text(at))

    def test_sidebar_button_opens_the_form(self):
        at = self.run_app()
        at.button(key="btn_talk_to_person").click().run()
        at = self.submit_form(at, summary="My hostel fees are wrong, 9876543210")
        self.assertIn("[REDACTED_PHONE]", self.store.list_tickets()[0].summary)

    def test_no_consent_stores_nothing(self):
        at = self.run_app()
        at.button(key="btn_talk_to_person").click().run()
        at = self.submit_form(at, summary="help", consent=False)
        self.assertTrue(any("consent" in w.value.lower() for w in at.warning))
        self.assertFalse(self.db_path.exists())

    def test_negative_abstention_offers_and_records_context(self):
        at = self.send(self.run_app(), "this is useless, you are terrible")
        self.assertTrue(at.session_state["escalation_open"])
        self.assertIn("frustrating", _rendered_text(at))
        self.assertFalse(self.db_path.exists())
        self.submit_form(at)
        ticket = self.store.list_tickets()[0]
        self.assertEqual((ticket.reason, ticket.sentiment), ("auto_offer", "negative"))

    def test_neutral_abstention_makes_no_offer(self):
        at = self.send(self.run_app(), "zzz qqq")
        self.assertFalse(at.session_state["escalation_open"])

    def test_negative_answered_query_makes_no_offer(self):
        at = self.send(self.run_app(), "My financial aid package seems incorrect.")
        reply = at.session_state["messages"][-1]
        self.assertFalse(reply["result"].abstained)
        self.assertEqual(reply["sentiment"], "negative")
        self.assertFalse(at.session_state["escalation_open"])

    def test_offer_can_be_dismissed_and_expires_on_the_next_turn(self):
        at = self.send(self.run_app(), "this is useless, you are terrible")
        at.button(key="btn_escalation_dismiss").click().run()
        self.assertFalse(at.session_state["escalation_open"])
        at = self.send(at, "this is useless, you are terrible")
        at = self.send(at, "When will my financial aid be disbursed?")
        self.assertFalse(at.session_state["escalation_open"])

    def test_student_sees_status_change_made_by_staff(self):
        at = self.send(self.run_app(), "I want to talk to a human")
        at = self.submit_form(at)
        reference = self.store.list_tickets()[0].reference
        self.store.update_status(reference, "in_progress")
        at.run()
        self.assertIn(f"{reference}: in progress", _rendered_text(at))

    def test_store_failure_shows_no_fabricated_reference(self):
        at = AppTest.from_function(_failing_store_script, args=(str(BASE_DIR),), default_timeout=60).run()
        self.submit_form(at)
        self.assertTrue(any("could not be saved" in e.value for e in at.error))
        self.assertNotIn("ESC-", _rendered_text(at))


class TestPrivacyJourney(SupportAppTestCase):
    def test_notice_states_the_chat_is_not_monitored(self):
        self.assertIn("not monitored", _rendered_text(self.run_app()))

    def test_opt_out_erases_rows_and_hides_feedback(self):
        at = self.send(self.run_app(), "I want to talk to a human")
        at = self.submit_form(at)
        self.assertEqual(len(self.store.list_tickets()), 1)
        self.assertGreaterEqual(len(at.get("feedback")), 1)
        at.toggle(key="toggle_opt_out").set_value(True).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(self.store.list_tickets(), [])
        self.assertEqual(len(at.get("feedback")), 0)

    def test_typed_opt_out_only_points_to_the_toggle(self):
        at = self.send(self.run_app(), "I want to talk to a human")
        at = self.submit_form(at)
        at = self.send(at, "delete my data")
        self.assertFalse(at.session_state["opted_out"])
        self.assertEqual(len(self.store.list_tickets()), 1)
        self.assertIn("Don't keep my data", _rendered_text(at))

    def test_ordinary_use_writes_nothing(self):
        at = self.send(self.run_app(), "How can I connect to campus Wi-Fi?")
        self.send(at, "zzz qqq")
        self.assertFalse(self.db_path.exists())

    def test_feedback_is_optional_and_redacted(self):
        at = self.run_app()
        at.get("feedback")[0].set_value(3)
        at.text_input(key="session_comment").set_value(f"mail {RAW_EMAIL}")
        at.button(key="btn_send_feedback").click().run()
        self.assertEqual(len(at.exception), 0)
        row = self.store.recent_feedback()[0]
        self.assertEqual(row.rating, 4)
        self.assertEqual(row.comment, "mail [REDACTED_EMAIL]")

    def test_feedback_without_rating_is_not_stored(self):
        at = self.run_app()
        at.button(key="btn_send_feedback").click().run()
        self.assertTrue(any("choose a rating" in w.value for w in at.warning))
        self.assertFalse(self.db_path.exists())


class TestAdminPortal(SupportAppTestCase):
    def seed_ticket(self, summary="Hostel fees look wrong"):
        return self.store.create_escalation("student", summary, consent=True, reason="manual")

    def test_queue_is_hidden_without_sign_in(self):
        self.seed_ticket()
        at = self.run_app()
        at.button(key="btn_portal_switch").click().run()
        self.assertNotIn("Hostel fees", _rendered_text(at))
        self.assertEqual(len(at.chat_input), 0)

    def test_wrong_password_shows_no_tickets(self):
        self.seed_ticket()
        at = self.sign_in(self.run_app(), "wrong password")
        self.assertTrue(any("Incorrect" in e.value for e in at.error))
        self.assertNotIn("Hostel fees", _rendered_text(at))

    def test_lockout(self):
        at = self.run_app()
        at.button(key="btn_portal_switch").click().run()
        for _ in range(MAX_ADMIN_LOGIN_ATTEMPTS):
            at.text_input[0].set_value("wrong password")
            next(b for b in at.button if b.label == "Sign in").click().run()
        self.assertTrue(any("Too many failed" in e.value for e in at.error))
        self.assertEqual(len(at.text_input), 0)

    def test_unconfigured_portal_stays_closed(self):
        self.seed_ticket()
        environment = {k: v for k, v in os.environ.items() if k != ADMIN_PASSWORD_HASH_ENV}
        with mock.patch.dict(os.environ, environment, clear=True):
            at = self.run_app()
            at.button(key="btn_portal_switch").click().run()
            self.assertTrue(any("not been configured" in w.value for w in at.warning))
            self.assertNotIn("Hostel fees", _rendered_text(at))

    def test_student_ticket_reaches_the_admin_queue_and_can_be_resolved(self):
        student = self.send(self.run_app(), "I want to talk to a human")
        self.submit_form(student, summary="Please check my hostel fees")
        reference = self.store.list_tickets()[0].reference

        admin = self.sign_in(self.run_app())
        self.assertIn("Please check my hostel fees", _rendered_text(admin))
        admin.button(key=f"ticket_{reference}_resolved").click().run()
        self.assertEqual(len(admin.exception), 0)
        self.assertEqual(self.store.list_tickets()[0].status, "resolved")
        self.assertNotIn("Please check my hostel fees", _rendered_text(admin))
        admin.selectbox(key="admin_status_filter").set_value("resolved").run()
        self.assertIn("Please check my hostel fees", _rendered_text(admin))

    def test_markup_in_a_summary_is_rendered_as_plain_text(self):
        payload = "<img src=x onerror=alert(1)> **bold**"
        self.seed_ticket(payload)
        at = self.sign_in(self.run_app())
        self.assertTrue(any(el.value == payload for el in at.text))
        self.assertFalse(any(payload in str(el.value) for el in at.markdown))

    def test_feedback_is_visible_to_staff(self):
        self.store.add_feedback("student", 4, "Clear answers")
        at = self.sign_in(self.run_app())
        self.assertIn("Clear answers", _rendered_text(at))

    def test_sign_out_hides_the_queue(self):
        self.seed_ticket()
        at = self.sign_in(self.run_app())
        at.button(key="btn_admin_sign_out").click().run()
        self.assertNotIn("Hostel fees", _rendered_text(at))

    def test_database_failure_shows_a_generic_message(self):
        self.db_path.mkdir()
        at = self.sign_in(self.run_app())
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("unavailable" in e.value for e in at.error))
        self.assertNotIn(str(self.db_path), _rendered_text(at))


class TestStoredRows(SupportAppTestCase):
    def test_no_conversation_text_is_stored_beyond_the_summary(self):
        at = self.send(self.run_app(), "How can I connect to campus Wi-Fi?")
        at = self.send(at, "I want to talk to a human")
        self.submit_form(at, summary="short summary only")
        with sqlite3.connect(self.db_path) as connection:
            dump = "\n".join(connection.iterdump())
        self.assertIn("short summary only", dump)
        self.assertNotIn("Wi-Fi", dump)


if __name__ == "__main__":
    unittest.main()
