"""
Unit and integration tests for Admin Analytics, Knowledge Base Viewer, and CSV Data Export.

Tests cover:
- Accurate ticket counts by status.
- Feedback aggregation and empty-data handling.
- Sentiment distribution from stored records.
- Ticket creation trends over time.
- Knowledge-base loading and multi-field search.
- CSV contents, UTF-8 encoding, and exclusion of database internals.
- PII redaction defense-in-depth on export.
- Opt-out session exclusion.
- Spreadsheet formula injection neutralization (CWE-1236).
- Application-level unauthorized admin access rejection.
- End-to-end admin portal interactions via AppTest.
"""
import csv
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from streamlit.testing.v1 import AppTest

from src.ml.pipeline import get_pipeline
from src.support.admin_auth import hash_password
from src.support.analytics import get_admin_analytics, require_admin_auth
from src.support.config import ADMIN_PASSWORD_HASH_ENV, SUPPORT_DB_ENV
from src.support.export import (
    export_feedback_csv,
    export_tickets_csv,
    sanitize_csv_cell,
)
from src.support.session import init_support_state
from src.support.store import SupportStore
from src.ui.helpers import (
    load_evaluation_knowledge_base,
    load_runtime_knowledge_base,
    search_knowledge_base,
)

APP_PATH = str(BASE_DIR / "app.py")
ADMIN_PASSWORD = "test-admin-secret"
ADMIN_HASH = hash_password(ADMIN_PASSWORD)


class BaseSupportStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "support.sqlite3"
        self.store = SupportStore(self.db_path)
        self.state = {}
        init_support_state(self.state)


class TestAdminAnalytics(BaseSupportStoreTestCase):
    def test_accurate_ticket_counts_by_status(self):
        t1 = self.store.create_escalation("s1", "Issue 1", consent=True, reason="manual")
        t2 = self.store.create_escalation("s2", "Issue 2", consent=True, reason="manual")
        t3 = self.store.create_escalation("s3", "Issue 3", consent=True, reason="manual")
        t4 = self.store.create_escalation("s4", "Issue 4", consent=True, reason="manual")

        self.store.update_status(t2, "in_progress")
        self.store.update_status(t3, "resolved")
        self.store.update_status(t4, "resolved")

        counts = self.store.ticket_counts()
        self.assertEqual(counts["open"], 1)
        self.assertEqual(counts["in_progress"], 1)
        self.assertEqual(counts["resolved"], 2)
        self.assertEqual(counts["total"], 4)

    def test_empty_ticket_counts(self):
        counts = self.store.ticket_counts()
        self.assertEqual(counts["open"], 0)
        self.assertEqual(counts["in_progress"], 0)
        self.assertEqual(counts["resolved"], 0)
        self.assertEqual(counts["total"], 0)

    def test_feedback_aggregation(self):
        self.store.add_feedback("s1", 5, "Excellent service")
        self.store.add_feedback("s2", 4, "Good")
        self.store.add_feedback("s3", 4, "Helpful")
        self.store.add_feedback("s4", 1, "Poor")

        metrics = self.store.feedback_metrics()
        self.assertEqual(metrics["total"], 4)
        self.assertAlmostEqual(metrics["average"], 3.5, places=2)
        self.assertEqual(metrics["distribution"][5], 1)
        self.assertEqual(metrics["distribution"][4], 2)
        self.assertEqual(metrics["distribution"][3], 0)
        self.assertEqual(metrics["distribution"][2], 0)
        self.assertEqual(metrics["distribution"][1], 1)

    def test_empty_feedback_aggregation(self):
        metrics = self.store.feedback_metrics()
        self.assertEqual(metrics["total"], 0)
        self.assertIsNone(metrics["average"])
        for star in range(1, 6):
            self.assertEqual(metrics["distribution"][star], 0)

    def test_sentiment_distribution(self):
        self.store.create_escalation(
            "s1", "Frustrated student", consent=True, reason="auto_offer", sentiment="negative"
        )
        self.store.create_escalation(
            "s2", "Another complaint", consent=True, reason="auto_offer", sentiment="negative"
        )
        self.store.create_escalation(
            "s3", "Neutral inquiry", consent=True, reason="manual", sentiment="neutral"
        )

        sentiments = self.store.sentiment_distribution()
        self.assertEqual(sentiments.get("negative"), 2)
        self.assertEqual(sentiments.get("neutral"), 1)
        self.assertNotIn("positive", sentiments)

    def test_empty_sentiment_distribution(self):
        sentiments = self.store.sentiment_distribution()
        self.assertEqual(sentiments, {})

    def test_ticket_creation_trends(self):
        self.store.create_escalation("s1", "Trend 1", consent=True, reason="manual")
        self.store.create_escalation("s2", "Trend 2", consent=True, reason="manual")

        trends = self.store.ticket_creation_trends()
        self.assertGreaterEqual(len(trends), 1)
        day, count = trends[0]
        self.assertEqual(len(day), 10)  # YYYY-MM-DD
        self.assertEqual(count, 2)

    def test_unauthorized_admin_access_rejected(self):
        self.state["admin_authenticated"] = False
        with self.assertRaises(PermissionError):
            require_admin_auth(self.state)

        with self.assertRaises(PermissionError):
            get_admin_analytics(self.store, self.state)


class TestKnowledgeBaseViewer(unittest.TestCase):
    def setUp(self):
        self.kb_df = load_runtime_knowledge_base()

    def test_runtime_knowledge_base_matches_the_live_retriever(self):
        self.assertEqual(len(self.kb_df), 200)
        self.assertEqual(len(self.kb_df), len(get_pipeline().retriever.df))
        self.assertEqual(
            self.kb_df["User Message"].tolist(), get_pipeline().retriever.df["User Message"].tolist()
        )

    def test_evaluation_subset_is_the_160_row_training_split(self):
        subset = load_evaluation_knowledge_base()
        self.assertEqual(len(subset), 160)
        self.assertTrue(set(subset.index) < set(self.kb_df.index))

    def test_runtime_knowledge_base_columns(self):
        for required_col in ["User Message", "Bot Response", "Intent"]:
            self.assertIn(required_col, self.kb_df.columns)

    def test_search_knowledge_base_by_query(self):
        results = search_knowledge_base(self.kb_df, query="financial aid")
        self.assertGreaterEqual(len(results), 1)
        for _, row in results.iterrows():
            text_match = (
                "financial aid" in str(row["User Message"]).lower()
                or "financial aid" in str(row["Bot Response"]).lower()
                or "financial aid" in str(row["Intent"]).lower()
                or "financial aid" in str(row.get("Topic", "")).lower()
            )
            self.assertTrue(text_match)

    def test_search_knowledge_base_by_intent_filter(self):
        results = search_knowledge_base(self.kb_df, intent_filter="financial_aid")
        self.assertGreaterEqual(len(results), 1)
        for _, row in results.iterrows():
            self.assertEqual(row["Intent"], "financial_aid")

    def test_search_knowledge_base_no_match(self):
        results = search_knowledge_base(self.kb_df, query="nonexistent_gibberish_query_xyz_123")
        self.assertEqual(len(results), 0)

    def test_search_knowledge_base_type_error(self):
        with self.assertRaises(TypeError):
            search_knowledge_base("not_a_dataframe")


class TestCSVExport(BaseSupportStoreTestCase):
    def setUp(self):
        super().setUp()
        self.state["admin_authenticated"] = True

    def test_formula_injection_neutralization(self):
        # Triggers: =, +, -, @, \t, \r
        self.assertEqual(sanitize_csv_cell("=SUM(A1:A10)"), "'=SUM(A1:A10)")
        self.assertEqual(sanitize_csv_cell("+12345"), "'+12345")
        self.assertEqual(sanitize_csv_cell("-12345"), "'-12345")
        self.assertEqual(sanitize_csv_cell("@alert()"), "'@alert()")
        self.assertEqual(sanitize_csv_cell("\tcmd"), "'\tcmd")
        self.assertEqual(sanitize_csv_cell("\rcmd"), "'\rcmd")
        # Leading whitespace followed by trigger
        self.assertEqual(sanitize_csv_cell("   =CMD|' /C calc'!A0"), "'   =CMD|' /C calc'!A0")
        # Benign values untouched
        self.assertEqual(sanitize_csv_cell("Normal Question Text"), "Normal Question Text")
        self.assertEqual(sanitize_csv_cell(""), "")
        self.assertEqual(sanitize_csv_cell(None), "")
        self.assertEqual(sanitize_csv_cell(5), "5")

    def test_csv_contents_and_utf8_encoding(self):
        self.store.create_escalation(
            "sess_1",
            "Need help with transcript",
            consent=True,
            reason="manual",
            intent="administration",
            sentiment="neutral",
        )
        self.store.add_feedback("sess_1", 5, "Very helpful response")

        tickets_csv = export_tickets_csv(self.store, self.state)
        feedback_csv = export_feedback_csv(self.store, self.state)

        # UTF-8 decoding
        self.assertIsInstance(tickets_csv, bytes)
        self.assertIsInstance(feedback_csv, bytes)
        tickets_text = tickets_csv.decode("utf-8")
        feedback_text = feedback_csv.decode("utf-8")

        # Check headers
        tickets_reader = csv.reader(io.StringIO(tickets_text))
        tickets_headers = next(tickets_reader)
        self.assertEqual(
            tickets_headers,
            ["Reference", "Created At (UTC)", "Status", "Reason", "Intent", "Sentiment", "Summary"],
        )

        feedback_reader = csv.reader(io.StringIO(feedback_text))
        feedback_headers = next(feedback_reader)
        self.assertEqual(feedback_headers, ["Created At (UTC)", "Rating", "Comment"])

        # Check absence of database internals and session IDs
        self.assertNotIn("session_id", tickets_text)
        self.assertNotIn("sess_1", tickets_text)
        self.assertNotIn("session_id", feedback_text)
        self.assertNotIn("sess_1", feedback_text)
        self.assertNotIn("password", tickets_text.lower())
        self.assertNotIn("hash", tickets_text.lower())

    def test_csv_pii_exclusion(self):
        raw_email = "student.john@university.edu"
        raw_phone = "+1-555-839-2019"
        # Even if unredacted text were passed, export enforces redaction
        self.store.create_escalation(
            "s_pii",
            f"Contact me at {raw_email} or {raw_phone}",
            consent=True,
            reason="manual",
        )
        self.store.add_feedback("s_pii", 4, f"Follow up at {raw_email}")

        tickets_text = export_tickets_csv(self.store, self.state).decode("utf-8")
        feedback_text = export_feedback_csv(self.store, self.state).decode("utf-8")

        self.assertNotIn(raw_email, tickets_text)
        self.assertNotIn(raw_phone, tickets_text)
        self.assertIn("[REDACTED_EMAIL]", tickets_text)
        self.assertIn("[REDACTED_PHONE]", tickets_text)

        self.assertNotIn(raw_email, feedback_text)
        self.assertIn("[REDACTED_EMAIL]", feedback_text)

    def test_csv_opt_out_exclusion(self):
        s_opted_out = "sess_opted_out"
        self.store.create_escalation(
            s_opted_out, "Opted out question", consent=True, reason="manual"
        )
        self.store.add_feedback(s_opted_out, 3, "Opted out comment")

        # Normal export includes it if not listed
        text_before = export_tickets_csv(self.store, self.state).decode("utf-8")
        self.assertIn("Opted out question", text_before)

        # Explicitly excluded session
        text_after = export_tickets_csv(
            self.store, self.state, exclude_session_ids=[s_opted_out]
        ).decode("utf-8")
        self.assertNotIn("Opted out question", text_after)

        fb_after = export_feedback_csv(
            self.store, self.state, exclude_session_ids=[s_opted_out]
        ).decode("utf-8")
        self.assertNotIn("Opted out comment", fb_after)

    def test_unauthorized_export_rejected(self):
        self.state["admin_authenticated"] = False
        with self.assertRaises(PermissionError):
            export_tickets_csv(self.store, self.state)

        with self.assertRaises(PermissionError):
            export_feedback_csv(self.store, self.state)


class TestAdminPortalAppE2E(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "support.sqlite3"
        patcher = mock.patch.dict(
            os.environ,
            {SUPPORT_DB_ENV: str(self.db_path), ADMIN_PASSWORD_HASH_ENV: ADMIN_HASH},
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = SupportStore(self.db_path)

    def run_app(self) -> AppTest:
        return AppTest.from_file(APP_PATH, default_timeout=60).run()

    def test_admin_portal_tabs_analytics_kb_export_render(self):
        self.store.create_escalation(
            "session_admin_test",
            "Wi-Fi troubleshooting ticket",
            consent=True,
            reason="manual",
            intent="campus_facilities",
            sentiment="neutral",
        )
        self.store.add_feedback("session_admin_test", 5, "Great help")

        at = self.run_app()
        # Switch to admin portal
        at.button(key="btn_portal_switch").click().run()
        # Enter password and sign in
        at.text_input[0].set_value(ADMIN_PASSWORD)
        next(b for b in at.button if b.label == "Sign in").click().run()
        self.assertEqual(len(at.exception), 0)

        rendered_text = "\n".join(
            str(el.value)
            for group in (at.markdown, at.caption, at.text, at.code, at.info, at.metric)
            for el in group
        )

        # Check Ticket Queue elements
        self.assertIn("Wi-Fi troubleshooting ticket", rendered_text)
        self.assertIn("Great help", rendered_text)

        # Check Analytics Dashboard elements
        self.assertIn("Operational Support Analytics", rendered_text)
        metric_labels = [m.label for m in at.metric]
        self.assertIn("Total Tickets", metric_labels)
        self.assertIn("Total Submissions", metric_labels)

        # Check Knowledge Base elements
        self.assertIn("Runtime FAQ Knowledge Base (200 Records)", rendered_text)
        self.assertIn("Displaying 200 of 200", rendered_text)
        # FAQ answers may use these words themselves, so check the interface wording only
        self.assertNotIn("Approved Knowledge Base", rendered_text)
        self.assertNotIn("approved records", rendered_text)
        self.assertNotIn("verified question-answer", rendered_text)

        # The 160-row evaluation subset is a separate, clearly labelled view
        at.selectbox(key="kb_record_set").set_value("Evaluation Subset (160 Records)").run()
        self.assertEqual(len(at.exception), 0)
        subset_text = "\n".join(str(m.value) for m in at.markdown) + "\n".join(
            str(c.value) for c in at.caption
        )
        self.assertIn("Evaluation Subset (160 Records)", subset_text)
        self.assertIn("Displaying 160 of 160", subset_text)
        at.selectbox(key="kb_record_set").set_value("Runtime FAQ Knowledge Base (200 Records)").run()
        self.assertIn("kb-viewer-container", rendered_text)
        self.assertIn("kb-viewer-table", rendered_text)
        self.assertIn("kb-th-question", rendered_text)
        self.assertIn("kb-th-answer", rendered_text)

        # Check Export elements
        self.assertIn("Export Support Data", rendered_text)
        self.assertTrue(any(b.key == "btn_download_tickets_csv" for b in at.download_button))
        self.assertTrue(any(b.key == "btn_download_feedback_csv" for b in at.download_button))

        # Sign out
        at.button(key="btn_admin_sign_out").click().run()
        self.assertEqual(len(at.exception), 0)
        rendered_after = "\n".join(
            str(el.value)
            for group in (at.markdown, at.caption, at.text, at.code, at.info, at.metric)
            for el in group
        )
        self.assertNotIn("Wi-Fi troubleshooting ticket", rendered_after)

    def test_admin_portal_empty_analytics_and_honest_states(self):
        """Verify honest empty states when zero tickets exist and no query volume claims."""
        at = self.run_app()
        at.button(key="btn_portal_switch").click().run()
        at.text_input[0].set_value(ADMIN_PASSWORD)
        next(b for b in at.button if b.label == "Sign in").click().run()
        self.assertEqual(len(at.exception), 0)

        info_texts = [str(el.value) for el in at.info]
        self.assertTrue(
            any("No support tickets are currently recorded in the 30-day store." in t for t in info_texts)
        )
        self.assertTrue(
            any("No student feedback has been submitted." in t for t in info_texts)
        )
        self.assertTrue(
            any("Insufficient data to display creation trends." in t for t in info_texts)
        )

        all_text = "\n".join(
            str(el.value)
            for group in (at.markdown, at.caption, at.text, at.code, at.info, at.metric)
            for el in group
        )
        self.assertIn("Operational Support Analytics", all_text)
        self.assertIn("Telemetry is computed strictly from stored support tickets", all_text)
        self.assertIn("Unescalated student chat interactions are processed ephemerally and never logged.", all_text)
        self.assertIn("Insufficient data: No stored escalation tickets currently contain tone or sentiment labels.", all_text)
        # Ensure it does not claim total chatbot query volume
        self.assertNotIn("total student queries", all_text.lower())


if __name__ == "__main__":
    unittest.main()
