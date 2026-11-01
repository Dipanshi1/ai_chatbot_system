"""
Unit tests for Streamlit UI helpers, validation logic, and presentation mapping (Phase 5C).
"""
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.config import MAX_MESSAGE_CHARS
from src.ml.pipeline import answer
from src.ml.schemas import Outcome
from src.support.config import RETENTION_DAYS
from src.ui.helpers import (
    CUSTOM_CSS,
    DISCLAIMER_TEXT,
    EXAMPLE_QUESTIONS,
    PAGE_TITLE,
    PRIVACY_NOTICE,
    SUGGESTED_TILES,
    WELCOME_MESSAGE,
    extract_diagnostics,
    format_outcome_badge,
    get_initial_messages,
    validate_user_message,
)


class TestUIHelpers(unittest.TestCase):
    def test_initial_messages_structure(self):
        """Validate that session state initialization creates standard assistant welcome message."""
        messages = get_initial_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["content"], WELCOME_MESSAGE)
        self.assertIsNone(messages[0]["result"])

    def test_message_validation_valid_input(self):
        """Validate that normal questions within 500 characters pass validation."""
        valid_query = "How do I apply for financial aid?"
        is_valid, error = validate_user_message(valid_query)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_message_validation_empty_and_whitespace(self):
        """Validate that empty or whitespace-only inputs fail validation."""
        for empty_val in ["", "   ", "\n\t  \n", None]:
            with self.subTest(val=empty_val):
                is_valid, error = validate_user_message(empty_val)
                self.assertFalse(is_valid)
                self.assertIsNotNone(error)
                self.assertIn("enter a question", error.lower())

    def test_message_validation_500_char_boundary(self):
        """Validate exact 500-character boundary limits."""
        # 499 characters -> valid
        msg_499 = "a" * 499
        is_valid, error = validate_user_message(msg_499)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

        # 500 characters -> valid
        msg_500 = "a" * 500
        is_valid, error = validate_user_message(msg_500)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

        # 501 characters -> invalid
        msg_501 = "a" * 501
        is_valid, error = validate_user_message(msg_501)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIn("exceeds the maximum limit of 500", error)
        self.assertIn("501", error)

    def test_outcome_badge_mapping(self):
        """Validate user-friendly presentation descriptions for all three outcomes."""
        for outcome in [Outcome.ANSWERED, Outcome.INTENT_FALLBACK, Outcome.ABSTAINED]:
            with self.subTest(outcome=outcome):
                badge = format_outcome_badge(outcome)
                self.assertEqual(badge["status"], outcome.name)
                self.assertTrue(len(badge["title"]) > 0)
                self.assertTrue(len(badge["explanation"]) > 0)

    def test_diagnostics_extraction_and_uncalibrated_margin(self):
        """Validate diagnostics formatting and explicit uncalibrated margin label."""
        result = answer("when will my financial aid be disbursed?")
        diagnostics = extract_diagnostics(result)

        self.assertIn("Outcome", diagnostics)
        self.assertIn("Committed Intent", diagnostics)
        self.assertIn("Classification margin (uncalibrated)", diagnostics)

        # Guarantee no fake probability / confidence % labels exist
        for key in diagnostics.keys():
            self.assertNotIn("confidence %", key.lower())
            self.assertNotIn("probability", key.lower())

        # Check versions are included
        self.assertIn("Model Version", diagnostics)
        self.assertIn("Knowledge Base Version", diagnostics)
        self.assertIn("Policy Version", diagnostics)

    def test_disclaimer_and_examples_present(self):
        """Validate independence disclaimer and example questions."""
        self.assertIn("independently", DISCLAIMER_TEXT.lower())
        self.assertIn("conversational memory", DISCLAIMER_TEXT.lower())
        self.assertGreater(len(EXAMPLE_QUESTIONS), 2)
        for eg in EXAMPLE_QUESTIONS:
            self.assertTrue(len(eg) > 5)

    def test_suggested_tiles_and_theme_css(self):
        """Validate welcome suggested question tiles and scoped design system CSS."""
        self.assertGreaterEqual(len(SUGGESTED_TILES), 4)
        for tile in SUGGESTED_TILES:
            self.assertEqual(len(tile), 3)
            icon, title, query = tile
            self.assertTrue(len(icon) > 0)
            self.assertTrue(len(title) > 0)
            is_valid, _ = validate_user_message(query)
            self.assertTrue(is_valid)

        self.assertIn("max-width: 860px", CUSTOM_CSS)
        self.assertIn("product-header", CUSTOM_CSS)
        self.assertIn("outcome-badge", CUSTOM_CSS)

    def test_outcome_badge_contains_visual_indicators(self):
        """Validate icons and badge class keys for accessible, non-color-only identification."""
        for outcome in [Outcome.ANSWERED, Outcome.INTENT_FALLBACK, Outcome.ABSTAINED]:
            badge = format_outcome_badge(outcome)
            self.assertIn("icon", badge)
            self.assertIn("badge_class", badge)
            self.assertTrue(len(badge["icon"]) > 0)
            self.assertTrue(len(badge["badge_class"]) > 0)

    def test_kb_viewer_css_and_styling(self):
        """Validate KB viewer container styling, text wrapping, and scroll containment CSS."""
        self.assertIn(".kb-viewer-container", CUSTOM_CSS)
        self.assertIn(".kb-viewer-table", CUSTOM_CSS)
        self.assertIn("overflow-x: auto", CUSTOM_CSS)
        self.assertIn("white-space: normal !important", CUSTOM_CSS)
        self.assertIn("word-break: break-word !important", CUSTOM_CSS)
        self.assertIn("overflow-wrap: break-word !important", CUSTOM_CSS)

    def test_displayed_sample_prompts_return_direct_faq_answers(self):
        """Validate that all displayed sample questions and tiles yield direct FAQ answers."""
        expected_matches = {
            "How do I add a course to my schedule?": ("course_registration", 0),
            "When will my financial aid be disbursed?": ("financial_aid", 65),
            "When is the library open on weekends?": ("campus_facilities", 1),
            "Who do I contact about accessibility services?": ("student_services", 13),
        }

        # Check EXAMPLE_QUESTIONS in sidebar
        for prompt in EXAMPLE_QUESTIONS:
            self.assertIn(prompt, expected_matches, f"Unexpected sample question: {prompt}")
            res = answer(prompt)
            exp_intent, exp_row = expected_matches[prompt]
            self.assertEqual(
                res.outcome,
                Outcome.ANSWERED,
                f"Sample question '{prompt}' did not receive direct FAQ answer: {res.outcome}",
            )
            self.assertEqual(res.answer_source, "knowledge_base")
            self.assertEqual(res.intent, exp_intent)
            self.assertEqual(res.matched_row_id, exp_row)
            self.assertGreaterEqual(res.margin, 0.20)
            self.assertGreaterEqual(res.overlap, 0.60)
            self.assertGreater(len(res.answer), 20)

        # Check SUGGESTED_TILES on welcome stage
        for icon, title, query in SUGGESTED_TILES:
            self.assertIn(query, expected_matches, f"Unexpected tile query: {query}")
            res = answer(query)
            self.assertEqual(
                res.outcome,
                Outcome.ANSWERED,
                f"Tile query '{query}' did not receive direct FAQ answer: {res.outcome}",
            )
            self.assertEqual(res.answer_source, "knowledge_base")

    def test_unsupported_sample_prompts_safety_abstention_audit(self):
        """Verify safety gating on historical unsupported prompts without lowering thresholds."""
        # 1. Broad query with low margin -> ABSTAINED (low_margin)
        res_services = answer("Where can I find student services?")
        self.assertEqual(res_services.outcome, Outcome.ABSTAINED)
        self.assertLess(res_services.margin, 0.20)

        # 2. Financial aid application (missing in KB) -> INTENT_FALLBACK (no_reliable_match)
        res_aid = answer("How do I apply for financial aid?")
        self.assertEqual(res_aid.outcome, Outcome.INTENT_FALLBACK)
        self.assertEqual(res_aid.intent, "financial_aid")
        self.assertLess(res_aid.overlap, 0.60)

        # 3. Registration date question (missing in KB) -> INTENT_FALLBACK (no_reliable_match)
        res_reg = answer("When is course registration?")
        self.assertEqual(res_reg.outcome, Outcome.INTENT_FALLBACK)
        self.assertEqual(res_reg.intent, "course_registration")
        self.assertLess(res_reg.overlap, 0.60)

    def test_privacy_notice_distinguishes_temporary_session_and_storage(self):
        """Validate privacy notice accurately states conversation is held temporarily and not written to support DB."""
        self.assertIn("held temporarily", PRIVACY_NOTICE)
        self.assertIn("not written to the support database", PRIVACY_NOTICE)
        self.assertIn(f"for up to {RETENTION_DAYS} days", PRIVACY_NOTICE)


if __name__ == "__main__":
    unittest.main()
