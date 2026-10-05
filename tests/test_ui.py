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
from src.ui.helpers import (
    DISCLAIMER_TEXT,
    EXAMPLE_QUESTIONS,
    PAGE_TITLE,
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
        self.assertGreater(len(EXAMPLE_QUESTIONS), 2)
        for eg in EXAMPLE_QUESTIONS:
            self.assertTrue(len(eg) > 5)


if __name__ == "__main__":
    unittest.main()
