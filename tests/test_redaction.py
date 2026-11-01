"""Unit tests for PII redaction (src/support/redaction.py)."""
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.data import load_dataset, split_dataset
from src.ml.lexical import tokenize
from src.ml.pipeline import get_pipeline
from src.ml.preprocessing import clean_text
from src.support.redaction import PLACEHOLDERS, redact


class TestDetection(unittest.TestCase):
    def assert_redacted(self, text, placeholder, secret):
        result = redact(text)
        self.assertIn(placeholder, result.text)
        self.assertNotIn(secret, result.text)

    def test_email(self):
        self.assert_redacted(
            "reach me at Jane.Doe+x@mail.uni.ac.in please",
            "[REDACTED_EMAIL]",
            "Jane.Doe",
        )

    def test_phone_formats(self):
        for text, secret in (
            ("call +91 98765 43210", "98765"),
            ("call 9876543210 today", "9876543210"),
            ("(555) 010-1000", "010-1000"),
            ("+1 415-555-2671", "555-2671"),
            ("my number is 555-0100", "555-0100"),
        ):
            with self.subTest(text=text):
                self.assert_redacted(text, "[REDACTED_PHONE]", secret)

    def test_card_requires_valid_checksum(self):
        self.assert_redacted("card 4111 1111 1111 1111", "[REDACTED_CARD]", "4111")
        self.assertNotIn("[REDACTED_CARD]", redact("ref 1234 5678 9012 3456").text)

    def test_labelled_identifiers(self):
        for text, secret in (
            ("student id: 2021CS1042", "2021CS1042"),
            ("my id is 2021CS1042", "2021CS1042"),
            ("my roll no 21BCE1042", "21BCE1042"),
            ("enrollment number AB12345", "AB12345"),
            ("aadhaar 1234-5678-9012", "1234-5678-9012"),
            ("my ssn is 123-45-6789", "123-45-6789"),
        ):
            with self.subTest(text=text):
                self.assert_redacted(text, "[REDACTED_ID]", secret)

    def test_bare_student_id_shape(self):
        self.assert_redacted("it is 2021CS1042 thanks", "[REDACTED_ID]", "2021CS1042")

    def test_stated_name(self):
        self.assert_redacted("My name is Priya Sharma", "[REDACTED_NAME]", "Sharma")
        result = redact("my name is priya and i forgot my password")
        self.assertEqual(result.text, "my name is [REDACTED_NAME] and i forgot my password")

    def test_multiple_types_are_counted(self):
        result = redact("a@b.edu, c@d.edu and 9876543210")
        self.assertEqual(result.counts, {"EMAIL": 2, "PHONE": 1})
        self.assertTrue(result.found)


class TestNoFalsePositives(unittest.TestCase):
    def test_ordinary_text_is_unchanged(self):
        for text in (
            "Where can I find the syllabus for MATH201?",
            "Is room 1204 open?",
            "Exam on 2026-10-15",
            "Exams run 2026-10-15 2026-12-10",
            "I paid 15000 in fees",
            "the deadline is 12/10/2026",
            "course CS101 section 2",
            "my id card is lost",
            "my student id is lost",
            "student id number please",
            "call 2025 hotline",
            "reset before 11:30",
            "my name is not on the list",
            "my name is missing from the roster",
        ):
            with self.subTest(text=text):
                result = redact(text)
                self.assertEqual(result.text, text)
                self.assertFalse(result.found)

    def test_canonical_training_messages_are_unchanged(self):
        train_df, _ = split_dataset(load_dataset())
        for message in train_df["User Message"].astype(str):
            self.assertEqual(redact(message).text, message)


class TestInputHandling(unittest.TestCase):
    def test_empty_and_non_string(self):
        for value in ("", "   ", None, 42):
            with self.subTest(value=value):
                result = redact(value)
                self.assertEqual(result.counts, {})
                self.assertFalse(result.found)
        self.assertEqual(redact(None).text, "")

    def test_idempotent(self):
        once = redact("my name is Priya, id 2021CS1042, a@b.edu, 9876543210").text
        self.assertEqual(redact(once).text, once)

    def test_deterministic(self):
        text = "a@b.edu 9876543210"
        self.assertEqual(redact(text), redact(text))


class TestPipelineSafety(unittest.TestCase):
    """Placeholders must not leak signal into the frozen model."""

    def test_each_placeholder_is_one_token_outside_both_vocabularies(self):
        pipeline = get_pipeline()
        kb_vocabulary = pipeline.retriever._vectorizer.vocabulary_
        for placeholder in PLACEHOLDERS.values():
            with self.subTest(placeholder=placeholder):
                tokens = tokenize(clean_text(placeholder))
                self.assertEqual(len(tokens), 1)
                self.assertNotIn(tokens[0], pipeline.classifier.vocabulary)
                self.assertNotIn(tokens[0], kb_vocabulary)

    def test_placeholder_does_not_change_classifier_margin(self):
        pipeline = get_pipeline()
        base = pipeline.answer("I forgot my password")
        padded = pipeline.answer(f"I forgot my password {PLACEHOLDERS['EMAIL']}")
        self.assertAlmostEqual(base.margin, padded.margin, places=9)
        self.assertEqual(base.predicted_intent, padded.predicted_intent)

    def test_redacted_result_holds_no_raw_pii(self):
        result = get_pipeline().answer(redact("my phone is 9876543210, jane.doe@uni.edu").text)
        dumped = str(result.to_dict())
        self.assertNotIn("9876543210", dumped)
        self.assertNotIn("jane", dumped)


if __name__ == "__main__":
    unittest.main()
