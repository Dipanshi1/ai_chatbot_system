"""Unit tests for the heuristic sentiment signal (src/support/sentiment.py)."""
import ast
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.pipeline import get_pipeline
from src.support.sentiment import analyze_sentiment


class TestSentiment(unittest.TestCase):
    def test_labels(self):
        for text, label in (
            ("this is useless, you are terrible", "negative"),
            ("I am so stressed and frustrated", "negative"),
            ("thanks, that was really helpful", "positive"),
            ("Where is the library?", "neutral"),
        ):
            with self.subTest(text=text):
                self.assertEqual(analyze_sentiment(text).label, label)

    def test_negation_flips_polarity(self):
        self.assertEqual(analyze_sentiment("this was not helpful").label, "negative")
        self.assertEqual(analyze_sentiment("not bad at all").label, "positive")
        self.assertEqual(analyze_sentiment("that doesn’t help me, it isn't good").label, "negative")

    def test_asking_for_help_does_not_offset_a_complaint(self):
        self.assertEqual(analyze_sentiment("I need help, this is useless").label, "negative")

    def test_negation_window_is_bounded(self):
        result = analyze_sentiment("not sure why but the portal is great")
        self.assertEqual(result.label, "positive")

    def test_mixed_tie_is_neutral(self):
        result = analyze_sentiment("great portal but terrible wifi")
        self.assertEqual(result.label, "neutral")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.matched_terms, ("great", "terrible"))

    def test_score_range_and_determinism(self):
        text = "awful awful thanks"
        first = analyze_sentiment(text)
        self.assertEqual(first, analyze_sentiment(text))
        self.assertTrue(-1.0 <= first.score <= 1.0)

    def test_empty_and_non_string(self):
        for value in ("", "   ", None, 7):
            with self.subTest(value=value):
                result = analyze_sentiment(value)
                self.assertEqual(result.label, "neutral")
                self.assertEqual(result.matched_terms, ())

    def test_placeholders_carry_no_sentiment(self):
        self.assertEqual(analyze_sentiment("[REDACTED_EMAIL] [REDACTED_ID]").label, "neutral")


class TestIndependenceFromMl(unittest.TestCase):
    def test_support_package_does_not_import_ml(self):
        for path in (BASE_DIR / "src" / "support").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    with self.subTest(file=path.name, module=module):
                        self.assertFalse(module.startswith("src.ml"))
                        self.assertFalse(module.startswith("sklearn"))

    def test_answer_is_unchanged_by_running_sentiment(self):
        message = "this financial aid process is terrible and useless"
        pipeline = get_pipeline()
        before = pipeline.answer(message)
        analyze_sentiment(message)
        after = pipeline.answer(message)
        self.assertEqual(before.to_dict(), after.to_dict())


if __name__ == "__main__":
    unittest.main()
