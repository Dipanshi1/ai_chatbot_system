import sys
from pathlib import Path
import unittest

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml import (
    ClassifierResult,
    IntentClassifier,
    MODEL_PATH,
    VECTORIZER_PATH,
    clean_text,
    predict_intent,
)


class TestMLInferencePipeline(unittest.TestCase):
    """Classifier evidence layer: loading, preprocessing and score structure."""

    def test_artifact_loading(self):
        """Verify model and vectorizer artifacts exist and load correctly."""
        self.assertTrue(MODEL_PATH.exists(), f"Model missing at {MODEL_PATH}")
        self.assertTrue(
            VECTORIZER_PATH.exists(), f"Vectorizer missing at {VECTORIZER_PATH}"
        )

        classifier = IntentClassifier()
        self.assertIsNotNone(classifier.model)
        self.assertIsNotNone(classifier.vectorizer)
        self.assertEqual(len(classifier.model.classes_), 26)
        self.assertEqual(len(classifier.vocabulary), 360)

    def test_missing_artifact_raises(self):
        with self.assertRaises(FileNotFoundError):
            IntentClassifier(model_path=Path("/nonexistent/model.pkl"))

    def test_preprocessing(self):
        """Verify centralized text cleaning rules."""
        raw = "  WHERE can   I find  FINANCIAL Aid? \n\t "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "where can i find financial aid?")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(123), "")

    def test_empty_input_returns_none(self):
        self.assertIsNone(predict_intent(""))
        self.assertIsNone(predict_intent("  \n\t"))

    def test_prediction_structure(self):
        """Classifier reports evidence only: scores, margin, ranked predictions."""
        result = predict_intent("when will my financial aid be disbursed?")

        self.assertIsInstance(result, ClassifierResult)
        self.assertEqual(result.predicted_intent, "financial_aid")
        self.assertEqual(len(result.top_predictions), 5)
        self.assertEqual(result.top_predictions[0][0], result.predicted_intent)

        scores = [score for _, score in result.top_predictions]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(result.margin, result.top_score - result.second_score)
        self.assertAlmostEqual(result.second_score, scores[1])
        self.assertGreaterEqual(result.margin, 0.0)

        # No decision-making fields on classifier evidence.
        for field in ("abstained", "abstention_threshold", "threshold"):
            self.assertFalse(hasattr(result, field))

    def test_predicted_intent_matches_model_predict(self):
        classifier = IntentClassifier()
        for query in [
            "how do i connect to campus wi-fi?",
            "where can i find internship postings?",
            "zzz qqq",
        ]:
            expected = classifier.model.predict(
                classifier.vectorizer.transform([clean_text(query)])
            )[0]
            self.assertEqual(classifier.predict(query).predicted_intent, expected)

    def test_top_k(self):
        self.assertEqual(len(predict_intent("parking", top_k=3).top_predictions), 3)
        self.assertEqual(len(predict_intent("parking", top_k=0).top_predictions), 0)


if __name__ == "__main__":
    unittest.main()
