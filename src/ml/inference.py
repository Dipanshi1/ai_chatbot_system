from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

from .config import MODEL_PATH, VECTORIZER_PATH
from .preprocessing import clean_text
from .schemas import ClassifierResult


class IntentClassifier:
    """
    Reusable inference engine for LinearSVC-based intent classification.

    Loads the saved TfidfVectorizer and LinearSVC model artifacts and reports
    classifier evidence only (scores and margin). It makes no acceptance or
    abstention decisions; those belong to policy.decide().
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None,
    ):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.vectorizer_path = (
            Path(vectorizer_path) if vectorizer_path else VECTORIZER_PATH
        )
        self.model = None
        self.vectorizer = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained model and vectorizer from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}"
            )
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(
                f"Vectorizer artifact not found at {self.vectorizer_path}"
            )

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

    @property
    def vocabulary(self) -> Dict[str, int]:
        """Vocabulary of the saved classifier vectorizer (used for lexical analysis)."""
        return self.vectorizer.vocabulary_

    def predict(self, message: str, top_k: int = 5) -> Optional[ClassifierResult]:
        """
        Score a message against all intents.

        The margin is the difference between the highest and second-highest
        LinearSVC decision_function scores. It is an uncalibrated margin, not a
        probability. Returns None if the message is empty after cleaning.
        """
        cleaned_message = clean_text(message)
        if not cleaned_message:
            return None

        message_tfidf = self.vectorizer.transform([cleaned_message])
        scores = np.asarray(self.model.decision_function(message_tfidf))

        # A binary LinearSVC returns a 1-D array of shape (1,); the margin
        # definition below requires one score per class.
        if scores.ndim != 2 or scores.shape[1] != len(self.model.classes_):
            raise ValueError(
                "Expected one decision score per class; got shape "
                f"{scores.shape} for {len(self.model.classes_)} classes"
            )
        if scores.shape[1] < 2:
            raise ValueError("At least two classes are required to compute a margin")

        scores = scores[0]

        # Stable descending order: ties resolve to the lower class index.
        ranked_indices = np.argsort(-scores, kind="stable")
        top_idx, second_idx = ranked_indices[0], ranked_indices[1]
        top_score = float(scores[top_idx])
        second_score = float(scores[second_idx])

        top_predictions = tuple(
            (str(self.model.classes_[idx]), float(scores[idx]))
            for idx in ranked_indices[: max(top_k, 0)]
        )

        return ClassifierResult(
            predicted_intent=str(self.model.classes_[top_idx]),
            margin=top_score - second_score,
            top_score=top_score,
            second_score=second_score,
            top_predictions=top_predictions,
        )


# Module-level singleton instance for convenient reuse
_default_classifier: Optional[IntentClassifier] = None


def get_classifier() -> IntentClassifier:
    """Get or initialize the shared IntentClassifier instance."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = IntentClassifier()
    return _default_classifier


def predict_intent(message: str, top_k: int = 5) -> Optional[ClassifierResult]:
    """Convenience function to score a message using the default classifier."""
    return get_classifier().predict(message, top_k=top_k)
