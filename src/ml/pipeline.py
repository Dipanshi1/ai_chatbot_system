"""
Orchestrator: the single public entry point for answering a message.

Collects evidence (lexical, classifier, retrieval), delegates the decision to
policy.decide(), and assembles a ChatResult. Contains no decision logic.
"""
import hashlib
from pathlib import Path
from typing import Optional

from .config import (
    CLARIFICATION_MESSAGE,
    EMPTY_INPUT_MESSAGE,
    INTENT_FALLBACK_TEMPLATE,
    POLICY_VERSION,
)
from .inference import IntentClassifier
from .lexical import analyze_lexical
from .policy import DEFAULT_POLICY, decide
from .preprocessing import clean_text
from .retrieval import ResponseRetriever
from .schemas import AbstainReason, ChatResult, Outcome, PolicyConfig


def _sha256_prefix(path: Path, length: int = 12) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def _area_name(intent: str) -> str:
    return intent.replace("_", " ")


class ChatPipeline:
    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        retriever: Optional[ResponseRetriever] = None,
    ):
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever or ResponseRetriever()
        self.model_version = (
            f"linsvc-{_sha256_prefix(self.classifier.model_path)}"
            f"+tfidf-{_sha256_prefix(self.classifier.vectorizer_path)}"
        )
        self.kb_version = self.retriever.kb_version
        self.policy_version = POLICY_VERSION

    def answer(self, message: str, policy: Optional[PolicyConfig] = None) -> ChatResult:
        policy = policy or DEFAULT_POLICY
        cleaned = clean_text(message)

        lexical = classifier_result = candidate = None
        if cleaned:
            lexical = analyze_lexical(cleaned, self.classifier.vocabulary)
            classifier_result = self.classifier.predict(cleaned)
            candidate = self.retriever.best_match(
                cleaned, classifier_result.predicted_intent
            )

        decision = decide(lexical, classifier_result, candidate, policy)

        if decision.outcome is Outcome.ANSWERED:
            answer_text, answer_source = candidate.response, "knowledge_base"
        elif decision.outcome is Outcome.INTENT_FALLBACK:
            answer_text = INTENT_FALLBACK_TEMPLATE.format(
                area=_area_name(decision.committed_intent)
            )
            answer_source = "intent_fallback"
        elif decision.reason is AbstainReason.EMPTY_INPUT:
            answer_text, answer_source = EMPTY_INPUT_MESSAGE, "clarification"
        else:
            answer_text, answer_source = CLARIFICATION_MESSAGE, "clarification"

        return ChatResult(
            outcome=decision.outcome,
            answer=answer_text,
            answer_source=answer_source,
            abstained=decision.outcome is not Outcome.ANSWERED,
            abstain_reason=decision.reason,
            intent=decision.committed_intent,
            predicted_intent=classifier_result.predicted_intent if classifier_result else None,
            margin=classifier_result.margin if classifier_result else None,
            top_predictions=classifier_result.top_predictions if classifier_result else (),
            similarity=candidate.cosine if candidate else None,
            overlap=candidate.dice if candidate else None,
            matched_question=candidate.matched_question if candidate else None,
            matched_row_id=candidate.row_id if candidate else None,
            shared_content_tokens=candidate.shared_content_tokens if candidate else (),
            lexical=lexical,
            policy=policy,
            model_version=self.model_version,
            kb_version=self.kb_version,
            policy_version=self.policy_version,
        )


# Module-level singleton instance for convenient reuse
_default_pipeline: Optional[ChatPipeline] = None


def get_pipeline() -> ChatPipeline:
    """Get or initialize the shared ChatPipeline instance."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = ChatPipeline()
    return _default_pipeline


def answer(message: str, policy: Optional[PolicyConfig] = None) -> ChatResult:
    """Answer a message using the default pipeline."""
    return get_pipeline().answer(message, policy=policy)
