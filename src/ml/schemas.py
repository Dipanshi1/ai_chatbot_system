"""
Typed contracts for the Phase 4 inference pipeline.

Evidence types (LexicalEvidence, ClassifierResult, RetrievalCandidate) are
produced by components that report facts only. Decision is produced solely by
policy.decide(). ChatResult is the single public result of pipeline.answer().
"""
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class Outcome(str, Enum):
    ANSWERED = "answered"
    INTENT_FALLBACK = "intent_fallback"
    ABSTAINED = "abstained"


class AbstainReason(str, Enum):
    EMPTY_INPUT = "empty_input"
    NO_LEXICAL_EVIDENCE = "no_lexical_evidence"
    LOW_MARGIN = "low_margin"
    WEAK_LEXICAL_EVIDENCE = "weak_lexical_evidence"
    NO_RELIABLE_MATCH = "no_reliable_match"


@dataclass(frozen=True)
class LexicalEvidence:
    tokens: Tuple[str, ...]
    content_tokens: Tuple[str, ...]
    in_vocab_content_tokens: Tuple[str, ...]
    oov_content_tokens: Tuple[str, ...]
    in_vocab_content_count: int
    content_oov_ratio: float


@dataclass(frozen=True)
class ClassifierResult:
    predicted_intent: str
    margin: float
    top_score: float
    second_score: float
    top_predictions: Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class RetrievalCandidate:
    row_id: int
    matched_question: str
    response: str
    cosine: float
    shared_content_tokens: Tuple[str, ...]
    dice: float


@dataclass(frozen=True)
class PolicyConfig:
    """
    Policy thresholds. All values are prototype policy constants;
    not fitted; not optimal.
    """

    margin_threshold: float
    paraphrase_min_dice: float
    weak_evidence_max_in_vocab_content: int
    weak_evidence_min_oov_ratio: float
    retrieval_generic_tokens: frozenset

    def __post_init__(self) -> None:
        if self.margin_threshold < 0:
            raise ValueError("margin_threshold must be >= 0")
        if not 0.0 <= self.paraphrase_min_dice <= 1.0:
            raise ValueError("paraphrase_min_dice must be in [0, 1]")
        if self.weak_evidence_max_in_vocab_content < 0:
            raise ValueError("weak_evidence_max_in_vocab_content must be >= 0")
        if not 0.0 <= self.weak_evidence_min_oov_ratio <= 1.0:
            raise ValueError("weak_evidence_min_oov_ratio must be in [0, 1]")


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    reason: Optional[AbstainReason]
    committed_intent: Optional[str]


@dataclass(frozen=True)
class ChatResult:
    outcome: Outcome
    answer: str
    answer_source: str  # "knowledge_base" | "intent_fallback" | "clarification"
    abstained: bool
    abstain_reason: Optional[AbstainReason]
    intent: Optional[str]
    predicted_intent: Optional[str]
    margin: Optional[float]
    top_predictions: Tuple[Tuple[str, float], ...]
    similarity: Optional[float]
    overlap: Optional[float]
    matched_question: Optional[str]
    matched_row_id: Optional[int]
    shared_content_tokens: Tuple[str, ...]
    lexical: Optional[LexicalEvidence]
    policy: PolicyConfig
    model_version: str
    kb_version: str
    policy_version: str

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly representation (enums → str, tuples/sets → lists)."""
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, frozenset):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
