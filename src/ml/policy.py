"""
Safety policy: the SINGLE place where answer / fallback / abstention is decided.

decide() is a pure function over evidence produced by lexical.py, inference.py
and retrieval.py. It performs no I/O and holds no state.

Gate order (first failing gate wins, so exactly one reason is reported):
    1. EMPTY_INPUT            -> ABSTAINED
    2. NO_LEXICAL_EVIDENCE    -> ABSTAINED
    3. LOW_MARGIN             -> ABSTAINED   (margin == threshold is accepted)
    4. paraphrase passes      -> ANSWERED
    5. WEAK_LEXICAL_EVIDENCE  -> ABSTAINED
    6. NO_RELIABLE_MATCH      -> INTENT_FALLBACK

Retrieval can never override a low classifier margin.
"""
from typing import Optional

from .config import (
    MARGIN_THRESHOLD,
    PARAPHRASE_MIN_DICE,
    RETRIEVAL_GENERIC_TOKENS,
    WEAK_EVIDENCE_MAX_IN_VOCAB_CONTENT,
    WEAK_EVIDENCE_MIN_OOV_RATIO,
)
from .schemas import (
    AbstainReason,
    ClassifierResult,
    Decision,
    LexicalEvidence,
    Outcome,
    PolicyConfig,
    RetrievalCandidate,
)

# All values are prototype policy constants; not fitted; not optimal.
DEFAULT_POLICY = PolicyConfig(
    margin_threshold=MARGIN_THRESHOLD,
    paraphrase_min_dice=PARAPHRASE_MIN_DICE,
    weak_evidence_max_in_vocab_content=WEAK_EVIDENCE_MAX_IN_VOCAB_CONTENT,
    weak_evidence_min_oov_ratio=WEAK_EVIDENCE_MIN_OOV_RATIO,
    retrieval_generic_tokens=RETRIEVAL_GENERIC_TOKENS,
)


def paraphrase_passes(
    candidate: Optional[RetrievalCandidate], policy: PolicyConfig = DEFAULT_POLICY
) -> bool:
    """
    True only if the matched knowledge-base question is a plausible paraphrase:
    cosine > 0, Dice >= minimum, and at least one shared non-generic content token.
    Cosine is used for ranking only; Dice is the gate.
    """
    if candidate is None or candidate.cosine <= 0.0:
        return False
    if candidate.dice < policy.paraphrase_min_dice:
        return False
    specific_shared = set(candidate.shared_content_tokens) - policy.retrieval_generic_tokens
    return bool(specific_shared)


def _is_weak_lexical(lexical: LexicalEvidence, policy: PolicyConfig) -> bool:
    return (
        lexical.in_vocab_content_count <= policy.weak_evidence_max_in_vocab_content
        and lexical.content_oov_ratio >= policy.weak_evidence_min_oov_ratio
    )


def _abstain(reason: AbstainReason) -> Decision:
    return Decision(outcome=Outcome.ABSTAINED, reason=reason, committed_intent=None)


def decide(
    lexical: Optional[LexicalEvidence],
    classifier: Optional[ClassifierResult],
    candidate: Optional[RetrievalCandidate],
    policy: PolicyConfig = DEFAULT_POLICY,
) -> Decision:
    """
    Decide the outcome for one message.

    `lexical is None` means the message was empty after clean_text().
    """
    if lexical is None:
        return _abstain(AbstainReason.EMPTY_INPUT)

    if lexical.in_vocab_content_count == 0:
        return _abstain(AbstainReason.NO_LEXICAL_EVIDENCE)

    if classifier is None:
        raise ValueError("classifier evidence is required when lexical evidence exists")

    if classifier.margin < policy.margin_threshold:
        return _abstain(AbstainReason.LOW_MARGIN)

    if paraphrase_passes(candidate, policy):
        return Decision(
            outcome=Outcome.ANSWERED,
            reason=None,
            committed_intent=classifier.predicted_intent,
        )

    if _is_weak_lexical(lexical, policy):
        return _abstain(AbstainReason.WEAK_LEXICAL_EVIDENCE)

    return Decision(
        outcome=Outcome.INTENT_FALLBACK,
        reason=AbstainReason.NO_RELIABLE_MATCH,
        committed_intent=classifier.predicted_intent,
    )
