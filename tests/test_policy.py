"""Pure unit tests for policy.decide(); no artifacts or data required."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.policy import DEFAULT_POLICY, decide, paraphrase_passes
from src.ml.schemas import (
    AbstainReason,
    ClassifierResult,
    LexicalEvidence,
    Outcome,
    PolicyConfig,
    RetrievalCandidate,
)


def lexical(in_vocab=("aid", "financial"), oov=()):
    content = tuple(in_vocab) + tuple(oov)
    return LexicalEvidence(
        tokens=content,
        content_tokens=content,
        in_vocab_content_tokens=tuple(in_vocab),
        oov_content_tokens=tuple(oov),
        in_vocab_content_count=len(in_vocab),
        content_oov_ratio=len(oov) / len(content) if content else 1.0,
    )


def classifier(margin=1.0, intent="financial_aid"):
    return ClassifierResult(
        predicted_intent=intent,
        margin=margin,
        top_score=0.5,
        second_score=0.5 - margin,
        top_predictions=((intent, 0.5), ("other", 0.5 - margin)),
    )


def candidate(dice=0.9, cosine=0.8, shared=("aid", "financial")):
    return RetrievalCandidate(
        row_id=0,
        matched_question="q",
        response="canned response",
        cosine=cosine,
        shared_content_tokens=tuple(shared),
        dice=dice,
    )


class TestGateOrder(unittest.TestCase):
    def test_empty_input(self):
        d = decide(None, None, None)
        self.assertEqual((d.outcome, d.reason), (Outcome.ABSTAINED, AbstainReason.EMPTY_INPUT))

    def test_no_lexical_evidence(self):
        d = decide(lexical(in_vocab=(), oov=("zzz",)), classifier(), candidate())
        self.assertEqual(d.reason, AbstainReason.NO_LEXICAL_EVIDENCE)

    def test_stopword_only_has_no_lexical_evidence(self):
        d = decide(lexical(in_vocab=(), oov=()), classifier(), candidate())
        self.assertEqual(d.reason, AbstainReason.NO_LEXICAL_EVIDENCE)

    def test_no_lexical_evidence_precedes_low_margin(self):
        d = decide(lexical(in_vocab=()), classifier(margin=0.0), None)
        self.assertEqual(d.reason, AbstainReason.NO_LEXICAL_EVIDENCE)

    def test_low_margin(self):
        d = decide(lexical(), classifier(margin=0.1), candidate())
        self.assertEqual((d.outcome, d.reason), (Outcome.ABSTAINED, AbstainReason.LOW_MARGIN))

    def test_retrieval_cannot_override_low_margin(self):
        perfect = candidate(dice=1.0, cosine=1.0, shared=("aid", "financial"))
        d = decide(lexical(), classifier(margin=0.01), perfect)
        self.assertEqual(d.reason, AbstainReason.LOW_MARGIN)

    def test_answered(self):
        d = decide(lexical(), classifier(), candidate())
        self.assertEqual(d.outcome, Outcome.ANSWERED)
        self.assertIsNone(d.reason)
        self.assertEqual(d.committed_intent, "financial_aid")

    def test_paraphrase_precedes_weak_lexical(self):
        weak = lexical(in_vocab=("password",), oov=("portalx",))
        d = decide(weak, classifier(), candidate(shared=("password",)))
        self.assertEqual(d.outcome, Outcome.ANSWERED)

    def test_weak_lexical_evidence(self):
        weak = lexical(in_vocab=("connect",), oov=("college", "faculty"))
        d = decide(weak, classifier(), candidate(dice=0.29, shared=("connect",)))
        self.assertEqual((d.outcome, d.reason), (Outcome.ABSTAINED, AbstainReason.WEAK_LEXICAL_EVIDENCE))
        self.assertIsNone(d.committed_intent)

    def test_oov_ratio_alone_does_not_abstain(self):
        # High OOV ratio but two in-vocab content tokens -> fallback, not abstention.
        lex = lexical(in_vocab=("campus", "dining"), oov=("card", "isn", "working", "hall"))
        d = decide(lex, classifier(intent="campus_facilities"), candidate(dice=0.36))
        self.assertEqual((d.outcome, d.reason), (Outcome.INTENT_FALLBACK, AbstainReason.NO_RELIABLE_MATCH))
        self.assertEqual(d.committed_intent, "campus_facilities")

    def test_no_candidate_gives_fallback(self):
        d = decide(lexical(), classifier(), None)
        self.assertEqual((d.outcome, d.reason), (Outcome.INTENT_FALLBACK, AbstainReason.NO_RELIABLE_MATCH))

    def test_no_candidate_with_weak_lexical_abstains(self):
        d = decide(lexical(in_vocab=("x1",), oov=("y1",)), classifier(), None)
        self.assertEqual(d.reason, AbstainReason.WEAK_LEXICAL_EVIDENCE)

    def test_missing_classifier_is_contract_violation(self):
        with self.assertRaises(ValueError):
            decide(lexical(), None, None)


class TestBoundaries(unittest.TestCase):
    def test_margin_equal_to_threshold_is_accepted(self):
        policy = replace(DEFAULT_POLICY, margin_threshold=0.5)
        d = decide(lexical(), classifier(margin=0.5), candidate(), policy)
        self.assertNotEqual(d.reason, AbstainReason.LOW_MARGIN)

    def test_margin_just_below_threshold_abstains(self):
        policy = replace(DEFAULT_POLICY, margin_threshold=0.5)
        d = decide(lexical(), classifier(margin=0.4999), candidate(), policy)
        self.assertEqual(d.reason, AbstainReason.LOW_MARGIN)

    def test_dice_equal_to_minimum_passes(self):
        self.assertTrue(paraphrase_passes(candidate(dice=DEFAULT_POLICY.paraphrase_min_dice)))

    def test_dice_below_minimum_fails(self):
        self.assertFalse(paraphrase_passes(candidate(dice=DEFAULT_POLICY.paraphrase_min_dice - 1e-9)))

    def test_custom_threshold_changes_outcome(self):
        d_default = decide(lexical(), classifier(margin=0.3), candidate())
        d_strict = decide(lexical(), classifier(margin=0.3), candidate(),
                          replace(DEFAULT_POLICY, margin_threshold=0.4))
        self.assertEqual(d_default.outcome, Outcome.ANSWERED)
        self.assertEqual(d_strict.reason, AbstainReason.LOW_MARGIN)


class TestParaphraseGate(unittest.TestCase):
    def test_high_cosine_low_dice_not_answered(self):
        self.assertFalse(paraphrase_passes(candidate(cosine=0.95, dice=0.3)))

    def test_generic_only_shared_tokens_not_answered(self):
        self.assertFalse(paraphrase_passes(candidate(dice=1.0, shared=("campus", "student"))))

    def test_zero_cosine_not_answered(self):
        self.assertFalse(paraphrase_passes(candidate(cosine=0.0, dice=1.0)))

    def test_none_candidate(self):
        self.assertFalse(paraphrase_passes(None))


class TestPolicyConfig(unittest.TestCase):
    def test_default_values(self):
        self.assertEqual(DEFAULT_POLICY.margin_threshold, 0.20)
        self.assertEqual(DEFAULT_POLICY.paraphrase_min_dice, 0.60)
        self.assertEqual(DEFAULT_POLICY.weak_evidence_max_in_vocab_content, 1)
        self.assertEqual(DEFAULT_POLICY.weak_evidence_min_oov_ratio, 0.5)

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            replace(DEFAULT_POLICY, margin_threshold=-0.1)
        with self.assertRaises(ValueError):
            replace(DEFAULT_POLICY, paraphrase_min_dice=1.5)

    def test_frozen(self):
        with self.assertRaises(Exception):
            DEFAULT_POLICY.margin_threshold = 0.9


class TestInvariants(unittest.TestCase):
    def test_reason_and_intent_invariants_over_grid(self):
        lexicals = [None, lexical(in_vocab=()), lexical(in_vocab=("a1",), oov=("b1",)), lexical()]
        margins = [0.0, 0.2, 1.0]
        candidates = [None, candidate(dice=0.3), candidate(), candidate(shared=("campus",))]
        for lex in lexicals:
            for m in margins:
                for cand in candidates:
                    d = decide(lex, classifier(margin=m), cand)
                    with self.subTest(lex=lex, margin=m, cand=cand):
                        if d.outcome is Outcome.ANSWERED:
                            self.assertIsNone(d.reason)
                        else:
                            self.assertIsNotNone(d.reason)
                        if d.outcome is Outcome.ABSTAINED:
                            self.assertIsNone(d.committed_intent)
                        else:
                            self.assertIsNotNone(d.committed_intent)
                        self.assertEqual(
                            d.outcome is Outcome.INTENT_FALLBACK,
                            d.reason is AbstainReason.NO_RELIABLE_MATCH,
                        )


if __name__ == "__main__":
    unittest.main()
