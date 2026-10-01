"""
Behavioural regression tests for pipeline.answer() against the frozen artifacts.

Several queries are near-duplicates of knowledge-base rows (some from the test
split). They pin behaviour; they are not evidence of model accuracy.
"""
import io
import json
import re
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import chatbot
from src.ml import (
    DEFAULT_POLICY,
    AbstainReason,
    Outcome,
    answer,
    get_retriever,
)

FACULTY = "how do i connect to college faculty?"
CAMPUS_CARD = "my campus card isn't working at the dining hall."

REGRESSION_CASES = [
    ("", Outcome.ABSTAINED, AbstainReason.EMPTY_INPUT, None),
    ("   \n\t ", Outcome.ABSTAINED, AbstainReason.EMPTY_INPUT, None),
    ("zzz qqq", Outcome.ABSTAINED, AbstainReason.NO_LEXICAL_EVIDENCE, None),
    ("what is the", Outcome.ABSTAINED, AbstainReason.NO_LEXICAL_EVIDENCE, None),
    (FACULTY, Outcome.ABSTAINED, AbstainReason.WEAK_LEXICAL_EVIDENCE, None),
    ("where can i find internship postings?", Outcome.ABSTAINED, AbstainReason.LOW_MARGIN, None),
    (CAMPUS_CARD, Outcome.INTENT_FALLBACK, AbstainReason.NO_RELIABLE_MATCH, "campus_facilities"),
    ("when will my financial aid be disbursed?", Outcome.ANSWERED, None, "financial_aid"),
    ("how do i connect to campus wi-fi?", Outcome.ANSWERED, None, "technical_issue"),
    ("i forgot my portal password", Outcome.ANSWERED, None, "technical_issue"),
]

ALL_QUERIES = [q for q, *_ in REGRESSION_CASES] + [
    "are there support groups for grief?",
    "how can i apply for financial aid?",
]


def kb_responses():
    return set(get_retriever()._responses)


class TestRegressionOutcomes(unittest.TestCase):
    def test_regression_cases(self):
        for query, outcome, reason, intent in REGRESSION_CASES:
            with self.subTest(query=query):
                r = answer(query)
                self.assertEqual(r.outcome, outcome)
                self.assertEqual(r.abstain_reason, reason)
                self.assertEqual(r.intent, intent)

    def test_faculty_query_does_not_return_vpn_answer(self):
        r = answer(FACULTY)
        self.assertNotIn("vpn", r.answer.lower())
        self.assertNotIn(r.answer, kb_responses())
        self.assertIsNone(r.intent)
        self.assertEqual(r.predicted_intent, "technical_issue")  # diagnostics kept

    def test_campus_card_query_does_not_return_dining_menu_answer(self):
        r = answer(CAMPUS_CARD)
        self.assertNotIn("menu", r.answer.lower())
        self.assertNotIn(r.answer, kb_responses())
        self.assertEqual(r.answer_source, "intent_fallback")
        self.assertIn("campus facilities", r.answer)

    def test_answered_returns_knowledge_base_response_with_trace(self):
        r = answer("when will my financial aid be disbursed?")
        self.assertIn(r.answer, kb_responses())
        self.assertEqual(r.answer_source, "knowledge_base")
        self.assertIsNotNone(r.matched_question)
        self.assertIsInstance(r.matched_row_id, int)
        self.assertGreaterEqual(r.overlap, DEFAULT_POLICY.paraphrase_min_dice)

    def test_empty_input_has_no_classifier_evidence(self):
        r = answer("   ")
        self.assertIsNone(r.margin)
        self.assertIsNone(r.predicted_intent)
        self.assertIsNone(r.lexical)
        self.assertEqual(r.top_predictions, ())


class TestRetrievalEvidence(unittest.TestCase):
    def test_no_usable_candidate(self):
        retriever = get_retriever()
        self.assertIsNone(retriever.best_match("zzz qqq", "technical_issue"))
        self.assertIsNone(retriever.best_match("what is the", "technical_issue"))
        self.assertIsNone(retriever.best_match("", "technical_issue"))
        self.assertIsNone(retriever.best_match("password", "no_such_intent"))

    def test_candidate_restricted_to_intent(self):
        retriever = get_retriever()
        query = "how do i request financial aid for campus housing and parking?"
        found = 0
        for intent in retriever.intents:
            cand = retriever.best_match(query, intent)
            if cand is not None:
                found += 1
                self.assertEqual(retriever.df.loc[cand.row_id, "Intent"], intent)
        self.assertGreater(found, 1)

    def test_dice_computed_on_content_tokens(self):
        cand = get_retriever().best_match("how do i connect to college faculty?", "technical_issue")
        self.assertIsNotNone(cand)
        self.assertEqual(cand.shared_content_tokens, ("connect",))
        self.assertLess(cand.dice, DEFAULT_POLICY.paraphrase_min_dice)


class TestThresholds(unittest.TestCase):
    def test_custom_threshold(self):
        default = answer(FACULTY)
        strict = answer(FACULTY, policy=replace(DEFAULT_POLICY, margin_threshold=0.40))
        self.assertEqual(default.abstain_reason, AbstainReason.WEAK_LEXICAL_EVIDENCE)
        self.assertEqual(strict.abstain_reason, AbstainReason.LOW_MARGIN)
        self.assertEqual(strict.policy.margin_threshold, 0.40)
        self.assertEqual(default.policy, DEFAULT_POLICY)

    def test_margin_equal_to_threshold_is_accepted(self):
        query = "when will my financial aid be disbursed?"
        margin = answer(query).margin
        at_boundary = answer(query, policy=replace(DEFAULT_POLICY, margin_threshold=margin))
        self.assertEqual(at_boundary.outcome, Outcome.ANSWERED)

    def test_raising_threshold_never_creates_answers(self):
        thresholds = [0.0, 0.1, 0.2, 0.4, 1.0, 10.0]
        for query in ALL_QUERIES:
            answered = [
                answer(query, policy=replace(DEFAULT_POLICY, margin_threshold=t)).outcome
                is Outcome.ANSWERED
                for t in thresholds
            ]
            with self.subTest(query=query):
                # Once a threshold stops an answer, every higher threshold must too.
                self.assertEqual(answered, sorted(answered, reverse=True))


class TestContract(unittest.TestCase):
    def test_invariants(self):
        for query in ALL_QUERIES:
            r = answer(query)
            with self.subTest(query=query):
                self.assertEqual(r.abstained, r.outcome is not Outcome.ANSWERED)
                self.assertEqual(r.abstain_reason is None, not r.abstained)
                self.assertTrue(r.answer.strip())
                if r.outcome is Outcome.ABSTAINED:
                    self.assertIsNone(r.intent)
                    self.assertEqual(r.answer_source, "clarification")
                else:
                    self.assertEqual(r.intent, r.predicted_intent)
                if r.outcome is Outcome.ANSWERED:
                    self.assertEqual(r.answer_source, "knowledge_base")
                    self.assertIn(r.answer, kb_responses())
                else:
                    self.assertNotIn(r.answer, kb_responses())

    def test_deterministic(self):
        for query in ALL_QUERIES:
            with self.subTest(query=query):
                self.assertEqual(answer(query).to_dict(), answer(query).to_dict())

    def test_to_dict_is_json_serialisable(self):
        d = answer("i forgot my portal password").to_dict()
        roundtrip = json.loads(json.dumps(d))
        self.assertEqual(roundtrip["outcome"], "answered")
        self.assertIsNone(roundtrip["abstain_reason"])
        self.assertIsInstance(roundtrip["policy"]["retrieval_generic_tokens"], list)
        self.assertIsInstance(roundtrip["lexical"]["content_tokens"], list)

    def test_versions(self):
        r = answer("zzz qqq")
        self.assertRegex(r.model_version, r"^linsvc-[0-9a-f]{12}\+tfidf-[0-9a-f]{12}$")
        self.assertRegex(r.kb_version, r"^kb-[0-9a-f]{12}$")
        self.assertEqual(r.policy_version, "phase4-v1")


class TestCLI(unittest.TestCase):
    def run_cli(self, message, debug=False):
        out = io.StringIO()
        with mock.patch("builtins.input", return_value=message), redirect_stdout(out):
            chatbot.run_cli(debug=debug)
        return out.getvalue()

    def test_cli_prints_result_answer(self):
        for query in ALL_QUERIES:
            with self.subTest(query=query):
                output = self.run_cli(query)
                self.assertEqual(output, f"Bot: {answer(query).answer}\n")

    def test_cli_debug_shows_diagnostics(self):
        output = self.run_cli(FACULTY, debug=True)
        self.assertIn("weak_lexical_evidence", output)
        self.assertIn("technical_issue", output)
        self.assertNotIn("vpn", output.split("Bot:")[-1].lower())

    def test_cli_has_no_threshold_logic(self):
        source = Path(chatbot.__file__).read_text()
        self.assertIsNone(re.search(r"THRESHOLD|margin\s*<", source))


if __name__ == "__main__":
    unittest.main()
