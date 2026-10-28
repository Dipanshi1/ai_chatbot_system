"""Unit tests for B1 Phase 3 evaluation logic (lib/evaluation.py). No model loading."""
import math
import sys
import unittest
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
for _p in (str(REPO_ROOT), str(EXPERIMENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
from lib import evaluation as ev  # noqa: E402
from lib.metrics import QueryLabels  # noqa: E402

INTENT_OF = {1: "a", 2: "a", 3: "b", 4: "c"}


class FakeRetriever:
    """Returns a fixed ranking per (query, intent); records calls."""

    def __init__(self, name, table):
        self.name, self.table, self.calls = name, table, []

    def rank(self, query, k, intent=None):
        self.calls.append((query, intent))
        ranking = self.table.get(query, [])
        if intent is not None:
            ranking = [(r, s) for r, s in ranking if INTENT_OF[r] == intent]
        return ranking[:k]


def item(cid, query, target, acc, pref=None, category="synonym_substitution", primary=True, slice_="B1"):
    return ev.BenchmarkItem(cid, query, category, QueryLabels(cid, target, frozenset(acc), pref),
                            positive=category != config.NEGATIVE_CATEGORY, primary=primary, slice=slice_)


class TestRunPasses(unittest.TestCase):
    def test_rows_for_every_query_pass_and_retriever(self):
        items = [item("CH_001", "q1", "a", {1}), item("CH_002", "q2", "c", (), category=config.NEGATIVE_CATEGORY,
                                                        primary=False, slice_="NEG")]
        table = {"q1": [(3, 0.9), (1, 0.8)], "q2": [(4, 0.5)]}
        rets = [FakeRetriever("base", table), FakeRetriever("cand", table)]
        rows = ev.run_passes(rets, items, predicted_intent={"CH_001": "b", "CH_002": "c"}, intent_of=INTENT_OF, k=10)
        self.assertEqual(len(rows), 2 * 3 * 2)
        p1 = next(r for r in rows if r["challenge_id"] == "CH_001" and r["pass"] == "pass1_global" and r["retriever"] == "base")
        self.assertEqual(p1["rank1_row_id"], 3)
        self.assertEqual(p1["outcome_class"], "WRONG_INTENT")
        p2 = next(r for r in rows if r["challenge_id"] == "CH_001" and r["pass"] == "pass2_oracle" and r["retriever"] == "base")
        self.assertEqual(p2["rank1_row_id"], 1)  # restricted to oracle intent "a"
        p3 = next(r for r in rows if r["challenge_id"] == "CH_001" and r["pass"] == "pass3_predicted" and r["retriever"] == "base")
        self.assertEqual(p3["rank1_row_id"], 3)  # restricted to predicted intent "b"
        self.assertIn(("q1", "b"), rets[0].calls)


class TestSummaries(unittest.TestCase):
    def setUp(self):
        self.items = [item(f"CH_{i:03d}", f"q{i}", "a", {1}) for i in range(1, 5)]
        self.rows = []
        for it in self.items:
            for name, ranking in (("base", [(3, 0.5)]), ("cand", [(1, 0.9), (3, 0.2)])):
                self.rows.append(ev.make_row(it, "pass1_global", name, ranking, INTENT_OF, predicted=None))

    def test_population_metrics_and_paired_deltas(self):
        s = ev.population_summary(self.rows, self.items, "pass1_global", "base", "cand", lambda it: it.primary)
        self.assertEqual(s["n_queries"], 4)
        self.assertEqual(s["retrievers"]["base"]["AcceptableRecall@1"]["value"], 0.0)
        self.assertEqual(s["retrievers"]["cand"]["AcceptableRecall@1"]["value"], 1.0)
        d = s["deltas"]["AcceptableRecall@1"]
        self.assertEqual((d["delta"], d["n"]), (1.0, 4))
        self.assertEqual(s["deltas"]["WrongIntent@1"]["delta"], -1.0)
        self.assertNotIn("CosineMargin", s["deltas"])

    def test_empty_population(self):
        s = ev.population_summary(self.rows, self.items, "pass1_global", "base", "cand", lambda it: False)
        self.assertEqual(s["n_queries"], 0)
        self.assertEqual(s["deltas"], {})


def stats(delta, lo, hi):
    return {"delta": delta, "ci_low": lo, "ci_high": hi, "n": 40}


class TestOutcomeCategory(unittest.TestCase):
    def cat(self, ar, mrr, wi=stats(0.0, -0.05, 0.05), p=0.01, resources_ok=True):
        return ev.assign_outcome_category(ar, mrr, wi, {"p_value": p}, resources_ok)

    def test_meaningful_improvement(self):
        c = self.cat(stats(0.20, 0.05, 0.35), stats(0.20, 0.05, 0.30))
        self.assertEqual(c["category"], "Meaningful improvement")

    def test_threshold_not_met_is_modest(self):
        c = self.cat(stats(0.10, 0.02, 0.20), stats(0.08, 0.01, 0.15))
        self.assertEqual(c["category"], "Modest improvement")

    def test_regression_on_ci_upper_below_zero(self):
        self.assertEqual(self.cat(stats(-0.2, -0.3, -0.1), stats(0.0, -0.1, 0.1))["category"], "Regression")

    def test_regression_on_safety_guardrail(self):
        c = self.cat(stats(0.3, 0.1, 0.5), stats(0.3, 0.1, 0.5), wi=stats(0.06, 0.0, 0.12))
        self.assertEqual(c["category"], "Regression")
        self.assertFalse(c["safety_guardrail_holds"])

    def test_safety_boundary_is_inclusive(self):
        c = self.cat(stats(0.3, 0.1, 0.5), stats(0.3, 0.1, 0.5), wi=stats(config.S_PP / 100, 0.0, 0.1))
        self.assertTrue(c["safety_guardrail_holds"])

    def test_underpowered_label(self):
        c = self.cat(stats(0.20, -0.02, 0.40), stats(0.20, -0.01, 0.40))
        self.assertEqual(c["category"], "No meaningful difference / inconclusive")
        self.assertTrue(c["underpowered"])

    def test_wilcoxon_required_but_not_sufficient(self):
        self.assertEqual(self.cat(stats(0.2, 0.05, 0.3), stats(0.2, 0.05, 0.3), p=0.2)["category"], "Modest improvement")
        self.assertEqual(self.cat(stats(0.2, 0.05, 0.3), stats(0.2, 0.05, 0.3), p="n/a")["category"], "Modest improvement")

    def test_resources_not_measured_label(self):
        c = self.cat(stats(0.2, 0.05, 0.3), stats(0.2, 0.05, 0.3), resources_ok=None)
        self.assertEqual(c["category"], "Meaningful retrieval improvement — resources not measured")

    def test_resource_failure_label(self):
        c = self.cat(stats(0.2, 0.05, 0.3), stats(0.2, 0.05, 0.3), resources_ok=False)
        self.assertEqual(c["category"], "Meaningful retrieval improvement — resource limit exceeded")


class TestFailureTagging(unittest.TestCase):
    def test_signatures_and_precedence(self):
        self.assertEqual(ev.failure_signatures("x y", 0.9, "WRONG_INTENT", negative=False), ["F3", "F1"])
        self.assertEqual(ev.primary_failure_tag(["F3", "F1"]), "F3")
        self.assertEqual(ev.failure_signatures("alpha beta gamma", 0.2, "INTENT_ONLY", negative=False), ["F4"])
        self.assertEqual(ev.failure_signatures("alpha beta gamma", 0.5, "WRONG_INTENT", negative=True), ["F5"])
        self.assertEqual(ev.primary_failure_tag([]), "UNASSIGNED")

    def test_is_failure(self):
        self.assertTrue(ev.is_failure("INTENT_ONLY", negative=False))
        self.assertFalse(ev.is_failure("INTENT_ONLY", negative=True))
        self.assertTrue(ev.is_failure("WRONG_INTENT", negative=True))
        self.assertFalse(ev.is_failure("ACCEPTABLE_HIT", negative=False))


class TestScoreDistribution(unittest.TestCase):
    def test_distribution(self):
        d = ev.score_distribution([0.1, 0.2, 0.3, None])
        self.assertEqual(d["n"], 3)
        self.assertAlmostEqual(d["max"], 0.3)
        self.assertTrue(math.isfinite(d["p95"]))
        self.assertEqual(ev.score_distribution([])["n"], 0)


if __name__ == "__main__":
    unittest.main()
