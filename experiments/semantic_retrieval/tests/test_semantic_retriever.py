"""
B1 experiment tests (unittest only). Run from the repo root:

  HF_HUB_OFFLINE=1 experiments/semantic_retrieval/.venv-b1/bin/python -m unittest \
      discover -s experiments/semantic_retrieval/tests -v

Requires the KB index (scripts/01_build_index.py) and the cached pinned model.
Anything that needs a verified benchmark uses synthetic fixtures.
"""
import importlib.util
import json
import math
import re
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
for _p in (str(REPO_ROOT), str(EXPERIMENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

import config  # noqa: E402
from lib import benchmark as bm  # noqa: E402
from lib import data as b1_data  # noqa: E402
from lib import dice as b1_dice  # noqa: E402
from lib import metrics as m  # noqa: E402
from lib import provenance  # noqa: E402
from lib import resources  # noqa: E402
from lib import retrievers  # noqa: E402
from lib import stats as b1_stats  # noqa: E402
from src.ml.preprocessing import clean_text  # noqa: E402
from src.ml.retrieval import ResponseRetriever  # noqa: E402

SPLIT = b1_data.load_canonical_split()
_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = retrievers.load_embedding_model()
    return _MODEL


# ---------------------------------------------------------------------------
# Stage A baseline adapter
# ---------------------------------------------------------------------------
class TestStageAAdapterParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = retrievers.StageARetrieverAdapter(SPLIT.train_df)
        cls.reference = ResponseRetriever(df=SPLIT.train_df)

    def test_rank1_matches_best_match_for_every_query_and_intent(self):
        queries = list(SPLIT.train_df["User Message"])
        if config.BENCHMARK_PATH.exists():
            queries += bm.read_benchmark()["query_text"].tolist()
        queries += ["", "   ", "zzz qqq", "what is the"]
        intents = list(SPLIT.all_intents()) + ["no_such_intent"]
        checked = 0
        for q in queries:
            for intent in intents:
                ours = self.adapter.rank(q, 1, intent=intent)
                ref = self.reference.best_match(q, intent)
                if ref is None:
                    self.assertEqual(ours, [], f"{q!r} / {intent}")
                else:
                    self.assertEqual(len(ours), 1, f"{q!r} / {intent}")
                    self.assertEqual(ours[0][0], ref.row_id, f"{q!r} / {intent}")
                    self.assertLessEqual(abs(ours[0][1] - ref.cosine), config.PARITY_COSINE_TOL)
                checked += 1
        self.assertGreater(checked, 160 * 26)

    def test_adapter_is_read_only(self):
        r = self.adapter.retriever
        before = (len(r._vectorizer.vocabulary_), r._matrix.shape, r._matrix.nnz, r.kb_version)
        for q in SPLIT.train_df["User Message"][:20]:
            self.adapter.rank(q, config.RETRIEVAL_DEPTH)
        after = (len(r._vectorizer.vocabulary_), r._matrix.shape, r._matrix.nnz, r.kb_version)
        self.assertEqual(before, after)

    def test_zero_score_rows_are_not_retrieved(self):
        self.assertEqual(self.adapter.rank("zzz qqq", 10), [])
        ranking = self.adapter.rank("forgot portal password", 160)
        self.assertTrue(all(score > 0 for _, score in ranking))

    def test_global_ranking_is_sorted_and_deterministic(self):
        r1 = self.adapter.rank("How do I apply for a transcript?", 10)
        r2 = self.adapter.rank("How do I apply for a transcript?", 10)
        self.assertEqual(r1, r2)
        scores = [s for _, s in r1]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ---------------------------------------------------------------------------
# KB index and embeddings
# ---------------------------------------------------------------------------
class TestKBIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.semantic = retrievers.SemanticRetriever(SPLIT.train_df, model=model())
        cls.adapter = retrievers.StageARetrieverAdapter(SPLIT.train_df)

    def test_kb_is_exactly_the_160_train_ids_ascending(self):
        for ids in (self.semantic.row_ids, self.adapter.row_ids):
            self.assertEqual(len(ids), 160)
            self.assertEqual(list(ids), sorted(ids))
            self.assertEqual(set(ids), set(SPLIT.train_ids))
            self.assertFalse(set(ids) & set(SPLIT.test_ids))
            b1_data.assert_kb_is_train_only(ids, SPLIT)

    def test_kb_guard_rejects_test_ids(self):
        bad = list(SPLIT.train_ids[:-1]) + [SPLIT.test_ids[0]]
        with self.assertRaises(b1_data.LeakageError):
            b1_data.assert_kb_is_train_only(bad, SPLIT)

    def test_index_file_shape_dtype_and_unit_norm(self):
        index = np.load(config.INDEX_PATH)
        self.assertEqual(index.shape, (160, config.EMBEDDING_DIM))
        self.assertEqual(index.dtype, np.float32)
        norms = np.linalg.norm(index.astype(np.float64), axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=config.UNIT_NORM_TOL))

    def test_index_rows_follow_train_row_order(self):
        questions = SPLIT.train_df["User Message"].tolist()
        fresh = retrievers.embed_texts(model(), questions)
        self.assertTrue(np.allclose(fresh, np.load(config.INDEX_PATH), atol=1e-4))

    def test_query_embeddings_are_unit_norm(self):
        emb = retrievers.embed_texts(model(), ["finals are freaking me out", "Can I change my major?"])
        self.assertTrue(np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=config.UNIT_NORM_TOL))

    def test_semantic_ranking_deterministic_full_length(self):
        r1 = self.semantic.rank("lost my uni card", 10)
        r2 = self.semantic.rank("lost my uni card", 10)
        self.assertEqual(r1, r2)
        self.assertEqual(len(r1), 10)
        restricted = self.semantic.rank("lost my uni card", 10, intent="housing")
        self.assertTrue(all(SPLIT.intent_of(rid) == "housing" for rid, _ in restricted))
        self.assertEqual(self.semantic.rank("x", 5, intent="no_such_intent"), [])

    def test_offline_required(self):
        import os
        saved = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            with self.assertRaises(RuntimeError):
                retrievers.require_offline()
        finally:
            if saved is not None:
                os.environ["HF_HUB_OFFLINE"] = saved


class TestTieBreaking(unittest.TestCase):
    def test_semantic_tolerance_ties_go_to_lowest_row_id(self):
        row_ids = [9, 3, 7, 1]
        scores = [0.5, 0.5000005, 0.4, 0.5]
        order = retrievers.order_with_tolerance(row_ids, scores, 1e-6)
        self.assertEqual([row_ids[p] for p in order], [1, 3, 9, 7])

    def test_semantic_scores_beyond_tolerance_are_not_tied(self):
        order = retrievers.order_with_tolerance([1, 2], [0.5, 0.50001], 1e-6)
        self.assertEqual([[1, 2][p] for p in order], [2, 1])

    def test_stage_a_exact_ties_go_to_lowest_row_id(self):
        df = pd.DataFrame(
            {
                "User Message": ["parking permit renewal", "dorm wifi", "parking permit renewal", "gym hours"],
                "Intent": ["campus_facilities", "technical_issue", "campus_facilities", "campus_facilities"],
                "Bot Response": ["r5", "r8", "r3", "r9"],
            },
            index=[5, 8, 3, 9],
        )
        adapter = retrievers.StageARetrieverAdapter(df)
        ranking = adapter.rank("parking permit renewal", 2)
        self.assertEqual([rid for rid, _ in ranking], [3, 5])
        self.assertEqual(ResponseRetriever(df=df).best_match("parking permit renewal", "campus_facilities").row_id, 3)


# ---------------------------------------------------------------------------
# Dice, buckets and P*
# ---------------------------------------------------------------------------
class TestDiceAndBuckets(unittest.TestCase):
    def test_exact_dice_values(self):
        f = frozenset
        self.assertEqual(b1_dice.dice(f("abcd"), f("aefg")), Fraction(1, 4))
        self.assertEqual(b1_dice.dice(f("ab"), f("ac")), Fraction(1, 2))
        self.assertEqual(b1_dice.dice(f("abcde"), f("abcxy")), Fraction(3, 5))
        self.assertEqual(b1_dice.dice(f("abcd"), f("abcx")), Fraction(3, 4))
        self.assertEqual(b1_dice.dice(f(), f()), 0)

    def test_stage_a_content_tokens(self):
        self.assertEqual(
            b1_dice.content_set("How do I connect to college faculty?"),
            frozenset({"connect", "college", "faculty"}),
        )

    def test_bucket_boundaries_are_exact(self):
        pos = "synonym_substitution"
        cases = {
            Fraction(0): "B1",
            Fraction(1, 4) - Fraction(1, 10**9): "B1",
            Fraction(1, 4): "B2",
            Fraction(1, 2): "B3a",
            Fraction(3, 5) - Fraction(1, 10**9): "B3a",
            Fraction(3, 5): "B3b",
            Fraction(3, 4): "B4",
            Fraction(1): "B4",
        }
        for value, expected in cases.items():
            self.assertEqual(b1_dice.reporting_slice(pos, value), expected, str(value))

    def test_primary_population_boundary(self):
        pos = "slang_informal"
        self.assertTrue(b1_dice.in_primary_population(pos, Fraction(3, 5) - Fraction(1, 10**9)))
        self.assertFalse(b1_dice.in_primary_population(pos, Fraction(3, 5)))
        self.assertFalse(b1_dice.in_primary_population(pos, None))

    def test_category4_is_neg_and_never_primary(self):
        neg = config.NEGATIVE_CATEGORY
        for value in (None, Fraction(0), Fraction(1, 10)):
            self.assertEqual(b1_dice.dice_bucket(neg, value), "NEG")
            self.assertFalse(b1_dice.in_primary_population(neg, value))

    def test_positive_without_acceptable_cannot_be_bucketed(self):
        with self.assertRaises(ValueError):
            b1_dice.dice_bucket("synonym_substitution", None)


# ---------------------------------------------------------------------------
# Benchmark validators (synthetic fixtures)
# ---------------------------------------------------------------------------
def _synthetic_benchmark() -> pd.DataFrame:
    """A valid, verified 72-row benchmark built from real training ids."""
    tech_ids = [rid for rid in SPLIT.train_ids if SPLIT.intent_of(rid) == "technical_issue"]
    rows = []
    i = 0
    for category, count in config.CATEGORY_COUNTS.items():
        for _ in range(count):
            i += 1
            positive = category in config.POSITIVE_CATEGORIES
            rows.append({
                "challenge_id": f"CH_{i:03d}",
                "query_text": f"zzqx synthetic probe number {i} kwrp",
                "target_intent": "technical_issue",
                "acceptable_row_ids": str(tech_ids[0]) if positive else "",
                "preferred_row_id": str(tech_ids[0]) if positive and i % 2 else "",
                "difficulty_category": category,
                "dice_to_preferred": "",
                "max_dice_to_acceptable": "",
                "max_dice_to_any_kb_row": "",
                "dice_bucket": "",
                "review_status": "verified",
                "notes": "synthetic",
            })
    return bm.annotate(pd.DataFrame(rows, columns=list(config.BENCHMARK_COLUMNS)), SPLIT)


def _all_errors(df):
    errors = bm.schema_errors(df, SPLIT)
    if errors:
        return errors
    return (bm.annotation_errors(df, SPLIT) + bm.contamination_errors(df, SPLIT)
            + bm.composition_errors(df, SPLIT))


class TestBenchmarkValidators(unittest.TestCase):
    def test_valid_synthetic_benchmark_passes(self):
        df = _synthetic_benchmark()
        self.assertEqual(_all_errors(df), [])
        counts = bm.population_counts(df, SPLIT)
        self.assertEqual((counts["Q"], counts["P"], counts["N"], counts["P_star"]), (72, 58, 14, 58))
        self.assertEqual(df.loc[df.difficulty_category == config.NEGATIVE_CATEGORY, "dice_bucket"].unique().tolist(), ["NEG"])

    def test_dice_to_preferred_only_when_preferred_exists(self):
        df = _synthetic_benchmark()
        no_pref = df["preferred_row_id"] == ""
        self.assertTrue((df.loc[no_pref, "dice_to_preferred"] == "").all())
        self.assertTrue((df.loc[~no_pref, "dice_to_preferred"] != "").all())

    def test_wrong_size_fails(self):
        self.assertTrue(any("exactly 72" in e for e in _all_errors(_synthetic_benchmark().iloc[:-1])))

    def test_category4_with_preferred_fails(self):
        df = _synthetic_benchmark()
        i = df.index[df.difficulty_category == config.NEGATIVE_CATEGORY][0]
        df.at[i, "acceptable_row_ids"] = df.at[0, "acceptable_row_ids"]
        df.at[i, "preferred_row_id"] = df.at[0, "acceptable_row_ids"]
        self.assertTrue(any("Category 4" in e for e in bm.schema_errors(df, SPLIT)))

    def test_positive_without_acceptable_fails(self):
        df = _synthetic_benchmark()
        df.at[0, "acceptable_row_ids"] = ""
        df.at[0, "preferred_row_id"] = ""
        self.assertTrue(any("at least one acceptable" in e for e in bm.schema_errors(df, SPLIT)))

    def test_test_row_id_is_rejected(self):
        df = _synthetic_benchmark()
        df.at[1, "acceptable_row_ids"] = str(SPLIT.test_ids[0])
        df.at[1, "preferred_row_id"] = ""
        self.assertTrue(any("not a training row id" in e for e in bm.schema_errors(df, SPLIT)))
        self.assertTrue(any("TEST row id" in e for e in bm.contamination_errors(df, SPLIT)))

    def test_preferred_must_be_acceptable(self):
        df = _synthetic_benchmark()
        other = next(r for r in SPLIT.train_ids if str(r) != df.at[0, "acceptable_row_ids"])
        df.at[0, "preferred_row_id"] = str(other)
        self.assertTrue(any("preferred_row_id must be one of" in e for e in bm.schema_errors(df, SPLIT)))

    def test_duplicates_of_train_or_test_questions_fail(self):
        df = _synthetic_benchmark()
        df.at[2, "query_text"] = "  " + SPLIT.train_df["User Message"].iloc[0].upper() + "  "
        df.at[3, "query_text"] = SPLIT.test_df["User Message"].iloc[0]
        errors = bm.contamination_errors(df, SPLIT)
        self.assertTrue(any("training (KB) question" in e for e in errors))
        self.assertTrue(any("test question" in e for e in errors))

    def test_primary_shortfall_fails(self):
        df = _synthetic_benchmark()
        positives = df.index[df.difficulty_category.isin(config.POSITIVE_CATEGORIES)][:19]
        for i in positives:
            rid = int(df.at[i, "acceptable_row_ids"])
            df.at[i, "query_text"] = f"{SPLIT.train_df.at[rid, 'User Message']} please {i}"
        df = bm.annotate(df, SPLIT)
        self.assertTrue(any("|P*|" in e for e in bm.composition_errors(df, SPLIT)))

    def test_tampered_dice_field_fails(self):
        df = _synthetic_benchmark()
        df.at[0, "max_dice_to_acceptable"] = "0.999999"
        self.assertTrue(any("does not match" in e for e in bm.annotation_errors(df, SPLIT)))

    def test_wrong_columns_fail(self):
        df = _synthetic_benchmark().drop(columns=["notes"])
        self.assertTrue(bm.schema_errors(df, SPLIT))

    def test_current_draft_file_is_schema_and_contamination_clean(self):
        if not config.BENCHMARK_PATH.exists():
            self.skipTest("benchmark file not created yet")
        df = bm.read_benchmark()
        self.assertEqual(bm.schema_errors(df, SPLIT) + bm.contamination_errors(df, SPLIT), [])

    def test_review_pack_contains_only_training_rows(self):
        if not config.BENCHMARK_PATH.exists():
            self.skipTest("benchmark file not created yet")
        spec = importlib.util.spec_from_file_location(
            "build_benchmark", config.BENCHMARK_DIR / "build_benchmark.py")
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        with tempfile.TemporaryDirectory() as tmp:
            original = config.REVIEW_PACK_PATH
            config.REVIEW_PACK_PATH = Path(tmp) / "pack.md"
            try:
                self.assertEqual(tool.cmd_review_pack(SPLIT), 0)
                text = config.REVIEW_PACK_PATH.read_text()
            finally:
                config.REVIEW_PACK_PATH = original
        ids = {int(x) for x in re.findall(r"\*\*(\d+)\*\*", text)}
        self.assertEqual(ids, set(SPLIT.train_ids))
        for q in SPLIT.test_df["User Message"]:
            self.assertNotIn(q, text)


# ---------------------------------------------------------------------------
# Metrics (hand-built cases)
# ---------------------------------------------------------------------------
INTENT_OF = {1: "a", 2: "a", 3: "b", 4: "c", **{i: "z" for i in range(10, 30)}}


def L(target="a", acceptable=(), preferred=None):
    return m.QueryLabels("CH_X", target, frozenset(acceptable), preferred)


class TestMetrics(unittest.TestCase):
    def test_no_candidate(self):
        labels = L(acceptable={1})
        self.assertEqual(m.outcome_class([], labels, INTENT_OF), m.NO_CANDIDATE)
        v = m.per_query_values([], labels, INTENT_OF)
        self.assertEqual(v["AcceptableRecall@1"], 0.0)
        self.assertEqual(v["IntentRecall@1"], 0.0)
        self.assertEqual(v["WrongIntent@1"], 0.0)  # no candidate is not a wrong-intent error

    def test_acceptable_vs_preferred_hit(self):
        labels = L(acceptable={1, 2}, preferred=2)
        ranking = [(1, 0.9), (2, 0.8)]
        self.assertEqual(m.outcome_class(ranking, labels, INTENT_OF), m.ACCEPTABLE_HIT)
        v = m.per_query_values(ranking, labels, INTENT_OF)
        self.assertEqual(v["AcceptableRecall@1"], 1.0)
        self.assertEqual(v["PreferredRecall@1"], 0.0)
        self.assertEqual(v["PreferredRecall@3"], 1.0)
        self.assertEqual(v["PreferredMRR@10"], 0.5)
        self.assertEqual(m.outcome_class([(2, 0.9)], labels, INTENT_OF), m.PREFERRED_HIT)

    def test_cross_intent_acceptable_is_not_wrong_intent(self):
        labels = L(target="a", acceptable={3})
        v = m.per_query_values([(3, 0.7)], labels, INTENT_OF)
        self.assertEqual(m.outcome_class([(3, 0.7)], labels, INTENT_OF), m.ACCEPTABLE_HIT)
        self.assertEqual(v["WrongIntent@1"], 0.0)
        self.assertEqual(v["IntentRecall@1"], 0.0)
        self.assertEqual(v["AcceptableRecall@1"], 1.0)

    def test_intent_only_and_wrong_intent(self):
        labels = L(target="a", acceptable={1})
        self.assertEqual(m.outcome_class([(2, 0.5)], labels, INTENT_OF), m.INTENT_ONLY)
        self.assertEqual(m.outcome_class([(4, 0.5)], labels, INTENT_OF), m.WRONG_INTENT)
        self.assertEqual(m.per_query_values([(4, 0.5), (1, 0.4)], labels, INTENT_OF)["WrongIntent@1"], 1.0)
        self.assertEqual(m.per_query_values([(4, 0.5), (1, 0.4)], labels, INTENT_OF)["IntentRecall@3"], 1.0)

    def test_missing_preferred_excluded_from_preferred_denominator(self):
        v = m.per_query_values([(1, 0.9)], L(acceptable={1}), INTENT_OF)
        self.assertTrue(math.isnan(v["PreferredRecall@1"]))
        self.assertTrue(math.isnan(v["PreferredMRR@10"]))
        agg = m.aggregate([v, m.per_query_values([(2, 0.9)], L(acceptable={2}, preferred=2), INTENT_OF)])
        self.assertEqual(agg["PreferredRecall@1"]["n"], 1)
        self.assertEqual(agg["AcceptableRecall@1"]["n"], 2)

    def test_negative_probe_without_acceptable(self):
        v = m.per_query_values([(4, 0.3)], L(target="c", acceptable=()), INTENT_OF)
        self.assertTrue(math.isnan(v["AcceptableRecall@1"]))
        self.assertEqual(v["IntentRecall@1"], 1.0)
        self.assertEqual(v["WrongIntent@1"], 0.0)

    def test_mrr_cut_off_at_10(self):
        filler = [(i, 1.0 - i / 100) for i in range(10, 19)]  # 9 non-acceptable rows
        at10 = filler + [(1, 0.1)]
        at11 = filler + [(10 + 9, 0.2), (1, 0.1)]
        self.assertAlmostEqual(m.per_query_values(at10, L(acceptable={1}), INTENT_OF)["AcceptableMRR@10"], 0.1)
        self.assertEqual(m.per_query_values(at11, L(acceptable={1}), INTENT_OF)["AcceptableMRR@10"], 0.0)

    def test_cosine_margin_needs_two_candidates(self):
        self.assertAlmostEqual(m.per_query_values([(1, 0.9), (2, 0.6)], L(acceptable={1}), INTENT_OF)["CosineMargin"], 0.3)
        self.assertTrue(math.isnan(m.per_query_values([(1, 0.9)], L(acceptable={1}), INTENT_OF)["CosineMargin"]))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
class TestStats(unittest.TestCase):
    def test_bootstrap_reproducible_with_seed_42(self):
        rng = np.random.default_rng(7)
        a, b = rng.integers(0, 2, 50).astype(float), rng.integers(0, 2, 50).astype(float)
        r1, r2 = b1_stats.paired_bootstrap(a, b), b1_stats.paired_bootstrap(a, b)
        self.assertEqual(r1, r2)
        self.assertEqual((r1["resamples"], r1["seed"], r1["n"]), (1000, 42, 50))
        self.assertLessEqual(r1["ci_low"], r1["ci_high"])

    def test_bootstrap_identical_systems(self):
        a = np.array([1.0, 0.0, 1.0, 1.0])
        r = b1_stats.paired_bootstrap(a, a)
        self.assertEqual((r["delta"], r["ci_low"], r["ci_high"]), (0.0, 0.0, 0.0))

    def test_bootstrap_edge_cases(self):
        self.assertIsNone(b1_stats.paired_bootstrap([], []))
        with self.assertRaises(ValueError):
            b1_stats.paired_bootstrap([1.0], [1.0, 0.0])
        with self.assertRaises(ValueError):
            b1_stats.paired_bootstrap([float("nan")], [1.0])

    def test_wilcoxon_all_zero_is_na(self):
        r = b1_stats.paired_wilcoxon([0.5, 1.0], [0.5, 1.0])
        self.assertEqual((r["n_total"], r["n_zero"], r["n_nonzero"]), (2, 2, 0))
        self.assertEqual((r["statistic"], r["p_value"]), ("n/a", "n/a"))

    def test_wilcoxon_zero_accounting_matches_scipy(self):
        base = np.array([1.0, 0.5, 0.0, 0.0, 1.0, 0.25, 0.0, 0.5])
        cand = np.array([1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0])
        r = b1_stats.paired_wilcoxon(base, cand)
        self.assertEqual((r["n_total"], r["n_zero"], r["n_nonzero"]), (8, 3, 5))
        d = cand - base
        ref = scipy_stats.wilcoxon(d[d != 0], alternative="two-sided")
        self.assertAlmostEqual(r["p_value"], float(ref.pvalue))


# ---------------------------------------------------------------------------
# Resources (fresh-process probe)
# ---------------------------------------------------------------------------
class TestResourceProbe(unittest.TestCase):
    def test_probe_takes_rss0_before_torch(self):
        queries = [
            {"query": "lost my uni card", "target_intent": "administration"},
            {"query": "finals are stressing me", "target_intent": "mental_health"},
            {"query": "zzz qqq", "target_intent": "technical_issue"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = resources.run_probe(queries, tmp)
        cp = out["checkpoints_mb"]
        for key in ("rss0_baseline_process", "rss_stage_a_data_loaded", "rss_import",
                    "rss_model", "rss_index", "rss_after_retrieval"):
            self.assertIn(key, cp)
        self.assertGreater(cp["rss_import"], cp["rss0_baseline_process"])
        self.assertAlmostEqual(out["incremental_peak_rss_mb"],
                               out["peak_rss_mb"] - cp["rss0_baseline_process"], places=1)
        self.assertEqual(out["environment"]["hf_hub_offline"], "1")
        self.assertEqual(out["n_queries"], 3)

    def test_latency_stats(self):
        s = resources.latency_stats([1.0, 2.0, 3.0, 4.0, 100.0])
        self.assertEqual(s["n"], 5)
        self.assertEqual(s["p50_ms"], 3.0)
        self.assertEqual(s["max_ms"], 100.0)
        self.assertLessEqual(s["p50_ms"], s["p95_ms"])


# ---------------------------------------------------------------------------
# Stage A integrity
# ---------------------------------------------------------------------------
class TestStageAIntegrity(unittest.TestCase):
    def test_stage_a_artifacts_match_manifest(self):
        manifest = json.loads((REPO_ROOT / "artifacts" / "manifest.json").read_text())
        self.assertEqual(provenance.sha256_file(REPO_ROOT / "artifacts" / "intent_model.pkl"),
                         manifest["model_sha256"])
        self.assertEqual(provenance.sha256_file(REPO_ROOT / "artifacts" / "tfidf_vectorizer.pkl"),
                         manifest["vectorizer_sha256"])

    def test_config_composition_is_consistent(self):
        self.assertEqual(sum(config.CATEGORY_COUNTS.values()), config.BENCHMARK_SIZE)
        self.assertEqual(sum(config.CATEGORY_COUNTS[c] for c in config.POSITIVE_CATEGORIES), 58)
        self.assertEqual(config.PRIMARY_DICE_LT, Fraction(3, 5))
        self.assertEqual(config.TORCH_THREADS, min(4, config._PHYSICAL_CORES))


if __name__ == "__main__":
    unittest.main()
