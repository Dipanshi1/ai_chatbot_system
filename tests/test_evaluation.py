"""
Unit tests for honest evaluation library and metrics (Phase 5B).
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.artifacts import compute_sha256
from src.ml.config import (
    FEATURE_COLUMN,
    MANIFEST_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    VECTORIZER_PATH,
    WILSON_Z,
)
from src.ml.data import load_dataset
from src.ml.evaluation import (
    check_saved_model_equivalence,
    classification_metrics,
    evaluate_models,
    majority_baseline,
    per_class_report,
    run_repeated_splits,
    summarize_repeated_splits,
    wilson_interval,
)
from src.ml.training import prepare_text, train_in_memory


class TestWilsonInterval(unittest.TestCase):
    def test_n_equals_zero_returns_none(self):
        """Validate that n=0 returns None."""
        self.assertIsNone(wilson_interval(0, 0))

    def test_invalid_counts_raise(self):
        """Validate that negative counts or k > n raise ValueError."""
        with self.assertRaises(ValueError):
            wilson_interval(-1, 10)
        with self.assertRaises(ValueError):
            wilson_interval(15, 10)
        with self.assertRaises(ValueError):
            wilson_interval(5, -1)

    def test_canonical_linear_svm_interval(self):
        """Validate 20/40 produces expected Wilson CI [0.351995, 0.648005]."""
        low, high = wilson_interval(20, 40)
        self.assertAlmostEqual(low, 0.351995, places=5)
        self.assertAlmostEqual(high, 0.648005, places=5)

    def test_clamped_bounds(self):
        """Validate bounds are clamped to [0.0, 1.0]."""
        low_zero, _ = wilson_interval(0, 10)
        _, high_all = wilson_interval(10, 10)
        self.assertGreaterEqual(low_zero, 0.0)
        self.assertLessEqual(high_all, 1.0)


class TestMajorityBaseline(unittest.TestCase):
    def test_majority_baseline_tie_resolution(self):
        """Validate highest training count wins and ties resolve in alphabetical order."""
        # 'campus_facilities' and 'student_services' tie with 16 in canonical training split
        df = load_dataset()
        res = train_in_memory(df)
        maj = majority_baseline(res.train_df[TARGET_COLUMN], res.test_df[TARGET_COLUMN])

        self.assertEqual(maj["majority_label"], "campus_facilities")
        self.assertIn("campus_facilities", maj["tied_labels"])
        self.assertIn("student_services", maj["tied_labels"])
        self.assertEqual(maj["correct"], 7)
        self.assertEqual(maj["n"], 40)
        self.assertAlmostEqual(maj["accuracy"], 0.175, places=3)
        self.assertAlmostEqual(maj["tied_accuracies"]["campus_facilities"], 0.175, places=3)
        self.assertAlmostEqual(maj["tied_accuracies"]["student_services"], 0.100, places=3)


class TestClassificationMetricsAndPerClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.all_labels = sorted(cls.df[TARGET_COLUMN].unique().tolist())
        cls.res = train_in_memory(cls.df)
        X_test_clean = prepare_text(cls.res.test_df[FEATURE_COLUMN])
        X_test_tfidf = cls.res.vectorizer.transform(X_test_clean)
        cls.y_test = cls.res.test_df[TARGET_COLUMN].values
        cls.svm_pred = cls.res.models["linear_svm"].predict(X_test_tfidf)

    def test_canonical_regression_anchors(self):
        """Validate canonical model comparison metrics against regression anchors."""
        comparison_df = evaluate_models(self.res, self.all_labels)

        # 1. Linear SVM
        svm_row = comparison_df[comparison_df["model"] == "linear_svm"].iloc[0]
        self.assertEqual(svm_row["correct"], 20)
        self.assertAlmostEqual(svm_row["accuracy"], 0.500, places=3)
        self.assertAlmostEqual(svm_row["macro_f1_present"], 0.334, places=3)
        self.assertAlmostEqual(svm_row["macro_f1_all_classes"], 0.231, places=3)
        self.assertAlmostEqual(svm_row["weighted_f1"], 0.510, places=3)

        # 2. Logistic Regression
        lr_row = comparison_df[comparison_df["model"] == "logistic_regression"].iloc[0]
        self.assertEqual(lr_row["correct"], 17)
        self.assertAlmostEqual(lr_row["accuracy"], 0.425, places=3)
        self.assertAlmostEqual(lr_row["macro_f1_present"], 0.237, places=3)
        self.assertAlmostEqual(lr_row["macro_f1_all_classes"], 0.146, places=3)
        self.assertAlmostEqual(lr_row["weighted_f1"], 0.374, places=3)

        # 3. Multinomial NB
        nb_row = comparison_df[comparison_df["model"] == "multinomial_nb"].iloc[0]
        self.assertEqual(nb_row["correct"], 15)
        self.assertAlmostEqual(nb_row["accuracy"], 0.375, places=3)
        self.assertAlmostEqual(nb_row["macro_f1_present"], 0.120, places=3)
        self.assertAlmostEqual(nb_row["macro_f1_all_classes"], 0.074, places=3)
        self.assertAlmostEqual(nb_row["weighted_f1"], 0.287, places=3)

    def test_per_class_report_structure(self):
        """Validate per-class report has exactly 26 rows and appropriate notes."""
        report = per_class_report(self.y_test, self.svm_pred, self.all_labels)
        self.assertEqual(len(report), 26)
        self.assertEqual(
            list(report.columns),
            ["intent", "precision", "recall", "f1", "support", "predicted_count", "in_test", "note"],
        )
        self.assertTrue(set(report["note"]).issubset({"", "support<=1", "absent from test"}))

        # Check absent classes
        absent = report[report["support"] == 0]
        self.assertTrue((absent["note"] == "absent from test").all())
        self.assertTrue((~absent["in_test"]).all())

        # Check singleton classes in test
        singletons = report[report["support"] == 1]
        self.assertTrue((singletons["note"] == "support<=1").all())


class TestEquivalenceAndRepeatedSplits(unittest.TestCase):
    def test_saved_model_prediction_equivalence(self):
        """Validate saved model equivalence checks pass with claim 'prediction-equivalent'."""
        equiv = check_saved_model_equivalence()
        self.assertTrue(equiv["is_prediction_equivalent"])
        self.assertEqual(equiv["claim"], "prediction-equivalent")
        self.assertEqual(equiv["prediction_agreement"], "40/40")
        self.assertTrue(equiv["vocabulary_equal"])
        self.assertTrue(equiv["idf_allclose"])
        self.assertLess(equiv["max_coef_difference"], 1e-4)
        self.assertIsNone(equiv["saved_model_random_state"])

    def test_repeated_splits_short_run(self):
        """Validate repeated splits logic using 3 seeds (does not write artifacts)."""
        df = load_dataset()
        test_seeds = [0, 1, 2]
        rep_df = run_repeated_splits(df, seeds=test_seeds)

        self.assertEqual(len(rep_df), 3 * 3)  # 3 seeds * 3 models
        summary = summarize_repeated_splits(rep_df)
        self.assertEqual(len(summary), 3)
        self.assertTrue((summary["mean"] >= 0.0).all())
        self.assertTrue((summary["mean"] <= 1.0).all())

    def test_evaluation_preserves_artifact_hashes(self):
        """Validate that running evaluation never alters canonical artifact files."""
        model_before = compute_sha256(MODEL_PATH)
        vec_before = compute_sha256(VECTORIZER_PATH)
        man_before = compute_sha256(MANIFEST_PATH)

        train_res = train_in_memory()
        _ = evaluate_models(train_res, sorted(train_res.train_df[TARGET_COLUMN].unique()))

        self.assertEqual(compute_sha256(MODEL_PATH), model_before)
        self.assertEqual(compute_sha256(VECTORIZER_PATH), vec_before)
        self.assertEqual(compute_sha256(MANIFEST_PATH), man_before)


if __name__ == "__main__":
    unittest.main()
