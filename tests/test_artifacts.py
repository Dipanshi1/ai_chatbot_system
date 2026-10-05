"""
Unit tests for artifact manifest and model reproducibility (Phase 5A).
"""
import json
import sys
import unittest
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.artifacts import (
    build_manifest,
    compute_sha256,
    verify_artifacts,
)
from src.ml.config import (
    MANIFEST_PATH,
    MODEL_PATH,
    MODEL_RANDOM_STATE,
    VECTORIZER_PATH,
)
from src.ml.data import load_dataset, split_dataset
from src.ml.pipeline import ChatPipeline
from src.ml.preprocessing import clean_text


class TestArtifactManifest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MANIFEST_PATH.exists(), f"Manifest missing at {MANIFEST_PATH}")
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

    def test_manifest_exists(self):
        """Validate that artifacts/manifest.json exists."""
        self.assertTrue(MANIFEST_PATH.exists())
        self.assertGreater(MANIFEST_PATH.stat().st_size, 0)

    def test_manifest_hashes_match_actual_files(self):
        """Validate that SHA-256 hashes recorded in the manifest match disk files."""
        actual_model_hash = compute_sha256(MODEL_PATH)
        actual_vec_hash = compute_sha256(VECTORIZER_PATH)

        self.assertEqual(self.manifest["model_sha256"], actual_model_hash)
        self.assertEqual(self.manifest["vectorizer_sha256"], actual_vec_hash)

        # Expected canonical SHA-256 hashes pinned for Phase 5A
        expected_model_hash = (
            "a473f4580f168c0ef0c7ea0c4812c4b1ca94aebce9f107d25d442c5fb30fa211"
        )
        expected_vec_hash = (
            "5f62a51adde831fbc634872c95672fd403d50b9c715266376315731d4de250f7"
        )
        self.assertEqual(actual_model_hash, expected_model_hash)
        self.assertEqual(actual_vec_hash, expected_vec_hash)

    def test_manifest_records_26_classes(self):
        """Validate that manifest records exactly 26 classes and matches loaded model."""
        num_classes = self.manifest.get("num_classes", self.manifest.get("number_of_classes"))
        self.assertEqual(num_classes, 26)

        model = joblib.load(MODEL_PATH)
        self.assertEqual(len(model.classes_), 26)
        if "classes" in self.manifest:
            self.assertEqual(len(self.manifest["classes"]), 26)
            self.assertEqual(list(model.classes_), self.manifest["classes"])

    def test_manifest_records_360_tfidf_features(self):
        """Validate that manifest records exactly 360 features and matches loaded vectorizer."""
        num_features = self.manifest.get(
            "num_features", self.manifest.get("number_of_tfidf_features")
        )
        self.assertEqual(num_features, 360)

        vectorizer = joblib.load(VECTORIZER_PATH)
        self.assertEqual(len(vectorizer.vocabulary_), 360)

    def test_model_version_matches_manifest(self):
        """Validate that pipeline.model_version matches the manifest's model_version."""
        pipeline = ChatPipeline()
        self.assertEqual(pipeline.model_version, self.manifest["model_version"])

    def test_artifact_verification_succeeds(self):
        """Validate that verify_artifacts() succeeds for canonical artifacts."""
        self.assertTrue(verify_artifacts())

    def test_artifact_verification_fails_on_tampered_hash(self):
        """Validate that verify_artifacts() detects tampered/invalid hashes."""
        tampered_manifest = dict(self.manifest)
        tampered_manifest["model_sha256"] = "0" * 64
        tmp_mf = MANIFEST_PATH.parent / "_test_tampered_manifest.json"
        try:
            with open(tmp_mf, "w", encoding="utf-8") as f:
                json.dump(tampered_manifest, f)
            with self.assertRaises(ValueError) as ctx:
                verify_artifacts(manifest_path=tmp_mf)
            self.assertIn("mismatch", str(ctx.exception).lower())
        finally:
            if tmp_mf.exists():
                tmp_mf.unlink()


class TestModelReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved_model = joblib.load(MODEL_PATH)
        cls.saved_vectorizer = joblib.load(VECTORIZER_PATH)

        df = load_dataset()
        cls.train_df, cls.test_df = split_dataset(df)

        cls.X_train_clean = cls.train_df["User Message"].apply(clean_text)
        cls.y_train = cls.train_df["Intent"]

        cls.X_test_clean = cls.test_df["User Message"].apply(clean_text)
        cls.y_test = cls.test_df["Intent"]

        # In-memory refit with canonical settings:
        # TfidfVectorizer with default settings, LinearSVC with MODEL_RANDOM_STATE
        cls.fresh_vectorizer = TfidfVectorizer()
        X_train_tfidf = cls.fresh_vectorizer.fit_transform(cls.X_train_clean)
        cls.X_test_tfidf = cls.fresh_vectorizer.transform(cls.X_test_clean)

        cls.fresh_model = LinearSVC(random_state=MODEL_RANDOM_STATE)
        cls.fresh_model.fit(X_train_tfidf, cls.y_train)

    def test_in_memory_refit_produces_identical_test_predictions(self):
        """Validate that in-memory refit with canonical settings produces identical predictions."""
        # Predictions using the saved model and vectorizer
        saved_test_tfidf = self.saved_vectorizer.transform(self.X_test_clean)
        saved_predictions = self.saved_model.predict(saved_test_tfidf)

        # Predictions using the in-memory refitted model and vectorizer
        fresh_predictions = self.fresh_model.predict(self.X_test_tfidf)

        np.testing.assert_array_equal(
            saved_predictions,
            fresh_predictions,
            err_msg="Predictions from in-memory refit do not match saved model predictions",
        )

    def test_model_coefficients_remain_within_1e_4_of_saved_model(self):
        """Validate that model coefficients remain within 1e-4 of the saved model."""
        max_coef_diff = np.max(np.abs(self.saved_model.coef_ - self.fresh_model.coef_))
        max_intercept_diff = np.max(
            np.abs(self.saved_model.intercept_ - self.fresh_model.intercept_)
        )

        self.assertLess(
            max_coef_diff,
            1e-4,
            f"Maximum coefficient difference {max_coef_diff:.6e} exceeds 1e-4 tolerance",
        )
        self.assertLess(
            max_intercept_diff,
            1e-4,
            f"Maximum intercept difference {max_intercept_diff:.6e} exceeds 1e-4 tolerance",
        )


if __name__ == "__main__":
    unittest.main()
