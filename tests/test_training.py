"""
Unit tests for pure training library and save safety (Phase 5B).
"""
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.config import (
    ARTIFACTS_DIR,
    FEATURE_COLUMN,
    FORBIDDEN_FEATURE_COLUMNS,
    MODEL_NAMES,
    MODEL_RANDOM_STATE,
    SELECTED_MODEL,
    SPLIT_RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.ml.data import load_dataset
from src.ml.training import (
    TrainingResult,
    build_models,
    fit_vectorizer,
    prepare_text,
    save_artifacts,
    train_in_memory,
)


class TestTrainingLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()

    def test_prepare_text(self):
        """Validate that prepare_text cleans text using clean_text rules."""
        raw = pd.Series(["  Hello World!  ", "HOW   DO  I   REGISTER???"])
        cleaned = prepare_text(raw)
        self.assertEqual(cleaned.iloc[0], "hello world!")
        self.assertEqual(cleaned.iloc[1], "how do i register???")

    def test_build_models_returns_fresh_instances(self):
        """Validate that build_models returns all three candidate models with explicit random_state."""
        models = build_models(random_state=42)
        self.assertEqual(set(models.keys()), set(MODEL_NAMES))
        self.assertEqual(models["linear_svm"].random_state, 42)
        self.assertEqual(models["logistic_regression"].random_state, 42)
        self.assertIsInstance(models["linear_svm"], LinearSVC)

    def test_deterministic_training_in_memory(self):
        """Validate that repeated in-memory training yields identical splits and predictions."""
        res1 = train_in_memory(self.df, split_random_state=42, model_random_state=42)
        res2 = train_in_memory(self.df, split_random_state=42, model_random_state=42)

        self.assertIsInstance(res1, TrainingResult)
        pd.testing.assert_frame_equal(res1.train_df, res2.train_df)
        pd.testing.assert_frame_equal(res1.test_df, res2.test_df)

        X_test_clean = prepare_text(res1.test_df[FEATURE_COLUMN])
        vec1 = res1.vectorizer.transform(X_test_clean)
        vec2 = res2.vectorizer.transform(X_test_clean)

        pred1 = res1.models[SELECTED_MODEL].predict(vec1)
        pred2 = res2.models[SELECTED_MODEL].predict(vec2)
        self.assertTrue((pred1 == pred2).all())

    def test_forbidden_columns_not_used(self):
        """Validate that modifying forbidden columns does not alter training or predictions."""
        df_corrupted = self.df.copy()
        for col in FORBIDDEN_FEATURE_COLUMNS:
            if col in df_corrupted.columns:
                df_corrupted[col] = "CORRUPTED_VALUE"

        res_clean = train_in_memory(self.df)
        res_corrupted = train_in_memory(df_corrupted)

        X_test_clean = prepare_text(res_clean.test_df[FEATURE_COLUMN])
        pred_clean = res_clean.models[SELECTED_MODEL].predict(
            res_clean.vectorizer.transform(X_test_clean)
        )
        pred_corrupted = res_corrupted.models[SELECTED_MODEL].predict(
            res_corrupted.vectorizer.transform(X_test_clean)
        )
        self.assertTrue((pred_clean == pred_corrupted).all())

    def test_single_intent_training_raises(self):
        """Validate that training split with fewer than 2 intents raises ValueError."""
        df_single = self.df.copy()
        df_single[TARGET_COLUMN] = "only_one_intent"
        with self.assertRaises(ValueError):
            train_in_memory(df_single)


class TestArtifactSaving(unittest.TestCase):
    def setUp(self):
        self.res = train_in_memory()

    def test_save_refuses_overwrite_by_default(self):
        """Validate that save_artifacts with overwrite=False raises FileExistsError on existing directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create a dummy existing artifact
            (tmp_path / "intent_model.pkl").touch()
            with self.assertRaises(FileExistsError):
                save_artifacts(self.res, output_dir=tmp_path, overwrite=False)

    def test_save_in_clean_directory(self):
        """Validate that save_artifacts persists model, vectorizer, and manifest into a clean directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_info = save_artifacts(self.res, output_dir=tmp_path, overwrite=False)

            self.assertTrue((tmp_path / "intent_model.pkl").exists())
            self.assertTrue((tmp_path / "tfidf_vectorizer.pkl").exists())
            self.assertTrue((tmp_path / "manifest.json").exists())
            self.assertIn("model_sha256", save_info)
            self.assertIn("vectorizer_sha256", save_info)
            self.assertIn("model_version", save_info)

    def test_save_with_overwrite_enabled(self):
        """Validate that save_artifacts with overwrite=True successfully replaces existing files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "intent_model.pkl").write_text("old")
            save_info = save_artifacts(self.res, output_dir=tmp_path, overwrite=True)
            self.assertTrue((tmp_path / "intent_model.pkl").exists())
            self.assertNotEqual((tmp_path / "intent_model.pkl").read_bytes(), b"old")


if __name__ == "__main__":
    unittest.main()
