"""
Unit tests for dataset loading, validation, and deterministic splitting (Phase 5A).
"""
import sys
import unittest
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.config import (
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    SPLIT_RANDOM_STATE,
    TEST_SIZE,
)
from src.ml.data import load_dataset, split_dataset


class TestDataLoading(unittest.TestCase):
    def setUp(self):
        self.df = load_dataset()

    def test_required_columns_exist(self):
        """Validate that all REQUIRED_COLUMNS exist in the loaded dataset."""
        for col in REQUIRED_COLUMNS:
            self.assertIn(
                col,
                self.df.columns,
                f"Required column '{col}' is missing from loaded dataset",
            )

    def test_dataset_contains_no_required_nulls(self):
        """Validate that there are no null values in required columns."""
        for col in REQUIRED_COLUMNS:
            null_count = int(self.df[col].isnull().sum())
            self.assertEqual(
                null_count,
                0,
                f"Required column '{col}' has {null_count} null values",
            )

    def test_missing_required_column_raises_clear_error(self):
        """Validate that missing any required column raises a clear descriptive ValueError."""
        incomplete_df = self.df.drop(columns=["Intent"])
        tmp_path = BASE_DIR / "data" / "raw" / "_test_temp_missing_col.xlsx"
        try:
            incomplete_df.to_excel(tmp_path, index=False)
            with self.assertRaises(ValueError) as ctx:
                load_dataset(tmp_path)
            error_msg = str(ctx.exception).lower()
            self.assertIn("missing required column", error_msg)
            self.assertIn("intent", error_msg)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_null_in_required_column_raises_clear_error(self):
        """Validate that null values in a required column raise a clear descriptive ValueError."""
        corrupted_df = self.df.copy()
        corrupted_df.loc[0, "User Message"] = None
        tmp_path = BASE_DIR / "data" / "raw" / "_test_temp_null_col.xlsx"
        try:
            corrupted_df.to_excel(tmp_path, index=False)
            with self.assertRaises(ValueError) as ctx:
                load_dataset(tmp_path)
            error_msg = str(ctx.exception).lower()
            self.assertIn("null", error_msg)
            self.assertIn("user message", error_msg)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_missing_file_raises_clear_error(self):
        """Validate that a non-existent file path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_dataset(BASE_DIR / "nonexistent_dataset_file.xlsx")


class TestDataSplitting(unittest.TestCase):
    def setUp(self):
        self.df = load_dataset()
        self.train_df, self.test_df = split_dataset(self.df)

    def test_split_is_160_40(self):
        """Validate that the split yields exactly 160 training and 40 testing samples."""
        self.assertEqual(len(self.train_df), 160)
        self.assertEqual(len(self.test_df), 40)
        self.assertEqual(len(self.train_df) + len(self.test_df), len(self.df))

    def test_train_test_indices_are_disjoint(self):
        """Validate that training and testing sets have strictly disjoint indices."""
        train_indices = set(self.train_df.index)
        test_indices = set(self.test_df.index)
        self.assertTrue(train_indices.isdisjoint(test_indices))
        self.assertEqual(train_indices | test_indices, set(self.df.index))

    def test_split_is_deterministic(self):
        """Validate that repeated splitting produces identical train and test splits."""
        train_df2, test_df2 = split_dataset(self.df)
        pd.testing.assert_frame_equal(self.train_df, train_df2)
        pd.testing.assert_frame_equal(self.test_df, test_df2)

    def test_accidental_split_changes_are_detected(self):
        """
        Validate that the split matches canonical expected indices.
        Guarantees that changes to TEST_SIZE, SPLIT_RANDOM_STATE, or accidental
        stratification are caught immediately.
        """
        expected_first_5_test_indices = [95, 15, 30, 158, 128]
        expected_first_5_train_indices = [79, 197, 38, 24, 122]
        self.assertEqual(self.test_df.index[:5].tolist(), expected_first_5_test_indices)
        self.assertEqual(self.train_df.index[:5].tolist(), expected_first_5_train_indices)

    def test_stratification_not_used(self):
        """
        Validate documentation and structural rationale: stratification is intentionally
        not used because the dataset contains singleton classes that prevent stratified
        splitting.
        """
        intent_counts = self.df["Intent"].value_counts()
        singleton_intents = intent_counts[intent_counts == 1].index.tolist()
        self.assertGreater(
            len(singleton_intents),
            0,
            "Expected at least one singleton class justifying the unstratified split",
        )


if __name__ == "__main__":
    unittest.main()
