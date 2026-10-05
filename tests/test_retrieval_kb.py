"""
Unit tests for ResponseRetriever DataFrame mode and train-only KB evaluation (Phase 5B).
"""
import sys
import unittest
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.data import load_dataset, split_dataset
from src.ml.evaluation import evaluate_pipeline_train_only_kb
from src.ml.retrieval import ResponseRetriever


class TestResponseRetrieverModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.train_df, cls.test_df = split_dataset(cls.df)

    def test_both_args_raises(self):
        """Validate that passing both data_path and df raises ValueError."""
        with self.assertRaises(ValueError):
            ResponseRetriever(data_path="dummy.xlsx", df=self.train_df)

    def test_invalid_df_validations(self):
        """Validate input checks on DataFrame mode."""
        # Non-DataFrame
        with self.assertRaises(TypeError):
            ResponseRetriever(df="not a dataframe")

        # Empty DataFrame
        with self.assertRaises(ValueError):
            ResponseRetriever(df=pd.DataFrame())

        # Missing required column
        missing_col_df = self.train_df.drop(columns=["Bot Response"])
        with self.assertRaises(ValueError):
            ResponseRetriever(df=missing_col_df)

        # Non-unique index
        dup_index_df = self.train_df.copy()
        dup_index_df.index = [0] * len(dup_index_df)
        with self.assertRaises(ValueError):
            ResponseRetriever(df=dup_index_df)

        # Non-integer index
        str_index_df = self.train_df.copy()
        str_index_df.index = [f"row_{i}" for i in range(len(str_index_df))]
        with self.assertRaises(ValueError):
            ResponseRetriever(df=str_index_df)

    def test_file_mode_behavior_unchanged(self):
        """Validate file mode retains kb- prefix and correct row count."""
        retriever = ResponseRetriever()
        self.assertEqual(len(retriever.df), 200)
        self.assertTrue(retriever.kb_version.startswith("kb-"))
        self.assertFalse(retriever.kb_version.startswith("kb-df-"))

    def test_dataframe_mode_properties(self):
        """Validate DataFrame mode properties, row IDs, and version string."""
        retriever = ResponseRetriever(df=self.train_df)
        self.assertEqual(len(retriever.df), 160)
        self.assertTrue(retriever.kb_version.startswith("kb-df-"))

        # Test best_match preserves original dataset index as row_id
        cand = retriever.best_match("when will my financial aid be disbursed?", "financial_aid")
        self.assertIsNotNone(cand)
        self.assertIn(cand.row_id, self.train_df.index)
        self.assertEqual(retriever.df.loc[cand.row_id, "Intent"], "financial_aid")


class TestLeakageSafePipelineEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.train_df, cls.test_df = split_dataset(cls.df)
        cls.pipeline_res = evaluate_pipeline_train_only_kb(cls.train_df, cls.test_df)

    def test_leakage_assertion_failure_raises_runtime_error(self):
        """Validate that any test-set leakage in the KB raises RuntimeError."""
        leaky_kb_df = pd.concat([self.train_df, self.test_df.iloc[:1]])
        with self.assertRaises(RuntimeError):
            evaluate_pipeline_train_only_kb(leaky_kb_df, self.test_df)

    def test_pipeline_outcome_counts(self):
        """Validate exact train-only KB pipeline outcome counts."""
        outcomes = self.pipeline_res["outcome_counts"]
        self.assertEqual(outcomes["ANSWERED"], 1)
        self.assertEqual(outcomes["INTENT_FALLBACK"], 21)
        self.assertEqual(outcomes["ABSTAINED"], 18)

    def test_wrong_intent_and_wrong_area_counts(self):
        """Validate safety gating counts: wrong-intent ANSWERED=0, wrong-area FALLBACK=6."""
        self.assertEqual(self.pipeline_res["wrong_intent_answered"], 0)
        self.assertEqual(self.pipeline_res["wrong_area_fallback"], 6)

    def test_committed_intent_accuracy(self):
        """Validate committed intent accuracy is 16/22."""
        self.assertEqual(self.pipeline_res["committed_intents_total"], 22)
        self.assertEqual(self.pipeline_res["committed_intents_correct"], 16)
        self.assertAlmostEqual(self.pipeline_res["committed_intent_accuracy"], 16 / 22, places=5)

    def test_predicted_intent_accuracy(self):
        """Validate predicted intent accuracy is 20/40."""
        self.assertEqual(self.pipeline_res["predicted_intent_correct"], 20)
        self.assertEqual(self.pipeline_res["predicted_intent_accuracy"], 0.5)

    def test_returned_own_heldout_is_zero(self):
        """Validate that zero test messages returned their own held-out canned response."""
        self.assertEqual(self.pipeline_res["returned_own_heldout_response"], 0)


if __name__ == "__main__":
    unittest.main()
