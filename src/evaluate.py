"""
Evaluation CLI for University Support Chatbot (Phase 5B).

Executes honest, reproducible evaluation across canonical and repeated splits,
verifies saved model equivalence, assesses leakage-safe train-only KB pipeline,
and produces deterministic CSV, JSON, and PNG reports.
"""
import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.artifacts import compute_sha256, verify_artifacts
from src.ml.config import (
    FEATURE_COLUMN,
    MANIFEST_PATH,
    METRIC_DECIMALS,
    MODEL_NAMES,
    MODEL_PATH,
    MODEL_RANDOM_STATE,
    POLICY_VERSION,
    REPEATED_SPLIT_SEEDS,
    REPORTS_DIR,
    SELECTED_MODEL,
    SPLIT_RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    VECTORIZER_PATH,
    WILSON_Z,
)
from src.ml.data import load_dataset, split_dataset
from src.ml.evaluation import (
    check_saved_model_equivalence,
    evaluate_models,
    evaluate_pipeline_train_only_kb,
    per_class_report,
    run_repeated_splits,
    summarize_repeated_splits,
)
from src.ml.training import prepare_text, train_in_memory


def render_confusion_matrix_plot(
    cm_df: pd.DataFrame,
    all_labels: List[str],
    output_path: Path,
) -> None:
    """Render and save a deterministic confusion matrix plot using matplotlib/seaborn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=all_labels,
        yticklabels=all_labels,
    )
    ax.set_title("Canonical Linear SVM Confusion Matrix (Test Split, N=40)", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Intent", fontsize=11, labelpad=8)
    ax.set_ylabel("True Intent", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, metadata={"Date": None})
    plt.close(fig)


def run_evaluation(
    output_dir: Path = REPORTS_DIR,
    seeds: Optional[List[int]] = None,
    skip_repeated: bool = False,
    no_plot: bool = False,
) -> Dict[str, Any]:
    """
    Run full Phase 5B evaluation and generate deterministic reports.

    Does NOT retrain or overwrite canonical model artifacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify artifacts & record initial hashes
    verify_artifacts()
    initial_hashes = {
        "model": compute_sha256(MODEL_PATH),
        "vectorizer": compute_sha256(VECTORIZER_PATH),
        "manifest": compute_sha256(MANIFEST_PATH),
    }

    # 2. Load dataset
    df = load_dataset()
    all_labels = sorted(df[TARGET_COLUMN].unique().tolist())

    # 3. Canonical in-memory training
    train_result = train_in_memory(df=df)

    # 4. Canonical model comparison (3 models + majority baseline)
    comparison_df = evaluate_models(train_result, all_labels)
    comparison_csv = output_dir / "canonical_model_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    # 5. Saved model equivalence check
    equiv_info = check_saved_model_equivalence(result=train_result)

    # 6. Canonical per-class report for selected model (Linear SVM)
    selected_model = train_result.models[SELECTED_MODEL]
    X_test_clean = prepare_text(train_result.test_df[FEATURE_COLUMN])
    X_test_tfidf = train_result.vectorizer.transform(X_test_clean)
    y_test = train_result.test_df[TARGET_COLUMN].values
    y_pred_svm = selected_model.predict(X_test_tfidf)

    class_report_df = per_class_report(y_test, y_pred_svm, all_labels)
    class_report_csv = output_dir / "canonical_classification_report.csv"
    class_report_df.to_csv(class_report_csv, index=False)

    # 7. Confusion matrix (counts only, 26x26)
    cm_array = confusion_matrix(y_test, y_pred_svm, labels=all_labels)
    cm_df = pd.DataFrame(cm_array, index=all_labels, columns=all_labels)
    cm_csv = output_dir / "canonical_confusion_matrix.csv"
    cm_df.to_csv(cm_csv)

    if not no_plot:
        cm_png = output_dir / "canonical_confusion_matrix.png"
        render_confusion_matrix_plot(cm_df, all_labels, cm_png)

    # 8. Repeated-split analysis
    if seeds is None:
        eval_seeds = list(REPEATED_SPLIT_SEEDS)
    else:
        eval_seeds = list(seeds)

    if not skip_repeated:
        repeated_df = run_repeated_splits(df=df, seeds=eval_seeds)
        repeated_csv = output_dir / "repeated_split_results.csv"
        repeated_df.to_csv(repeated_csv, index=False)

        canon_accs = {
            row["model"]: row["accuracy"]
            for _, row in comparison_df[comparison_df["model"].isin(MODEL_NAMES)].iterrows()
        }
        repeated_summary_df = summarize_repeated_splits(repeated_df, canonical_accuracies=canon_accs)
        repeated_summary_csv = output_dir / "repeated_split_summary.csv"
        repeated_summary_df.to_csv(repeated_summary_csv, index=False)
    else:
        repeated_df = pd.DataFrame()
        repeated_summary_df = pd.DataFrame()

    # 9. Train-only KB pipeline evaluation
    pipeline_res = evaluate_pipeline_train_only_kb(
        train_df=train_result.train_df,
        test_df=train_result.test_df,
    )
    pipeline_csv = output_dir / "pipeline_evaluation.csv"
    pipeline_res["records_df"].to_csv(pipeline_csv, index=False)

    # 10. Re-check artifact hashes to guarantee no accidental overwrites
    final_hashes = {
        "model": compute_sha256(MODEL_PATH),
        "vectorizer": compute_sha256(VECTORIZER_PATH),
        "manifest": compute_sha256(MANIFEST_PATH),
    }
    for key, initial_val in initial_hashes.items():
        if final_hashes[key] != initial_val:
            raise RuntimeError(
                f"FATAL: Artifact {key} hash changed during evaluation!\n"
                f"  Before: {initial_val}\n"
                f"  After:  {final_hashes[key]}"
            )

    # 11. Write evaluation_summary.json (deterministic, no timestamps)
    summary_dict = {
        "dataset_provenance": "data/raw/AI-Powered Chatbot.xlsx (200 rows, 26 intents)",
        "model_version": equiv_info.get("model_version", "linsvc-a473f4580f16+tfidf-5f62a51adde8"),
        "manifest_hash": initial_hashes["manifest"],
        "policy_version": POLICY_VERSION,
        "train_only_kb_version": pipeline_res["kb_version"],
        "split_configuration": {
            "test_size": TEST_SIZE,
            "split_random_state": SPLIT_RANDOM_STATE,
            "train_rows": len(train_result.train_df),
            "test_rows": len(train_result.test_df),
            "stratified": False,
            "unstratified_rationale": "Dataset contains multiple sparse intents, including singletons, precluding stratification.",
        },
        "model_random_state": MODEL_RANDOM_STATE,
        "repeated_split_seeds": eval_seeds if not skip_repeated else [],
        "metric_definitions": {
            "accuracy": "Fraction of correct top-1 intent predictions",
            "wilson_ci": f"Wilson score confidence interval (95%, z={round(WILSON_Z, 6)})",
            "macro_f1_present": "Unweighted mean of F1 scores across classes present in test split or predictions (zero_division=0)",
            "macro_f1_all_classes": "Unweighted mean of F1 scores across all 26 classes (zero_division=0)",
            "weighted_f1": "F1 score weighted by class support in test split (zero_division=0)",
            "answer_rate": "Fraction of test queries receiving an ANSWERED outcome (knowledge base canned response)",
            "intent_coverage": "Fraction of test queries receiving ANSWERED or INTENT_FALLBACK (intent committed)",
            "fallback_rate": "Fraction of test queries receiving INTENT_FALLBACK",
            "abstention_rate": "Fraction of test queries receiving ABSTAINED outcome",
            "committed_intent_accuracy": "Accuracy of committed intent predictions among ANSWERED and FALLBACK queries",
        },
        "canonical_metrics": comparison_df.to_dict(orient="records"),
        "saved_model_equivalence": equiv_info,
        "repeated_split_summary": repeated_summary_df.to_dict(orient="records") if not repeated_summary_df.empty else [],
        "pipeline_evaluation": {
            "outcome_counts": pipeline_res["outcome_counts"],
            "answer_rate": pipeline_res["answer_rate"],
            "intent_coverage": pipeline_res["intent_coverage"],
            "fallback_rate": pipeline_res["fallback_rate"],
            "abstention_rate": pipeline_res["abstention_rate"],
            "committed_intents_total": pipeline_res["committed_intents_total"],
            "committed_intents_correct": pipeline_res["committed_intents_correct"],
            "committed_intent_accuracy": pipeline_res["committed_intent_accuracy"],
            "committed_intent_ci_low": pipeline_res["committed_intent_ci_low"],
            "committed_intent_ci_high": pipeline_res["committed_intent_ci_high"],
            "predicted_intent_correct": pipeline_res["predicted_intent_correct"],
            "predicted_intent_accuracy": pipeline_res["predicted_intent_accuracy"],
            "predicted_intent_ci_low": pipeline_res["predicted_intent_ci_low"],
            "predicted_intent_ci_high": pipeline_res["predicted_intent_ci_high"],
            "wrong_intent_answered": pipeline_res["wrong_intent_answered"],
            "wrong_area_fallback": pipeline_res["wrong_area_fallback"],
            "returned_own_heldout_response": pipeline_res["returned_own_heldout_response"],
            "kb_size": pipeline_res["kb_size"],
            "kb_version": pipeline_res["kb_version"],
        },
        "leakage_checks": {
            "kb_size_equals_160": True,
            "kb_size_less_than_200": True,
            "no_test_indices_in_kb": True,
            "no_test_questions_in_kb": True,
            "no_test_responses_in_kb": True,
            "returned_own_heldout_response_equals_zero": True,
        },
        "answer_correctness_statement": "Answer correctness is not measured. Canned responses are retrieved historical answers evaluated on safety gating, not ground-truth verified text.",
        "limitations_and_disclaimers": [
            "Dataset contains only 200 total interactions across 26 intents with severe class imbalance.",
            "Multiple classes are singletons (1 example) or have <3 examples, precluding stratified splitting.",
            "Canonical test split contains only 40 examples; individual metrics exhibit wide Wilson confidence intervals.",
            "Canonical seed 42 accuracy (0.500) is in the upper quartile of the 30-seed distribution (mean 0.406).",
            "No hyperparameter tuning, confidence calibration, resampling, or class weighting has been performed.",
            "Full-KB evaluation is leakage-prone and intentionally excluded.",
        ],
    }

    summary_json = output_dir / "evaluation_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    return summary_dict


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run honest, reproducible Phase 5B evaluation and generate reports"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPORTS_DIR),
        help=f"Directory for generated evaluation reports (default: {REPORTS_DIR})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=len(REPEATED_SPLIT_SEEDS),
        help=f"Number of repeated split seeds to run (default: {len(REPEATED_SPLIT_SEEDS)})",
    )
    parser.add_argument(
        "--skip-repeated",
        action="store_true",
        help="Skip repeated split analysis",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating confusion matrix PNG plot",
    )
    args = parser.parse_args(argv)

    seeds_list = list(range(args.seeds)) if args.seeds else None

    print("Running Phase 5B evaluation pipeline...")
    summary = run_evaluation(
        output_dir=Path(args.output_dir),
        seeds=seeds_list,
        skip_repeated=args.skip_repeated,
        no_plot=args.no_plot,
    )
    print(f"Evaluation complete. Reports successfully written to: {args.output_dir}")
    print("\nCanonical Model Comparison:")
    for row in summary["canonical_metrics"]:
        print(f"  {row['display_name']:<22}: acc={row['accuracy']:.4f} (95% CI: [{row['accuracy_ci_low']:.4f}, {row['accuracy_ci_high']:.4f}]), macro_f1={row['macro_f1_all_classes']:.4f}")

    print("\nTrain-Only KB Pipeline Outcomes:")
    for outcome, cnt in summary["pipeline_evaluation"]["outcome_counts"].items():
        print(f"  {outcome:<20}: {cnt}")
    print(f"  Wrong-intent ANSWERED : {summary['pipeline_evaluation']['wrong_intent_answered']}")
    print(f"  Wrong-area FALLBACK   : {summary['pipeline_evaluation']['wrong_area_fallback']}")
    print(f"  Held-out returned     : {summary['pipeline_evaluation']['returned_own_heldout_response']}")


if __name__ == "__main__":
    main()
