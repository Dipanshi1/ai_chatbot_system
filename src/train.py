"""
Safe training CLI for University Support Chatbot (Phase 5B).

Trains candidate intent classification models in memory and displays comparisons.
By default, writes NOTHING to disk and does NOT modify artifacts, reports, or manifest.
"""
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.artifacts import write_manifest
from src.ml.config import (
    ARTIFACTS_DIR,
    FEATURE_COLUMN,
    SELECTED_MODEL,
    TARGET_COLUMN,
)
from src.ml.training import prepare_text, save_artifacts, train_in_memory


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Safe training CLI for University Support Chatbot"
    )
    parser.add_argument(
        "--write-manifest-only",
        action="store_true",
        help="Generate artifacts/manifest.json from existing artifacts without retraining",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist model, vectorizer, and manifest to output directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ARTIFACTS_DIR),
        help=f"Target directory for saved artifacts (default: {ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing artifacts in output directory (requires --save)",
    )
    args = parser.parse_args(argv)

    if args.overwrite and not args.save:
        parser.error("--overwrite requires --save.")

    if args.write_manifest_only:
        manifest_file = write_manifest()
        print(f"Artifact manifest successfully written to {manifest_file}")
        return

    # Train all 3 models in memory
    print("Training candidate models in memory (canonical 80/20 split, seed=42)...")
    result = train_in_memory()

    # Model comparison on test split
    X_test_clean = prepare_text(result.test_df[FEATURE_COLUMN])
    X_test_tfidf = result.vectorizer.transform(X_test_clean)
    y_test = result.test_df[TARGET_COLUMN]

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (In-Memory Evaluation on Test Split, N=40)")
    print("=" * 60)
    print(f"{'Model':<25} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 60)

    for name, model in result.models.items():
        y_pred = model.predict(X_test_tfidf)
        correct = int((y_pred == y_test).sum())
        total = len(y_test)
        acc = correct / total
        selected_marker = " (SELECTED)" if name == SELECTED_MODEL else ""
        print(f"{name + selected_marker:<25} {f'{correct}/{total}':<10} {acc:.4f}")
    print("=" * 60)

    if args.save:
        print(f"\nPersisting selected model ('{SELECTED_MODEL}') to {args.output_dir}...")
        save_info = save_artifacts(
            result,
            model_name=SELECTED_MODEL,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        print(f"Artifacts successfully saved to: {args.output_dir}")
        print(f"  Model SHA-256:      {save_info['model_sha256']}")
        print(f"  Vectorizer SHA-256: {save_info['vectorizer_sha256']}")
        print(f"  Model Version:      {save_info['model_version']}")
    else:
        print("\n[Safe Mode] In-memory run completed. No files were written to disk.")


if __name__ == "__main__":
    main()