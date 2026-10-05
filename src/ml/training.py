"""
Pure training library for the University Support Chatbot (Phase 5B).

Provides deterministic in-memory training and safe artifact persistence.
This module does NOT calculate evaluation metrics, touch Phase 4 inference
classes, or perform file I/O outside of save_artifacts().
"""
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Optional, Union

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .artifacts import build_manifest, compute_sha256
from .config import (
    ARTIFACTS_DIR,
    FEATURE_COLUMN,
    MODEL_NAMES,
    MODEL_RANDOM_STATE,
    SELECTED_MODEL,
    SPLIT_RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .data import load_dataset, split_dataset
from .preprocessing import clean_text


@dataclass(frozen=True)
class TrainingResult:
    """Immutable container holding fitted models, vectorizer, splits, and seeds."""

    vectorizer: TfidfVectorizer
    models: Dict[str, Any]
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    split_random_state: int
    model_random_state: int
    test_size: float


def prepare_text(texts: Union[pd.Series, Iterable[str]]) -> pd.Series:
    """
    Clean input texts element-wise using the frozen clean_text() preprocessor.

    Only 'User Message' may be used as the feature source.
    """
    if isinstance(texts, pd.Series):
        return texts.apply(clean_text)
    return pd.Series([clean_text(t) for t in texts])


def fit_vectorizer(train_texts: Union[pd.Series, Iterable[str]]) -> TfidfVectorizer:
    """
    Fit a scikit-learn TfidfVectorizer on training texts using default settings.

    Reproduces the canonical vectorizer artifact.
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_texts)
    return vectorizer


def build_models(random_state: int = MODEL_RANDOM_STATE) -> Dict[str, Any]:
    """
    Return fresh unfitted model instances for comparison.

    No class weighting, resampling, or hyperparameter tuning is added.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "linear_svm": LinearSVC(
            random_state=random_state,
        ),
        "multinomial_nb": MultinomialNB(),
    }


def train_in_memory(
    df: Optional[pd.DataFrame] = None,
    *,
    split_random_state: int = SPLIT_RANDOM_STATE,
    model_random_state: int = MODEL_RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> TrainingResult:
    """
    Train all three candidate models entirely in memory.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Dataset to use. If None, loaded via load_dataset().
    split_random_state : int, optional
        Seed for train_test_split (default: SPLIT_RANDOM_STATE = 42).
    model_random_state : int, optional
        Seed for model training (default: MODEL_RANDOM_STATE = 42).
    test_size : float, optional
        Fraction of data for testing (default: TEST_SIZE = 0.20).

    Returns
    -------
    TrainingResult
        Container holding fitted vectorizer, fitted models, and split DataFrames.
    """
    if df is None:
        df = load_dataset()

    train_df, test_df = split_dataset(
        df,
        test_size=test_size,
        random_state=split_random_state,
    )

    y_train = train_df[TARGET_COLUMN]
    distinct_intents = y_train.nunique()
    if distinct_intents < 2:
        raise ValueError(
            f"Training split must contain at least 2 distinct intents, found {distinct_intents}"
        )

    # Use only FEATURE_COLUMN
    X_train_raw = train_df[FEATURE_COLUMN]
    X_train_clean = prepare_text(X_train_raw)

    # Fit vectorizer only on training text
    vectorizer = fit_vectorizer(X_train_clean)
    X_train_tfidf = vectorizer.transform(X_train_clean)

    # Build and fit models
    models = build_models(random_state=model_random_state)
    for model in models.values():
        model.fit(X_train_tfidf, y_train)

    return TrainingResult(
        vectorizer=vectorizer,
        models=models,
        train_df=train_df,
        test_df=test_df,
        split_random_state=split_random_state,
        model_random_state=model_random_state,
        test_size=test_size,
    )


def save_artifacts(
    result: TrainingResult,
    *,
    model_name: str = SELECTED_MODEL,
    output_dir: Union[Path, str] = ARTIFACTS_DIR,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Safely persist model, vectorizer, and manifest artifacts.

    Guarantees:
    - Default overwrite=False.
    - If any target already exists and overwrite=False, raises FileExistsError.
    - Uses temporary files and atomic replacement to prevent partial writes.

    Parameters
    ----------
    result : TrainingResult
        In-memory training result container.
    model_name : str, optional
        Name of model to save (default: SELECTED_MODEL = 'linear_svm').
    output_dir : Path or str, optional
        Target directory for artifacts.
    overwrite : bool, optional
        Whether to overwrite existing artifact files (default: False).

    Returns
    -------
    Dict[str, Any]
        Saved paths and SHA-256 hashes.
    """
    if model_name not in result.models:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Expected one of {list(result.models.keys())}"
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_model = out_path / "intent_model.pkl"
    target_vectorizer = out_path / "tfidf_vectorizer.pkl"
    target_manifest = out_path / "manifest.json"

    targets = [target_model, target_vectorizer, target_manifest]

    if not overwrite:
        existing = [str(p) for p in targets if p.exists()]
        if existing:
            raise FileExistsError(
                f"Artifact targets already exist and overwrite=False: {existing}"
            )

    selected_model_instance = result.models[model_name]
    vectorizer_instance = result.vectorizer

    # Use staging temporary directory on the same filesystem/parent
    temp_dir = tempfile.mkdtemp(dir=out_path, prefix=".tmp_artifacts_")
    try:
        tmp_model = Path(temp_dir) / "intent_model.pkl"
        tmp_vec = Path(temp_dir) / "tfidf_vectorizer.pkl"
        tmp_manifest = Path(temp_dir) / "manifest.json"

        # Dump to temporary files
        joblib.dump(selected_model_instance, tmp_model)
        joblib.dump(vectorizer_instance, tmp_vec)

        # Build manifest dictionary from the temporary artifacts
        manifest_dict = build_manifest(
            model_path=tmp_model,
            vectorizer_path=tmp_vec,
        )
        # Ensure filenames in manifest match final target names
        manifest_dict["model_filename"] = target_model.name
        manifest_dict["vectorizer_filename"] = target_vectorizer.name

        import json

        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, indent=2)

        # Atomic replacements
        os.replace(tmp_model, target_model)
        os.replace(tmp_vec, target_vectorizer)
        os.replace(tmp_manifest, target_manifest)

        model_sha256 = manifest_dict["model_sha256"]
        vectorizer_sha256 = manifest_dict["vectorizer_sha256"]

        return {
            "model_path": target_model,
            "vectorizer_path": target_vectorizer,
            "manifest_path": target_manifest,
            "model_sha256": model_sha256,
            "vectorizer_sha256": vectorizer_sha256,
            "model_version": manifest_dict["model_version"],
        }
    finally:
        # Clean up temporary directory if it still exists
        if os.path.exists(temp_dir):
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
