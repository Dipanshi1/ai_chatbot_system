"""
Honest evaluation library for University Support Chatbot (Phase 5B).

Provides deterministic metrics, Wilson confidence intervals, majority baseline,
per-class reports, saved model equivalence checking, repeated-split analysis,
and leakage-safe train-only KB pipeline evaluation.
Contains NO file I/O operations.
"""
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .artifacts import compute_sha256
from .config import (
    FEATURE_COLUMN,
    METRIC_DECIMALS,
    MODEL_NAMES,
    MODEL_PATH,
    MODEL_RANDOM_STATE,
    REPEATED_SPLIT_SEEDS,
    SELECTED_MODEL,
    SPLIT_RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    VECTORIZER_PATH,
    WILSON_Z,
)
from .data import load_dataset, split_dataset
from .inference import IntentClassifier
from .pipeline import ChatPipeline
from .preprocessing import clean_text
from .retrieval import ResponseRetriever
from .schemas import Outcome
from .training import TrainingResult, build_models, fit_vectorizer, prepare_text, train_in_memory


def wilson_interval(
    k: int,
    n: int,
    z: float = WILSON_Z,
) -> Optional[Tuple[float, float]]:
    """
    Calculate the Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Total number of trials.
    z : float, optional
        Standard normal quantile (default: WILSON_Z = 1.959963984540054 for 95% CI).

    Returns
    -------
    Optional[Tuple[float, float]]
        (lower_bound, upper_bound) clamped to [0.0, 1.0], or None if n == 0.

    Raises
    ------
    ValueError
        If k or n are negative or k > n.
    """
    if n == 0:
        return None
    if k < 0 or n < 0 or k > n:
        raise ValueError(f"Invalid success/trial counts: k={k}, n={n}")

    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    half_width = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))

    low = max(0.0, center - half_width)
    high = min(1.0, center + half_width)
    return (round(low, METRIC_DECIMALS), round(high, METRIC_DECIMALS))


def majority_baseline(
    y_train: Sequence[str],
    y_test: Sequence[str],
) -> Dict[str, Any]:
    """
    Compute majority class baseline using training set frequencies.

    Rule:
    - Highest training count wins.
    - Ties resolve to the first label in sorted alphabetical order.
    - Reports all tied labels and their respective test accuracies.
    """
    train_series = pd.Series(list(y_train))
    test_series = pd.Series(list(y_test))

    counts = train_series.value_counts()
    max_count = int(counts.max())
    tied_labels = tuple(sorted(counts[counts == max_count].index.tolist()))
    majority_label = tied_labels[0]

    tied_accuracies = {}
    for label in tied_labels:
        acc = float((test_series == label).mean())
        tied_accuracies[label] = round(acc, METRIC_DECIMALS)

    correct = int((test_series == majority_label).sum())
    n = len(test_series)
    acc = correct / n if n > 0 else 0.0

    ci = wilson_interval(correct, n)
    ci_low = ci[0] if ci else 0.0
    ci_high = ci[1] if ci else 0.0

    return {
        "majority_label": majority_label,
        "tied_labels": tied_labels,
        "tied_accuracies": tied_accuracies,
        "correct": correct,
        "n": n,
        "accuracy": round(acc, METRIC_DECIMALS),
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
    }


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    all_labels: Sequence[str],
) -> Dict[str, Any]:
    """
    Calculate classification metrics adhering to Phase 5B rules.

    Important rules:
    - macro_f1_present: macro F1 over union of present classes in y_true and y_pred (zero_division=0)
    - macro_f1_all_classes: macro F1 across all 26 candidate intents (zero_division=0)
    - weighted_f1: weighted F1 across all classes (zero_division=0)
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n = len(y_true_arr)

    correct = int((y_true_arr == y_pred_arr).sum())
    acc = correct / n if n > 0 else 0.0

    ci = wilson_interval(correct, n)
    ci_low = ci[0] if ci else 0.0
    ci_high = ci[1] if ci else 0.0

    labels_present = sorted(set(y_true_arr) | set(y_pred_arr))
    macro_f1_present = float(
        f1_score(y_true_arr, y_pred_arr, labels=labels_present, average="macro", zero_division=0)
    )
    macro_f1_all_classes = float(
        f1_score(y_true_arr, y_pred_arr, labels=all_labels, average="macro", zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_true_arr, y_pred_arr, labels=all_labels, average="weighted", zero_division=0)
    )

    return {
        "accuracy": round(acc, METRIC_DECIMALS),
        "correct": correct,
        "n": n,
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
        "macro_f1_present": round(macro_f1_present, METRIC_DECIMALS),
        "n_labels_present": len(labels_present),
        "macro_f1_all_classes": round(macro_f1_all_classes, METRIC_DECIMALS),
        "weighted_f1": round(weighted_f1, METRIC_DECIMALS),
    }


def per_class_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    all_labels: Sequence[str],
) -> pd.DataFrame:
    """
    Construct a per-class report with exactly 26 rows.

    Columns:
    intent, precision, recall, f1, support, predicted_count, in_test, note
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=all_labels,
        zero_division=0,
    )

    rows = []
    for idx, intent in enumerate(all_labels):
        sup = int(support[idx])
        pred_count = int((y_pred_arr == intent).sum())
        in_test = sup > 0

        if sup == 0:
            note = "absent from test"
        elif sup <= 1:
            note = "support<=1"
        else:
            note = ""

        rows.append(
            {
                "intent": intent,
                "precision": round(float(precision[idx]), METRIC_DECIMALS),
                "recall": round(float(recall[idx]), METRIC_DECIMALS),
                "f1": round(float(f1[idx]), METRIC_DECIMALS),
                "support": sup,
                "predicted_count": pred_count,
                "in_test": in_test,
                "note": note,
            }
        )

    return pd.DataFrame(rows)


def evaluate_models(
    result: TrainingResult,
    all_labels: Sequence[str],
) -> pd.DataFrame:
    """
    Evaluate the 3 trained models and majority baseline on the test split.

    Does NOT retrain models.
    """
    X_test_clean = prepare_text(result.test_df[FEATURE_COLUMN])
    X_test_tfidf = result.vectorizer.transform(X_test_clean)
    y_test = result.test_df[TARGET_COLUMN].values
    y_train = result.train_df[TARGET_COLUMN].values

    n_train = len(y_train)
    n_test = len(y_test)

    display_names = {
        "logistic_regression": "Logistic Regression",
        "linear_svm": "Linear SVM",
        "multinomial_nb": "Multinomial NB",
    }

    records = []
    # 1. Candidate classifiers
    for model_key in MODEL_NAMES:
        model = result.models[model_key]
        y_pred = model.predict(X_test_tfidf)
        metrics = classification_metrics(y_test, y_pred, all_labels)
        records.append(
            {
                "model": model_key,
                "display_name": display_names.get(model_key, model_key),
                "selected": model_key == SELECTED_MODEL,
                "n_train": n_train,
                "n_test": n_test,
                "correct": metrics["correct"],
                "accuracy": metrics["accuracy"],
                "accuracy_ci_low": metrics["accuracy_ci_low"],
                "accuracy_ci_high": metrics["accuracy_ci_high"],
                "macro_f1_present": metrics["macro_f1_present"],
                "n_labels_present": metrics["n_labels_present"],
                "macro_f1_all_classes": metrics["macro_f1_all_classes"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )

    # 2. Majority baseline
    maj = majority_baseline(y_train, y_test)
    y_pred_maj = [maj["majority_label"]] * n_test
    maj_metrics = classification_metrics(y_test, y_pred_maj, all_labels)
    records.append(
        {
            "model": "majority_baseline",
            "display_name": "Majority Baseline",
            "selected": False,
            "n_train": n_train,
            "n_test": n_test,
            "correct": maj["correct"],
            "accuracy": maj["accuracy"],
            "accuracy_ci_low": maj["accuracy_ci_low"],
            "accuracy_ci_high": maj["accuracy_ci_high"],
            "macro_f1_present": maj_metrics["macro_f1_present"],
            "n_labels_present": maj_metrics["n_labels_present"],
            "macro_f1_all_classes": maj_metrics["macro_f1_all_classes"],
            "weighted_f1": maj_metrics["weighted_f1"],
        }
    )

    return pd.DataFrame(records)


def check_saved_model_equivalence(
    saved_model_path: Optional[Union[Path, str]] = None,
    saved_vectorizer_path: Optional[Union[Path, str]] = None,
    result: Optional[TrainingResult] = None,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compare committed saved model artifacts against an in-memory canonical seeded training run.

    Validates:
    - Vocabulary is equal
    - IDF values are allclose
    - Prediction agreement is exactly 40/40 on the canonical test split
    - Coefficient absolute difference is < 1e-4
    - Saved model random_state is None (claim: 'prediction-equivalent', not 'bit-reproducible')
    """
    m_path = Path(saved_model_path) if saved_model_path else MODEL_PATH
    v_path = Path(saved_vectorizer_path) if saved_vectorizer_path else VECTORIZER_PATH

    saved_model = joblib.load(m_path)
    saved_vec = joblib.load(v_path)

    if result is None:
        result = train_in_memory(df=df)

    # 1. Vectorizer vocabulary & IDF
    vocab_equal = saved_vec.vocabulary_ == result.vectorizer.vocabulary_
    idf_allclose = bool(np.allclose(saved_vec.idf_, result.vectorizer.idf_))

    # 2. Predictions on canonical test split
    X_test_clean = prepare_text(result.test_df[FEATURE_COLUMN])
    saved_pred = saved_model.predict(saved_vec.transform(X_test_clean))
    refit_pred = result.models["linear_svm"].predict(result.vectorizer.transform(X_test_clean))

    agreement_count = int((saved_pred == refit_pred).sum())
    total_test = len(saved_pred)
    agreement_str = f"{agreement_count}/{total_test}"

    # 3. Coefficient difference
    max_coef_diff = float(np.max(np.abs(saved_model.coef_ - result.models["linear_svm"].coef_)))
    max_intercept_diff = float(
        np.max(np.abs(saved_model.intercept_ - result.models["linear_svm"].intercept_))
    )

    # 4. Check saved random_state
    saved_random_state = getattr(saved_model, "random_state", None)

    is_equivalent = (
        vocab_equal
        and idf_allclose
        and (agreement_count == total_test)
        and (max_coef_diff < 1e-4)
        and (max_intercept_diff < 1e-4)
    )

    return {
        "is_prediction_equivalent": is_equivalent,
        "claim": "prediction-equivalent",
        "vocabulary_equal": vocab_equal,
        "idf_allclose": idf_allclose,
        "prediction_agreement": agreement_str,
        "max_coef_difference": round(max_coef_diff, METRIC_DECIMALS),
        "max_intercept_difference": round(max_intercept_diff, METRIC_DECIMALS),
        "saved_model_random_state": saved_random_state,
        "note": "Committed model was created before random_state=42 was pinned; it is prediction-equivalent, not bit-reproducible.",
    }


def run_repeated_splits(
    df: Optional[pd.DataFrame] = None,
    *,
    seeds: Sequence[int] = REPEATED_SPLIT_SEEDS,
    test_size: float = TEST_SIZE,
    model_random_state: int = MODEL_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Perform supplementary repeated 80/20 train/test splits across specified seeds.

    Operates entirely in memory and never modifies or saves artifacts.
    """
    if df is None:
        df = load_dataset()

    records = []
    for seed in seeds:
        train_df, test_df = split_dataset(
            df,
            test_size=test_size,
            random_state=seed,
        )

        y_train = train_df[TARGET_COLUMN].values
        y_test = test_df[TARGET_COLUMN].values
        n_test = len(y_test)

        X_train_clean = prepare_text(train_df[FEATURE_COLUMN])
        X_test_clean = prepare_text(test_df[FEATURE_COLUMN])

        vec = fit_vectorizer(X_train_clean)
        X_train_tfidf = vec.transform(X_train_clean)
        X_test_tfidf = vec.transform(X_test_clean)

        models = build_models(random_state=model_random_state)
        for model_name, model in models.items():
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            correct = int((y_pred == y_test).sum())
            acc = correct / n_test if n_test > 0 else 0.0

            records.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "accuracy": round(acc, METRIC_DECIMALS),
                    "correct": correct,
                    "n_test": n_test,
                }
            )

    return pd.DataFrame(records)


def summarize_repeated_splits(
    repeated_df: pd.DataFrame,
    canonical_accuracies: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Summarize repeated split results across models.

    Columns:
    model, mean, std, min, median, max, n_seeds_best_strict, n_seeds_best_or_tied,
    canonical_accuracy, share_of_seeds_at_or_above_canonical
    """
    if canonical_accuracies is None:
        canonical_accuracies = {
            "linear_svm": 0.500000,
            "logistic_regression": 0.425000,
            "multinomial_nb": 0.375000,
        }

    # Determine best model per seed
    pivot = repeated_df.pivot(index="seed", columns="model", values="accuracy")

    best_strict_counts = {m: 0 for m in MODEL_NAMES}
    best_or_tied_counts = {m: 0 for m in MODEL_NAMES}

    for _, row in pivot.iterrows():
        max_val = row.max()
        best_models = row[row == max_val].index.tolist()
        if len(best_models) == 1:
            best_strict_counts[best_models[0]] += 1
        for bm in best_models:
            best_or_tied_counts[bm] += 1

    summary_rows = []
    for model_name in MODEL_NAMES:
        sub = repeated_df[repeated_df["model"] == model_name]
        accs = sub["accuracy"]
        n_seeds = len(accs)

        mean_val = float(accs.mean())
        std_val = float(accs.std(ddof=1)) if n_seeds > 1 else 0.0
        min_val = float(accs.min())
        median_val = float(accs.median())
        max_val = float(accs.max())

        canon_acc = canonical_accuracies.get(model_name, 0.0)
        at_or_above = int((accs >= canon_acc).sum())
        share_at_or_above = at_or_above / n_seeds if n_seeds > 0 else 0.0

        summary_rows.append(
            {
                "model": model_name,
                "mean": round(mean_val, METRIC_DECIMALS),
                "std": round(std_val, METRIC_DECIMALS),
                "min": round(min_val, METRIC_DECIMALS),
                "median": round(median_val, METRIC_DECIMALS),
                "max": round(max_val, METRIC_DECIMALS),
                "n_seeds_best_strict": best_strict_counts[model_name],
                "n_seeds_best_or_tied": best_or_tied_counts[model_name],
                "canonical_accuracy": round(canon_acc, METRIC_DECIMALS),
                "share_of_seeds_at_or_above_canonical": round(share_at_or_above, METRIC_DECIMALS),
            }
        )

    return pd.DataFrame(summary_rows)


def evaluate_pipeline_train_only_kb(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    classifier: Optional[IntentClassifier] = None,
) -> Dict[str, Any]:
    """
    Evaluate the full chatbot pipeline using a leakage-free training-only knowledge base.

    Strict Leakage Assertions:
    - KB size == 160
    - KB size < 200
    - No test indices in KB
    - No test questions in KB
    - No test responses in KB

    Raises RuntimeError if any leakage check fails.
    """
    retriever = ResponseRetriever(df=train_df)
    clf = classifier if classifier is not None else IntentClassifier()
    pipeline = ChatPipeline(classifier=clf, retriever=retriever)

    # 1. Leakage assertions
    kb_size = len(retriever.df)
    if kb_size != 160:
        raise RuntimeError(f"Leakage check failed: KB size is {kb_size}, expected 160")
    if kb_size >= 200:
        raise RuntimeError(f"Leakage check failed: KB size {kb_size} is not strictly less than 200")

    overlap_indices = set(test_df.index).intersection(set(retriever.df.index))
    if overlap_indices:
        raise RuntimeError(f"Leakage check failed: test indices found in KB: {overlap_indices}")

    overlap_questions = set(test_df[FEATURE_COLUMN]).intersection(set(retriever.df[FEATURE_COLUMN]))
    if overlap_questions:
        raise RuntimeError(f"Leakage check failed: test questions found in KB: {overlap_questions}")

    overlap_responses = set(test_df["Bot Response"]).intersection(set(retriever.df["Bot Response"]))
    if overlap_responses:
        raise RuntimeError(f"Leakage check failed: test responses found in KB: {overlap_responses}")

    # 2. Row-by-row evaluation
    rows = []
    answered_count = 0
    fallback_count = 0
    abstained_count = 0

    wrong_intent_answered_count = 0
    wrong_area_fallback_count = 0
    returned_own_heldout_count = 0

    for idx, row in test_df.iterrows():
        msg = str(row[FEATURE_COLUMN])
        true_intent = str(row[TARGET_COLUMN])
        heldout_resp = str(row["Bot Response"])

        res = pipeline.answer(msg)

        outcome_val = res.outcome.value
        abstain_reason_val = res.abstain_reason.value if res.abstain_reason else None

        if res.outcome == Outcome.ANSWERED:
            answered_count += 1
        elif res.outcome == Outcome.INTENT_FALLBACK:
            fallback_count += 1
        elif res.outcome == Outcome.ABSTAINED:
            abstained_count += 1

        pred_intent = res.predicted_intent
        pred_correct = pred_intent == true_intent if pred_intent else False

        committed_intent = res.intent
        committed_correct = (committed_intent == true_intent) if committed_intent is not None else None

        wrong_intent_answered = bool(res.outcome == Outcome.ANSWERED and committed_intent != true_intent)
        if wrong_intent_answered:
            wrong_intent_answered_count += 1

        wrong_area_fallback = bool(res.outcome == Outcome.INTENT_FALLBACK and committed_intent != true_intent)
        if wrong_area_fallback:
            wrong_area_fallback_count += 1

        returned_own_heldout = bool(res.answer == heldout_resp)
        if returned_own_heldout:
            returned_own_heldout_count += 1

        matched_row_id = res.matched_row_id

        rows.append(
            {
                "row_id": int(idx),
                "message": msg,
                "true_intent": true_intent,
                "predicted_intent": pred_intent or "",
                "predicted_correct": pred_correct,
                "margin": round(float(res.margin), METRIC_DECIMALS) if res.margin is not None else None,
                "outcome": outcome_val,
                "abstain_reason": abstain_reason_val or "",
                "committed_intent": committed_intent or "",
                "committed_correct": committed_correct if committed_correct is not None else "",
                "similarity": round(float(res.similarity), METRIC_DECIMALS) if res.similarity is not None else None,
                "overlap": round(float(res.overlap), METRIC_DECIMALS) if res.overlap is not None else None,
                "matched_question": res.matched_question or "",
                "matched_row_id": matched_row_id if matched_row_id is not None else "",
                "answer_returned": res.answer,
                "wrong_intent_answered": wrong_intent_answered,
                "wrong_area_fallback": wrong_area_fallback,
                "returned_own_heldout_response": returned_own_heldout,
            }
        )

    pipeline_df = pd.DataFrame(rows)
    n_test = len(test_df)

    committed_total = answered_count + fallback_count
    committed_correct_count = (
        int((pipeline_df["committed_correct"] == True).sum()) if committed_total > 0 else 0
    )
    committed_acc = committed_correct_count / committed_total if committed_total > 0 else 0.0
    committed_ci = wilson_interval(committed_correct_count, committed_total)

    pred_correct_count = int(pipeline_df["predicted_correct"].sum())
    pred_acc = pred_correct_count / n_test if n_test > 0 else 0.0
    pred_ci = wilson_interval(pred_correct_count, n_test)

    summary = {
        "records_df": pipeline_df,
        "n_test": n_test,
        "outcome_counts": {
            "ANSWERED": answered_count,
            "INTENT_FALLBACK": fallback_count,
            "ABSTAINED": abstained_count,
        },
        "answer_rate": round(answered_count / n_test, METRIC_DECIMALS),
        "intent_coverage": round((answered_count + fallback_count) / n_test, METRIC_DECIMALS),
        "fallback_rate": round(fallback_count / n_test, METRIC_DECIMALS),
        "abstention_rate": round(abstained_count / n_test, METRIC_DECIMALS),
        "committed_intents_total": committed_total,
        "committed_intents_correct": committed_correct_count,
        "committed_intent_accuracy": round(committed_acc, METRIC_DECIMALS),
        "committed_intent_ci_low": committed_ci[0] if committed_ci else 0.0,
        "committed_intent_ci_high": committed_ci[1] if committed_ci else 0.0,
        "predicted_intent_correct": pred_correct_count,
        "predicted_intent_accuracy": round(pred_acc, METRIC_DECIMALS),
        "predicted_intent_ci_low": pred_ci[0] if pred_ci else 0.0,
        "predicted_intent_ci_high": pred_ci[1] if pred_ci else 0.0,
        "wrong_intent_answered": wrong_intent_answered_count,
        "wrong_area_fallback": wrong_area_fallback_count,
        "returned_own_heldout_response": returned_own_heldout_count,
        "kb_size": kb_size,
        "kb_version": retriever.kb_version,
    }

    return summary
