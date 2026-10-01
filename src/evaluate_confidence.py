import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.config import ABSTENTION_THRESHOLD, MODEL_PATH, RAW_DATA_PATH, VECTORIZER_PATH
from src.ml.preprocessing import clean_text


def run_confidence_analysis():
    # -----------------------------
    # Load dataset
    # -----------------------------
    df = pd.read_excel(RAW_DATA_PATH)
    X = df["User Message"]
    y = df["Intent"]

    # -----------------------------
    # Text cleaning
    # -----------------------------
    X_clean = X.apply(clean_text)

    # -----------------------------
    # Same train/test split used during training
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------
    # Load trained model and vectorizer
    # -----------------------------
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # -----------------------------
    # Convert test messages to TF-IDF
    # -----------------------------
    X_test_tfidf = vectorizer.transform(X_test)
    predictions = model.predict(X_test_tfidf)
    scores = model.decision_function(X_test_tfidf)

    # -----------------------------
    # Calculate margin for each prediction
    # -----------------------------
    margins = []
    top_scores = []
    second_scores = []

    for row in scores:
        ranked_scores = sorted(row, reverse=True)
        margin = ranked_scores[0] - ranked_scores[1]
        margins.append(margin)
        top_scores.append(ranked_scores[0])
        second_scores.append(ranked_scores[1])

    # -----------------------------
    # Build results table
    # -----------------------------
    results = pd.DataFrame({
        "Message": X_test.values,
        "Actual Intent": y_test.values,
        "Predicted Intent": predictions,
        "Top Score": top_scores,
        "Second Score": second_scores,
        "Margin": margins
    })

    results["Correct"] = (
        results["Actual Intent"] == results["Predicted Intent"]
    )
    results["Accepted_at_0.20"] = results["Margin"] >= ABSTENTION_THRESHOLD

    # -----------------------------
    # Display Margin Analysis
    # -----------------------------
    print("\n" + "=" * 80)
    print("--- 1. TEST SET MARGIN ANALYSIS (N = 40) ---")
    print("=" * 80)
    print(results.sort_values("Margin").to_string(index=False))

    total = len(results)
    correct_count = results["Correct"].sum()
    overall_acc = correct_count / total

    accepted_df = results[results["Accepted_at_0.20"]]
    acc_count = len(accepted_df)
    corr_in_acc = accepted_df["Correct"].sum()
    acc_among_accepted = (corr_in_acc / acc_count) if acc_count > 0 else 0.0
    coverage = acc_count / total
    abstention_rate = 1.0 - coverage

    correct_margins = results[results["Correct"]]["Margin"]
    incorrect_margins = results[~results["Correct"]]["Margin"]

    print("\n" + "=" * 80)
    print(f"--- 2. SUMMARY METRICS AT BASELINE THRESHOLD ({ABSTENTION_THRESHOLD}) ---")
    print("=" * 80)
    print(f"Total Test Samples          : {total}")
    print(f"Overall Accuracy            : {overall_acc:.4f} ({correct_count}/{total})")
    print(f"Coverage                    : {coverage:.4f} ({acc_count}/{total})")
    print(f"Abstention Rate             : {abstention_rate:.4f} ({total - acc_count}/{total})")
    print(f"Accuracy Among Accepted     : {acc_among_accepted:.4f} ({corr_in_acc}/{acc_count})")
    print("")
    print(f"Mean Margin (Correct)       : {correct_margins.mean():.4f}")
    print(f"Median Margin (Correct)     : {correct_margins.median():.4f}")
    print(f"Min / Max Margin (Correct)  : {correct_margins.min():.4f} / {correct_margins.max():.4f}")
    print("")
    print(f"Mean Margin (Incorrect)     : {incorrect_margins.mean():.4f}")
    print(f"Median Margin (Incorrect)   : {incorrect_margins.median():.4f}")
    print(f"Min / Max Margin (Incorrect): {incorrect_margins.min():.4f} / {incorrect_margins.max():.4f}")

    # -----------------------------
    # Threshold Sweep
    # -----------------------------
    thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    print("\n" + "=" * 80)
    print("--- 3. THRESHOLD SWEEP (0.00 to 0.50) ---")
    print("=" * 80)
    print(f"{'Threshold':<11} | {'Accepted':<8} | {'Abstained':<9} | {'Coverage':<10} | {'Abstention':<10} | {'Correct':<8} | {'Incorrect':<9} | {'Acc Among Accepted'}")
    print("-" * 88)

    for t in thresholds:
        t_acc = results[results["Margin"] >= t]
        n_acc = len(t_acc)
        n_abs = total - n_acc
        cov = n_acc / total
        abr = n_abs / total
        n_corr = t_acc["Correct"].sum()
        n_incorr = n_acc - n_corr
        acc_rate = (n_corr / n_acc) if n_acc > 0 else 0.0
        print(f"{t:<11.2f} | {n_acc:<8} | {n_abs:<9} | {cov*100:>8.1f}%  | {abr*100:>8.1f}%  | {n_corr:<8} | {n_incorr:<9} | {acc_rate*100:>17.2f}%")

    # -----------------------------
    # Per-Intent Analysis
    # -----------------------------
    all_intents = sorted(df["Intent"].unique())
    dataset_counts = df["Intent"].value_counts()
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()

    intent_rows = []
    for intent in all_intents:
        tot = dataset_counts.get(intent, 0)
        n_tr = train_counts.get(intent, 0)
        n_te = test_counts.get(intent, 0)
        in_te = n_te > 0
        if in_te:
            sub = results[results["Actual Intent"] == intent]
            n_corr = sub["Correct"].sum()
            n_inc = len(sub) - n_corr
            mean_m = sub["Margin"].mean()
        else:
            n_corr = 0
            n_inc = 0
            mean_m = None

        intent_rows.append({
            "Intent": intent,
            "Total": tot,
            "Train": n_tr,
            "Test": n_te,
            "In Test": in_te,
            "Correct": n_corr,
            "Incorrect": n_inc,
            "Mean Margin": f"{mean_m:.4f}" if mean_m is not None else "N/A"
        })

    intent_summary = pd.DataFrame(intent_rows)
    print("\n" + "=" * 80)
    print("--- 4. PER-INTENT DISTRIBUTION AND TEST PERFORMANCE ---")
    print("=" * 80)
    print(intent_summary.to_string(index=False))

    # -----------------------------
    # Diagnostic Queries Evaluation
    # -----------------------------
    diagnostic_queries = [
        "how do i connect to campus wi-fi?",
        "when will my financial aid be disbursed?",
        "where can i find internship postings?",
        "how do i connect to college faculty?"
    ]

    print("\n" + "=" * 80)
    print("--- 5. DIAGNOSTIC QUERIES EVALUATION ---")
    print("=" * 80)
    for q in diagnostic_queries:
        q_clean = clean_text(q)
        q_tfidf = vectorizer.transform([q_clean])
        q_scores = model.decision_function(q_tfidf)[0]
        q_ranked = q_scores.argsort()[::-1]
        top_c = model.classes_[q_ranked[0]]
        m = q_scores[q_ranked[0]] - q_scores[q_ranked[1]]
        is_abs = m < ABSTENTION_THRESHOLD
        print(f"Query: '{q}'")
        print(f"  Predicted Intent : {top_c}")
        print(f"  Decision Margin  : {m:.4f}")
        print(f"  Abstained (<{ABSTENTION_THRESHOLD}): {is_abs}")
        print(f"  Top 3 Scores     : {[(model.classes_[i], round(float(q_scores[i]), 3)) for i in q_ranked[:3]]}")
        print("-" * 60)


if __name__ == "__main__":
    run_confidence_analysis()
