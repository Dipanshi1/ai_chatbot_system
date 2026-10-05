import argparse
import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.config import MODEL_PATH, RAW_DATA_PATH, VECTORIZER_PATH
from src.ml.preprocessing import clean_text


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train intent models or manage artifacts")
    parser.add_argument(
        "--write-manifest-only",
        action="store_true",
        help="Generate artifacts/manifest.json from existing artifacts without retraining",
    )
    args = parser.parse_args(argv)

    if args.write_manifest_only:
        from src.ml.artifacts import write_manifest

        manifest_file = write_manifest()
        print(f"Artifact manifest successfully written to {manifest_file}")
        return

    # Load dataset
    df = pd.read_excel(RAW_DATA_PATH)

    # Select input and target
    X = df["User Message"]
    y = df["Intent"]

    # Clean text
    X_clean = X.apply(clean_text)
    print("\nIntent distribution:")
    print(y.value_counts())

    print("\nIntents with fewer than 3 examples:")
    print(y.value_counts()[y.value_counts() < 3])

    print("Original:")
    print(X.head())

    print("\nCleaned:")
    print(X_clean.head())

    # TF-IDF
    from sklearn.model_selection import train_test_split

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # TF-IDF
    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\nTraining TF-IDF shape:")
    print(X_train_tfidf.shape)

    print("\nTesting TF-IDF shape:")
    print(X_test_tfidf.shape)

    print("\nVocabulary size:")
    print(len(vectorizer.vocabulary_))

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB

    # =========================
    # Model Training
    # =========================

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
        "Naive Bayes": MultinomialNB()
    }

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        print(f"{name} trained successfully.")

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # =========================
    # Model Evaluation
    # =========================

    for name, model in models.items():
        # Predict intents for unseen test messages
        y_pred = model.predict(X_test_tfidf)

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    # =========================
    # Inspect SVM Predictions
    # =========================

    svm_model = models["Linear SVM"]

    y_pred = svm_model.predict(X_test_tfidf)

    results = pd.DataFrame({
        "Message": X_test.values,
        "Actual Intent": y_test.values,
        "Predicted Intent": y_pred
    })

    print("\n--- Prediction Results ---")
    print(results.to_string(index=False))

    # =========================
    # Save Model and Vectorizer
    # =========================

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(svm_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nModel and vectorizer saved successfully.")


if __name__ == "__main__":
    main()