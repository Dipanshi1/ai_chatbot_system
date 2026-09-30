import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


# Load dataset
df = pd.read_excel("data/raw/AI-Powered Chatbot.xlsx")


# Select input and target
X = df["User Message"]
y = df["Intent"]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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

import os

os.makedirs("artifacts", exist_ok=True)

joblib.dump(svm_model, "artifacts/intent_model.pkl")
joblib.dump(vectorizer, "artifacts/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")