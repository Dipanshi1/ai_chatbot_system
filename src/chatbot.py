import re
import joblib
import pandas as pd
import random

df = pd.read_excel("data/raw/AI-Powered Chatbot.xlsx")
def get_response(intent):
    responses = df[df["Intent"] == intent]["Bot Response"].tolist()

    if not responses:
        return "I'm sorry, I don't have information about that yet."

    return random.choice(responses)


# Load trained model and vectorizer
model = joblib.load("artifacts/intent_model.pkl")
vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_intent(message):
    # Clean the user's message
    cleaned_message = clean_text(message)

    # Convert text into TF-IDF features
    message_tfidf = vectorizer.transform([cleaned_message])

    # Predict intent
    intent = model.predict(message_tfidf)[0]

    return intent

if __name__ == "__main__":
    message = input("You: ")

    intent = predict_intent(message)
    response = get_response(intent)

    print("Predicted intent:", intent)
    print("Bot:", response)

import pandas as pd


df = pd.read_excel("data/raw/AI-Powered Chatbot.xlsx")

print("\n--- Responses per Intent ---")

response_counts = df.groupby("Intent")["Bot Response"].nunique()

print(response_counts)