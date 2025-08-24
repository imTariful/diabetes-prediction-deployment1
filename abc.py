import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import nltk

# Download movie_reviews only if not present
nltk.download("movie_reviews", quiet=True)
from nltk.corpus import movie_reviews

st.title("🎬 IMDb Sentiment Analysis (Scikit-learn Version)")

@st.cache_resource
def train_model():
    # Load NLTK movie_reviews dataset
    docs = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
    labels = [1 if fileid.split("/")[0] == "pos" else 0 for fileid in movie_reviews.fileids()]
    df = pd.DataFrame({"review": docs, "sentiment": labels})

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=42
    )

    # Build pipeline (TF-IDF + Logistic Regression)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    return pipe, acc

with st.spinner("Training model..."):
    model, acc = train_model()

st.success(f"✅ Model trained! Accuracy on test set: {acc:.2f}")

st.subheader("🔎 Try it yourself")
user_input = st.text_area("Enter a movie review:", "This movie was amazing, I really loved it!")

if st.button("Predict Sentiment"):
    prediction = model.predict([user_input])[0]
    prob = model.predict_proba([user_input])[0][prediction]
    sentiment = "😊 Positive" if prediction == 1 else "☹️ Negative"
    st.write(f"**Prediction:** {sentiment} (Confidence: {prob:.2f})")
