import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="IMDb Sentiment Analysis", page_icon="🎬", layout="wide")
st.markdown("<h1 style='text-align:center; color:#FF4B4B;'>🎬 IMDb Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Predict if a movie review is Positive or Negative</p>", unsafe_allow_html=True)
st.write("---")

# ----------------------
# Load IMDb dataset
# ----------------------
@st.cache_resource
def load_imdb_data(num_words=10000):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
    word_index = imdb.get_word_index()
    index_word = {v + 3: k for k, v in word_index.items()}
    index_word[0] = "<PAD>"
    index_word[1] = "<START>"
    index_word[2] = "<UNK>"
    index_word[3] = "<UNUSED>"

    def decode_review(encoded):
        return " ".join([index_word.get(i, "?") for i in encoded])

    X_train_text = [decode_review(x) for x in X_train]
    X_test_text = [decode_review(x) for x in X_test]
    return X_train_text, X_test_text, y_train, y_test

with st.spinner("Loading and preparing data..."):
    X_train, X_test, y_train, y_test = load_imdb_data()

# ----------------------
# Train model
# ----------------------
@st.cache_resource
def train_model(X_train, y_train, X_test, y_test):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    return pipe, acc

with st.spinner("Training model with TF-IDF + Logistic Regression..."):
    model, acc = train_model(X_train, y_train, X_test, y_test)

st.success(f"✅ Model trained! Test Accuracy: {acc:.2f}")
st.write("---")

# ----------------------
# User input & prediction
# ----------------------
st.subheader("🔎 Try it Yourself")
user_input = st.text_area("Enter a movie review:", "This movie was amazing, I really loved it!")

if st.button("Predict Sentiment"):
    prediction = model.predict([user_input])[0]
    prob = model.predict_proba([user_input])[0][prediction]
    
    col1, col2 = st.columns([1,2])
    with col1:
        if prediction == 1:
            st.markdown("<h2 style='color:green;'>😊 Positive</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:red;'>☹️ Negative</h2>", unsafe_allow_html=True)
        st.write(f"Confidence: **{prob*100:.2f}%**")
    with col2:
        st.progress(prob if prediction==1 else 1-prob)

    with st.expander("Show Your Review"):
        st.write(user_input)

st.write("---")
st.markdown("<p style='text-align:center; color:gray;'>Made with ❤️ by You | Powered by Streamlit & Scikit-learn</p>", unsafe_allow_html=True)
