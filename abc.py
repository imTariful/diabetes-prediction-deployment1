import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from datasets import load_dataset

# ----------------------
# Page configuration
# ----------------------
st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
)

# Header
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎬 IMDb Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Predict if a movie review is Positive or Negative</p>", unsafe_allow_html=True)
st.write("---")

# ----------------------
# Model training
# ----------------------
@st.cache_resource
def train_model():
    dataset = load_dataset("imdb")
    df = pd.DataFrame(dataset["train"])
    df = df.sample(5000, random_state=42)  # smaller subset for speed

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    return pipe, acc

with st.spinner("Training model... (this may take ~1 minute)"):
    model, acc = train_model()

st.success(f"✅ Model trained! Accuracy on test set: {acc:.2f}")
st.write("---")

# ----------------------
# User input & prediction
# ----------------------
st.subheader("🔎 Try it Yourself")

# Input box
user_input = st.text_area("Enter a movie review here:", "This movie was amazing, I really loved it!")

if st.button("Predict Sentiment"):
    prediction = model.predict([user_input])[0]
    prob = model.predict_proba([user_input])[0][prediction]

    # Sentiment visualization
    col1, col2 = st.columns([1,2])
    
    with col1:
        if prediction == 1:
            st.markdown("<h2 style='color: green;'>😊 Positive</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: red;'>☹️ Negative</h2>", unsafe_allow_html=True)
        st.write(f"Confidence: **{prob*100:.2f}%**")
    
    with col2:
        st.progress(prob) if prediction == 1 else st.progress(1-prob)
    
    # Show original review in expander
    with st.expander("Show Your Review"):
        st.write(user_input)

# ----------------------
# Footer
# ----------------------
st.write("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ by YourName | Powered by Streamlit & Scikit-learn</p>", 
    unsafe_allow_html=True
)
