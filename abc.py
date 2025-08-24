import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
# Sample IMDb data (replace with larger CSV if needed)
# ----------------------
data = {
    "review": [
        "I loved this movie, it was fantastic and thrilling!",
        "Worst movie ever, completely boring and slow.",
        "An amazing experience, would watch again.",
        "Terrible plot and bad acting, do not recommend.",
        "A wonderful film, very entertaining.",
        "Awful movie, wasted my time."
    ],
    "sentiment": [1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}
df = pd.DataFrame(data)

# ----------------------
# Train TF-IDF + Logistic Regression
# ----------------------
@st.cache_resource
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df["review"], df["sentiment"], test_size=0.2, random_state=42)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    return pipe, acc

with st.spinner("Training model..."):
    model, acc = train_model(df)

st.success(f"✅ Model trained! Accuracy on sample data: {acc:.2f}")
st.write("---")

# ----------------------
# User input & prediction
# ----------------------
st.subheader("🔎 Try it Yourself")
user_input = st.text_area("Enter a movie review:", "This movie was amazing, I really loved it!")

if st.button("Predict Sentiment"):
    prediction = model.predict([user_input])[0]
    prob = model.predict_proba([user_input])[0][prediction]

    col1, col2 = st.columns([1, 2])
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
