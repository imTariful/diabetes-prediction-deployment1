import streamlit as st
from transformers import pipeline

# Load HuggingFace Sentiment Pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Streamlit App
st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊", layout="centered")
st.title("📝 Sentiment Analysis App")
st.write("Enter a review and see if it's **Positive** or **Negative**!")

# Input text
user_input = st.text_area("✍️ Write your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        result = sentiment_pipeline(user_input)[0]
        label = result["label"]
        score = round(result["score"] * 100, 2)

        if label == "POSITIVE":
            st.success(f"✅ Positive Review ({score}%)")
        else:
            st.error(f"❌ Negative Review ({score}%)")
    else:
        st.warning("⚠️ Please enter some text before analyzing.")
