import os
import subprocess

# Ensure TextBlob is installed
try:
    from textblob import TextBlob
except ImportError:
    subprocess.check_call(["pip", "install", "textblob"])
    from textblob import TextBlob

import streamlit as st

# Streamlit App
st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊", layout="centered")

st.title("😊 Sentiment Analysis App")

# Input
user_input = st.text_area("Enter your review:")

if st.button("Analyze"):
    if user_input.strip() != "":
        blob = TextBlob(user_input)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            st.success("This is a Positive Review! 👍")
        elif sentiment < 0:
            st.error("This is a Negative Review! 👎")
        else:
            st.info("This review seems Neutral 😐")
    else:
        st.warning("Please enter some text to analyze.")
