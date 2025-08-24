import streamlit as st
from textblob import TextBlob

# Streamlit App
st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊", layout="centered")

st.title("📝 Sentiment Analysis App")
st.write("Enter a review and see if it's **Positive** or **Negative**!")

# Input text
user_input = st.text_area("✍️ Write your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Sentiment Analysis
        blob = TextBlob(user_input)
        polarity = blob.sentiment.polarity
        
        # Display result
        if polarity > 0:
            st.success("✅ Positive Review")
            st.progress(min(1.0, polarity))  # show positivity level
        elif polarity < 0:
            st.error("❌ Negative Review")
            st.progress(min(1.0, abs(polarity)))  # show negativity level
        else:
            st.info("😐 Neutral Review")

# Footer
st.write("---")
st.caption("Built with ❤️ using Streamlit & TextBlob")
