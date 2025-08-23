# app.py
import streamlit as st
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ========== Setup ==========
st.set_page_config(page_title="NLP Preprocessing App", layout="wide")

# Download resources
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")
hf_sentiment = pipeline("sentiment-analysis")

# ========== Functions ==========
def preprocess_text(text):
    # Sentence Tokenization
    sentences = sent_tokenize(text)

    # Word Tokenization
    words = word_tokenize(text.lower())

    # Remove Stopwords
    filtered_words = [w for w in words if w.isalpha() and w not in stop_words]

    # Lemmatization with spaCy
    doc = nlp(" ".join(filtered_words))
    lemmas = [token.lemma_ for token in doc]

    return sentences, words, filtered_words, lemmas


def plot_word_frequency(words):
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1

    df = [{"word": k, "count": v} for k, v in word_freq.items()]
    fig = px.bar(df, x="word", y="count", title="Word Frequency", text="count")
    return fig


def plot_wordcloud(words):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig


# ========== Streamlit UI ==========
st.title("📝 NLP Preprocessing & Visualization")
st.markdown("An **all-in-one text preprocessing app** with Tokenization, Lemmatization, Stopwords Removal, WordCloud, and Sentiment Analysis.")

# Input Text
text_input = st.text_area("Enter your text:", height=200)

if st.button("Process Text"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Preprocess
        sentences, words, filtered_words, lemmas = preprocess_text(text_input)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔹 Original Sentences")
            st.write(sentences)

            st.subheader("🔹 Tokenized Words")
            st.write(words)

            st.subheader("🔹 Without Stopwords")
            st.write(filtered_words)

            st.subheader("🔹 Lemmatized Words")
            st.write(lemmas)

        with col2:
            st.subheader("📊 Word Frequency Visualization")
            st.plotly_chart(plot_word_frequency(filtered_words))

            st.subheader("☁️ WordCloud")
            st.pyplot(plot_wordcloud(filtered_words))

        # Sentiment Analysis
        st.subheader("💡 Sentiment Analysis (Hugging Face)")
        sentiment_results = hf_sentiment(text_input)
        st.json(sentiment_results)
