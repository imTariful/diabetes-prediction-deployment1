# all_in_one_imdb_sentiment_analysis.py
# This script combines data loading, preprocessing, training of TF-IDF, Word2Vec, and BERT models (using embeddings + LR),
# saves the models, and provides a Streamlit UI for sentiment prediction on user input reviews.
# Run this script with: streamlit run all_in_one_imdb_sentiment_analysis.py
# Models are trained and saved only if they don't exist in the 'models' directory.
# BERT uses a subset for training due to computational constraints.
# Improvements: Caching for model loading, user-friendly UI with model selection, confidence scores, examples, and error handling.

import os
import joblib
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch
import random
import streamlit as st

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Create models directory if it doesn't exist
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Function to train and save models if not already saved
@st.cache_resource(show_spinner="Training models if not already done...")
def train_and_save_models():
    # Check if all models are already saved
    if (os.path.exists(os.path.join(MODELS_DIR, "tfidf_vec.joblib")) and
        os.path.exists(os.path.join(MODELS_DIR, "tfidf_lr.joblib")) and
        os.path.exists(os.path.join(MODELS_DIR, "w2v.model")) and
        os.path.exists(os.path.join(MODELS_DIR, "w2v_lr.joblib")) and
        os.path.exists(os.path.join(MODELS_DIR, "bert_model")) and
        os.path.exists(os.path.join(MODELS_DIR, "bert_tokenizer")) and
        os.path.exists(os.path.join(MODELS_DIR, "bert_lr.joblib"))):
        st.info("Models already trained and saved. Skipping training.")
        return

    st.info("Training models... This may take a while.")

    # Load IMDB dataset
    imdb = load_dataset("imdb")
    train_data = imdb['train']
    test_data = imdb['test']

    # Convert to DataFrames
    train_df = pd.DataFrame({'text': train_data['text'], 'label': train_data['label']})
    test_df = pd.DataFrame({'text': test_data['text'], 'label': test_data['label']})

    # Split train into train and val
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    # Preprocess
    train_df['processed_text'] = train_df['text'].apply(preprocess_text)
    val_df['processed_text'] = val_df['text'].apply(preprocess_text)
    test_df['processed_text'] = test_df['text'].apply(preprocess_text)

    y_train = train_df['label']
    y_test = test_df['label']

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['processed_text'])
    X_test_tfidf = tfidf_vectorizer.transform(test_df['processed_text'])

    tfidf_model = LogisticRegression(max_iter=1000, random_state=42)
    tfidf_model.fit(X_train_tfidf, y_train)

    joblib.dump(tfidf_vectorizer, os.path.join(MODELS_DIR, "tfidf_vec.joblib"))
    joblib.dump(tfidf_model, os.path.join(MODELS_DIR, "tfidf_lr.joblib"))

    # Word2Vec
    train_tokens = [word_tokenize(text) for text in train_df['processed_text']]
    word2vec_model = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=1, workers=4, seed=42)

    def get_doc_embedding(tokens, model):
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    X_train_w2v = np.array([get_doc_embedding(tokens, word2vec_model) for tokens in train_tokens])
    X_test_tokens = [word_tokenize(text) for text in test_df['processed_text']]
    X_test_w2v = np.array([get_doc_embedding(tokens, word2vec_model) for tokens in X_test_tokens])

    w2v_model = LogisticRegression(max_iter=1000, random_state=42)
    w2v_model.fit(X_train_w2v, y_train)

    word2vec_model.save(os.path.join(MODELS_DIR, "w2v.model"))
    joblib.dump(w2v_model, os.path.join(MODELS_DIR, "w2v_lr.joblib"))

    # BERT (using subset for efficiency)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    def get_bert_embeddings(texts, batch_size=16):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', max_length=128, truncation=True, padding=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())
        return np.vstack(embeddings)

    train_subset = train_df['processed_text'].iloc[:2000]
    test_subset = test_df['processed_text'].iloc[:1000]
    y_train_subset = train_df['label'].iloc[:2000]
    y_test_subset = test_df['label'].iloc[:1000]

    X_train_bert = get_bert_embeddings(train_subset.tolist())
    X_test_bert = get_bert_embeddings(test_subset.tolist())

    bert_model_lr = LogisticRegression(max_iter=1000, random_state=42)
    bert_model_lr.fit(X_train_bert, y_train_subset)

    bert_model.save_pretrained(os.path.join(MODELS_DIR, "bert_model"))
    tokenizer.save_pretrained(os.path.join(MODELS_DIR, "bert_tokenizer"))
    joblib.dump(bert_model_lr, os.path.join(MODELS_DIR, "bert_lr.joblib"))

    st.success("Models trained and saved successfully!")

# Load models with caching
@st.cache_resource
def load_models(model_type):
    if model_type == "TF-IDF":
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vec.joblib"))
        model = joblib.load(os.path.join(MODELS_DIR, "tfidf_lr.joblib"))
        return vectorizer, model
    elif model_type == "Word2Vec":
        w2v = Word2Vec.load(os.path.join(MODELS_DIR, "w2v.model"))
        model = joblib.load(os.path.join(MODELS_DIR, "w2v_lr.joblib"))
        return w2v, model
    elif model_type == "BERT":
        tokenizer = BertTokenizer.from_pretrained(os.path.join(MODELS_DIR, "bert_tokenizer"))
        bert = BertModel.from_pretrained(os.path.join(MODELS_DIR, "bert_model"))
        model = joblib.load(os.path.join(MODELS_DIR, "bert_lr.joblib"))
        return tokenizer, bert, model

# Prediction functions
def predict_tfidf(review, vectorizer, model):
    processed = preprocess_text(review)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1] if pred == 1 else model.predict_proba(vec)[0][0]
    return "Positive" if pred == 1 else "Negative", prob

def predict_w2v(review, w2v, model):
    processed = preprocess_text(review)
    tokens = word_tokenize(processed)
    embedding = np.mean([w2v.wv[word] for word in tokens if word in w2v.wv], axis=0) if tokens else np.zeros(w2v.vector_size)
    pred = model.predict([embedding])[0]
    prob = model.predict_proba([embedding])[0][1] if pred == 1 else model.predict_proba([embedding])[0][0]
    return "Positive" if pred == 1 else "Negative", prob

def predict_bert(review, tokenizer, bert, model):
    processed = preprocess_text(review)
    inputs = tokenizer([processed], return_tensors='pt', max_length=128, truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    pred = model.predict(embedding)[0]
    prob = model.predict_proba(embedding)[0][1] if pred == 1 else model.predict_proba(embedding)[0][0]
    return "Positive" if pred == 1 else "Negative", prob

# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎥", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.markdown("Train models (if not done), select a model, enter a review, and get the sentiment prediction!")

# Train models button (trains only if not saved)
if st.button("Train Models (if not already trained)"):
    train_and_save_models()

# Model selection
model_type = st.selectbox("Select Model:", ["TF-IDF", "Word2Vec", "BERT"])

# Load selected model
if model_type == "BERT":
    tokenizer, bert, lr_model = load_models(model_type)
else:
    if model_type == "TF-IDF":
        vectorizer, lr_model = load_models(model_type)
    else:  # Word2Vec
        w2v, lr_model = load_models(model_type)

# Sidebar for examples
st.sidebar.header("Quick Examples")
example_reviews = [
    "This movie was absolutely fantastic! The acting was top-notch and the plot kept me on the edge of my seat.",
    "What a waste of time. The story was predictable and the characters were poorly developed.",
    "An average film with some good moments but overall forgettable."
]
selected_example = st.sidebar.selectbox("Choose an example:", ["None"] + example_reviews)
if selected_example != "None":
    st.session_state.review = selected_example

# Input
review = st.text_area("Enter your movie review:", value=st.session_state.get("review", ""), height=150, placeholder="Type your review...")

# Analyze
if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.error("Please enter a review.")
    else:
        with st.spinner("Analyzing..."):
            if model_type == "TF-IDF":
                label, score = predict_tfidf(review, vectorizer, lr_model)
            elif model_type == "Word2Vec":
                label, score = predict_w2v(review, w2v, lr_model)
            else:  # BERT
                label, score = predict_bert(review, tokenizer, bert, lr_model)
            
            color = "success" if label == "Positive" else "error"
            icon = "✅" if label == "Positive" else "❌"
            getattr(st, color)(f"{icon} This review is **{label}** with a confidence of {score:.2%}.")
            
            st.markdown("---")
            st.info("Try another review or switch models!")

# Footer
st.markdown("---")
st.caption("Built with Streamlit. Models trained on IMDB dataset.")
