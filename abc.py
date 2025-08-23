import os
import sys
import subprocess
import joblib
import pandas as pd
import numpy as np
import re
import string
import streamlit as st
try:
    from datasets import load_dataset
except ModuleNotFoundError:
    st.error("Installing datasets...")
    subprocess.run([sys.executable, "-m", "pip", "install", "datasets"])
    os.execv(sys.executable, ['python'] + sys.argv)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    st.error("Installing scikit-learn...")
    subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn"])
    os.execv(sys.executable, ['python'] + sys.argv)
try:
    from gensim.models import Word2Vec
except ModuleNotFoundError:
    st.error("Installing gensim...")
    subprocess.run([sys.executable, "-m", "pip", "install", "gensim"])
    os.execv(sys.executable, ['python'] + sys.argv)
try:
    from transformers import BertTokenizer, BertModel
    import torch
except ModuleNotFoundError:
    st.error("Installing transformers and torch...")
    subprocess.run([sys.executable, "-m", "pip", "install", "transformers", "torch"])
    os.execv(sys.executable, ['python'] + sys.argv)
try:
    import joblib
except ModuleNotFoundError:
    st.error("Installing joblib...")
    subprocess.run([sys.executable, "-m", "pip", "install", "joblib"])
    os.execv(sys.executable, ['python'] + sys.argv)
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Create models directory
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Hardcoded stopwords list
stop_words = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you',
    'your', 'yours', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'doing', 'but',
    'if', 'or', 'because', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once'
}

# Preprocessing function (no NLTK)
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens), tokens

# Train and save models
@st.cache_resource(show_spinner="Training models if not already done...")
def train_and_save_models():
    if (os.path.exists(os.path.join(MODELS_DIR, "tfidf_vec.joblib")) and
        os.path.exists(os.path.join(MODELS_DIR, "tfidf_lr.joblib")) and
        os.path.exists(os.path.join(MODELS_DIR, "w2v.model")) and
        os.path.exists(os.path.join(MODELS_DIR, "w2v_lr.joblib")) and
        os.path.exists(os.path.join(MODELS_DIR, "bert_model")) and
        os.path.exists(os.path.join(MODELS_DIR, "bert_tokenizer")) and
        os.path.exists(os.path.join(MODELS_DIR, "bert_lr.joblib"))):
        st.info("Models already trained and saved.")
        return

    st.info("Training models... This may take a while.")

    # Load IMDB dataset
    try:
        imdb = load_dataset("imdb")
    except Exception as e:
        st.error(f"Failed to load IMDB dataset: {e}")
        st.stop()

    train_data = imdb['train']
    test_data = imdb['test']

    # Convert to DataFrames
    train_df = pd.DataFrame({'text': train_data['text'], 'label': train_data['label']})
    test_df = pd.DataFrame({'text': test_data['text'], 'label': test_data['label']})

    # Split train into train and val
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    # Preprocess
    train_df['processed_text'], train_df['tokens'] = zip(*train_df['text'].apply(preprocess_text))
    val_df['processed_text'], val_df['tokens'] = zip(*val_df['text'].apply(preprocess_text))
    test_df['processed_text'], test_df['tokens'] = zip(*test_df['text'].apply(preprocess_text))

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
    word2vec_model = Word2Vec(sentences=train_df['tokens'], vector_size=100, window=5, min_count=1, workers=4, seed=42)

    def get_doc_embedding(tokens, model):
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    X_train_w2v = np.array([get_doc_embedding(tokens, word2vec_model) for tokens in train_df['tokens']])
    X_test_w2v = np.array([get_doc_embedding(tokens, word2vec_model) for tokens in test_df['tokens']])

    w2v_model = LogisticRegression(max_iter=1000, random_state=42)
    w2v_model.fit(X_train_w2v, y_train)

    word2vec_model.save(os.path.join(MODELS_DIR, "w2v.model"))
    joblib.dump(w2v_model, os.path.join(MODELS_DIR, "w2v_lr.joblib"))

    # BERT (subset for efficiency)
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

    X_train_bert = get_bert_embeddings(train_subset.tolist())
    X_test_bert = get_bert_embeddings(test_subset.tolist())

    bert_model_lr = LogisticRegression(max_iter=1000, random_state=42)
    bert_model_lr.fit(X_train_bert, y_train_subset)

    bert_model.save_pretrained(os.path.join(MODELS_DIR, "bert_model"))
    tokenizer.save_pretrained(os.path.join(MODELS_DIR, "bert_tokenizer"))
    joblib.dump(bert_model_lr, os.path.join(MODELS_DIR, "bert_lr.joblib"))

    st.success("Models trained and saved!")

# Load models
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
    processed, _ = preprocess_text(review)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1] if pred == 1 else model.predict_proba(vec)[0][0]
    return "Positive" if pred == 1 else "Negative", prob

def predict_w2v(review, w2v, model):
    processed, tokens = preprocess_text(review)
    embedding = np.mean([w2v.wv[word] for word in tokens if word in w2v.wv], axis=0) if tokens else np.zeros(w2v.vector_size)
    pred = model.predict([embedding])[0]
    prob = model.predict_proba([embedding])[0][1] if pred == 1 else model.predict_proba([embedding])[0][0]
    return "Positive" if pred == 1 else "Negative", prob

def predict_bert(review, tokenizer, bert, model):
    processed, _ = preprocess_text(review)
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

# Train models button
if st.button("Train Models (if not already trained)"):
    train_and_save_models()

# Model selection
model_type = st.selectbox("Select Model:", ["TF-IDF", "Word2Vec", "BERT"])

# Load selected model
try:
    if model_type == "BERT":
        tokenizer, bert, lr_model = load_models(model_type)
    elif model_type == "TF-IDF":
        vectorizer, lr_model = load_models(model_type)
    else:  # Word2Vec
        w2v, lr_model = load_models(model_type)
except FileNotFoundError:
    st.error("Models not found. Please train the models first by clicking 'Train Models'.")
    st.stop()

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
