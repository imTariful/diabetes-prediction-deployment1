# -----------------------------------------------------------
# IMDB Sentiment Lab — Streamlit App
# Clean UI • Flexible Preprocessing • TF-IDF • Word2Vec • BERT
# -----------------------------------------------------------

import os
import re
import string
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from typing import List, Tuple, Literal, Optional

# ML / NLP
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from gensim.models import Word2Vec

# Optional heavy deps (SpaCy + Transformers). Handled lazily.
SPACY_OK = False
BERT_OK = False

# ---------- Page & Theme ----------
st.set_page_config(
    page_title="IMDB Sentiment Lab",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a cleaner, modern look
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .metric-card {border-radius: 16px; padding: 16px; background: #f8f9fb; box-shadow: 0 4px 20px rgba(0,0,0,0.06);}
    .section-card {border-radius: 20px; padding: 22px; background: white; box-shadow: 0 6px 28px rgba(0,0,0,0.07);}
    .title-glow {font-weight: 800; letter-spacing: .3px;}
    .subtle {color: #6b7280;}
    .tag {display:inline-block; padding: 2px 10px; border-radius: 999px; background:#eef2ff; color:#4338ca; font-size:12px; margin-left:8px;}
    .divider {height:1px; background: #eef0f4; margin: 12px 0 18px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-glow'>🎬 IMDB Sentiment Lab</h1>", unsafe_allow_html=True)
st.write("A clean, interactive workspace to preprocess reviews, train models, and visualize performance.")

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("Dataset")
    source = st.radio("Choose data source:", ["IMDB (Hugging Face)", "Upload CSV"], index=0)
    text_col_name = st.text_input("Text column (CSV mode)", value="text")
    label_col_name = st.text_input("Label column (CSV mode: 0/1)", value="label")

    st.subheader("Sampling & Split")
    sample_size = st.slider("Use N samples (per split):", 1000, 25000, 10000, step=1000,
                            help="For IMDB: train/test have 25k each. Sampling helps speed.")
    test_size = st.slider("Test size (%)", 10, 40, 20, step=5) / 100
    val_size = st.slider("Validation size (%)", 0, 20, 10, step=5) / 100
    random_state = st.number_input("Random seed", value=42, step=1)

    st.subheader("Preprocessing")
    use_spacy = st.checkbox("Use spaCy for tokenization + lemmatization", value=True)
    remove_stop = st.checkbox("Remove stopwords (NLTK)", value=True)
    remove_nums = st.checkbox("Remove numbers", value=True)
    min_token_len = st.slider("Min token length", 1, 5, 3)
    output_mode = st.radio("Preprocess output for:", ["TF-IDF / BERT (string)", "Word2Vec (tokens)"], index=0)

    st.subheader("Models")
    run_tfidf = st.checkbox("TF-IDF + Logistic Regression", value=True)
    run_w2v   = st.checkbox("Word2Vec + Random Forest", value=True)
    run_bert  = st.checkbox("BERT embeddings + Logistic Regression (subset)", value=False)
    max_len_bert = st.slider("BERT max length", 64, 256, 128, step=32)
    subset_train_bert = st.slider("BERT train subset", 1000, 10000, 5000, step=500)
    subset_test_bert  = st.slider("BERT test subset", 500, 5000, 2000, step=500)

# ---------- Utilities ----------
@st.cache_data(show_spinner=False)
def load_imdb_dataframe(n_per_split: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ds = load_dataset("imdb")
    train = pd.DataFrame(ds["train"])
    test  = pd.DataFrame(ds["test"])

    train = train.sample(n=min(n_per_split, len(train)), random_state=seed)
    test  = test.sample(n=min(n_per_split, len(test)), random_state=seed)

    # labels are already 0/1
    return train, test

@st.cache_data(show_spinner=False)
def load_csv_dataframe(file, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    assert text_col in df.columns and label_col in df.columns, "Invalid columns."
    return df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

# Lazy import for spaCy + stopwords
def init_spacy_stopwords(remove_stopwords: bool):
    global SPACY_OK, nlp, STOP_WORDS
    try:
        import spacy, nltk
        from nltk.corpus import stopwords
        try:
            _ = stopwords.words("english")
        except:
            nltk.download("stopwords")
        STOP_WORDS = set(stopwords.words("english")) if remove_stopwords else set()
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception:
            # attempt to download in runtime environments that allow it
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        SPACY_OK = True
    except Exception as e:
        SPACY_OK = False
        st.warning(f"spaCy/stopwords not available: {e}")

def clean_text(
    text: str,
    use_spacy_tok: bool = True,
    remove_stopwords: bool = True,
    remove_numbers: bool = True,
    min_len: int = 3,
    return_tokens: bool = False
):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    if remove_numbers:
        text = re.sub(r"\d+", " ", text)

    if use_spacy_tok and SPACY_OK:
        doc = nlp(text)
        toks = [t.lemma_ for t in doc if not t.is_punct]
    else:
        toks = text.split()

    if remove_stopwords:
        toks = [t for t in toks if t not in STOP_WORDS]

    toks = [t for t in toks if len(t) >= min_len]

    if return_tokens:
        return toks
    return " ".join(toks)

def split_train_val_test(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    X = df["text"].tolist()
    y = df["label"].astype(int).tolist()

    # First split off test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Then split train/val
    if val_size > 0:
        rel_val = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=rel_val, random_state=seed, stratify=y_trainval
        )
    else:
        X_train, y_train = X_trainval, y_trainval
        X_val, y_val = [], []
    return X_train, y_train, X_val, y_val, X_test, y_test

def metrics_table(y_true, y_pred) -> pd.DataFrame:
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return pd.DataFrame([{
        "Accuracy": round(acc, 4),
        "Precision": round(pr, 4),
        "Recall": round(rc, 4),
        "F1": round(f1, 4),
    }])

def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Negative","Positive"], y=["Negative","Positive"],
                    text_auto=True)
    fig.update_layout(title=title, margin=dict(l=0,r=0,t=40,b=0))
    return fig

def plot_roc_pr(y_true, y_score, title_suffix=""):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC={roc_auc:.3f}"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
    roc_fig.update_layout(title=f"ROC Curve {title_suffix}", xaxis_title="FPR", yaxis_title="TPR",
                          margin=dict(l=0,r=0,t=40,b=0))

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR AUC={pr_auc:.3f}"))
    pr_fig.update_layout(title=f"Precision–Recall Curve {title_suffix}",
                         xaxis_title="Recall", yaxis_title="Precision",
                         margin=dict(l=0,r=0,t=40,b=0))
    return roc_fig, pr_fig

# ---------- Data Ingestion ----------
with st.container():
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        st.subheader("📥 Data")
        if source == "IMDB (Hugging Face)":
            with st.spinner("Loading IMDB from 🤗 Datasets…"):
                train_df, test_df = load_imdb_dataframe(sample_size, random_state)
                df = pd.concat([train_df, test_df], ignore_index=True)
        else:
            file = st.file_uploader("Upload a CSV (must include text + label columns)", type=["csv"])
            if file:
                df = load_csv_dataframe(file, text_col_name, label_col_name)
            else:
                st.stop()

        st.write(f"Loaded **{len(df):,}** rows.")
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)

    with c2:
        st.subheader("🔖 Labels")
        label_counts = df["label"].value_counts().sort_index()
        st.plotly_chart(px.bar(label_counts, title="Label Distribution", labels={"value":"Count","index":"Label"}),
                        use_container_width=True)

    with c3:
        st.subheader("ℹ️ Notes")
        st.caption("• IMDB labels: 0=Negative, 1=Positive")
        st.caption("• Use the sidebar to change preprocessing & models")
        st.caption("• BERT uses a subset (configurable) for speed")

# ---------- Preprocessing ----------
with st.container():
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("🧼 Preprocessing")

    # Initialize spaCy + stopwords on demand
    if use_spacy and not SPACY_OK:
        init_spacy_stopwords(remove_stop)

    return_tokens = (output_mode == "Word2Vec (tokens)")
    st.write("Applying preprocessing…")
    pb = st.progress(0)

    cleaned = []
    step = max(1, len(df)//50)
    for i, t in enumerate(df["text"].tolist()):
        cleaned.append(
            clean_text(
                t,
                use_spacy_tok=use_spacy and SPACY_OK,
                remove_stopwords=remove_stop,
                remove_numbers=remove_nums,
                min_len=min_token_len,
                return_tokens=return_tokens
            )
        )
        if i % step == 0:
            pb.progress(min(100, int(i/len(df)*100)))

    pb.progress(100)
    df_proc = pd.DataFrame({"text": cleaned, "label": df["label"].astype(int)})
    st.success("Preprocessing complete.")
    st.dataframe(df_proc.head(8), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Split ----------
X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(df_proc, test_size, val_size, random_state)

# ---------- Tabs for Models ----------
tabs = st.tabs(["TF-IDF + LR", "Word2Vec + RF", "BERT + LR", "📊 Compare & Analyze"])
results_rows = []

# ===== TF-IDF =====
with tabs[0]:
    if not run_tfidf:
        st.info("Enable TF-IDF in the sidebar to run.")
    else:
        st.subheader("TF-IDF + Logistic Regression")
        with st.spinner("Vectorizing with TF-IDF…"):
            vectorizer = TfidfVectorizer(max_features=10000)
            Xtr = vectorizer.fit_transform(X_train if isinstance(X_train[0], str) else [" ".join(x) for x in X_train])
            Xte = vectorizer.transform(X_test if isinstance(X_test[0], str) else [" ".join(x) for x in X_test])

        with st.spinner("Training Logistic Regression…"):
            lr = LogisticRegression(max_iter=300, n_jobs=None if hasattr(LogisticRegression,'n_jobs') else None)
            lr.fit(Xtr, y_train)
            ypred = lr.predict(Xte)
            if hasattr(lr, "predict_proba"):
                yscore = lr.predict_proba(Xte)[:,1]
            else:
                # fall back to decision function if not available
                yscore = lr.decision_function(Xte)

        mt = metrics_table(y_test, ypred)
        st.markdown("### Metrics")
        st.dataframe(mt, use_container_width=True, hide_index=True)

        cm_fig = plot_confusion(y_test, ypred, "TF-IDF Confusion Matrix")
        st.plotly_chart(cm_fig, use_container_width=True)

        roc_fig, pr_fig = plot_roc_pr(y_test, yscore, "(TF-IDF)")
        c1, c2 = st.columns(2)
        c1.plotly_chart(roc_fig, use_container_width=True)
        c2.plotly_chart(pr_fig, use_container_width=True)

        # Top features (most influential)
        try:
            coef = lr.coef_[0]
            feats = np.array(vectorizer.get_feature_names_out())
            top_pos_idx = np.argsort(coef)[-20:][::-1]
            top_neg_idx = np.argsort(coef)[:20]
            top_df = pd.DataFrame({
                "Positive features": feats[top_pos_idx],
                "Coef(+)": coef[top_pos_idx],
                "Negative features": feats[top_neg_idx],
                "Coef(-)": coef[top_neg_idx],
            })
            st.markdown("### Top Features")
            st.dataframe(top_df, use_container_width=True, hide_index=True)
        except Exception:
            pass

        acc = float(mt["Accuracy"][0])
        prc = float(mt["Precision"][0]); rec = float(mt["Recall"][0]); f1 = float(mt["F1"][0])
        results_rows.append(("TF-IDF + LR", acc, prc, rec, f1))

# ===== Word2Vec =====
with tabs[1]:
    if not run_w2v:
        st.info("Enable Word2Vec in the sidebar to run.")
    else:
        st.subheader("Word2Vec + Random Forest")
        # Ensure token lists
        X_train_tok = X_train if isinstance(X_train[0], list) else [x.split() for x in X_train]
        X_test_tok  = X_test  if isinstance(X_test[0], list)  else [x.split() for x in X_test]

        with st.spinner("Training Word2Vec…"):
            w2v = Word2Vec(sentences=X_train_tok, vector_size=100, window=5, min_count=2, workers=4, seed=random_state)

        def sent_embed(tokens: List[str], model: Word2Vec) -> np.ndarray:
            pool = [model.wv[w] for w in tokens if w in model.wv]
            return np.mean(pool, axis=0) if len(pool) else np.zeros(model.vector_size)

        with st.spinner("Embedding & training Random Forest…"):
            Xtr_w2v = np.vstack([sent_embed(t, w2v) for t in X_train_tok])
            Xte_w2v = np.vstack([sent_embed(t, w2v) for t in X_test_tok])

            rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
            rf.fit(Xtr_w2v, y_train)
            ypred = rf.predict(Xte_w2v)
            if hasattr(rf, "predict_proba"):
                yscore = rf.predict_proba(Xte_w2v)[:,1]
            else:
                # probability not available
                yscore = (ypred.astype(float))

        mt = metrics_table(y_test, ypred)
        st.markdown("### Metrics")
        st.dataframe(mt, use_container_width=True, hide_index=True)

        cm_fig = plot_confusion(y_test, ypred, "Word2Vec Confusion Matrix")
        st.plotly_chart(cm_fig, use_container_width=True)

        roc_fig, pr_fig = plot_roc_pr(y_test, yscore, "(Word2Vec)")
        c1, c2 = st.columns(2)
        c1.plotly_chart(roc_fig, use_container_width=True)
        c2.plotly_chart(pr_fig, use_container_width=True)

        acc = float(mt["Accuracy"][0])
        prc = float(mt["Precision"][0]); rec = float(mt["Recall"][0]); f1 = float(mt["F1"][0])
        results_rows.append(("Word2Vec + RF", acc, prc, rec, f1))

# ===== BERT =====
with tabs[2]:
    if not run_bert:
        st.info("Enable BERT in the sidebar to run.")
    else:
        st.subheader("BERT (CLS embeddings) + Logistic Regression")
        st.caption("Runs on a subset for speed (configurable in sidebar).")
        # Lazy import for transformers
        try:
            from transformers import BertTokenizer, TFBertModel
            import tensorflow as tf
            BERT_OK = True
        except Exception as e:
            BERT_OK = False
            st.warning(f"Transformers/TensorFlow not available: {e}")

        if BERT_OK:
            # Prepare cleaned strings for BERT (if word tokens, join back)
            X_train_str = X_train if isinstance(X_train[0], str) else [" ".join(x) for x in X_train]
            X_test_str  = X_test  if isinstance(X_test[0], str)  else [" ".join(x) for x in X_test]

            sub_tr = min(subset_train_bert, len(X_train_str))
            sub_te = min(subset_test_bert, len(X_test_str))

            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            bert = TFBertModel.from_pretrained("bert-base-uncased")

            @st.cache_resource(show_spinner=False)
            def encode_batch(texts: List[str], max_len: int):
                return tokenizer(texts, padding=True, truncation=True, return_tensors="tf", max_length=max_len)

            with st.spinner("Encoding with BERT tokenizer…"):
                enc_tr = encode_batch(X_train_str[:sub_tr], max_len_bert)
                enc_te = encode_batch(X_test_str[:sub_te], max_len_bert)
                ytr = np.array(y_train[:sub_tr])
                yte = np.array(y_test[:sub_te])

            with st.spinner("Extracting CLS embeddings…"):
                tr_emb = bert(enc_tr).last_hidden_state[:,0,:].numpy()
                te_emb = bert(enc_te).last_hidden_state[:,0,:].numpy()

            with st.spinner("Training Logistic Regression on embeddings…"):
                lr = LogisticRegression(max_iter=300)
                lr.fit(tr_emb, ytr)
                ypred = lr.predict(te_emb)
                if hasattr(lr, "predict_proba"):
                    yscore = lr.predict_proba(te_emb)[:,1]
                else:
                    yscore = lr.decision_function(te_emb)

            mt = metrics_table(yte, ypred)
            st.markdown("### Metrics")
            st.dataframe(mt, use_container_width=True, hide_index=True)

            cm_fig = plot_confusion(yte, ypred, "BERT Confusion Matrix")
            st.plotly_chart(cm_fig, use_container_width=True)

            roc_fig, pr_fig = plot_roc_pr(yte, yscore, "(BERT)")
            c1, c2 = st.columns(2)
            c1.plotly_chart(roc_fig, use_container_width=True)
            c2.plotly_chart(pr_fig, use_container_width=True)

            acc = float(mt["Accuracy"][0])
            prc = float(mt["Precision"][0]); rec = float(mt["Recall"][0]); f1 = float(mt["F1"][0])
            results_rows.append(("BERT + LR", acc, prc, rec, f1))

# ===== Compare & Analyze =====
with tabs[3]:
    st.subheader("📊 Results & Analysis")
    if results_rows:
        res_df = pd.DataFrame(results_rows, columns=["Model","Accuracy","Precision","Recall","F1"])
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        # Radar chart of metrics
        try:
            melt = res_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
            radar = px.line_polar(melt, r="Score", theta="Metric", color="Model", line_close=True,
                                  range_r=[0,1], title="Metric Radar")
            st.plotly_chart(radar, use_container_width=True)
        except Exception:
            pass

        # Download button
        csv = res_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results CSV", csv, file_name="results_summary.csv", mime="text/csv")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.subheader("🔍 Error Analysis")
    show_k = st.slider("Show K misclassified examples", 5, 50, 10, step=5)
    which = st.selectbox("Choose model for error analysis", [r[0] for r in results_rows] if results_rows else [])
    if which:
        # Recompute predictions for selected (kept minimal to avoid storing all preds)
        if which.startswith("TF-IDF"):
            vectorizer = TfidfVectorizer(max_features=10000)
            Xtr = vectorizer.fit_transform(X_train if isinstance(X_train[0], str) else [" ".join(x) for x in X_train])
            Xte = vectorizer.transform(X_test if isinstance(X_test[0], str) else [" ".join(x) for x in X_test])
            lr = LogisticRegression(max_iter=300)
            lr.fit(Xtr, y_train)
            ypred = lr.predict(Xte)
            texts = X_test if isinstance(X_test[0], str) else [" ".join(x) for x in X_test]
        elif which.startswith("Word2Vec"):
            X_train_tok = X_train if isinstance(X_train[0], list) else [x.split() for x in X_train]
            X_test_tok  = X_test  if isinstance(X_test[0], list)  else [x.split() for x in X_test]
            w2v = Word2Vec(sentences=X_train_tok, vector_size=100, window=5, min_count=2, workers=2, seed=random_state)
            def sent_embed(tokens, model): 
                pool = [model.wv[w] for w in tokens if w in model.wv]
                return np.mean(pool, axis=0) if len(pool) else np.zeros(model.vector_size)
            Xtr_w2v = np.vstack([sent_embed(t, w2v) for t in X_train_tok])
            Xte_w2v = np.vstack([sent_embed(t, w2v) for t in X_test_tok])
            rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
            rf.fit(Xtr_w2v, y_train)
            ypred = rf.predict(Xte_w2v)
            texts = [" ".join(t) for t in X_test_tok]
        else:
            st.info("For BERT, run detailed error analysis in the BERT tab to avoid heavy recompute.")
            st.stop()

        mism_idx = [i for i,(yt,yp) in enumerate(zip(y_test, ypred)) if yt!=yp]
        k = min(show_k, len(mism_idx))
        if k == 0:
            st.success("No misclassifications found in the first pass 🎉")
        else:
            sample_idx = mism_idx[:k]
            err_df = pd.DataFrame({
                "True": [y_test[i] for i in sample_idx],
                "Pred": [ypred[i] for i in sample_idx],
                "Text": [texts[i] for i in sample_idx]
            })
            st.dataframe(err_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Tip: Switch between preprocessing modes (string/tokens) to compare TF-IDF vs Word2Vec fairly. Use BERT when accuracy matters most and you can afford the runtime.")
