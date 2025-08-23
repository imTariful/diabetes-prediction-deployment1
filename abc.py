# abc.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ============ STREAMLIT APP ============
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

st.title("🩺 Diabetes Prediction with ML")
st.markdown("Upload your dataset, preprocess, train a model, and predict diabetes risk.")

# -------- File Upload --------
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Raw Dataset Preview")
    st.dataframe(df.head())

    # -------- Data Preprocessing --------
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
        text = re.sub(r"\d+", "", text)  # remove numbers
        tokens = text.split()
        tokens = [w for w in tokens if w not in stop_words]
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(tokens)

    st.subheader("⚙️ Data Preprocessing")
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if text_cols:
        st.write(f"Cleaning text columns: {text_cols}")
        for col in text_cols:
            df[col] = df[col].astype(str).apply(clean_text)

    st.success("✅ Preprocessing Completed")
    st.dataframe(df.head())

    # -------- Train/Test Split --------
    st.subheader("📊 Train Model")
    target_col = st.selectbox("Select Target Column", df.columns)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Handle categorical
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric("🎯 Model Accuracy", f"{acc:.2%}")

        # -------- Confusion Matrix --------
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # -------- Feature Importance --------
        feature_importance = pd.DataFrame(
            {"Feature": X.columns, "Importance": model.feature_importances_}
        ).sort_values(by="Importance", ascending=False)

        fig_imp = px.bar(
            feature_importance.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Important Features",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # -------- Make Prediction --------
        st.subheader("🧑‍⚕️ Predict Diabetes")
        user_input = {}
        for col in X.columns:
            val = st.number_input(f"Enter {col}", float(X[col].min()), float(X[col].max()))
            user_input[col] = val

        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Prediction: {prediction}")
