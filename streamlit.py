import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import altair as alt

# Dataset URL
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"

# Metrics file path (optional, for deployment)
METRICS_PATH = 'metrics.json'  # Relative path for Streamlit Cloud

# Cache dataset loading to improve performance
@st.cache_data
def load_data():
    names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(DATASET_URL, names=names)
    return data

# Cache model training to avoid retraining on every interaction
@st.cache_resource
def train_models():
    data = load_data()
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=200),
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'DecisionTree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier()
    }

    best_model = None
    best_f1 = 0
    best_name = None
    metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        metrics[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    # Save metrics to file (optional, for persistence)
    try:
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f)
    except Exception as e:
        st.warning(f"Could not save metrics to {METRICS_PATH}: {str(e)}")

    return best_model, best_name, metrics, X_train, X_test, y_train, y_test

# Load data and train model
try:
    data = load_data()
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    best_model, best_model_name, metrics, X_train, X_test, y_train, y_test = train_models()
except Exception as e:
    st.error(f"Failed to load data or train model: {str(e)}")
    st.stop()

# Load metrics from file ifSamir
try:
    with open(METRICS_PATH, 'r') as f:
        saved_metrics = json.load(f)
    metrics.update(saved_metrics)
except FileNotFoundError:
    st.warning(f"Metrics file not found at {METRICS_PATH}. Using computed metrics.")

# App Title and Description
st.title("Advanced Diabetes Prediction System")
st.markdown("""
This application predicts diabetes risk using a machine learning model trained on the Pima Indians Diabetes Dataset. 
Enter patient details to get a prediction, explore dataset statistics, view model performance, and learn about diabetes risk factors.
""")

# Sidebar for Navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Choose a section to explore:")
page = st.sidebar.radio("", ["Prediction", "Dataset Overview", "Model Metrics", "Visualizations", "Risk Factors & Education"])

# Prediction Page
if page == "Prediction":
    st.header("Predict Diabetes Risk")
    st.markdown("""
    Enter patient details below to predict the likelihood of diabetes. The model provides a prediction and confidence score based on the input features.
    """)

    # Input Form with Columns and Tooltips
    st.subheader("Patient Details")
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, value=3, help="Number of times the patient has been pregnant")
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, value=145, help="Plasma glucose concentration (2-hour oral glucose tolerance test)")
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, value=70, help="Diastolic blood pressure")
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, value=20, help="Triceps skin fold thickness")

    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, value=85, help="2-Hour serum insulin level")
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, value=33.6, help="Body mass index (weight in kg / (height in m)²)")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.35, help="Genetic likelihood of diabetes")
        age = st.number_input("Age (years)", min_value=0, value=29, help="Age of the patient")

    # Predict Button
    if st.button("Predict", key="predict_button"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        try:
            prediction = best_model.predict(input_data)[0]
            confidence = best_model.predict_proba(input_data)[0][prediction]
            result = "Diabetic" if prediction == 1 else "Not Diabetic"

            # Display Results with Styling
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"**{result}** - Confidence: {confidence:.2%}")
                st.markdown("**Recommendation:** Consult a healthcare professional for further evaluation and management.")
            else:
                st.success(f"**{result}** - Confidence: {confidence:.2%}")
                st.markdown("**Recommendation:** Continue a healthy lifestyle to minimize risk.")

            # Radar Chart for Input vs. Dataset Average
            st.subheader("Your Input vs. Dataset Averages")
            avg_data = X.mean().values
            features = X.columns
            user_input = input_data[0]

            # Normalize for radar chart
            max_vals = np.maximum(avg_data, user_input)
            norm_avg = avg_data / max_vals
            norm_user = user_input / max_vals

            radar_df = pd.DataFrame({
                'Feature': features,
                'Average': norm_avg,
                'User': norm_user
            }).melt(id_vars='Feature')

            chart = alt.Chart(radar_df).mark_line(point=True).encode(
                theta=alt.Theta(field="Feature", type="nominal"),
                radius=alt.Radius(field="value", scale=alt.Scale(type="linear", zero=True, rangeMin=0)),
                color=alt.Color(field="variable", type="nominal", legend=alt.Legend(title="Data Source")),
                order=alt.Order("variable", sort="ascending")
            ).properties(
                title="Input Features vs. Dataset Averages (Normalized)",
                width=400,
                height=400
            )
            st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Dataset Overview Page
elif page == "Dataset Overview":
    st.header("Pima Indians Diabetes Dataset")
    st.markdown("""
    The dataset contains medical records of 768 female patients of Pima Indian heritage, aged 21 and above.
    It is used to predict the onset of diabetes within 5 years based on diagnostic measurements.
    """)

    # Sample Data
    st.subheader("Sample Data")
    st.dataframe(data.head(10), use_container_width=True)

    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe(), use_container_width=True)

    # Class Distribution
    st.subheader("Diabetes Outcome Distribution")
    outcome_counts = data['Outcome'].value_counts(normalize=True) * 100
    outcome_df = pd.DataFrame({
        'Outcome': ['Non-Diabetic', 'Diabetic'],
        'Percentage': [outcome_counts[0], outcome_counts[1]]
    })
    chart = alt.Chart(outcome_df).mark_bar().encode(
        x=alt.X('Outcome', title='Outcome'),
        y=alt.Y('Percentage', title='Percentage (%)'),
        color=alt.Color('Outcome', scale=alt.Scale(scheme='set2'))
    ).properties(
        title="Distribution of Diabetes Outcomes",
        width=400,
        height=300
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown(f"**Non-Diabetic:** {outcome_counts[0]:.2f}% | **Diabetic:** {outcome_counts[1]:.2f}%")

# Model Metrics Page
elif page == "Model Metrics":
    st.header("Model Performance Metrics")
    st.markdown("""
    The model was trained and evaluated using multiple classifiers. Below are the performance metrics (accuracy, precision, recall, F1 score) on the test set.
    """)

    if metrics:
        # Display Metrics in Table
        metrics_df = pd.DataFrame(metrics).T
        st.subheader("Metrics for All Models")
        st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

        # Highlight Best Model
        st.subheader(f"Best Model: {best_model_name}")
        st.markdown(f"**F1 Score:** {metrics[best_model_name]['f1_score']:.4f} | **Accuracy:** {metrics[best_model_name]['accuracy']:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix for Best Model")
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

        # ROC Curve
        st.subheader("ROC Curve")
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)
    else:
        st.error("Metrics data unavailable. Model metrics were computed but not saved.")

# Visualizations Page
elif page == "Visualizations":
    st.header("Interactive Data Visualizations")
    st.markdown("Explore feature distributions, correlations, and feature importance to understand the dataset and model.")

    # Feature Histograms
    st.subheader("Feature Distributions by Outcome")
    feature = st.selectbox("Select Feature", X.columns, key="feature_select")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=data, x=feature, hue='Outcome', kde=True, ax=ax, palette='Set2')
    ax.set_title(f'Distribution of {feature}')
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, cbar=True)
    ax.set_title('Correlation Between Features')
    st.pyplot(fig)

    # Feature Importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importances = best_model.feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
        chart = alt.Chart(feat_imp_df).mark_bar().encode(
            x=alt.X('Importance', title='Importance'),
            y=alt.Y('Feature', title='Feature', sort='-x'),
            color=alt.Color('Importance', scale=alt.Scale(scheme='viridis'))
        ).properties(
            title="Feature Importance for Prediction",
            width=400,
            height=300
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Feature importance is not available for this model type.")

# Risk Factors & Education Page
elif page == "Risk Factors & Education":
    st.header("Diabetes Risk Factors & Education")
    st.markdown("""
    Understanding diabetes risk factors and prevention strategies is crucial for managing health. Below are key insights and resources.
    """)

    # Risk Factors
    st.subheader("Key Risk Factors")
    st.markdown("""
    - **High Glucose Levels:** Elevated blood sugar is a strong indicator of diabetes risk.
    - **Obesity (High BMI):** Increases insulin resistance, a precursor to Type 2 diabetes.
    - **Age:** Risk increases significantly after age 45.
    - **Family History (DPF):** Genetic predisposition, as measured by the Diabetes Pedigree Function.
    - **Sedentary Lifestyle:** Inferred from BMI and insulin levels; lack of exercise increases risk.
    """)

    # Prevention Tips
    st.subheader("Prevention Tips")
    st.markdown("""
    - **Healthy Diet:** Focus on whole grains, vegetables, lean proteins, and low sugar intake.
    - **Regular Exercise:** Aim for at least 150 minutes of moderate aerobic activity per week.
    - **Weight Management:** Maintain a healthy BMI through diet and exercise.
    - **Regular Check-ups:** Monitor blood pressure, cholesterol, and glucose levels.
    - **Avoid Smoking:** Smoking increases diabetes complications.
    """)

    # Educational Resources
    with st.expander("Learn More About Diabetes"):
        st.markdown("""
        - **Type 1 Diabetes:** Autoimmune condition, typically diagnosed in childhood or adolescence.
        - **Type 2 Diabetes:** Most common, linked to lifestyle and genetics; often manageable with lifestyle changes.
        - **Gestational Diabetes:** Occurs during pregnancy and may increase future Type 2 risk.
        
        **Resources:**
        - [American Diabetes Association](https://www.diabetes.org/)
        - [CDC Diabetes Information](https://www.cdc.gov/diabetes/index.html)
        - Consult a healthcare provider for personalized advice.
        """)

# Footer
st.markdown("---")
st.markdown("""
**Developed with Streamlit** | Data Source: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | Model Trained with scikit-learn
""")
