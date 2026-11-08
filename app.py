
import os
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="BeanVision | ML Studio",
    page_icon="🌱",
    layout="wide"
)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

TARGET = "Class"

# ---------------- CUSTOM STYLING ----------------

st.markdown("""
<style>
.stApp {
    background-color: #F4F7FA;
}
h1, h2, h3 {
    color: #17324D;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #DCE5EC;
}
[data-testid="stSidebar"] {
    background-color: #E8F0F5;
}
.stButton button {
    background-color: #087F8C;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD SAVED MODELS ----------------

@st.cache_resource
def load_models():
    loaded = {}

    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {filename}"
            )

        loaded[name] = joblib.load(path)

    return loaded


@st.cache_data
def load_features():
    path = os.path.join(MODEL_DIR, "feature_columns.json")

    with open(path, "r") as file:
        return json.load(file)["features"]


# ---------------- DATA PREPARATION ----------------

def prepare_data(data, features):

    required_columns = features + [TARGET]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        return None, None, missing

    data = data[required_columns].copy()

    if data.isnull().any().any():
        return None, None, ["Missing values found in uploaded data"]

    try:
        X = data[features].astype(float)
    except (ValueError, TypeError):
        return None, None, ["Input features must be numerical"]

    y = data[TARGET]

    return X, y, []


# ---------------- MODEL EVALUATION ----------------

def evaluate_model(model, X, y):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    classes = model.classes_

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(
            y,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="macro"
        ),
        "Precision": precision_score(
            y, predictions,
            average="macro",
            zero_division=0
        ),
        "Recall": recall_score(
            y, predictions,
            average="macro",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y, predictions,
            average="macro",
            zero_division=0
        ),
        "MCC": matthews_corrcoef(y, predictions)
    }

    return predictions, metrics


# ---------------- HEADER ----------------

st.title("🌱 BeanVision")
st.caption(
    "MACHINE LEARNING CLASSIFICATION STUDIO | "
    "Dry Bean Variety Prediction"
)

st.markdown("""
Evaluate five machine learning algorithms using physical
characteristics of dry beans and explore their classification
performance across seven varieties.
""")

# ---------------- LOAD RESOURCES ----------------

try:
    models = load_models()
    features = load_features()

except Exception as error:
    st.error(f"Unable to load application resources: {error}")
    st.stop()


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("⚙️ Analysis Controls")

    uploaded_file = st.file_uploader(
        "Upload test dataset",
        type=["csv"]
    )

    sample_path = os.path.join(ROOT, "test_data.csv")

    if os.path.exists(sample_path):
        with open(sample_path, "rb") as file:
            st.download_button(
                "⬇ Download sample CSV",
                file,
                file_name="test_data.csv",
                mime="text/csv"
            )

    st.divider()

    selected_model = st.selectbox(
        "Select ML Algorithm",
        list(MODEL_FILES.keys())
    )

    st.caption("Target column: Class")

    st.divider()

    st.info(
        "Models are pre-trained. "
        "Uploading data does not trigger retraining."
    )


# ---------------- UPLOAD VALIDATION ----------------

if uploaded_file is None:

    st.info(
        "Please upload a test CSV file using the sidebar "
        "to start the classification analysis."
    )

    with st.expander("Required dataset columns"):
        st.write(features + [TARGET])

    st.stop()


try:
    raw_data = pd.read_csv(uploaded_file)

except Exception as error:
    st.error(f"Unable to read CSV: {error}")
    st.stop()


X, y, missing = prepare_data(raw_data, features)

if missing:
    st.error(f"Dataset validation failed: {missing}")
    st.stop()


# ---------------- DATASET OVERVIEW ----------------

st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Test Records", f"{len(X):,}")
col2.metric("Input Features", X.shape[1])
col3.metric("Bean Varieties", y.nunique())

with st.expander("Preview uploaded dataset"):
    st.dataframe(raw_data.head(10), use_container_width=True)


# ---------------- TABS ----------------

tab1, tab2, tab3 = st.tabs([
    "🔍 Model Evaluation",
    "📈 Model Comparison",
    "📋 Dataset Insights"
])


# ---------------- TAB 1: MODEL EVALUATION ----------------

with tab1:

    model = models[selected_model]

    predictions, metrics = evaluate_model(model, X, y)

    st.subheader(f"Performance: {selected_model}")

    metric_cols = st.columns(6)

    for col, (name, value) in zip(
        metric_cols, metrics.items()
    ):
        col.metric(name, f"{value:.4f}")

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y,
            predictions,
            labels=model.classes_
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            ax=ax
        )

        ax.set_xlabel("Predicted Variety")
        ax.set_ylabel("Actual Variety")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with right:

        st.subheader("Classification Report")

        report = classification_report(
            y,
            predictions,
            labels=model.classes_,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).T

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

    # Download predictions

    output = raw_data.copy()
    output["Predicted_Class"] = predictions

    st.download_button(
        "⬇ Download Prediction Results",
        output.to_csv(index=False),
        file_name="dry_bean_predictions.csv",
        mime="text/csv"
    )


# ---------------- TAB 2: MODEL COMPARISON ----------------

with tab2:

    st.subheader("Compare All Five Models")

    comparison = []

    for name, model in models.items():

        _, result = evaluate_model(model, X, y)

        comparison.append({
            "Model": name,
            **result
        })

    comparison_df = pd.DataFrame(comparison)

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    selected_metric = st.selectbox(
        "Select metric for comparison",
        [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=comparison_df,
        x="Model",
        y=selected_metric,
        hue="Model",
        palette="crest",
        legend=False,
        ax=ax
    )

    ax.set_title(f"Model Comparison - {selected_metric}")
    ax.set_xlabel("")
    ax.set_ylabel(selected_metric)
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    best_model = comparison_df.loc[
        comparison_df[selected_metric].idxmax()
    ]

    st.success(
        f"Best model for {selected_metric}: "
        f"{best_model['Model']} "
        f"({best_model[selected_metric]:.4f})"
    )


# ---------------- TAB 3: DATASET INSIGHTS ----------------

with tab3:

    st.subheader("Bean Variety Distribution")

    distribution = y.value_counts()

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.barplot(
        x=distribution.index,
        y=distribution.values,
        hue=distribution.index,
        palette="crest",
        legend=False,
        ax=ax
    )

    ax.set_xlabel("Bean Variety")
    ax.set_ylabel("Number of Records")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        distribution.rename("Count").to_frame(),
        use_container_width=True
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "BeanVision | Dry Bean Classification | "
    "BITS Pilani WILP Machine Learning Assignment 2"
)
