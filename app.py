import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Heart Disease ML Benchmark", layout="wide")

st.title("📊 Multi-Model Classification Dashboard")
st.write("Upload test dataset CSV, select a model, and evaluate performance without retraining.")

st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload 'test_data.csv'", type=["csv"])

st.sidebar.header("2. Select Model")
model_options = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Support Vector Machine"
]
selected_model_name = st.sidebar.selectbox("Choose Classification Model", model_options)

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    
    if "target" not in df_test.columns:
        st.error("Uploaded CSV must contain the 'target' column.")
    else:
        X_test = df_test.drop(columns=["target"])
        y_test = df_test["target"]
        
        model_filename = selected_model_name.replace(" ", "_") + ".joblib"
        model_path = os.path.join("model", model_filename)
        scaler_path = os.path.join("model", "scaler.joblib")
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            if selected_model_name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
                scaler = joblib.load(scaler_path)
                X_eval = scaler.transform(X_test)
            else:
                X_eval = X_test

            y_pred = model.predict(X_eval)
            y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)

            st.subheader(f"⚡ Performance Metrics for {selected_model_name}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")

            st.markdown("---")
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("📌 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)

            with col_right:
                st.subheader("📋 Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report_dict).transpose().style.highlight_max(axis=0))
        else:
            st.error(f"Model file {model_path} not found.")
else:
    st.info("👈 Please upload `test_data.csv` from the sidebar to inspect predictions.")
