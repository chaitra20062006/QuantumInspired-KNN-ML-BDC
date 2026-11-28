"""Dataset Explorer page."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from person_c.data_loader import load_dataset


DATASETS = {
    "Pima Indians Diabetes": "pima",
    "Heart Disease (Cleveland)": "heart",
    "Breast Cancer Wisconsin": "breast_cancer",
}

FEATURE_NAMES = {
    "pima": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
             "Insulin", "BMI", "DiabetesPedigree", "Age"],
    "heart": ["Age", "Sex", "ChestPain", "RestBP", "Cholesterol", "FastingBS",
              "RestECG", "MaxHR", "ExAngina", "Oldpeak", "Slope", "CA", "Thal"],
    "breast_cancer": [f"Feature {i}" for i in range(1, 31)],
}

CLASS_NAMES = {
    "pima": {0: "No Diabetes", 1: "Diabetes"},
    "heart": {0: "Healthy", 1: "Disease"},
    "breast_cancer": {0: "Benign", 1: "Malignant"},
}


def render():
    st.title("📊 Dataset Explorer")

    dataset_name = st.selectbox("Select a Dataset", list(DATASETS.keys()))
    key = DATASETS[dataset_name]

    with st.spinner("Loading dataset..."):
        X_tr, X_te, y_tr, y_te, folds = load_dataset(key, seed=42)

    X_all = np.vstack([X_tr, X_te])
    y_all = np.concatenate([y_tr, y_te])
    feat_names = FEATURE_NAMES.get(key, [])
    # Always ensure feat_names exactly matches the number of columns
    if len(feat_names) < X_all.shape[1]:
        feat_names = feat_names + [f"Feature {i+1}" for i in range(len(feat_names), X_all.shape[1])]
    feat_names = feat_names[:X_all.shape[1]]
    cls_names = CLASS_NAMES.get(key, {0: "Class 0", 1: "Class 1"})

    # ── Overview metrics ──────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", len(y_all))
    c2.metric("Features", X_all.shape[1])
    c3.metric("Train / Test Split", f"{len(y_tr)} / {len(y_te)}")
    c4.metric("CV Folds", len(folds))

    # ── Class distribution ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Class Distribution")
    unique, counts = np.unique(y_all, return_counts=True)
    labels = [cls_names.get(int(u), str(u)) for u in unique]
    fig = px.pie(values=counts, names=labels, color_discrete_sequence=px.colors.qualitative.Set2,
                 title=f"Class Balance — {dataset_name}")
    st.plotly_chart(fig, use_container_width=True)

    # ── Feature distributions ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Feature Distribution")
    selected_feat = st.selectbox("Choose a feature to inspect", feat_names)
    feat_idx = feat_names.index(selected_feat)

    df_feat = pd.DataFrame({
        selected_feat: X_all[:, feat_idx],
        "Class": [cls_names.get(int(y), str(y)) for y in y_all],
    })
    fig2 = px.histogram(df_feat, x=selected_feat, color="Class", barmode="overlay",
                        nbins=30, color_discrete_sequence=px.colors.qualitative.Set2,
                        title=f"Distribution of {selected_feat}")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Feature Correlation Heatmap")
    df_corr = pd.DataFrame(X_all, columns=feat_names)
    corr = df_corr.corr()
    fig3 = px.imshow(corr, text_auto=".2f", aspect="auto",
                     color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Pearson Correlation Between Features")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Raw data preview ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Raw Data Preview (Training Set)")
    df_preview = pd.DataFrame(X_tr, columns=feat_names)
    df_preview["Label"] = [cls_names.get(int(y), str(y)) for y in y_tr]
    st.dataframe(df_preview.head(50), use_container_width=True)
