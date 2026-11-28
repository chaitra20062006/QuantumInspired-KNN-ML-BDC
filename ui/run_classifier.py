"""Run Classifier page — live QIEA-KNN prediction with controls."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.interfaces import bind_fitness
from person_c.data_loader import load_dataset
from person_b.classifier import predict
from person_b.fitness import fitness
from person_a import qiea, ga
from person_b.pso import run as pso_run


DATASETS = {
    "Pima Indians Diabetes": "pima",
    "Heart Disease (Cleveland)": "heart",
    "Breast Cancer Wisconsin": "breast_cancer",
}

METHODS = {
    "QIEA-KNN (Quantum)": "qiea",
    "QIEA Fuzzy-KNN": "qiea_fuzzy",
    "Genetic Algorithm (GA)": "ga",
    "PSO": "pso",
    "Baseline KNN (All features, K=5)": "baseline",
}


def render():
    st.title("⚙️ Run Classifier")
    st.caption("Run QIEA-KNN live on a selected dataset and see predictions in real time.")

    st.markdown("---")
    st.subheader("1️⃣ Choose Data Source")
    data_source = st.radio("", ["Use a built-in dataset", "Upload your own CSV"], horizontal=True)

    X_tr = X_te = y_tr = y_te = folds = feat_names = None

    if data_source == "Use a built-in dataset":
        dataset_name = st.selectbox("Dataset", list(DATASETS.keys()))
        key = DATASETS[dataset_name]
        seed = st.number_input("Random Seed", min_value=0, max_value=999, value=42, step=1)
        with st.spinner("Loading and preprocessing..."):
            X_tr, X_te, y_tr, y_te, folds = load_dataset(key, seed=int(seed))
        st.success(f"Loaded {len(y_tr) + len(y_te)} samples, {X_tr.shape[1]} features.")
        feat_names = [f"Feature {i+1}" for i in range(X_tr.shape[1])]

    else:
        uploaded = st.file_uploader("Upload CSV (last column = target label)", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head(), use_container_width=True)
            X_raw = df.iloc[:, :-1].values.astype(float)
            y_raw = df.iloc[:, -1].values.astype(int)
            feat_names = list(df.columns[:-1])
            from sklearn.model_selection import train_test_split, StratifiedKFold
            from sklearn.preprocessing import StandardScaler
            X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.25,
                                                        stratify=y_raw, random_state=42)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X_tr, y_tr))
            st.success(f"Loaded {len(y_raw)} samples, {X_tr.shape[1]} features.")
        else:
            st.info("Upload a CSV file to continue.")

    if X_tr is None:
        return

    # ── Algorithm settings ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("2️⃣ Configure Algorithm")

    col1, col2 = st.columns(2)
    with col1:
        method_name = st.selectbox("Search Method", list(METHODS.keys()))
        method = METHODS[method_name]
    with col2:
        fuzzy = "fuzzy" in method.lower()
        st.info(f"**Fuzzy mode:** {'ON 🟢' if fuzzy else 'OFF 🔴'}")

    col3, col4, col5 = st.columns(3)
    with col3:
        pop_size = st.slider("Population Size", 5, 50, 10, 5)
    with col4:
        generations = st.slider("Generations", 5, 50, 10, 5)
    with col5:
        run_seed = st.number_input("Algorithm Seed", 0, 999, 0, 1)

    n_features = X_tr.shape[1]
    st.caption(f"Total fitness evaluations: **{pop_size * generations}**")

    # ── Run ───────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("3️⃣ Run")

    if st.button("🚀 Run Classifier", type="primary", use_container_width=True):
        with st.spinner(f"Running {method_name}... This may take a moment."):
            t0 = time.time()
            fn = bind_fitness(fitness, X_tr, y_tr, folds)

            if method == "baseline":
                mask = np.ones(n_features, dtype=bool)
                best_k = 5
                history = []
            elif method in ("qiea", "qiea_fuzzy"):
                mask, best_k, history = qiea.run(fn, n_features, generations, int(run_seed), pop_size=pop_size)
            elif method == "ga":
                mask, best_k, history = ga.run(fn, n_features, generations, int(run_seed), pop_size=pop_size)
            elif method == "pso":
                mask, best_k, history = pso_run(fn, n_features, generations, int(run_seed), pop_size=pop_size)

            labels, memberships = predict(X_tr, y_tr, X_te, mask, best_k, fuzzy=fuzzy)
            elapsed = time.time() - t0

        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        avg = "binary" if len(np.unique(y_te)) == 2 else "weighted"
        f1   = f1_score(y_te, labels, average=avg, zero_division=0)
        acc  = accuracy_score(y_te, labels)
        prec = precision_score(y_te, labels, average=avg, zero_division=0)
        rec  = recall_score(y_te, labels, average=avg, zero_division=0)

        # ── Metrics ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Results")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("F1 Score", f"{f1:.4f}")
        m2.metric("Accuracy", f"{acc:.4f}")
        m3.metric("Precision", f"{prec:.4f}")
        m4.metric("Recall", f"{rec:.4f}")
        m5.metric("⏱️ Time", f"{elapsed:.1f}s")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Features Selected", f"{int(mask.sum())} / {n_features}")
        with col_b:
            st.metric("Optimal K", best_k)

        # ── Feature mask bar chart ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Selected Features")
        fn_list = feat_names if feat_names else [f"F{i}" for i in range(n_features)]
        fn_list = fn_list[:n_features]
        df_mask = pd.DataFrame({"Feature": fn_list, "Selected": mask.astype(int)})
        fig = px.bar(df_mask, x="Feature", y="Selected", color="Selected",
                     color_continuous_scale=["#d9534f", "#5cb85c"],
                     title="Feature Selection Mask (1 = selected, 0 = dropped)")
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # ── Convergence curve ─────────────────────────────────────────────────
        if history:
            st.markdown("---")
            st.subheader("Convergence Curve")
            df_hist = pd.DataFrame(history)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_hist["generation"], y=df_hist["best_fitness"],
                                      name="Best Fitness", line=dict(color="#2196F3", width=2)))
            fig2.add_trace(go.Scatter(x=df_hist["generation"], y=df_hist["mean_fitness"],
                                      name="Mean Fitness", line=dict(color="#FF9800", width=1, dash="dot")))
            fig2.update_layout(xaxis_title="Generation", yaxis_title="Fitness",
                                title="Best & Mean Fitness Per Generation")
            st.plotly_chart(fig2, use_container_width=True)

        # ── Membership confidence ─────────────────────────────────────────────
        if memberships is not None and memberships.ndim == 2 and memberships.shape[1] == 2:
            st.markdown("---")
            st.subheader("Prediction Confidence (Test Set)")
            conf_df = pd.DataFrame({
                "Confidence (Positive Class)": memberships[:, 1],
                "True Label": y_te,
                "Predicted": labels,
            })
            fig3 = px.histogram(conf_df, x="Confidence (Positive Class)",
                                color=conf_df["True Label"].astype(str),
                                nbins=20, barmode="overlay",
                                title="Fuzzy Membership Confidence Distribution",
                                color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig3, use_container_width=True)
