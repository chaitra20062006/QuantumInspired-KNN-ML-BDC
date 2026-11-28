"""Results & Comparison page — loads saved CSV results."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULT_FILES = {
    "Pima Diabetes": os.path.join(PROJECT_ROOT, "results", "pima", "pima_results.csv"),
    "Heart Disease":  os.path.join(PROJECT_ROOT, "results", "heart", "full_results.csv"),
    "Breast Cancer":  os.path.join(PROJECT_ROOT, "results", "full_results.csv"),
}

METHOD_ORDER = ["KNN Baseline", "KNN+GA", "KNN+PSO", "QIEA-KNN", "QIEA-Fuzzy-KNN"]
COLORS       = ["#78909C", "#FF7043", "#AB47BC", "#42A5F5", "#66BB6A"]


def render():
    st.title("📈 Results & Comparison")

    dataset_name = st.selectbox("Choose Results Dataset", list(RESULT_FILES.keys()))
    csv_path = RESULT_FILES[dataset_name]

    if not os.path.exists(csv_path):
        st.warning(
            f"Results file not found at `{csv_path}`.\n\n"
            "Run the pipeline first:\n"
            "```\npython scripts/run_pipeline_c.py --seeds 10 --pop 20 --gens 20\n```"
        )
        return

    df = pd.read_csv(csv_path)
    methods_in_data = [m for m in METHOD_ORDER if m in df["method"].unique()]
    if not methods_in_data:
        methods_in_data = df["method"].unique().tolist()

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Mean ± Std Performance Table")

    metrics = ["f1", "accuracy", "precision", "recall"]
    rows = []
    for m in methods_in_data:
        sub = df[df["method"] == m]
        row = {"Method": m}
        for met in metrics:
            if met in sub.columns:
                row[met.upper()] = f"{sub[met].mean():.4f} ± {sub[met].std():.4f}"
        if "n_features" in sub.columns:
            row["# Features"] = f"{sub['n_features'].mean():.1f}"
        if "k" in sub.columns:
            row["K"] = f"{sub['k'].mean():.1f}"
        rows.append(row)

    st.dataframe(pd.DataFrame(rows).set_index("Method"), use_container_width=True)

    # ── F1 bar chart ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("F1 Score Comparison")

    if "f1" in df.columns:
        means = [df[df["method"] == m]["f1"].mean() for m in methods_in_data]
        stds  = [df[df["method"] == m]["f1"].std()  for m in methods_in_data]
        fig = go.Figure()
        for i, (m, mean, std) in enumerate(zip(methods_in_data, means, stds)):
            fig.add_trace(go.Bar(
                name=m, x=[m], y=[mean],
                error_y=dict(type="data", array=[std], visible=True),
                marker_color=COLORS[i % len(COLORS)],
            ))
        fig.update_layout(
            title="Mean Test F1 Score (± std across seeds)",
            yaxis_title="F1 Score", showlegend=False,
            yaxis=dict(range=[max(0, min(means) - 0.05), 1.0]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Box plots ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Distribution of F1 Scores Across Seeds")
    if "f1" in df.columns:
        fig2 = px.box(
            df[df["method"].isin(methods_in_data)],
            x="method", y="f1", color="method",
            category_orders={"method": methods_in_data},
            color_discrete_sequence=COLORS,
            title="F1 Score Box Plot (each dot = one seed)",
            points="all",
        )
        fig2.update_layout(showlegend=False, xaxis_title="Method", yaxis_title="F1 Score")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Feature reduction ─────────────────────────────────────────────────────
    if "n_features" in df.columns:
        st.markdown("---")
        st.subheader("Feature Reduction")
        fig3 = px.box(
            df[df["method"].isin(methods_in_data)],
            x="method", y="n_features", color="method",
            category_orders={"method": methods_in_data},
            color_discrete_sequence=COLORS,
            title="Features Selected by Each Method",
            points="all",
        )
        fig3.update_layout(showlegend=False, xaxis_title="Method", yaxis_title="# Features")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Raw data table ────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📄 View Raw Results CSV"):
        st.dataframe(df, use_container_width=True)
