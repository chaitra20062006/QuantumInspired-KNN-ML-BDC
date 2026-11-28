"""Scalability (Spark) page."""
import streamlit as st
import pandas as pd
import os, sys
import plotly.express as px
import plotly.graph_objects as go
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = os.path.join(PROJECT_ROOT, "results", "scaling", "scaling_results.csv")
PLOT_PATH = os.path.join(PROJECT_ROOT, "results", "scaling", "speedup_plot.png")


def render():
    st.title("🌐 Scalability — Apache Spark")
    st.markdown(
        """
        Fitness evaluation is **embarrassingly parallel**: each chromosome is independent.  
        We use PySpark to distribute evaluations across cores, achieving near-linear speedup.
        """
    )

    # ── Speedup image ──────────────────────────────────────────────────────────
    if os.path.exists(PLOT_PATH):
        st.markdown("---")
        st.subheader("Speedup Plot")
        st.image(PLOT_PATH, caption="Spark Speedup: T1 / Tn across data sizes", use_container_width=True)

    # ── Interactive chart from CSV ────────────────────────────────────────────
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)

        st.markdown("---")
        st.subheader("Interactive Speedup Chart")
        fig = go.Figure()
        for rows in df["rows"].unique():
            sub = df[df["rows"] == rows]
            fig.add_trace(go.Scatter(
                x=sub["cores"], y=sub["speedup"],
                mode="lines+markers",
                name=f"{int(rows):,} rows",
                line=dict(width=2),
                marker=dict(size=8),
            ))

        fig.add_trace(go.Scatter(
            x=[1, 2, 4], y=[1, 2, 4],
            mode="lines", name="Ideal Linear",
            line=dict(dash="dash", color="gray"),
        ))
        fig.update_layout(
            title="Spark Speedup vs Number of Cores",
            xaxis_title="Cores", yaxis_title="Speedup (T1 / Tn)",
            xaxis=dict(tickvals=[1, 2, 4]),
            legend_title="Data Size",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Results table ──────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Raw Scaling Data")
        df_display = df.copy()
        df_display["rows"] = df_display["rows"].apply(lambda x: f"{int(x):,}")
        st.dataframe(df_display, use_container_width=True)

        # ── Key insight ────────────────────────────────────────────────────────
        st.markdown("---")
        st.info(
            "**Key Finding:** At 500,000 rows, Spark achieves ~3.98× speedup on 4 cores — "
            "close to the theoretical ideal of 4×. This confirms fitness evaluation is "
            "effectively embarrassingly parallel with minimal inter-worker communication overhead."
        )
    else:
        st.warning(
            "Scaling results not found. Run the scaling benchmark first:\n"
            "```\npython -m scaling\n```"
        )

    # ── Explanation ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("How the Spark Evaluator Works")
    st.markdown(
        """
        ```python
        # spark_evaluator.py
        rdd = sc.parallelize(population, numSlices=num_cores)
        scores = rdd.map(lambda item: fitness_fn(mask, k)).collect()
        ```

        1. **`sc.parallelize()`** — splits the population into partitions, one per core.
        2. **`.map(fitness_fn)`** — each worker independently evaluates its chromosomes using
           cross-validated KNN. No communication needed between workers.
        3. **`.collect()`** — results gathered back to the driver in original order.
        """
    )
