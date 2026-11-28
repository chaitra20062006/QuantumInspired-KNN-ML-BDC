"""Home page."""
import streamlit as st


def render():
    st.title("🧬 QIEA-KNN: Quantum-Inspired Medical Classifier")
    st.markdown(
        """
        > A hybrid classifier that uses a **Quantum-Inspired Evolutionary Algorithm (QIEA)**  
        > to jointly select features and tune K for a **Fuzzy K-Nearest Neighbours** model,  
        > with fitness evaluation parallelized on **Apache Spark**.
        """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔬 Datasets", "3 Medical Datasets")
        st.caption("Pima Diabetes · Heart Disease · Breast Cancer")
    with col2:
        st.metric("⚡ Methods Compared", "5 Algorithms")
        st.caption("Baseline · GA · PSO · QIEA · QIEA-Fuzzy")
    with col3:
        st.metric("🌐 Max Speedup", "~4× on 4 cores")
        st.caption("Via Apache Spark distributed evaluation")

    st.markdown("---")
    st.subheader("🗺️ How the Pipeline Works")

    st.markdown(
        """
        ```
        ┌──────────────┐     ┌─────────────────────┐     ┌──────────────────┐
        │  Data Loader │     │  QIEA / GA Optimizer  │     │   Fuzzy KNN      │
        │  (Pima etc.) │────►│  (feature mask + K)   │────►│  (predict)       │
        └──────────────┘     └─────────────────────┘     └──────────────────┘
               │                       │                          │
               └───────────────────────┴──────────────────────────┘
                             Spark parallel fitness evaluator
        ```
        """
    )

    st.markdown("---")
    st.subheader("📋 Navigate Using the Sidebar")
    st.markdown(
        """
        | Page | What You Can Do |
        |------|----------------|
        | 📊 Dataset Explorer | View dataset stats, feature distributions, class balance |
        | ⚙️ Run Classifier | Upload CSV or pick a dataset and run QIEA-KNN live |
        | 📈 Results & Comparison | F1 scores, convergence curves, feature selection comparison |
        | 🌐 Scalability | Spark speedup plots across 1 / 2 / 4 cores |
        | ℹ️ About | Full explanation of QIEA, Fuzzy KNN, and the team |
        """
    )
