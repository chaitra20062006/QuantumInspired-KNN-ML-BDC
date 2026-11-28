"""About page — project explanation."""
import streamlit as st


def render():
    st.title("ℹ️ About the Project")

    st.markdown(
        """
        ## QIEA-KNN: Quantum-Inspired Evolutionary Feature Selection for Medical Diagnosis

        This project is a **hybrid classifier** designed for medical datasets.
        Instead of using all available features and a fixed K, it uses a smart optimization
        algorithm to simultaneously find the **best subset of features** and the **ideal K**,
        then classifies using a **Fuzzy KNN** that handles borderline patients gracefully.
        """
    )

    st.markdown("---")
    st.subheader("🔬 The Three Core Components")

    tab1, tab2, tab3 = st.tabs(["⚛️ QIEA", "🤖 Fuzzy KNN", "🌐 Data & Spark"])

    with tab1:
        st.markdown(
            """
            ### Quantum-Inspired Evolutionary Algorithm (QIEA)
            Finds the best feature mask and K using quantum probability.

            **Q-bits (Quantum Bits):**  
            Unlike a classical bit (0 or 1), a Q-bit exists in superposition:
            > α|0⟩ + β|1⟩, where α² + β² = 1
            
            Stored as an angle θ: α = cos(θ), β = sin(θ), so P(bit=1) = sin²(θ).

            **Each Generation:**
            1. **Measurement** — Q-bits collapse to 0/1 by sampling sin²(θ)
            2. **Evaluate** — Cross-validated fitness score for each candidate
            3. **Rotation Gate** — θ steered toward the best-so-far solution
            4. **Mutation** — Random NOT gate: θ → π/2 − θ (prevents local traps)

            **Why better than GA?**  
            QIEA maintains a probability distribution over the entire search space.
            One Q-bit represents all feature combinations simultaneously.
            """
        )

    with tab2:
        st.markdown(
            """
            ### Fuzzy K-Nearest Neighbours

            **Standard KNN Problem:**  
            If 3 distant neighbours say "Healthy" and 2 very close neighbours say "Diabetic",
            standard KNN still predicts "Healthy" (majority vote ignores distance).

            **Fuzzy KNN Solution:**  
            Each neighbour is weighted by inverse distance:
            > w_i = 1 / d_i^(2/(m−1))

            Closer neighbours have far more voting power.
            Output is a **membership degree** ∈ [0,1] per class — not just a rigid label.
            Perfect for borderline medical cases.

            **Fitness Function:**
            > Fitness = 0.9 × F1 + 0.1 × (1 − selected/total)
            
            Rewards high F1 AND using fewer features simultaneously.
            """
        )

    with tab3:
        st.markdown(
            """
            ### Data Pipeline & Apache Spark

            **Leakage-Safe Preprocessing (`data_loader.py`):**
            - Missing value imputation: median/mode computed on **training rows only**
            - Feature scaling: StandardScaler fit on **training data only**
            - Stratified train/test split: 25% held out for final evaluation
            - 5-fold Stratified CV: used only within training data

            **Spark Parallel Evaluator (`spark_evaluator.py`):**
            - Chromosome evaluation is embarrassingly parallel (each is independent)
            - `sc.parallelize(population)` → distributes across worker cores
            - Near-linear speedup: ~3.98× on 4 cores at 500k rows

            **Datasets:**
            - Pima Indians Diabetes (768 patients, 8 features)
            - Diabetes 130-US Hospitals (large scale, for Spark benchmarking)
            - LSH Approximate Search for 1M+ row nearest-neighbor lookups
            """
        )

    st.markdown("---")
    st.subheader("👥 Developers")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            **Search & Optimization**
            - QIEA core
            - Genetic Algorithm
            - Metrics toolkit
            - Breast Cancer dataset
            """
        )
    with c2:
        st.markdown(
            """
            **Classification**
            - Fuzzy KNN
            - PSO baseline
            - Fitness function
            - Heart Disease dataset
            """
        )
    with c3:
        st.markdown(
            """
            **Data & Scale**
            - Preprocessing pipeline
            - Spark evaluator
            - Scaling experiments
            - Pima Diabetes dataset
            """
        )

    st.markdown("---")
    st.subheader("📚 Key References")
    st.markdown(
        """
        - Han, K.H. & Kim, J.H. (2002). *Quantum-inspired evolutionary algorithm for a class of combinatorial optimization.*
        - Keller, J.M. et al. (1985). *A fuzzy K-nearest neighbor algorithm.*
        - Kennedy, J. & Eberhart, R. (1995). *Particle Swarm Optimization.*
        - Smith, J. et al. UCI Machine Learning Repository — Pima Indians Diabetes Dataset.
        """
    )
