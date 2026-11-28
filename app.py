"""
QIEA-KNN Streamlit App — Main Entry Point
Run with:  streamlit run app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QIEA-KNN Medical Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg",
    width=60,
)
st.sidebar.title("🧬 QIEA-KNN")
st.sidebar.caption("Quantum-Inspired Medical Classifier")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 Dataset Explorer",
        "⚙️ Run Classifier",
        "📈 Results & Comparison",
        "🌐 Scalability (Spark)",
        "ℹ️ About the Project",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**QIEA-KNN Medical Classifier**")

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "🏠 Home":
    from ui.home import render
    render()
elif page == "📊 Dataset Explorer":
    from ui.dataset_explorer import render
    render()
elif page == "⚙️ Run Classifier":
    from ui.run_classifier import render
    render()
elif page == "📈 Results & Comparison":
    from ui.results import render
    render()
elif page == "🌐 Scalability (Spark)":
    from ui.scalability import render
    render()
elif page == "ℹ️ About the Project":
    from ui.about import render
    render()
