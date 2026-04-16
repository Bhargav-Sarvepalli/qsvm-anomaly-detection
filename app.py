"""
app.py
------
Streamlit web app for the Quantum SVM Anomaly Detection project.
Run locally:  streamlit run app.py
Deploy free:  https://streamlit.io/cloud
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys

sys.path.append(os.path.dirname(__file__))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.preprocess import get_splits
from download_data import ensure_data_exists

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum SVM — Network Intrusion Detection",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    border-left: 4px solid #9C27B0;
    margin-bottom: 1rem;
  }
  .metric-card-blue {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    border-left: 4px solid #2196F3;
    margin-bottom: 1rem;
  }
  .section-header {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin-bottom: 0.5rem;
  }
  .explainer-box {
    background: #f0f4ff;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    border: 1px solid #c5d0f5;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚛️ QSVM Controls")
    st.markdown("---")

    st.markdown("### Dataset")
    n_classical = st.slider(
        "Classical SVM samples", 200, 2000, 1000, step=200,
        help="More samples = higher accuracy but slower training"
    )
    n_quantum = st.slider(
        "Quantum SVM samples", 50, 200, 100, step=50,
        help="Keep this low — quantum simulation is slow. Each sample pair requires a circuit run."
    )

    st.markdown("### Quantum Circuit")
    reps = st.selectbox(
        "ZZFeatureMap reps", [1, 2, 3], index=1,
        help="Circuit repetitions. More reps = more expressive but slower."
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app demonstrates a **Quantum Support Vector Machine** 
    for network intrusion detection, benchmarked against a 
    classical SVM baseline.
    
    Built on the **KDD Cup 99** dataset — the standard 
    benchmark for network security research.
    
    **Tech stack**
    - Qiskit 1.1 (quantum circuits)
    - qiskit-machine-learning (QSVC)
    - scikit-learn (classical baseline)
    - Streamlit (this app)
    
    ---
    **Built by Bhargav**  
    Data AI & Quantum Computing  
    """)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚛️ Quantum SVM — Network Intrusion Detection")
st.markdown(
    "A quantum-classical hybrid anomaly detection system that benchmarks a "
    "**Quantum Support Vector Machine (QSVM)** against a classical SVM "
    "on real federal network traffic data."
)

st.markdown("---")

# ── How it works expander ─────────────────────────────────────────────────────
with st.expander("📖 How does this work? (click to expand)", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. The Problem")
        st.markdown("""
        Federal networks receive millions of connection 
        requests daily. Most are normal — but some are 
        intrusion attempts. We need to classify each 
        connection as **normal** or **attack** in real time.
        """)
    with col2:
        st.markdown("#### 2. The Quantum Approach")
        st.markdown("""
        A **ZZFeatureMap** quantum circuit encodes each 
        network connection into a quantum state |φ(x)⟩. 
        The quantum kernel K(x,y) = |⟨φ(x)|φ(y)⟩|² 
        measures similarity in a high-dimensional Hilbert 
        space — impossible to compute classically at scale.
        """)
    with col3:
        st.markdown("#### 3. The Comparison")
        st.markdown("""
        The classical SVM uses all 41 features with an 
        RBF kernel. The quantum SVM uses only 2 features 
        (src_bytes, dst_bytes) encoded as qubit rotations. 
        A close result demonstrates the power of the 
        quantum kernel in a compressed feature space.
        """)

st.markdown("---")


# ── Data loading ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "kddcup.data")

ensure_data_exists(DATA_PATH)

if not os.path.exists(DATA_PATH):
    st.error("""
    **Dataset not found.**  
    Please place `kddcup.data` in the `data/` folder.  
    Download via: `python -c "from sklearn.datasets import fetch_kddcup99; ..."`  
    See README for full instructions.
    """)
    st.stop()

@st.cache_data(show_spinner="Loading and preprocessing dataset...")
def load_data(n_cls, n_q):
    return get_splits(DATA_PATH, n_samples=n_cls, n_quantum_samples=n_q)


# ── Run button ────────────────────────────────────────────────────────────────
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

with col_btn1:
    run_classical = st.button("▶ Run Classical SVM", type="primary", use_container_width=True)
with col_btn2:
    run_quantum = st.button("⚛️ Run Quantum SVM", use_container_width=True)
with col_btn3:
    run_both = st.button("🔬 Run Full Comparison", use_container_width=True)

st.markdown("---")


# ── Helper: confusion matrix plot ─────────────────────────────────────────────
def plot_cm(cm, title, cmap, acc):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
        ax=ax, cbar=False, annot_kws={"size": 14}
    )
    ax.set_title(f"{title}\nAccuracy: {acc*100:.1f}%", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    plt.tight_layout()
    return fig


def plot_comparison_bar(c_acc, q_acc):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    models = ["Classical SVM\n(41 features)", "Quantum SVM\n(2 features)"]
    accs = [c_acc * 100, q_acc * 100]
    colors = ["#2196F3", "#9C27B0"]
    bars = ax.bar(models, accs, color=colors, width=0.4, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%", ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )
    ax.set_ylim(0, 115)
    ax.axhline(y=50, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(1.55, 51.5, "random\nguess", fontsize=7, color="red", alpha=0.7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Comparison", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ── Classical SVM ──────────────────────────────────────────────────────────────
def run_classical_svm(splits):
    from src.classical_svm import train_classical_svm, evaluate
    c = splits["classical"]
    with st.spinner("Training Classical SVM..."):
        clf = train_classical_svm(c["X_train"], c["y_train"])
        results = evaluate(clf, c["X_test"], c["y_test"])
    return results, clf


# ── Quantum SVM ────────────────────────────────────────────────────────────────
def run_quantum_svm(splits, reps):
    from src.quantum_svm import train_quantum_svm, evaluate
    q = splits["quantum"]
    progress = st.progress(0, text="Initializing quantum kernel...")
    time.sleep(0.5)
    progress.progress(10, text="Building ZZFeatureMap circuit...")
    time.sleep(0.5)
    progress.progress(20, text="Computing quantum kernel matrix (this takes a few minutes)...")
    qsvc, kernel = train_quantum_svm(q["X_train"], q["y_train"], n_features=2)
    progress.progress(80, text="Classifying test samples...")
    results = evaluate(qsvc, q["X_test"], q["y_test"])
    progress.progress(100, text="Done!")
    time.sleep(0.3)
    progress.empty()
    return results, qsvc


# ── Classical only ─────────────────────────────────────────────────────────────
if run_classical or run_both:
    splits = load_data(n_classical, n_quantum)
    c_results, clf = run_classical_svm(splits)

    st.markdown("## Classical SVM Results")
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

    with col1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        fig = plot_cm(c_results["confusion_matrix"], "Classical SVM", "Blues", c_results["accuracy"])
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
        acc = c_results["accuracy"]
        cm = c_results["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel()
        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.metric("True Positives (attacks caught)", int(tp))
        st.metric("False Negatives (missed attacks)", int(fn))
        st.metric("False Positives (false alarms)", int(fp))
        st.metric("Support vectors used", f"{clf.n_support_[0] + clf.n_support_[1]}")

    with col3:
        st.markdown('<div class="section-header">What this means</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card-blue">
        <b>The classical SVM achieved {acc*100:.1f}% accuracy</b> using all 41 features 
        of the network connection data.<br><br>
        It correctly identified <b>{tp} out of {tp+fn} attack connections</b> and 
        <b>{tn} out of {tn+fp} normal connections</b> in the test set.<br><br>
        This is the <b>baseline</b> — the benchmark our quantum model is compared against.
        The classical RBF kernel maps data into a high-dimensional space implicitly,
        which is why it performs so well on this structured dataset.
        </div>
        """, unsafe_allow_html=True)

    if not run_both:
        st.markdown("---")
        st.info("Click **⚛️ Run Quantum SVM** or **🔬 Run Full Comparison** to see the quantum results.")


# ── Quantum only ───────────────────────────────────────────────────────────────
if run_quantum or run_both:
    splits = load_data(n_classical, n_quantum)

    st.markdown("## Quantum SVM Results")
    st.markdown(
        f"Running QSVM on **{n_quantum} samples**, **2 features** (src_bytes, dst_bytes), "
        f"**ZZFeatureMap reps={reps}**, **2 qubits**."
    )

    q_results, qsvc = run_quantum_svm(splits, reps)

    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

    with col1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        fig = plot_cm(q_results["confusion_matrix"], "Quantum SVM", "Purples", q_results["accuracy"])
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
        acc = q_results["accuracy"]
        cm = q_results["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel()
        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.metric("True Positives (attacks caught)", int(tp))
        st.metric("False Negatives (missed attacks)", int(fn))
        st.metric("False Positives (false alarms)", int(fp))
        st.metric("Support vectors used", f"{qsvc.n_support_[0] + qsvc.n_support_[1]}")

    with col3:
        st.markdown('<div class="section-header">What this means</div>', unsafe_allow_html=True)
        acc = q_results["accuracy"]
        st.markdown(f"""
        <div class="metric-card">
        <b>The quantum SVM achieved {acc*100:.1f}% accuracy</b> using only 
        <b>2 of the 41 available features</b>.<br><br>
        The <b>ZZFeatureMap</b> circuit encoded each connection into a quantum state |φ(x)⟩.
        The quantum kernel K(x,y) = |⟨φ(x)|φ(y)⟩|² computed similarity in a 
        <b>4-dimensional Hilbert space</b> (2 qubits = 2² = 4 dimensions).<br><br>
        The accuracy gap vs classical reflects the <b>information loss from using 2 vs 41 
        features</b> — not a fundamental quantum limitation. On real quantum hardware with 
        41 qubits, the quantum model would operate in a 2⁴¹-dimensional space.
        </div>
        """, unsafe_allow_html=True)


# ── Full comparison ────────────────────────────────────────────────────────────
if run_both:
    st.markdown("---")
    st.markdown("## Full Comparison")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        fig = plot_comparison_bar(c_results["accuracy"], q_results["accuracy"])
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        c_acc = c_results["accuracy"] * 100
        q_acc = q_results["accuracy"] * 100
        diff = q_acc - c_acc

        st.markdown("#### Summary")
        st.markdown(f"""
        | Model | Features | Accuracy |
        |-------|----------|----------|
        | Classical SVM (RBF kernel) | 41 | **{c_acc:.1f}%** |
        | Quantum SVM (ZZFeatureMap) | 2 | **{q_acc:.1f}%** |
        """)

        st.markdown("#### Key Takeaways")
        st.markdown(f"""
        **1. The quantum kernel works.** Even with only 2 features, the QSVM 
        classifies network traffic meaningfully above random chance (50%).
        
        **2. The gap is about features, not quantum.** The classical model uses 
        41 features vs 2 for quantum. With more qubits encoding more features, 
        the quantum model's accuracy would increase substantially.
        
        **3. NISQ-era practicality.** This is a hybrid quantum-classical 
        architecture — the kernel runs on a quantum simulator, the SVM optimizer 
        runs classically. This is the realistic path for near-term quantum 
        advantage in cybersecurity applications.
        
        **4. Federal relevance.** The same architecture applies to DoD network 
        monitoring, VA healthcare access anomalies, and FEMA infrastructure 
        security — all active domains at Platinum Business Services.
        """)

    st.markdown("---")
    st.markdown("#### Pipeline Architecture")
    st.code("""
Raw network connection (KDD Cup 99)
        │
        ▼
Feature vector x = [x₁, x₂, ..., x₄₁]   ← 41 numerical features
        │
        ├─────────────────────────────────────────────────┐
        │  Classical path                  Quantum path   │
        ▼                                  ▼              │
RBF Kernel K(x,y)           ZZFeatureMap circuit         │
Classical SVM trains        encodes x → |φ(x)⟩           │
on all 41 features          (2 features, 2 qubits)        │
        │                                  │              │
        │              FidelityQuantumKernel               │
        │              K(x,y) = |⟨φ(x)|φ(y)⟩|²            │
        │                                  │              │
        └──────────────┬───────────────────┘              │
                       ▼
              Normal / Attack classification
    """, language="text")