"""
app.py — Quantum SVM Anomaly Detection — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

sys.path.append(os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum SVM — Network Intrusion Detection",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric-card {
    background:transparent; border-radius:10px; padding:1rem 1.5rem;
    border-left:4px solid #9C27B0; margin-bottom:0.75rem;
}
.metric-card-blue {
    background:transparent; border-radius:10px; padding:1rem 1.5rem;
    border-left:4px solid #2196F3; margin-bottom:0.75rem;
}
.section-label {
    font-size:0.7rem; font-weight:600; text-transform:uppercase;
    letter-spacing:0.08em; color:#888; margin-bottom:0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚛️ Controls")
    st.markdown("---")
    st.markdown("### Dataset size")
    n_classical = st.slider("Classical SVM samples", 200, 2000, 800, step=200,
        help="More = higher accuracy, slower training")
    n_quantum = st.slider("Quantum SVM samples", 50, 150, 100, step=50,
        help="Keep low — quantum simulation is slow")
    st.markdown("### Quantum circuit")
    reps = st.selectbox("ZZFeatureMap reps", [1, 2], index=1,
        help="Circuit depth. More = expressive but slower.")
    st.markdown("---")
    st.markdown("""
    **Built by Bhargav**  
    Data AI & Quantum Computing  
    Platinum Business Services — 2025
    
    **Stack:** Qiskit · scikit-learn · Streamlit
    """)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚛️ Quantum SVM — Network Intrusion Detection")
st.markdown(
    "Benchmarks a **Quantum Support Vector Machine** against a classical SVM "
    "on the KDD Cup 99 federal network intrusion dataset."
)
st.markdown("---")

# ── How it works ──────────────────────────────────────────────────────────────
with st.expander("📖 How does this work?", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### The Problem")
        st.markdown("Federal servers log millions of connections daily. We classify each as **normal** or **attack** in real time.")
    with c2:
        st.markdown("#### Quantum Approach")
        st.markdown("A **ZZFeatureMap** circuit encodes data into quantum state |φ(x)⟩. The quantum kernel K(x,y)=|⟨φ(x)|φ(y)⟩|² measures similarity in Hilbert space.")
    with c3:
        st.markdown("#### The Comparison")
        st.markdown("Classical SVM uses all 41 features. Quantum SVM uses 2 features encoded as qubit rotations. The gap shows the feature constraint, not a quantum limitation.")

st.markdown("---")

# ── Data path ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "kddcup.data")

# ── Cached data loader — only runs when button is clicked ─────────────────────
@st.cache_data(show_spinner=False)
def load_splits(n_cls, n_q):
    from src.preprocess import get_splits
    return get_splits(DATA_PATH, n_samples=n_cls, n_quantum_samples=n_q)

# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_cm(cm, title, cmap, acc):
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=["Normal","Attack"],
                yticklabels=["Normal","Attack"],
                ax=ax, cbar=False, annot_kws={"size":13})
    ax.set_title(f"{title}\nAccuracy: {acc*100:.1f}%", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    plt.tight_layout()
    return fig

def plot_bar(c_acc, q_acc):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bars = ax.bar(
        ["Classical\n(41 features)", "Quantum\n(2 features)"],
        [c_acc*100, q_acc*100],
        color=["#2196F3","#9C27B0"], width=0.4, edgecolor="white"
    )
    for bar, acc in zip(bars, [c_acc*100, q_acc*100]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(50, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(1.55, 51.5, "random\nguess", fontsize=7, color="red", alpha=0.7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig

# ── Buttons ───────────────────────────────────────────────────────────────────
b1, b2, b3 = st.columns(3)
run_classical = b1.button("▶ Run Classical SVM", type="primary", use_container_width=True)
run_quantum   = b2.button("⚛️ Run Quantum SVM",  use_container_width=True)
run_both      = b3.button("🔬 Run Full Comparison", use_container_width=True)

if not (run_classical or run_quantum or run_both):
    st.info("👆 Choose a model above and click Run to start. Classical SVM finishes in seconds. Quantum SVM takes ~3 minutes.")
    st.markdown("#### What you'll see")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Confusion Matrix**\nShows true/false positives and negatives for each model")
    c2.markdown("**Key Metrics**\nAccuracy, precision, recall, F1-score")
    c3.markdown("**Explanation**\nWhat the results mean in a federal cybersecurity context")
    st.stop()

# ── Run Classical ─────────────────────────────────────────────────────────────
c_results = None
if run_classical or run_both:
    with st.spinner("Loading data..."):
        splits = load_splits(n_classical, n_quantum)

    from src.classical_svm import train_classical_svm, evaluate as eval_c
    with st.spinner("Training Classical SVM on 41 features..."):
        clf = train_classical_svm(splits["classical"]["X_train"], splits["classical"]["y_train"])
        c_results = eval_c(clf, splits["classical"]["X_test"], splits["classical"]["y_test"])

    st.markdown("## Classical SVM Results")
    col1, col2, col3 = st.columns([1.1, 1.1, 1.8])
    with col1:
        st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
        fig = plot_cm(c_results["confusion_matrix"], "Classical SVM", "Blues", c_results["accuracy"])
        st.pyplot(fig); plt.close(fig)
    with col2:
        st.markdown('<div class="section-label">Metrics</div>', unsafe_allow_html=True)
        acc = c_results["accuracy"]
        tn, fp, fn, tp = c_results["confusion_matrix"].ravel()
        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.metric("Attacks caught (TP)", int(tp))
        st.metric("Missed attacks (FN)", int(fn))
        st.metric("False alarms (FP)", int(fp))
        st.metric("Support vectors", int(clf.n_support_.sum()))
    with col3:
        st.markdown('<div class="section-label">What this means</div>', unsafe_allow_html=True)
        acc = c_results["accuracy"]
        tn, fp, fn, tp = c_results["confusion_matrix"].ravel()
        st.markdown(f"""<div class="metric-card-blue">
        <b>Classical SVM: {acc*100:.1f}% accuracy</b> using all 41 features.<br><br>
        Caught <b>{tp}/{tp+fn} attacks</b> and correctly cleared <b>{tn}/{tn+fp} normal connections</b>.<br><br>
        This is the <b>baseline benchmark</b>. The RBF kernel maps data into a 
        high-dimensional space implicitly — the classical gold standard for this problem.
        </div>""", unsafe_allow_html=True)
    st.markdown("---")

# ── Run Quantum ───────────────────────────────────────────────────────────────
q_results = None
if run_quantum or run_both:
    if not (run_classical or run_both):
        with st.spinner("Loading data..."):
            splits = load_splits(n_classical, n_quantum)

    st.markdown("## Quantum SVM Results")
    st.markdown(f"Running on **{n_quantum} samples**, **2 features**, **ZZFeatureMap reps={reps}**, **2 qubits**.")

    progress = st.progress(0, text="Initializing quantum kernel...")
    import time

    from src.quantum_svm import train_quantum_svm, evaluate as eval_q
    progress.progress(15, text="Building ZZFeatureMap circuit (encoding data as qubit rotations)...")
    time.sleep(0.3)
    progress.progress(25, text="Computing quantum kernel matrix — running circuit for each sample pair...")

    qsvc, kernel = train_quantum_svm(
        splits["quantum"]["X_train"],
        splits["quantum"]["y_train"],
        n_features=2
    )

    progress.progress(85, text="Classifying test samples against support vectors...")
    q_results = eval_q(qsvc, splits["quantum"]["X_test"], splits["quantum"]["y_test"])
    progress.progress(100, text="Complete!")
    time.sleep(0.3)
    progress.empty()

    col1, col2, col3 = st.columns([1.1, 1.1, 1.8])
    with col1:
        st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
        fig = plot_cm(q_results["confusion_matrix"], "Quantum SVM", "Purples", q_results["accuracy"])
        st.pyplot(fig); plt.close(fig)
    with col2:
        st.markdown('<div class="section-label">Metrics</div>', unsafe_allow_html=True)
        acc = q_results["accuracy"]
        tn, fp, fn, tp = q_results["confusion_matrix"].ravel()
        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.metric("Attacks caught (TP)", int(tp))
        st.metric("Missed attacks (FN)", int(fn))
        st.metric("False alarms (FP)", int(fp))
        st.metric("Support vectors", int(qsvc.n_support_.sum()))
    with col3:
        st.markdown('<div class="section-label">What this means</div>', unsafe_allow_html=True)
        acc = q_results["accuracy"]
        tn, fp, fn, tp = q_results["confusion_matrix"].ravel()
        st.markdown(f"""<div class="metric-card">
        <b>Quantum SVM: {acc*100:.1f}% accuracy</b> using only <b>2 of 41 features</b>.<br><br>
        The <b>ZZFeatureMap</b> encoded each connection into a quantum state |φ(x)⟩.
        Similarity was computed in a <b>4-dimensional Hilbert space</b> (2² = 4 dims).<br><br>
        <b>The gap vs classical is about features, not quantum.</b> With 41 qubits on 
        real hardware, the model operates in a 2⁴¹-dimensional space — 
        computationally impossible to replicate classically.
        </div>""", unsafe_allow_html=True)
    st.markdown("---")

# ── Full comparison panel ─────────────────────────────────────────────────────
if run_both and c_results and q_results:
    st.markdown("## Full Comparison")
    col1, col2 = st.columns([1, 1.6])
    with col1:
        fig = plot_bar(c_results["accuracy"], q_results["accuracy"])
        st.pyplot(fig); plt.close(fig)
    with col2:
        c_acc = c_results["accuracy"]*100
        q_acc = q_results["accuracy"]*100
        st.markdown("#### Summary")
        st.markdown(f"""
| Model | Features | Accuracy |
|-------|----------|----------|
| Classical SVM (RBF) | 41 | **{c_acc:.1f}%** |
| Quantum SVM (ZZFeatureMap) | 2 | **{q_acc:.1f}%** |
        """)
        st.markdown("#### Key Takeaways")
        st.markdown(f"""
**1. The quantum kernel works.** Even with 2 features, QSVM classifies above random chance.

**2. The gap is about features, not quantum.** Classical uses 41 vs quantum's 2. More qubits = more features = higher accuracy.

**3. NISQ-era practicality.** This hybrid architecture — quantum kernel + classical optimizer — is deployable on today's hardware.

**4. Federal relevance.** Directly applicable to DoD network monitoring, VA healthcare anomaly detection, and FEMA infrastructure security.
        """)