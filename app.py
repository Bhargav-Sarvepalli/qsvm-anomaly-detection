"""
app.py — Quantum SVM Network Intrusion Detection
Sree Sai Bhargav Sarvepalli | bhargav.tech
"""

import streamlit as st
import numpy as np
import pandas as pd
import json, os, sys, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc)
from sklearn.svm import SVC

sys.path.append(os.path.dirname(__file__))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QSVM — Network Intrusion Detection",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme & CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  .block-container { padding-top: 1.5rem; }

  /* Section headers */
  .sec-header {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #a78bfa; margin-bottom: 0.3rem;
  }

  /* Cards */
  .card {
    background: #1e1b2e; border: 1px solid #2d2b45;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
  }
  .card-blue  { border-left: 4px solid #3b82f6; }
  .card-purple{ border-left: 4px solid #8b5cf6; }
  .card-green { border-left: 4px solid #10b981; }
  .card-amber { border-left: 4px solid #f59e0b; }

  /* Stat pills */
  .stat-row { display:flex; gap:12px; flex-wrap:wrap; margin:0.6rem 0; }
  .stat-pill {
    background:#2d2b45; border-radius:8px; padding:8px 14px;
    font-size:0.82rem; color:#e2e8f0; border:1px solid #3d3b55;
  }
  .stat-pill b { color:#a78bfa; font-size:1.1rem; display:block; }

  /* Step badges */
  .step-badge {
    display:inline-block; background:#4c1d95; color:#ddd6fe;
    font-size:0.7rem; font-weight:700; padding:2px 10px;
    border-radius:20px; margin-bottom:0.4rem; letter-spacing:0.06em;
  }

  /* Alert boxes */
  .alert-info {
    background:#1e3a5f; border:1px solid #3b82f6; border-radius:8px;
    padding:0.8rem 1rem; font-size:0.88rem; color:#bfdbfe; margin:0.5rem 0;
  }
  .alert-warn {
    background:#3f2d0a; border:1px solid #f59e0b; border-radius:8px;
    padding:0.8rem 1rem; font-size:0.88rem; color:#fde68a; margin:0.5rem 0;
  }
  .alert-success {
    background:#052e16; border:1px solid #10b981; border-radius:8px;
    padding:0.8rem 1rem; font-size:0.88rem; color:#a7f3d0; margin:0.5rem 0;
  }

  /* Result number */
  .big-number {
    font-size:2.4rem; font-weight:700; color:#a78bfa; line-height:1.1;
  }
  .big-label { font-size:0.8rem; color:#94a3b8; margin-top:2px; }

  /* Footer */
  .footer {
    text-align:center; padding:1.5rem 0 0.5rem;
    border-top:1px solid #2d2b45; font-size:0.82rem; color:#64748b;
    margin-top:2rem;
  }
  .footer a { color:#a78bfa; text-decoration:none; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CACHE_PATH = os.path.join(os.path.dirname(__file__), "results", "cached_results.json")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "kddcup.data")

KDD_FEATURES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚛️ QSVM App")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠 Introduction",
        "📊 Data Explorer",
        "🔍 Run Detection",
        "📈 Model Comparison",
        "🏗️ Architecture",
        "👤 About"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem;color:#94a3b8;'>
    <b style='color:#e2e8f0;'>Sree Sai Bhargav Sarvepalli</b><br>
    M.S. Computer Science, UMBC<br><br>
    <a href='https://bhargav.tech' style='color:#a78bfa;'>🌐 bhargav.tech</a><br>
    <a href='https://github.com/Bhargav-Sarvepalli/qsvm-anomaly-detection' style='color:#a78bfa;'>
    ⚙️ GitHub Repo</a>
    </div>
    """, unsafe_allow_html=True)

# ── Helper: load cache ─────────────────────────────────────────────────────────
def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return None

# ── Helper: plots ──────────────────────────────────────────────────────────────
def styled_cm(cm, title, cmap):
    fig, ax = plt.subplots(figsize=(4, 3.4))
    fig.patch.set_facecolor("#1e1b2e")
    ax.set_facecolor("#1e1b2e")
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap=cmap,
                xticklabels=["Normal","Attack"],
                yticklabels=["Normal","Attack"],
                ax=ax, cbar=False, annot_kws={"size":14,"color":"white"})
    ax.set_title(title, fontsize=10, fontweight="bold", color="white", pad=8)
    ax.set_xlabel("Predicted", fontsize=8, color="#94a3b8")
    ax.set_ylabel("Actual", fontsize=8, color="#94a3b8")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#2d2b45")
    plt.tight_layout()
    return fig

def metric_bar(metrics_dict):
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#1e1b2e")
    ax.set_facecolor("#1e1b2e")
    names  = list(metrics_dict.keys())
    c_vals = [v[0]*100 for v in metrics_dict.values()]
    q_vals = [v[1]*100 for v in metrics_dict.values()]
    x = np.arange(len(names)); w = 0.32
    ax.bar(x - w/2, c_vals, w, label="Classical SVM", color="#3b82f6", alpha=0.9)
    ax.bar(x + w/2, q_vals, w, label="Quantum SVM",   color="#8b5cf6", alpha=0.9)
    for i, (cv, qv) in enumerate(zip(c_vals, q_vals)):
        ax.text(i-w/2, cv+0.5, f"{cv:.1f}%", ha="center", fontsize=8, color="white")
        ax.text(i+w/2, qv+0.5, f"{qv:.1f}%", ha="center", fontsize=8, color="white")
    ax.set_xticks(x); ax.set_xticklabels(names, color="white", fontsize=9)
    ax.set_ylim(0, 115); ax.set_ylabel("Score (%)", color="#94a3b8")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#2d2b45", labelcolor="white", fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_edgecolor("#2d2b45")
    fig.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — INTRODUCTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Introduction":

    st.markdown("""
    <div style='margin-bottom:0.3rem;'>
      <span style='font-size:2.6rem;font-weight:800;color:#e2e8f0;'>
        ⚛️ Quantum SVM
      </span><br>
      <span style='font-size:1.2rem;color:#a78bfa;font-weight:500;'>
        Network Intrusion Detection using Quantum Machine Learning
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='alert-info'>
    This app benchmarks a <b>Quantum Support Vector Machine (QSVM)</b> against a 
    classical SVM on the KDD Cup 99 federal network intrusion dataset — 
    demonstrating quantum kernel methods on a real cybersecurity problem.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Who is this for
    st.markdown("### 👥 Who is this for?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='card card-purple'>
        <b>🔐 Security Analysts</b><br><br>
        See how ML classifies network connections as normal or malicious —
        and where quantum kernels offer a different perspective.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='card card-blue'>
        <b>🤖 ML Engineers</b><br><br>
        Explore quantum kernel methods, ZZFeatureMap encoding, and honest
        benchmarking against classical RBF-SVM baselines.
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='card card-green'>
        <b>🔬 Researchers</b><br><br>
        Examine a NISQ-era hybrid quantum-classical pipeline on a standard
        federal benchmark dataset with full reproducibility.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # The problem
    st.markdown("### 🎯 The Problem")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("""
        Federal servers process **millions of network connections daily**.
        The vast majority are legitimate — but some are intrusion attempts:
        port scans, denial-of-service floods, unauthorized access probes.

        A security operations team manually reviewing every connection is
        impossible. We need an automated classifier that can label each
        connection as **normal** or **attack** in real time, with:
        - **High recall** — never miss a real attack
        - **Low false alarms** — don't overwhelm analysts with noise
        - **Speed** — classify thousands of connections per second
        """)
    with c2:
        st.markdown("""<div class='card card-amber'>
        <b>📌 Real-world scenario</b><br><br>
        Out of <b>10,000 network requests</b> hitting a DoD server today:<br><br>
        • ~9,700 are normal user traffic<br>
        • ~200 are automated attack probes<br>
        • ~100 are sophisticated intrusion attempts<br><br>
        Missing even <b>10 of those 300</b> could mean a breach.
        The classifier needs near-perfect recall on attacks.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Why quantum
    st.markdown("### ⚛️ Why Quantum SVM?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class='card card-blue'>
        <b>Classical SVM — the problem</b><br><br>
        A classical SVM uses a kernel function to map data into a
        high-dimensional space where it becomes separable. The RBF kernel
        is powerful but operates in a fixed implicit feature space.<br><br>
        Think of it as trying to separate tangled headphone wires on a table.
        Sometimes you need to lift them into 3D space to untangle them.
        Classical kernels do this — but in a mathematically fixed way.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='card card-purple'>
        <b>Quantum SVM — the advantage</b><br><br>
        A quantum kernel maps data into <b>Hilbert space</b> — a space
        with 2ⁿ dimensions for n qubits. For 10 qubits: 1,024 dimensions.
        For 41 qubits: over 2 trillion dimensions.<br><br>
        This isn't just "bigger" — it's a fundamentally different geometry.
        Patterns invisible in classical feature spaces may become separable
        in quantum Hilbert space. That's the quantum advantage.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # How to use
    st.markdown("### 🧭 How to use this app")
    steps = [
        ("📊 Data Explorer",    "Understand the dataset — features, class balance, distributions"),
        ("🔍 Run Detection",    "Upload your own CSV or use sample data, run the classifier"),
        ("📈 Model Comparison", "See QSVM vs Classical SVM — all metrics, ROC curves, confusion matrices"),
        ("🏗️ Architecture",    "Understand the full pipeline from raw data to classification"),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div style='display:flex;align-items:flex-start;gap:12px;margin-bottom:10px;'>
          <div style='background:#4c1d95;color:#ddd6fe;font-weight:700;font-size:0.9rem;
                      border-radius:50%;width:28px;height:28px;display:flex;align-items:center;
                      justify-content:center;flex-shrink:0;margin-top:2px;'>{i}</div>
          <div><b style='color:#e2e8f0;'>{title}</b>
          <br><span style='color:#94a3b8;font-size:0.88rem;'>{desc}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='alert-success'>
    👈 Use the <b>sidebar navigation</b> to move between sections.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown("## 📊 Data Explorer")
    st.markdown("Understanding the data before training any model.")

    st.markdown("""<div class='card card-blue'>
    <b>📁 KDD Cup 1999 Dataset</b> — the standard federal benchmark for network intrusion detection research.<br><br>
    Collected from a simulated US Air Force LAN over 9 weeks. Each row represents one network connection
    with 41 features describing its behaviour. Widely used in cybersecurity ML research since 1999.
    </div>""", unsafe_allow_html=True)

    # Dataset stats
    st.markdown("### Dataset at a glance")
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown("<div class='stat-pill'><b>494,021</b>Total connections</div>", unsafe_allow_html=True)
    c2.markdown("<div class='stat-pill'><b>41</b>Features per connection</div>", unsafe_allow_html=True)
    c3.markdown("<div class='stat-pill'><b>2</b>Classes (Normal / Attack)</div>", unsafe_allow_html=True)
    c4.markdown("<div class='stat-pill'><b>23</b>Attack subtypes</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Class balance
    st.markdown("### Class distribution")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor("#1e1b2e")
        ax.set_facecolor("#1e1b2e")
        sizes  = [19.69, 80.31]
        colors = ["#3b82f6", "#8b5cf6"]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=["Normal (19.7%)", "Attack (80.3%)"],
            colors=colors, autopct="%1.1f%%", startangle=90,
            textprops={"color":"white","fontsize":9}
        )
        for at in autotexts: at.set_color("white")
        ax.set_title("Raw dataset — highly imbalanced", color="white", fontsize=9)
        st.pyplot(fig); plt.close(fig)
    with c2:
        st.markdown("""<div class='card card-amber'>
        <b>⚠️ Class imbalance problem</b><br><br>
        The raw dataset is 80% attacks — if we trained on this directly,
        a model that predicts "attack" for everything would get 80% accuracy
        while being completely useless.<br><br>
        <b>Our fix:</b> Balanced sampling — we take equal numbers of normal
        and attack connections for training. This forces the model to actually
        learn the difference rather than exploit the imbalance.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class='alert-success'>
        After balancing: <b>50% normal / 50% attack</b> in all train/test splits.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Features
    st.markdown("### Feature groups")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='card card-blue'>
        <b>🌐 Connection features (9)</b><br><br>
        Basic properties of the TCP/IP connection:<br>
        duration, protocol_type, service, flag,
        src_bytes, dst_bytes, land, wrong_fragment, urgent
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='card card-purple'>
        <b>👤 Content features (13)</b><br><br>
        Features of data in the connection payload:<br>
        hot, num_failed_logins, logged_in,
        num_compromised, root_shell, su_attempted,
        num_root, num_file_creations, num_shells...
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='card card-green'>
        <b>📡 Traffic features (19)</b><br><br>
        Statistical properties over a 2-second window:<br>
        count, srv_count, serror_rate, rerror_rate,
        same_srv_rate, diff_srv_rate, dst_host_count...
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Quantum features highlight
    st.markdown("### Features used in Quantum SVM")
    st.markdown("""
    Because each feature = 1 qubit, and simulating qubits is exponentially expensive,
    we restrict the quantum model to **2 features** chosen for their attack signal strength:
    """)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class='card card-purple'>
        <b>src_bytes</b> — bytes sent from source to destination<br><br>
        Attackers often send abnormally small (port probes) or large
        (buffer overflow attempts) payloads. This single feature
        carries strong signal for several attack types.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='card card-purple'>
        <b>dst_bytes</b> — bytes sent from destination to source<br><br>
        The server's response size reveals whether the connection
        was serviced normally or triggered an error/exploit response.
        Together with src_bytes, this forms a 2D fingerprint.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Preprocessing steps
    st.markdown("### Preprocessing pipeline")
    steps = [
        ("1. Load raw data", "494,021 rows, 42 columns including label"),
        ("2. Encode categoricals", "protocol_type, service, flag → integers via LabelEncoder"),
        ("3. Binarize labels", "All attack subtypes → 1, normal → 0"),
        ("4. Balanced sampling", "Equal normal/attack rows sampled (no imbalance bias)"),
        ("5. Train/test split", "80% training, 20% testing — same split for both models"),
        ("6. MinMax scaling", "All features scaled to [0, 1] — prevents large-value dominance"),
    ]
    for step, desc in steps:
        st.markdown(f"""
        <div style='display:flex;gap:12px;align-items:flex-start;margin-bottom:8px;'>
          <span style='background:#1e3a5f;color:#93c5fd;font-size:0.78rem;font-weight:600;
                       padding:2px 10px;border-radius:12px;white-space:nowrap;'>{step}</span>
          <span style='color:#94a3b8;font-size:0.88rem;padding-top:2px;'>{desc}</span>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RUN DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Run Detection":
    st.markdown("## 🔍 Run Detection")
    st.markdown("Run the classifier on sample data or your own CSV file.")

    # Data source
    st.markdown("### Step 1 — Choose your data source")
    data_source = st.radio("", ["Use built-in sample data (recommended)", "Upload my own CSV"],
                           horizontal=True, label_visibility="collapsed")

    df_input = None

    if data_source == "Use built-in sample data (recommended)":
        st.markdown("""<div class='alert-info'>
        Using 300 balanced samples from KDD Cup 99 (150 normal + 150 attack).
        Classical SVM trains on 240, tests on 60. Results are instant.
        </div>""", unsafe_allow_html=True)

        if os.path.exists(DATA_PATH):
            from src.preprocess import load_data, encode_categoricals, binarize_labels
            @st.cache_data(show_spinner=False)
            def get_sample():
                df = load_data(DATA_PATH)
                df = encode_categoricals(df)
                df = binarize_labels(df)
                normal = df[df["is_attack"]==0].sample(150, random_state=42)
                attack = df[df["is_attack"]==1].sample(150, random_state=42)
                return pd.concat([normal, attack]).sample(frac=1, random_state=42).reset_index(drop=True)
            df_input = get_sample()
            st.success(f"Sample loaded: {len(df_input)} connections ({df_input['is_attack'].value_counts()[0]} normal, {df_input['is_attack'].value_counts()[1]} attack)")
        else:
            st.error("Dataset not found locally. Please use the upload option.")

    else:
        st.markdown("""<div class='alert-warn'>
        <b>CSV format requirements:</b><br>
        • Must have exactly 41 feature columns + 1 label column (42 total)<br>
        • Column names must match KDD Cup 99 feature names<br>
        • Label column named <b>is_attack</b> with values 0 (normal) or 1 (attack)<br>
        • Maximum <b>500 rows</b> to prevent timeout<br>
        • No missing values
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                # Validate
                if len(df_up) > 500:
                    st.error("File has more than 500 rows. Please trim to 500 rows max.")
                elif "is_attack" not in df_up.columns:
                    st.error("Missing 'is_attack' column. Please add a column named 'is_attack' with values 0 or 1.")
                elif df_up.shape[1] != 42:
                    st.error(f"Expected 42 columns (41 features + is_attack), got {df_up.shape[1]}.")
                elif df_up.isnull().any().any():
                    st.error("File contains missing values. Please clean the data first.")
                else:
                    df_input = df_up
                    st.success(f"File loaded: {len(df_input)} rows, {df_input.shape[1]} columns.")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    if df_input is None:
        st.stop()

    st.markdown("---")

    # Model choice
    st.markdown("### Step 2 — Choose model")
    model_choice = st.radio("", ["Classical SVM (instant)", "Quantum SVM (uses pre-computed results)"],
                             horizontal=True, label_visibility="collapsed")

    st.markdown("---")

    # Run
    st.markdown("### Step 3 — Run")

    if st.button("▶ Run Detection", type="primary"):
        X = df_input.drop(columns=["is_attack"]).values
        y = df_input["is_attack"].values

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        if "Classical" in model_choice:
            with st.spinner("Training Classical SVM..."):
                clf = SVC(kernel="rbf", C=1.0, gamma="scale",
                          probability=True, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:,1]

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            cm   = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            st.markdown("### Results — Classical SVM")

            # Big numbers
            c1,c2,c3,c4 = st.columns(4)
            c1.markdown(f"<div class='big-number'>{acc*100:.1f}%</div><div class='big-label'>Accuracy</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='big-number'>{rec*100:.1f}%</div><div class='big-label'>Recall (attacks caught)</div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='big-number'>{prec*100:.1f}%</div><div class='big-label'>Precision</div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='big-number'>{f1*100:.1f}%</div><div class='big-label'>F1 Score</div>", unsafe_allow_html=True)

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Confusion Matrix**")
                fig = styled_cm(cm, "Classical SVM", "Blues")
                st.pyplot(fig); plt.close(fig)
                st.markdown(f"""
                <div class='alert-{"success" if fn==0 else "warn"}'>
                Caught <b>{tp}/{tp+fn} attacks</b> · Missed <b>{fn}</b> · 
                False alarms <b>{fp}</b> · Support vectors <b>{clf.n_support_.sum()}</b>
                </div>""", unsafe_allow_html=True)

            with col2:
                st.markdown("**ROC Curve**")
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(4, 3.4))
                fig.patch.set_facecolor("#1e1b2e"); ax.set_facecolor("#1e1b2e")
                ax.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"AUC = {roc_auc:.3f}")
                ax.plot([0,1],[0,1],"--",color="#4b5563",lw=1)
                ax.set_xlabel("False Positive Rate", color="#94a3b8", fontsize=8)
                ax.set_ylabel("True Positive Rate", color="#94a3b8", fontsize=8)
                ax.set_title("ROC Curve", color="white", fontsize=10)
                ax.tick_params(colors="white")
                ax.legend(facecolor="#2d2b45", labelcolor="white", fontsize=8)
                for sp in ax.spines.values(): sp.set_edgecolor("#2d2b45")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            st.markdown("---")
            st.markdown("**What do these results mean?**")
            if fn == 0:
                st.markdown(f"""<div class='alert-success'>
                The model caught <b>every single attack</b> in the test set with {acc*100:.1f}% overall accuracy.
                In a federal network context, zero missed attacks means zero undetected breaches from this classifier.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='alert-warn'>
                The model missed <b>{fn} attacks</b> out of {tp+fn} total. In production, each missed attack
                represents a potential undetected breach. Tuning the decision threshold could improve recall
                at the cost of more false alarms.
                </div>""", unsafe_allow_html=True)

        else:
            # Quantum — show cached
            cache = load_cache()
            if not cache:
                st.error("Pre-computed quantum results not found. Please run `python precompute_results.py` locally first.")
                st.stop()

            st.markdown("""<div class='alert-info'>
            Showing pre-computed QSVM results from a full 200-sample run with ZZFeatureMap (2 qubits, reps=2).
            Live quantum simulation takes ~10 minutes on a simulator — caching ensures consistent, 
            reproducible results for portfolio demonstration.
            </div>""", unsafe_allow_html=True)

            q = cache["quantum"]
            cm = np.array(q["confusion_matrix"])
            tn, fp, fn, tp = cm.ravel()
            acc = q["accuracy"]

            total = int(cm.sum())
            n_attacks = int(tp + fn)

            st.markdown("### Results — Quantum SVM")
            st.markdown(f"""<div class='card card-purple'>
            <b>Run config:</b> {cache['config']['n_quantum_samples']} training samples ·
            2 features (src_bytes, dst_bytes) · ZZFeatureMap reps={cache['config']['reps']} ·
            2 qubits · Hilbert space dimension = 2² = 4
            </div>""", unsafe_allow_html=True)

            c1,c2,c3,c4 = st.columns(4)
            c1.markdown(f"<div class='big-number'>{acc*100:.1f}%</div><div class='big-label'>Accuracy</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='big-number'>{tp}/{n_attacks}</div><div class='big-label'>Attacks caught</div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='big-number'>{fn}</div><div class='big-label'>Missed attacks</div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='big-number'>{q['n_support']}</div><div class='big-label'>Support vectors</div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = styled_cm(cm, "Quantum SVM (2 features)", "Purples")
                st.pyplot(fig); plt.close(fig)
            with col2:
                st.markdown("""<div class='card card-purple'>
                <b>Why this accuracy?</b><br><br>
                The quantum model uses <b>only 2 of 41 features</b>. The accuracy gap
                vs classical reflects this information loss — not a fundamental quantum limitation.<br><br>
                On real quantum hardware with 41 qubits, the model would operate in
                <b>2⁴¹-dimensional Hilbert space</b> — computationally impossible to
                replicate with any classical kernel.<br><br>
                The quantum kernel itself is working correctly — it's encoding data
                into quantum states and computing similarity via circuit fidelity.
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.markdown("## 📈 Model Comparison")
    st.markdown("Classical SVM vs Quantum SVM — full metrics side by side.")

    cache = load_cache()
    if not cache:
        st.warning("Pre-computed results not found. Run `python precompute_results.py` locally, commit results/cached_results.json, and redeploy.")
        st.stop()

    c = cache["classical"]
    q = cache["quantum"]
    cfg = cache.get("config", {})

    c_cm = np.array(c["confusion_matrix"])
    q_cm = np.array(q["confusion_matrix"])

    c_tn,c_fp,c_fn,c_tp = c_cm.ravel()
    q_tn,q_fp,q_fn,q_tp = q_cm.ravel()

    def safe_div(a,b): return a/b if b else 0

    c_prec = safe_div(c_tp, c_tp+c_fp)
    c_rec  = safe_div(c_tp, c_tp+c_fn)
    c_f1   = safe_div(2*c_prec*c_rec, c_prec+c_rec)
    q_prec = safe_div(q_tp, q_tp+q_fp)
    q_rec  = safe_div(q_tp, q_tp+q_fn)
    q_f1   = safe_div(2*q_prec*q_rec, q_prec+q_rec)

    # Summary cards
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class='card card-blue'>
        <b>🔵 Classical SVM</b> — 41 features, RBF kernel<br><br>
        <span style='font-size:2rem;font-weight:800;color:#3b82f6;'>{c['accuracy']*100:.1f}%</span> accuracy<br>
        {cfg.get('n_classical_samples',2000)} training samples · {c['n_support']} support vectors
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='card card-purple'>
        <b>⚛️ Quantum SVM</b> — 2 features, ZZFeatureMap kernel<br><br>
        <span style='font-size:2rem;font-weight:800;color:#8b5cf6;'>{q['accuracy']*100:.1f}%</span> accuracy<br>
        {cfg.get('n_quantum_samples',200)} training samples · {q['n_support']} support vectors
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Metrics comparison
    st.markdown("### All metrics")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        metrics = {
            "Accuracy":  (c["accuracy"], q["accuracy"]),
            "Precision": (c_prec, q_prec),
            "Recall":    (c_rec, q_rec),
            "F1 Score":  (c_f1, q_f1),
        }
        fig = metric_bar(metrics)
        st.pyplot(fig); plt.close(fig)
    with col2:
        st.markdown("##### Metric breakdown")
        rows = []
        for name, (cv, qv) in metrics.items():
            diff = qv - cv
            arrow = "▲" if diff > 0 else "▼"
            color = "#10b981" if diff > 0 else "#f87171"
            rows.append(f"""
            <tr>
              <td style='padding:6px 8px;color:#e2e8f0;'>{name}</td>
              <td style='padding:6px 8px;color:#3b82f6;font-weight:600;'>{cv*100:.1f}%</td>
              <td style='padding:6px 8px;color:#8b5cf6;font-weight:600;'>{qv*100:.1f}%</td>
              <td style='padding:6px 8px;color:{color};font-weight:600;'>{arrow}{abs(diff*100):.1f}%</td>
            </tr>""")
        st.markdown(f"""
        <table style='width:100%;border-collapse:collapse;font-size:0.88rem;'>
          <tr style='color:#94a3b8;border-bottom:1px solid #2d2b45;'>
            <th style='padding:6px 8px;text-align:left;'>Metric</th>
            <th style='padding:6px 8px;text-align:left;'>Classical</th>
            <th style='padding:6px 8px;text-align:left;'>Quantum</th>
            <th style='padding:6px 8px;text-align:left;'>Δ</th>
          </tr>
          {''.join(rows)}
        </table>""", unsafe_allow_html=True)

    st.markdown("---")

    # Confusion matrices
    st.markdown("### Confusion matrices")
    col1, col2 = st.columns(2)
    with col1:
        fig = styled_cm(c_cm, f"Classical SVM — {c['accuracy']*100:.1f}%", "Blues")
        st.pyplot(fig); plt.close(fig)
        st.markdown(f"TN={c_tn} · FP={c_fp} · FN={c_fn} · TP={c_tp}", unsafe_allow_html=True)
    with col2:
        fig = styled_cm(q_cm, f"Quantum SVM — {q['accuracy']*100:.1f}%", "Purples")
        st.pyplot(fig); plt.close(fig)
        st.markdown(f"TN={q_tn} · FP={q_fp} · FN={q_fn} · TP={q_tp}", unsafe_allow_html=True)

    st.markdown("---")

    # Interpretation
    st.markdown("### Interpreting the gap")
    st.markdown(f"""<div class='card card-amber'>
    <b>The accuracy gap ({(c['accuracy']-q['accuracy'])*100:.1f}%) is about features, not quantum.</b><br><br>
    Classical SVM uses <b>41 features</b>. Quantum SVM uses <b>2 features</b> (src_bytes, dst_bytes).
    That's an 95% reduction in information. The fact that the quantum model still classifies
    meaningfully above random chance (50%) with only 2 features demonstrates that
    the quantum kernel is extracting real signal.<br><br>
    On real quantum hardware with 41 qubits encoding all features, the quantum model would
    operate in <b>2⁴¹ ≈ 2.2 trillion dimensional Hilbert space</b> — no classical computer
    can even represent this space, let alone compute kernels in it efficiently.
    </div>""", unsafe_allow_html=True)

    # Model limitations
    with st.expander("⚠️ Model limitations — read before citing these results"):
        st.markdown("""
        **1. Simulation, not real hardware.** The quantum kernel runs on a statevector simulator
        (Qiskit Aer). Real quantum hardware introduces noise that would change results.

        **2. Small quantum training set.** 200 samples is small for SVM training. Classical uses 2000.
        This contributes to the accuracy gap beyond the feature difference.

        **3. KDD Cup 99 is dated.** This dataset is from 1999. Modern attack patterns differ significantly.
        Results should not be extrapolated to current threat landscapes without retraining.

        **4. 2-feature quantum encoding.** We use 2 features to keep simulation tractable.
        This is a deliberate NISQ-era compromise, not a fundamental limitation of QSVM.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Architecture":
    st.markdown("## 🏗️ System Architecture")
    st.markdown("How the full pipeline works — from raw network data to classification.")

    st.markdown("### End-to-end pipeline")
    st.code("""
┌─────────────────────────────────────────────────────────┐
│              Raw Network Connection (KDD Cup 99)         │
│   duration=0, protocol=tcp, src_bytes=181, ...          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   PREPROCESSING                          │
│  1. Encode categoricals (LabelEncoder)                  │
│  2. Binarize labels  (normal=0, attack=1)               │
│  3. Balanced sampling (50/50 class split)               │
│  4. Train/test split (80/20)                            │
│  5. MinMax scaling   (all features → [0,1])             │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
               ▼                      ▼
   ┌───────────────────┐   ┌───────────────────────────┐
   │   CLASSICAL PATH  │   │       QUANTUM PATH         │
   │                   │   │                            │
   │  41 features      │   │  2 features                │
   │  RBF kernel       │   │  ZZFeatureMap circuit      │
   │  K(x,y) computed  │   │  x → |φ(x)⟩  (2 qubits)  │
   │  analytically     │   │                            │
   │                   │   │  FidelityQuantumKernel     │
   │                   │   │  K(x,y)=|⟨φ(x)|φ(y)⟩|²   │
   └────────┬──────────┘   └────────────┬──────────────┘
            │                           │
            ▼                           ▼
   ┌───────────────────┐   ┌───────────────────────────┐
   │   sklearn SVC     │   │   qiskit-ml QSVC           │
   │   trains on       │   │   trains on quantum        │
   │   kernel matrix   │   │   kernel matrix            │
   └────────┬──────────┘   └────────────┬──────────────┘
            │                           │
            └──────────┬────────────────┘
                       ▼
            ┌──────────────────────┐
            │   CLASSIFICATION     │
            │   Normal / Attack    │
            │   + confidence score │
            └──────────────────────┘
    """, language="text")

    st.markdown("---")

    # ZZFeatureMap explanation
    st.markdown("### ZZFeatureMap — how data becomes a quantum state")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class='card card-purple'>
        <b>Circuit for 2 features x = [x₁, x₂]</b><br><br>
        <code>q₀ |0⟩ ──[H]──[Rz(x₁)]──●──[Rz(x₁·x₂)]── measure</code><br>
        <code>                          |</code><br>
        <code>q₁ |0⟩ ──[H]──[Rz(x₂)]──⊕──[Rz(x₁·x₂)]── measure</code><br><br>
        <b>H gate:</b> puts qubit in superposition (both 0 and 1)<br>
        <b>Rz(xᵢ):</b> rotates by data value — encodes your feature<br>
        <b>CNOT:</b> entangles qubits — creates correlation<br>
        <b>Rz(x₁·x₂):</b> encodes feature interaction — the key term
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='card card-blue'>
        <b>Why the interaction term matters</b><br><br>
        Classical kernels can encode individual features but computing
        the interaction x₁·x₂ requires explicit feature engineering.<br><br>
        The ZZFeatureMap naturally encodes all pairwise interactions
        through entanglement gates — this is what "ZZ" refers to.<br><br>
        For n features, a full ZZFeatureMap encodes all n(n-1)/2
        pairwise interactions simultaneously. For 41 features: 820
        interaction terms, all encoded in one circuit pass.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Tech stack
    st.markdown("### Tech stack")
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown("""<div class='card card-purple'>
    <b>Quantum</b><br><br>
    Qiskit 1.1<br>
    qiskit-machine-learning 0.7.2<br>
    ZZFeatureMap<br>
    FidelityQuantumKernel<br>
    QSVC
    </div>""", unsafe_allow_html=True)
    c2.markdown("""<div class='card card-blue'>
    <b>Classical ML</b><br><br>
    scikit-learn 1.5<br>
    SVC (RBF kernel)<br>
    MinMaxScaler<br>
    LabelEncoder<br>
    train_test_split
    </div>""", unsafe_allow_html=True)
    c3.markdown("""<div class='card card-green'>
    <b>Data</b><br><br>
    KDD Cup 99<br>
    494,021 connections<br>
    41 features<br>
    23 attack types<br>
    Balanced sampling
    </div>""", unsafe_allow_html=True)
    c4.markdown("""<div class='card card-amber'>
    <b>App</b><br><br>
    Streamlit 1.35<br>
    matplotlib<br>
    seaborn<br>
    Streamlit Cloud<br>
    GitHub CI
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 About":
    st.markdown("## 👤 About")

    c1, c2 = st.columns([1, 1.8])
    with c1:
        st.markdown("""
        <div style='background:#2d2b45;border-radius:50%;width:100px;height:100px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:2.5rem;margin-bottom:1rem;'>👨‍💻</div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <h2 style='color:#e2e8f0;margin-bottom:0.2rem;'>Sree Sai Bhargav Sarvepalli</h2>
        <p style='color:#a78bfa;font-size:1rem;margin-bottom:1rem;'>
        M.S. Computer Science · University of Maryland, Baltimore County (GPA: 3.74)
        </p>
        """, unsafe_allow_html=True)

        cols = st.columns(3)
        cols[0].link_button("🌐 bhargav.tech", "https://bhargav.tech")
        cols[1].link_button("⚙️ GitHub", "https://github.com/Bhargav-Sarvepalli")
        cols[2].link_button("💼 LinkedIn", "https://linkedin.com/in/bhargav-sarvepalli")

    st.markdown("---")

    st.markdown("### About this project")
    st.markdown("""<div class='card card-purple'>
    This project was built to demonstrate quantum machine learning on a real-world
    cybersecurity problem — going beyond coursework into a fully deployed, interactive application.<br><br>
    The goal was honest benchmarking: not claiming quantum superiority, but demonstrating
    that quantum kernels are a viable and implementable approach for anomaly detection,
    with a clear path to advantage as hardware scales.<br><br>
    Every component — preprocessing, classical baseline, quantum kernel, evaluation,
    and this app — was built from scratch and is fully documented in the GitHub repo.
    </div>""", unsafe_allow_html=True)

    st.markdown("### Relevant background")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class='card card-blue'>
        <b>Coursework</b><br><br>
        • Quantum Computing (UMBC)<br>
        • Machine Learning (UMBC)<br>
        • Natural Language Processing (UMBC)<br>
        • Malware Analysis (UMBC)<br>
        • Data Science (UMBC)
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='card card-green'>
        <b>Other ML Projects</b><br><br>
        • Pulsar Star Detection — FastAPI + LightGBM + AWS EC2<br>
        • Cross-Lingual Sentiment — mBERT, XLM-R, XLNet<br>
        • Chest X-Ray Classifier — CNN, TensorFlow, AUC 0.99<br>
        • NexTask — AI-powered Kanban SaaS (Anthropic API)
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div class='alert-info'>
    <b>Source code:</b>
    <a href='https://github.com/Bhargav-Sarvepalli/qsvm-anomaly-detection'
       style='color:#93c5fd;'>
    github.com/Bhargav-Sarvepalli/qsvm-anomaly-detection</a>
    </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  Built by <a href='https://bhargav.tech'>Sree Sai Bhargav Sarvepalli</a> ·
  <a href='https://github.com/Bhargav-Sarvepalli/qsvm-anomaly-detection'>GitHub</a> ·
  Powered by Qiskit & Streamlit
</div>
""", unsafe_allow_html=True)