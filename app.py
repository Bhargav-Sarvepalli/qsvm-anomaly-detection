"""
app.py — Quantum SVM Network Intrusion Detection
Sree Sai Bhargav Sarvepalli | bhargav.tech
"""

import streamlit as st
import numpy as np
import pandas as pd
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))

st.set_page_config(
    page_title="QSVM — Network Intrusion Detection",
    page_icon="⚛️",
    layout="wide",
)

st.markdown("""
<style>
  .block-container { padding-top: 2rem; max-width: 1000px; }
  .metric-row { display:flex; gap:12px; flex-wrap:wrap; margin:0.8rem 0; }
  .metric-box {
    flex:1; min-width:100px; background:#f8fafc;
    border:1px solid #e2e8f0; border-radius:8px; padding:12px 14px;
  }
  .metric-box .val { font-size:1.5rem; font-weight:700; color:#0f172a; line-height:1.1; }
  .metric-box .lbl { font-size:0.7rem; color:#94a3b8; margin-top:2px;
    text-transform:uppercase; letter-spacing:0.05em; }
  .note { font-size:0.83rem; color:#64748b; border-left:3px solid #e2e8f0;
    padding-left:0.75rem; margin:0.6rem 0; line-height:1.6; }
  .footer { margin-top:3rem; padding-top:1rem; border-top:1px solid #f1f5f9;
    font-size:0.78rem; color:#cbd5e1; text-align:center; }
  .footer a { color:#7c3aed; text-decoration:none; }
</style>
""", unsafe_allow_html=True)

DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "kddcup.data")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "results", "cached_results.json")

# ── Auto-download dataset if missing (Streamlit Cloud) ────────────────────────
@st.cache_resource(show_spinner="Downloading dataset (first run only)...")
def ensure_data():
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_kddcup99
        df = fetch_kddcup99(subset=None, shuffle=True, random_state=42, as_frame=True).frame
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

ensure_data()

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### QSVM Detection")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem;color:#94a3b8;line-height:1.8;'>
    <strong style='color:#e2e8f0;'>Sree Sai Bhargav Sarvepalli</strong><br>
    M.S. Computer Science, UMBC<br>
    <a href='https://bhargav.tech' style='color:#7c3aed;'>bhargav.tech</a>
    </div>
    """, unsafe_allow_html=True)

# ── Plot helpers ───────────────────────────────────────────────────────────────
def cm_plot(cm, title, cmap):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap=cmap,
                xticklabels=["Normal","Attack"],
                yticklabels=["Normal","Attack"],
                ax=ax, cbar=False, annot_kws={"size":13})
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Predicted", fontsize=8, color="#64748b")
    ax.set_ylabel("Actual", fontsize=8, color="#64748b")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig

def roc_plot(fpr, tpr, roc_auc, color):
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1],"--", color="#e2e8f0", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=8, color="#64748b")
    ax.set_ylabel("True Positive Rate", fontsize=8, color="#64748b")
    ax.set_title("ROC Curve", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig

def bar_comparison(metrics):
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    names  = list(metrics.keys())
    c_vals = [v[0]*100 for v in metrics.values()]
    q_vals = [v[1]*100 for v in metrics.values()]
    x = np.arange(len(names)); w = 0.3
    ax.bar(x-w/2, c_vals, w, label="Classical", color="#3b82f6", alpha=0.85)
    ax.bar(x+w/2, q_vals, w, label="Quantum",   color="#7c3aed", alpha=0.85)
    for i,(cv,qv) in enumerate(zip(c_vals,q_vals)):
        ax.text(i-w/2, cv+0.6, f"{cv:.0f}%", ha="center", fontsize=7.5, color="#1e293b")
        ax.text(i+w/2, qv+0.6, f"{qv:.0f}%", ha="center", fontsize=7.5, color="#1e293b")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 115); ax.set_ylabel("Score (%)", fontsize=8, color="#64748b")
    ax.axhline(50, color="#e2e8f0", linestyle="--", linewidth=1)
    ax.legend(fontsize=8, framealpha=0.5)
    ax.spines[["top","right"]].set_visible(False)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("QSVM — Network Intrusion Detection")
st.markdown("Classify network connections as normal or attack using classical or quantum SVM.")
st.markdown("---")

# ── Section 1: Run Detection ───────────────────────────────────────────────────
st.markdown("## Run Detection")

col_src, col_model = st.columns(2)
with col_src:
    st.markdown("**Data source**")
    source = st.radio("", ["Sample data", "Upload CSV"],
                      horizontal=True, label_visibility="collapsed")
with col_model:
    st.markdown("**Model**")
    model = st.radio("", ["Classical SVM", "Quantum SVM (pre-computed)"],
                     horizontal=True, label_visibility="collapsed")

df_input = None

if source == "Sample data":
    st.markdown(
        '<p class="note">300 balanced connections from KDD Cup 99 — 150 normal, 150 attack.</p>',
        unsafe_allow_html=True
    )
    from src.preprocess import load_data, encode_categoricals, binarize_labels
    @st.cache_data(show_spinner=False)
    def get_sample():
        df = load_data(DATA_PATH)
        df = encode_categoricals(df)
        df = binarize_labels(df)
        n = df[df["is_attack"]==0].sample(150, random_state=42)
        a = df[df["is_attack"]==1].sample(150, random_state=42)
        return pd.concat([n,a]).sample(frac=1,random_state=42).reset_index(drop=True)
    df_input = get_sample()
    vc = df_input["is_attack"].value_counts()
    st.success(f"{len(df_input)} connections loaded — {vc[0]} normal, {vc[1]} attack")

else:
    st.markdown("42 columns (41 KDD features + `is_attack`), values 0/1, max 500 rows, no missing values.")
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            errors = []
            if len(df_up) > 500:                  errors.append("Max 500 rows.")
            if "is_attack" not in df_up.columns:  errors.append("Missing 'is_attack' column.")
            if df_up.shape[1] != 42:              errors.append(f"Expected 42 columns, got {df_up.shape[1]}.")
            if df_up.isnull().any().any():         errors.append("File has missing values.")
            if errors:
                for e in errors: st.error(e)
            else:
                df_input = df_up
                st.success(f"{len(df_input)} rows loaded.")
        except Exception as e:
            st.error(f"Could not read file: {e}")

if "Quantum" in model:
    st.markdown(
        '<p class="note">Quantum results are from a full 200-sample local run — '
        'shown instantly. Live recomputation takes ~10 minutes on a simulator.</p>',
        unsafe_allow_html=True
    )

st.markdown("")
run = st.button("Run classifier", type="primary", disabled=(df_input is None and source == "Upload CSV"))

if run and df_input is not None:
    X = df_input.drop(columns=["is_attack"]).values
    y = df_input["is_attack"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te  = scaler.transform(X_te)

    if model == "Classical SVM":
        with st.spinner("Training..."):
            clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:,1]

        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec  = recall_score(y_te, y_pred, zero_division=0)
        f1   = f1_score(y_te, y_pred, zero_division=0)
        cm   = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        rauc = auc(fpr, tpr)

        st.markdown("### Results — Classical SVM")
        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-box"><div class="val">{acc*100:.1f}%</div><div class="lbl">Accuracy</div></div>'
            f'<div class="metric-box"><div class="val">{prec*100:.1f}%</div><div class="lbl">Precision</div></div>'
            f'<div class="metric-box"><div class="val">{rec*100:.1f}%</div><div class="lbl">Recall</div></div>'
            f'<div class="metric-box"><div class="val">{f1*100:.1f}%</div><div class="lbl">F1</div></div>'
            f'<div class="metric-box"><div class="val">{rauc:.3f}</div><div class="lbl">ROC AUC</div></div>'
            f'</div>', unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            fig = cm_plot(cm, "Confusion Matrix", "Blues")
            st.pyplot(fig); plt.close(fig)
            st.markdown(f'<p class="note">TN {tn} · FP {fp} · FN {fn} · TP {tp} · {clf.n_support_.sum()} support vectors</p>',
                        unsafe_allow_html=True)
        with col2:
            fig = roc_plot(fpr, tpr, rauc, "#3b82f6")
            st.pyplot(fig); plt.close(fig)
        if fn == 0:
            st.success(f"Caught all {tp} attacks — zero missed.")
        else:
            st.warning(f"Missed {fn} attack(s) out of {tp+fn}.")

    else:
        cache = load_cache()
        if not cache:
            st.error("Cached results not found. Run `python precompute_results.py` locally and commit the output.")
            st.stop()

        q = cache["quantum"]
        cm = np.array(q["confusion_matrix"])
        tn, fp, fn, tp = cm.ravel()
        acc = q["accuracy"]
        cfg = cache.get("config", {})

        st.markdown("### Results — Quantum SVM")
        st.markdown(
            f'<p class="note">'
            f'{cfg.get("n_quantum_samples","200")} samples · '
            f'2 features (src_bytes, dst_bytes) · '
            f'ZZFeatureMap reps={cfg.get("reps",2)} · 2 qubits</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-box"><div class="val">{acc*100:.1f}%</div><div class="lbl">Accuracy</div></div>'
            f'<div class="metric-box"><div class="val">{tp}/{tp+fn}</div><div class="lbl">Attacks caught</div></div>'
            f'<div class="metric-box"><div class="val">{fn}</div><div class="lbl">Missed</div></div>'
            f'<div class="metric-box"><div class="val">{fp}</div><div class="lbl">False alarms</div></div>'
            f'<div class="metric-box"><div class="val">{q["n_support"]}</div><div class="lbl">Support vectors</div></div>'
            f'</div>', unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            fig = cm_plot(cm, "Confusion Matrix", "Purples")
            st.pyplot(fig); plt.close(fig)
        with col2:
            st.markdown("**Run configuration**")
            st.code(
                f"Features:  src_bytes, dst_bytes\n"
                f"Qubits:    2\n"
                f"Circuit:   ZZFeatureMap (reps={cfg.get('reps',2)})\n"
                f"Kernel:    FidelityQuantumKernel\n"
                f"Samples:   {cfg.get('n_quantum_samples','200')} train",
                language="text"
            )
            st.markdown(
                '<p class="note">Accuracy gap = using 2 of 41 features, not a quantum limitation. '
                'With 41 qubits the model operates in 2⁴¹-dimensional Hilbert space.</p>',
                unsafe_allow_html=True
            )

# ── Section 2: Model Comparison ───────────────────────────────────────────────
st.markdown("---")
st.markdown("## Model Comparison")
st.markdown("Classical SVM vs Quantum SVM — full pre-computed metrics.")

cache = load_cache()
if not cache:
    st.markdown(
        '<p class="note">Comparison requires cached results. '
        'Run <code>python precompute_results.py</code> locally and commit '
        '<code>results/cached_results.json</code>.</p>',
        unsafe_allow_html=True
    )
else:
    c   = cache["classical"]
    q   = cache["quantum"]
    cfg = cache.get("config", {})
    c_cm = np.array(c["confusion_matrix"])
    q_cm = np.array(q["confusion_matrix"])
    c_tn,c_fp,c_fn,c_tp = c_cm.ravel()
    q_tn,q_fp,q_fn,q_tp = q_cm.ravel()

    def safe(a,b): return a/b if b else 0
    c_prec = safe(c_tp,c_tp+c_fp); c_rec = safe(c_tp,c_tp+c_fn)
    c_f1   = safe(2*c_prec*c_rec, c_prec+c_rec)
    q_prec = safe(q_tp,q_tp+q_fp); q_rec = safe(q_tp,q_tp+q_fn)
    q_f1   = safe(2*q_prec*q_rec, q_prec+q_rec)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Classical SVM** — 41 features, {cfg.get('n_classical_samples',2000)} samples")
        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-box"><div class="val">{c["accuracy"]*100:.1f}%</div><div class="lbl">Accuracy</div></div>'
            f'<div class="metric-box"><div class="val">{c_rec*100:.1f}%</div><div class="lbl">Recall</div></div>'
            f'<div class="metric-box"><div class="val">{c_f1*100:.1f}%</div><div class="lbl">F1</div></div>'
            f'</div>', unsafe_allow_html=True
        )
    with col2:
        st.markdown(f"**Quantum SVM** — 2 features, {cfg.get('n_quantum_samples',200)} samples")
        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-box"><div class="val">{q["accuracy"]*100:.1f}%</div><div class="lbl">Accuracy</div></div>'
            f'<div class="metric-box"><div class="val">{q_rec*100:.1f}%</div><div class="lbl">Recall</div></div>'
            f'<div class="metric-box"><div class="val">{q_f1*100:.1f}%</div><div class="lbl">F1</div></div>'
            f'</div>', unsafe_allow_html=True
        )

    col1, col2 = st.columns([1.4, 1])
    with col1:
        metrics = {
            "Accuracy":  (c["accuracy"], q["accuracy"]),
            "Precision": (c_prec, q_prec),
            "Recall":    (c_rec,  q_rec),
            "F1":        (c_f1,   q_f1),
        }
        fig = bar_comparison(metrics)
        st.pyplot(fig); plt.close(fig)
    with col2:
        rows = []
        for name,(cv,qv) in metrics.items():
            diff  = qv - cv
            color = "#16a34a" if diff >= 0 else "#dc2626"
            sign  = "+" if diff >= 0 else ""
            rows.append(
                f"<tr>"
                f"<td style='padding:5px 8px;color:#1e293b;'>{name}</td>"
                f"<td style='padding:5px 8px;color:#3b82f6;font-weight:600;'>{cv*100:.1f}%</td>"
                f"<td style='padding:5px 8px;color:#7c3aed;font-weight:600;'>{qv*100:.1f}%</td>"
                f"<td style='padding:5px 8px;color:{color};font-weight:600;'>{sign}{diff*100:.1f}%</td>"
                f"</tr>"
            )
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;font-size:0.84rem;'>"
            f"<tr style='color:#94a3b8;font-size:0.72rem;border-bottom:1px solid #f1f5f9;'>"
            f"<th style='padding:5px 8px;text-align:left;'>Metric</th>"
            f"<th style='padding:5px 8px;text-align:left;'>Classical</th>"
            f"<th style='padding:5px 8px;text-align:left;'>Quantum</th>"
            f"<th style='padding:5px 8px;text-align:left;'>Diff</th>"
            f"</tr>{''.join(rows)}</table>",
            unsafe_allow_html=True
        )

    col1, col2 = st.columns(2)
    with col1:
        fig = cm_plot(c_cm, "Classical SVM", "Blues")
        st.pyplot(fig); plt.close(fig)
        st.markdown(f'<p class="note">TN {c_tn} · FP {c_fp} · FN {c_fn} · TP {c_tp}</p>',
                    unsafe_allow_html=True)
    with col2:
        fig = cm_plot(q_cm, "Quantum SVM", "Purples")
        st.pyplot(fig); plt.close(fig)
        st.markdown(f'<p class="note">TN {q_tn} · FP {q_fp} · FN {q_fn} · TP {q_tp}</p>',
                    unsafe_allow_html=True)

    st.markdown(
        f'<p class="note">The {(c["accuracy"]-q["accuracy"])*100:.1f}% gap reflects '
        f'using 2 of 41 features — not a quantum limitation.</p>',
        unsafe_allow_html=True
    )

    with st.expander("Model limitations"):
        st.markdown(
            "**Simulation, not hardware.** Statevector simulator — real devices introduce noise.  \n"
            "**Small quantum training set.** 200 samples vs 2000 classical.  \n"
            "**KDD Cup 99 is dated.** 1999 data — retrain before applying to modern threats."
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>"
    "Sree Sai Bhargav Sarvepalli · "
    "<a href='https://bhargav.tech'>bhargav.tech</a> · "
    "<a href='https://github.com/Bhargav-Sarvepalli/qsvm-anomaly-detection'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)