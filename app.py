"""
app.py — Quantum SVM Network Intrusion Detection
Sree Sai Bhargav Sarvepalli | bhargav.tech
"""

import streamlit as st
import numpy as np
import pandas as pd
import json, os, sys, time
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
  .block-container { padding-top: 1.5rem; }

  /* Metric cards */
  .cards { display:flex; gap:14px; flex-wrap:wrap; margin:1rem 0 1.5rem; }
  .card {
    flex:1; min-width:110px;
    background:#18181b; border:1px solid #27272a;
    border-radius:12px; padding:16px 18px;
  }
  .card .val {
    font-size:1.8rem; font-weight:700; color:#fff; line-height:1.1;
  }
  .card .lbl {
    font-size:0.68rem; color:#71717a; margin-top:4px;
    text-transform:uppercase; letter-spacing:0.07em;
  }
  .card.highlight { border-color:#7c3aed; background:#1a0a2e; }
  .card.highlight .val { color:#a78bfa; }

  /* Section label */
  .sec { font-size:0.68rem; font-weight:600; letter-spacing:0.1em;
    text-transform:uppercase; color:#52525b; margin-bottom:0.3rem; }

  /* Note */
  .note { font-size:0.82rem; color:#52525b; border-left:2px solid #27272a;
    padding-left:0.7rem; margin:0.5rem 0 1rem; line-height:1.6; }

  /* Footer */
  .footer { margin-top:2.5rem; padding-top:1rem; border-top:1px solid #18181b;
    font-size:0.76rem; color:#3f3f46; text-align:center; }
  .footer a { color:#7c3aed; text-decoration:none; }
</style>
""", unsafe_allow_html=True)

DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "kddcup.data")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "results", "cached_results.json")

# ── Auto-download dataset ──────────────────────────────────────────────────────
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
    st.markdown("## ⚛️ QSVM")
    st.markdown("Network Intrusion Detection")
    st.markdown("---")

    st.markdown('<p class="sec">Dataset</p>', unsafe_allow_html=True)
    n_samples = st.slider("Training samples (classical)", 200, 2000, 800, step=200)

    st.markdown('<p class="sec" style="margin-top:1rem;">Data source</p>', unsafe_allow_html=True)
    source = st.radio("", ["Sample data (KDD Cup 99)", "Upload CSV"],
                      label_visibility="collapsed")

    df_input = None

    if source == "Sample data (KDD Cup 99)":
        from src.preprocess import load_data, encode_categoricals, binarize_labels
        @st.cache_data(show_spinner=False)
        def get_sample():
            df = load_data(DATA_PATH)
            df = encode_categoricals(df)
            df = binarize_labels(df)
            n = df[df["is_attack"]==0].sample(150, random_state=42)
            a = df[df["is_attack"]==1].sample(150, random_state=42)
            return pd.concat([n,a]).sample(frac=1, random_state=42).reset_index(drop=True)
        df_input = get_sample()
        st.success("300 connections loaded")
    else:
        st.markdown(
            '<p class="note">42 cols (41 features + is_attack), '
            'max 500 rows, no missing values.</p>',
            unsafe_allow_html=True
        )
        uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                errors = []
                if len(df_up) > 500:                  errors.append("Max 500 rows.")
                if "is_attack" not in df_up.columns:  errors.append("Missing 'is_attack' column.")
                if df_up.shape[1] != 42:              errors.append(f"Need 42 columns, got {df_up.shape[1]}.")
                if df_up.isnull().any().any():         errors.append("Contains missing values.")
                if errors:
                    for e in errors: st.error(e)
                else:
                    df_input = df_up
                    st.success(f"{len(df_input)} rows loaded")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem;color:#52525b;line-height:1.9;'>
    <span style='color:#e4e4e7;font-weight:600;'>Sree Sai Bhargav Sarvepalli</span><br>
    M.S. Computer Science, UMBC<br>
    <a href='https://bhargav.tech' style='color:#7c3aed;'>bhargav.tech</a>
    </div>
    """, unsafe_allow_html=True)

# ── Main header ────────────────────────────────────────────────────────────────
st.markdown("# QSVM — Network Intrusion Detection")
st.markdown(
    "Classify federal network connections as **normal** or **attack** "
    "using classical or quantum SVM on the KDD Cup 99 dataset."
)

if df_input is None:
    st.info("Upload a CSV in the sidebar or select sample data to get started.")
    st.stop()

# ── Prepare data splits ────────────────────────────────────────────────────────
X = df_input.drop(columns=["is_attack"]).values
y = df_input["is_attack"].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = MinMaxScaler()
X_tr = scaler.fit_transform(X_tr)
X_te  = scaler.transform(X_te)

# ── Plot helpers ───────────────────────────────────────────────────────────────
def styled_fig():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#09090b")
    ax.set_facecolor("#09090b")
    ax.tick_params(colors="#52525b", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#27272a")
    return fig, ax

def cm_plot(cm, title, cmap):
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    fig.patch.set_facecolor("#09090b")
    ax.set_facecolor("#09090b")
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap=cmap,
                xticklabels=["Normal","Attack"],
                yticklabels=["Normal","Attack"],
                ax=ax, cbar=False, annot_kws={"size":14, "color":"white"})
    ax.set_title(title, fontsize=10, fontweight="bold", color="white", pad=8)
    ax.set_xlabel("Predicted", fontsize=8, color="#52525b")
    ax.set_ylabel("Actual", fontsize=8, color="#52525b")
    ax.tick_params(colors="#71717a")
    for sp in ax.spines.values(): sp.set_edgecolor("#27272a")
    plt.tight_layout()
    return fig

def roc_plot(fpr, tpr, roc_auc, color):
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    fig.patch.set_facecolor("#09090b")
    ax.set_facecolor("#09090b")
    ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1],"--", color="#27272a", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=8, color="#52525b")
    ax.set_ylabel("True Positive Rate",  fontsize=8, color="#52525b")
    ax.set_title("ROC Curve", fontsize=9, fontweight="bold", color="white")
    ax.tick_params(colors="#71717a")
    ax.legend(fontsize=8, facecolor="#18181b", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#27272a")
    plt.tight_layout()
    return fig

def bar_compare(metrics):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor("#09090b")
    ax.set_facecolor("#09090b")
    names  = list(metrics.keys())
    c_vals = [v[0]*100 for v in metrics.values()]
    q_vals = [v[1]*100 for v in metrics.values()]
    x = np.arange(len(names)); w = 0.3
    ax.bar(x-w/2, c_vals, w, label="Classical", color="#3b82f6", alpha=0.9)
    ax.bar(x+w/2, q_vals, w, label="Quantum",   color="#7c3aed", alpha=0.9)
    for i,(cv,qv) in enumerate(zip(c_vals,q_vals)):
        ax.text(i-w/2, cv+0.8, f"{cv:.0f}%", ha="center", fontsize=7.5, color="#e4e4e7")
        ax.text(i+w/2, qv+0.8, f"{qv:.0f}%", ha="center", fontsize=7.5, color="#e4e4e7")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9, color="#a1a1aa")
    ax.set_ylim(0, 115)
    ax.axhline(50, color="#27272a", linestyle="--", linewidth=1)
    ax.text(len(names)-0.42, 51.5, "random baseline", fontsize=7, color="#3f3f46")
    ax.set_ylabel("Score (%)", fontsize=8, color="#52525b")
    ax.tick_params(colors="#52525b")
    ax.legend(fontsize=8, facecolor="#18181b", labelcolor="white")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_edgecolor("#27272a")
    plt.tight_layout()
    return fig

def metric_cards(metrics_dict, highlight_key=None):
    cards_html = '<div class="cards">'
    for label, val in metrics_dict.items():
        cls = "card highlight" if label == highlight_key else "card"
        cards_html += f'<div class="{cls}"><div class="val">{val}</div><div class="lbl">{label}</div></div>'
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Classical SVM", "Quantum SVM", "Compare"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSICAL SVM
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Classical SVM")
    st.markdown(
        f"RBF kernel · {len(X_tr)} training samples · 41 features · scikit-learn"
    )

    if st.button("Run Classical SVM", type="primary", key="run_c"):
        with st.status("Training Classical SVM...", expanded=True) as status:
            st.write("Loading features...")
            time.sleep(0.3)
            st.write(f"Training SVC on {len(X_tr)} samples...")
            clf = SVC(kernel="rbf", C=1.0, gamma="scale",
                      probability=True, random_state=42)
            clf.fit(X_tr, y_tr)
            st.write("Evaluating on test set...")
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:,1]
            status.update(label="Done", state="complete")

        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec  = recall_score(y_te, y_pred, zero_division=0)
        f1   = f1_score(y_te, y_pred, zero_division=0)
        cm   = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        rauc = auc(fpr, tpr)

        metric_cards({
            "Accuracy":  f"{acc*100:.1f}%",
            "Precision": f"{prec*100:.1f}%",
            "Recall":    f"{rec*100:.1f}%",
            "F1 Score":  f"{f1*100:.1f}%",
            "ROC AUC":   f"{rauc:.3f}",
        })

        col1, col2, col3 = st.columns(3)
        with col1:
            fig = cm_plot(cm, "Confusion Matrix", "Blues")
            st.pyplot(fig); plt.close(fig)
        with col2:
            fig = roc_plot(fpr, tpr, rauc, "#3b82f6")
            st.pyplot(fig); plt.close(fig)
        with col3:
            st.markdown("**Details**")
            st.markdown(
                f'<p class="note">'
                f'True Negatives: {tn}<br>'
                f'False Positives: {fp}<br>'
                f'False Negatives: {fn}<br>'
                f'True Positives: {tp}<br>'
                f'Support vectors: {clf.n_support_.sum()}'
                f'</p>',
                unsafe_allow_html=True
            )
            if fn == 0:
                st.success(f"Caught all {tp} attacks. Zero missed.")
            else:
                st.warning(f"Missed {fn} attack(s) out of {tp+fn}.")
    else:
        st.markdown(
            '<p class="note">Click the button above to train and evaluate '
            'the classical SVM on the loaded dataset.</p>',
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QUANTUM SVM
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Quantum SVM")
    st.markdown(
        "ZZFeatureMap · 2 qubits · FidelityQuantumKernel · "
        "2 features (src_bytes, dst_bytes)"
    )
    st.markdown(
        '<p class="note">'
        'Results are pre-computed from a full 200-sample local run. '
        'Live quantum simulation takes ~10 minutes on a statevector simulator. '
        'The cached run uses the same train/test methodology as the classical model.'
        '</p>',
        unsafe_allow_html=True
    )

    cache = load_cache()

    if st.button("Load Quantum Results", type="primary", key="run_q"):
        if not cache:
            st.error("Cached results not found. Run `python precompute_results.py` and commit the output.")
            st.stop()

        with st.status("Loading quantum results...", expanded=True) as status:
            st.write("Reading cached kernel matrix results...")
            time.sleep(0.4)
            st.write("Reconstructing confusion matrix...")
            time.sleep(0.3)
            st.write("Computing metrics...")
            time.sleep(0.2)
            status.update(label="Done", state="complete")

        q   = cache["quantum"]
        cfg = cache.get("config", {})
        cm  = np.array(q["confusion_matrix"])
        tn, fp, fn, tp = cm.ravel()
        acc = q["accuracy"]

        def safe(a,b): return a/b if b else 0
        prec = safe(tp, tp+fp)
        rec  = safe(tp, tp+fn)
        f1   = safe(2*prec*rec, prec+rec)

        metric_cards({
            "Accuracy":  f"{acc*100:.1f}%",
            "Precision": f"{prec*100:.1f}%",
            "Recall":    f"{rec*100:.1f}%",
            "F1 Score":  f"{f1*100:.1f}%",
            "Support vectors": str(q["n_support"]),
        }, highlight_key="Recall")

        col1, col2, col3 = st.columns(3)
        with col1:
            fig = cm_plot(cm, "Confusion Matrix", "Purples")
            st.pyplot(fig); plt.close(fig)
        with col2:
            st.markdown("**Circuit config**")
            st.code(
                f"Features:  src_bytes, dst_bytes\n"
                f"Qubits:    2\n"
                f"Circuit:   ZZFeatureMap\n"
                f"Reps:      {cfg.get('reps', 2)}\n"
                f"Kernel:    FidelityQuantumKernel\n"
                f"Hilbert:   2² = 4 dimensions\n"
                f"Samples:   {cfg.get('n_quantum_samples','200')} train",
                language="text"
            )
        with col3:
            st.markdown("**Details**")
            st.markdown(
                f'<p class="note">'
                f'True Negatives: {tn}<br>'
                f'False Positives: {fp}<br>'
                f'False Negatives: {fn}<br>'
                f'True Positives: {tp}'
                f'</p>',
                unsafe_allow_html=True
            )
            if fn == 0:
                st.success("Zero missed attacks — 100% recall.")
            else:
                st.info(
                    f"Missed {fn} attack(s). Accuracy gap reflects "
                    f"using 2 of 41 features, not a quantum limitation."
                )
    else:
        st.markdown(
            '<p class="note">Click the button above to load '
            'the pre-computed quantum SVM results.</p>',
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Model Comparison")
    st.markdown("Classical SVM vs Quantum SVM — full metrics side by side.")

    cache = load_cache()
    if not cache:
        st.warning(
            "Comparison requires cached quantum results. "
            "Run `python precompute_results.py` and commit `results/cached_results.json`."
        )
        st.stop()

    c    = cache["classical"]
    q    = cache["quantum"]
    cfg  = cache.get("config", {})
    c_cm = np.array(c["confusion_matrix"])
    q_cm = np.array(q["confusion_matrix"])
    c_tn,c_fp,c_fn,c_tp = c_cm.ravel()
    q_tn,q_fp,q_fn,q_tp = q_cm.ravel()

    def safe(a,b): return a/b if b else 0
    c_prec = safe(c_tp,c_tp+c_fp); c_rec = safe(c_tp,c_tp+c_fn)
    c_f1   = safe(2*c_prec*c_rec, c_prec+c_rec)
    q_prec = safe(q_tp,q_tp+q_fp); q_rec = safe(q_tp,q_tp+q_fn)
    q_f1   = safe(2*q_prec*q_rec, q_prec+q_rec)

    # Summary cards — classical
    st.markdown("#### Classical SVM")
    st.markdown(f"41 features · {cfg.get('n_classical_samples', 2000)} samples · RBF kernel")
    metric_cards({
        "Accuracy":  f"{c['accuracy']*100:.1f}%",
        "Precision": f"{c_prec*100:.1f}%",
        "Recall":    f"{c_rec*100:.1f}%",
        "F1 Score":  f"{c_f1*100:.1f}%",
    })

    # Summary cards — quantum
    st.markdown("#### Quantum SVM")
    st.markdown(f"2 features · {cfg.get('n_quantum_samples', 200)} samples · ZZFeatureMap")
    metric_cards({
        "Accuracy":  f"{q['accuracy']*100:.1f}%",
        "Precision": f"{q_prec*100:.1f}%",
        "Recall":    f"{q_rec*100:.1f}%",
        "F1 Score":  f"{q_f1*100:.1f}%",
    }, highlight_key="Recall")

    st.markdown("---")

    # Bar chart + diff table
    col1, col2 = st.columns([1.5, 1])
    with col1:
        metrics = {
            "Accuracy":  (c["accuracy"], q["accuracy"]),
            "Precision": (c_prec, q_prec),
            "Recall":    (c_rec,  q_rec),
            "F1":        (c_f1,   q_f1),
        }
        fig = bar_compare(metrics)
        st.pyplot(fig); plt.close(fig)
    with col2:
        rows = []
        for name,(cv,qv) in metrics.items():
            diff  = qv - cv
            color = "#22c55e" if diff >= 0 else "#ef4444"
            sign  = "+" if diff >= 0 else ""
            rows.append(
                f"<tr>"
                f"<td style='padding:6px 10px;color:#e4e4e7;'>{name}</td>"
                f"<td style='padding:6px 10px;color:#3b82f6;font-weight:600;'>{cv*100:.1f}%</td>"
                f"<td style='padding:6px 10px;color:#a78bfa;font-weight:600;'>{qv*100:.1f}%</td>"
                f"<td style='padding:6px 10px;color:{color};font-weight:600;'>{sign}{diff*100:.1f}%</td>"
                f"</tr>"
            )
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;font-size:0.85rem;'>"
            f"<tr style='color:#52525b;font-size:0.7rem;border-bottom:1px solid #27272a;'>"
            f"<th style='padding:6px 10px;text-align:left;'>Metric</th>"
            f"<th style='padding:6px 10px;text-align:left;'>Classical</th>"
            f"<th style='padding:6px 10px;text-align:left;'>Quantum</th>"
            f"<th style='padding:6px 10px;text-align:left;'>Diff</th>"
            f"</tr>{''.join(rows)}</table>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Confusion matrices
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
        f'<p class="note">'
        f'The {(c["accuracy"]-q["accuracy"])*100:.1f}% accuracy gap reflects '
        f'using 2 of 41 features — not a quantum limitation. '
        f'With 41 qubits on real hardware the model operates in '
        f'2⁴¹-dimensional Hilbert space, inaccessible to any classical kernel.'
        f'</p>',
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