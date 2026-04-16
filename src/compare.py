"""
compare.py
----------
Runs both classical and quantum SVMs, then produces a side-by-side
comparison dashboard. This is the main script you run end-to-end.

Run from the project root:
    python src/compare.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves file without opening a window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix

from src.preprocess import get_splits
from src.classical_svm import train_classical_svm, evaluate as eval_classical
from src.quantum_svm import train_quantum_svm, evaluate as eval_quantum


def run_comparison(data_path: str):
    """
    Full end-to-end pipeline:
      1. Load and preprocess data
      2. Train classical SVM
      3. Train quantum SVM
      4. Compare results side by side
      5. Save dashboard to results/
    """

    print("=" * 55)
    print("  QSVM Anomaly Detection — Quantum vs Classical")
    print("=" * 55)
    print("\n[1/4] Loading and preprocessing data...")

    splits = get_splits(data_path, n_samples=2000, n_quantum_samples=200)
    c = splits["classical"]
    q = splits["quantum"]

    print(f"  Classical dataset: {c['X_train'].shape[0]} train, {c['X_test'].shape[0]} test, {c['X_train'].shape[1]} features")
    print(f"  Quantum dataset:   {q['X_train'].shape[0]} train, {q['X_test'].shape[0]} test, {q['X_train'].shape[1]} features")

    print("\n[2/4] Training Classical SVM...")
    clf = train_classical_svm(c["X_train"], c["y_train"])
    classical_results = eval_classical(clf, c["X_test"], c["y_test"])

    print("\n[3/4] Training Quantum SVM...")
    qsvc, kernel = train_quantum_svm(q["X_train"], q["y_train"], n_features=2)
    quantum_results = eval_quantum(qsvc, q["X_test"], q["y_test"])

    print("\n[4/4] Generating comparison dashboard...")
    fig = build_dashboard(classical_results, quantum_results)

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/comparison_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Dashboard saved to results/comparison_dashboard.png")

    print_summary(classical_results, quantum_results)

    return classical_results, quantum_results


def build_dashboard(classical_results, quantum_results):
    """
    Build a 1x3 dashboard:
    [Classical confusion matrix] [Quantum confusion matrix] [Accuracy bar chart]
    No decision boundary — that requires 2500+ quantum circuit runs and is too slow.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "Quantum vs Classical SVM — Network Intrusion Detection\n"
        "Dataset: KDD Cup 99  |  Quantum: ZZFeatureMap (2 qubits, reps=2)",
        fontsize=13,
        fontweight="bold",
        y=1.02
    )

    _plot_cm(axes[0], classical_results["confusion_matrix"], "Classical SVM (41 features)", cmap="Blues")
    _plot_cm(axes[1], quantum_results["confusion_matrix"],   "Quantum SVM (2 features)",    cmap="Purples")
    _plot_accuracy_bars(axes[2], classical_results["accuracy"], quantum_results["accuracy"])

    plt.tight_layout()
    return fig


def _plot_cm(ax, cm, title, cmap):
    """Draw one confusion matrix panel."""
    import seaborn as sns
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
        ax=ax, cbar=False
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)

    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, -0.18,
        f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
        ha="center", fontsize=8, color="gray",
        transform=ax.transAxes
    )


def _plot_accuracy_bars(ax, classical_acc, quantum_acc):
    """Bar chart comparing accuracy of both models."""
    models = ["Classical SVM\n(41 features)", "Quantum SVM\n(2 features)"]
    accs   = [classical_acc * 100, quantum_acc * 100]
    colors = ["#2196F3", "#9C27B0"]

    bars = ax.bar(models, accs, color=colors, width=0.45, edgecolor="white", linewidth=1.5)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold"
        )

    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_title("Accuracy Comparison", fontsize=11, fontweight="bold", pad=8)
    ax.axhline(y=50, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(1.55, 51, "random\nguess", fontsize=7, color="red", alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def print_summary(classical_results, quantum_results):
    """Print a clean final comparison to the console."""
    c_acc = classical_results["accuracy"] * 100
    q_acc = quantum_results["accuracy"] * 100
    diff  = q_acc - c_acc

    print("\n" + "=" * 55)
    print("  FINAL COMPARISON")
    print("=" * 55)
    print(f"  Classical SVM  (41 features): {c_acc:.2f}%")
    print(f"  Quantum SVM    ( 2 features): {q_acc:.2f}%")
    print(f"  Difference:                  {diff:+.2f}%")
    print("=" * 55)
    print(f"\n  The quantum SVM used only 2 of 41 features.")
    print(f"  With access to the full feature set on real")
    print(f"  quantum hardware, the gap would close significantly.")
    print("=" * 55)


if __name__ == "__main__":
    DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "kddcup.data"
    )

    if not os.path.exists(DATA_PATH):
        print(f"\nDataset not found at: {DATA_PATH}")
        sys.exit(1)

    run_comparison(DATA_PATH)