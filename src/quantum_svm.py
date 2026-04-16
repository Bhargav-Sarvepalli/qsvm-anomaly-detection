"""
quantum_svm.py
--------------
Trains a Quantum Support Vector Machine using Qiskit.

Pipeline:
  classical data  →  ZZFeatureMap (quantum circuit)
                  →  FidelityQuantumKernel (kernel matrix)
                  →  QSVC (SVM trained on quantum kernel)
                  →  predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Qiskit imports — each one has a specific job
from qiskit.circuit.library import ZZFeatureMap
# ZZFeatureMap: the quantum circuit that encodes classical data into quantum states
# "ZZ" refers to the type of entanglement gates used between qubits
# This is the quantum feature map we discussed in Phase 2

from qiskit_machine_learning.kernels import FidelityQuantumKernel
# FidelityQuantumKernel: computes K(x,y) = |<φ(x)|φ(y)>|²
# i.e., the overlap (fidelity) between two quantum states
# This IS the quantum kernel matrix builder

from qiskit_machine_learning.algorithms import QSVC
# QSVC: Quantum Support Vector Classifier
# Works exactly like sklearn's SVC but uses a quantum kernel instead of RBF


def build_quantum_kernel(n_features: int, reps: int = 2) -> FidelityQuantumKernel:
    """
    Build the quantum kernel using ZZFeatureMap.

    Parameters
    ----------
    n_features : number of features in our data = number of qubits in the circuit.
                 We're using 2 (src_bytes, dst_bytes), so this will be 2.

    reps       : how many times the feature map circuit is repeated.
                 More reps = more expressive (can separate more complex patterns)
                 but also slower and noisier.
                 reps=2 is the standard starting point.

    What ZZFeatureMap does step by step (for n_features=2):
    ─────────────────────────────────────────────────────────
    Start:    |0⟩|0⟩          both qubits in ground state

    Layer 1 (repeated `reps` times):
      H gates:  put both qubits in superposition
      Rz gates: rotate each qubit by its data value x_i
                q0 rotates by x1 (src_bytes scaled)
                q1 rotates by x2 (dst_bytes scaled)
      ZZ gate:  entangle the qubits, encoding the interaction x1*x2
                this cross-term is what makes the quantum kernel powerful
                a classical kernel can't naturally encode this interaction

    Result:   |φ(x)⟩  — a quantum state encoding your data point
    """
    feature_map = ZZFeatureMap(
        feature_dimension=n_features,  # one qubit per feature
        reps=reps,                      # circuit depth (repetitions)
        entanglement="full"             # connect every pair of qubits
        # "full" entanglement means:
        # for 2 qubits: q0 <-> q1
        # for 4 qubits: q0<->q1, q0<->q2, q0<->q3, q1<->q2, q1<->q3, q2<->q3
        # each connection encodes an interaction between those two features
    )

    # FidelityQuantumKernel wraps the feature map and handles the
    # kernel computation K(x,y) = |<φ(x)|φ(y)>|²
    # Under the hood it:
    #   1. Runs the feature map circuit for x  → gets |φ(x)⟩
    #   2. Runs the feature map circuit for y  → gets |φ(y)⟩
    #   3. Computes their inner product (overlap)
    #   4. Squares it → gives a value between 0 and 1
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    return quantum_kernel


def train_quantum_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int = 2
) -> tuple:
    """
    Build the quantum kernel and train the QSVC on it.

    Note on speed: computing the kernel matrix requires running the
    quantum circuit for every pair of training samples.
    For 160 training samples → 160×160 = 25,600 kernel computations.
    Each computation runs the circuit twice and measures overlap.
    On a simulator this takes a few minutes. On real quantum hardware
    it would be much faster.

    This is why we keep n_quantum_samples=200 in preprocessing —
    it keeps training time manageable while still being meaningful.
    """
    print("Building quantum kernel (ZZFeatureMap, 2 qubits, reps=2)...")
    quantum_kernel = build_quantum_kernel(n_features=n_features, reps=2)

    print(f"Training QSVC on {X_train.shape[0]} samples...")
    print("(This will take a few minutes — the simulator is computing")
    print(" the quantum kernel matrix for every pair of training points)")

    qsvc = QSVC(quantum_kernel=quantum_kernel)
    # QSVC is a drop-in replacement for sklearn's SVC
    # The only difference: instead of computing K(x,y) with RBF formula,
    # it runs the quantum circuit to compute K(x,y) = |<φ(x)|φ(y)>|²

    qsvc.fit(X_train, y_train)
    # .fit() here:
    #   1. Computes the full quantum kernel matrix (slow — circuit runs many times)
    #   2. Passes that matrix to a classical SVM optimizer
    #   3. SVM finds optimal alpha weights for support vectors
    #   4. Stores support vectors for use at prediction time

    print(f"\nTraining complete.")
    print(f"Number of support vectors: {qsvc.n_support_}")

    return qsvc, quantum_kernel


def evaluate(qsvc, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the quantum SVM on the held-out test set.
    Identical structure to classical_svm.evaluate() —
    this makes direct comparison easy.
    """
    print("\nClassifying test samples...")
    print("(Each test point requires kernel computation against support vectors)")

    y_pred = qsvc.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n── Quantum SVM Results ───────────────────────────")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nDetailed report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Normal (0)", "Attack (1)"]
    ))

    return {
        "accuracy": acc,
        "y_pred": y_pred,
        "y_test": y_test,
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }


def plot_confusion_matrix(results: dict, save_path: str = None):
    """Confusion matrix for the quantum SVM results."""
    cm = results["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",   # purple theme to visually distinguish from classical
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
        ax=ax
    )

    ax.set_title("Quantum SVM — Confusion Matrix", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()
    return fig


def plot_decision_boundary(
    qsvc, quantum_kernel, X_test: np.ndarray, y_test: np.ndarray,
    save_path: str = None
):
    """
    Visualize the decision boundary the quantum SVM learned.

    Because we used exactly 2 features, we can plot in 2D.
    This is one of the best visuals for your portfolio — it shows
    the non-linear boundary that the quantum kernel enables.

    How it works:
    - Create a fine grid of points covering the feature space [0,1] x [0,1]
    - Ask the QSVC to classify every point on the grid
    - Color the grid by class → this reveals the decision boundary shape
    - Overlay the actual test points
    """
    print("\nPlotting decision boundary (this requires classifying ~10,000 grid points)...")

    # Create a 100x100 grid across the [0,1] x [0,1] feature space
    # (our features were scaled to [0,1] in preprocessing)
    h = 0.02  # grid step size — smaller = finer but slower
    xx, yy = np.meshgrid(
        np.arange(0, 1 + h, h),
        np.arange(0, 1 + h, h)
    )

    # Flatten grid into a list of (x1, x2) points and classify each one
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = qsvc.predict(grid_points)
    Z = Z.reshape(xx.shape)  # reshape back to grid for plotting

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Background shading showing decision regions
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlGn")
    # Green region = classified as Normal
    # Red region   = classified as Attack

    # Overlay actual test points
    scatter = ax.scatter(
        X_test[:, 0], X_test[:, 1],
        c=y_test,
        cmap="RdYlGn",
        edgecolors="black",
        linewidths=0.5,
        s=40,
        zorder=5   # draw on top of the background
    )

    ax.set_xlabel("src_bytes (scaled)", fontsize=11)
    ax.set_ylabel("dst_bytes (scaled)", fontsize=11)
    ax.set_title(
        "Quantum SVM Decision Boundary\n"
        "Network Traffic: Normal vs Attack",
        fontsize=12
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Normal traffic region"),
        Patch(facecolor="#F44336", label="Attack traffic region"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()
    return fig


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.preprocess import get_splits

    splits = get_splits("../data/kddcup.data")
    q = splits["quantum"]

    qsvc, kernel = train_quantum_svm(q["X_train"], q["y_train"], n_features=2)
    results = evaluate(qsvc, q["X_test"], q["y_test"])
    plot_confusion_matrix(results, save_path="../results/quantum_confusion_matrix.png")
    plot_decision_boundary(
        qsvc, kernel,
        q["X_test"], q["y_test"],
        save_path="../results/quantum_decision_boundary.png"
    )