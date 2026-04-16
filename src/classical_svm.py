"""
classical_svm.py
----------------
Trains a classical Support Vector Machine on the full 41-feature
dataset and evaluates it. This is our baseline — the benchmark
that the quantum SVM will be compared against.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def train_classical_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """
    Train a classical SVM with an RBF (Radial Basis Function) kernel.

    Why RBF kernel?
    The RBF kernel is the classical equivalent of what the quantum feature
    map does — it maps data into a high-dimensional space implicitly.
    It's the strongest classical kernel for this kind of problem, so
    it's the fairest opponent for our quantum SVM.

    Parameters
    ----------
    C  : regularization parameter. Higher C = tries harder to classify
         every training point correctly (risks overfitting).
         Lower C = allows some misclassifications for a smoother boundary.
         C=1 is a balanced default.

    gamma : controls how far the influence of a single training example
            reaches. 'scale' means gamma = 1/(n_features * X.var()),
            which automatically adjusts to the data scale.
    """
    clf = SVC(
        kernel="rbf",    # Radial Basis Function — the standard powerful kernel
        C=1.0,           # regularization strength
        gamma="scale",   # auto-scale gamma to the data
        random_state=42  # for reproducibility
    )

    clf.fit(X_train, y_train)
    # .fit() is where training happens:
    # - builds the kernel matrix from X_train
    # - finds the optimal hyperplane (max margin boundary)
    # - identifies support vectors
    # - stores weights (alpha values) for each support vector

    print(f"Training complete.")
    print(f"Number of support vectors: {clf.n_support_}")
    # clf.n_support_ gives [n_support_vectors_class_0, n_support_vectors_class_1]

    return clf


def evaluate(clf: SVC, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Run the trained SVM on unseen test data and compute metrics.

    Metrics explained:
    - Accuracy:  % of all predictions that were correct
    - Precision: of all connections we called "attack", what % actually were?
                 (low precision = many false alarms)
    - Recall:    of all actual attacks, what % did we catch?
                 (low recall = we missed real attacks — dangerous!)
    - F1 score:  harmonic mean of precision and recall. Best single number
                 to compare models when classes are balanced.
    """
    y_pred = clf.predict(X_test)
    # .predict() runs the decision function for each test point
    # and returns the class label (0 or 1)

    acc = accuracy_score(y_test, y_pred)

    print("\n── Classical SVM Results ─────────────────────────")
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
    """
    A confusion matrix shows the 4 possible outcomes:

                     Predicted Normal  |  Predicted Attack
    Actually Normal:   True Negative   |   False Positive  (false alarm)
    Actually Attack:   False Negative  |   True Positive   (caught it!)

    For cybersecurity, False Negatives (missed attacks) are the worst outcome.
    We want that bottom-right cell as high as possible.
    """
    cm = results["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,         # show numbers inside each cell
        fmt="d",            # format as integer (not scientific notation)
        cmap="Blues",       # blue color scale — darker = higher count
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
        ax=ax
    )

    ax.set_title("Classical SVM — Confusion Matrix", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

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
    c = splits["classical"]

    clf = train_classical_svm(c["X_train"], c["y_train"])
    results = evaluate(clf, c["X_test"], c["y_test"])
    plot_confusion_matrix(results, save_path="../results/classical_confusion_matrix.png")