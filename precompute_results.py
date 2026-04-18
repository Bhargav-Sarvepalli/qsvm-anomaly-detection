"""
precompute_results.py
---------------------
Run this once locally to generate cached results from a full 200-sample
quantum run. Saves to results/cached_results.json.
The Streamlit app loads these instantly instead of recomputing live.

Run from project root:
    python precompute_results.py
"""

import json, os, sys, numpy as np
sys.path.append(os.path.dirname(__file__))

from src.preprocess import get_splits
from src.classical_svm import train_classical_svm, evaluate as eval_c
from src.quantum_svm import train_quantum_svm, evaluate as eval_q

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "kddcup.data")

print("Loading data...")
splits = get_splits(DATA_PATH, n_samples=2000, n_quantum_samples=200)

print("Training classical SVM...")
clf = train_classical_svm(splits["classical"]["X_train"], splits["classical"]["y_train"])
c_res = eval_c(clf, splits["classical"]["X_test"], splits["classical"]["y_test"])

print("Training quantum SVM (this takes ~10 minutes)...")
qsvc, kernel = train_quantum_svm(splits["quantum"]["X_train"], splits["quantum"]["y_train"], n_features=2)
q_res = eval_q(qsvc, splits["quantum"]["X_test"], splits["quantum"]["y_test"])

cache = {
    "classical": {
        "accuracy": float(c_res["accuracy"]),
        "confusion_matrix": c_res["confusion_matrix"].tolist(),
        "n_support": int(clf.n_support_.sum())
    },
    "quantum": {
        "accuracy": float(q_res["accuracy"]),
        "confusion_matrix": q_res["confusion_matrix"].tolist(),
        "n_support": int(qsvc.n_support_.sum())
    },
    "config": {
        "n_classical_samples": 2000,
        "n_quantum_samples": 200,
        "n_features": 2,
        "reps": 2,
        "features_used": ["src_bytes", "dst_bytes"]
    }
}

os.makedirs("results", exist_ok=True)
with open("results/cached_results.json", "w") as f:
    json.dump(cache, f, indent=2)

print("\nCached results saved to results/cached_results.json")
print(f"Classical: {c_res['accuracy']*100:.2f}%")
print(f"Quantum:   {q_res['accuracy']*100:.2f}%")