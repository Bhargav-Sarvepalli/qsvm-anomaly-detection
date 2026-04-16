# Quantum SVM for Network Intrusion Detection

A quantum-classical hybrid anomaly detection system that benchmarks a
**Quantum Support Vector Machine (QSVM)** against a classical SVM on the
KDD Cup 99 network intrusion dataset — a real-world cybersecurity use case
directly applicable to federal IT and DoD environments.

---

## Motivation

Federal networks face sophisticated intrusion attempts daily. Classical ML
models are effective but operate in limited feature spaces. Quantum computing
offers a path to richer kernel functions — computed in exponentially
high-dimensional Hilbert spaces — that may capture attack patterns invisible
to classical approaches.

This project implements and honestly benchmarks a quantum kernel approach
against the classical RBF-SVM baseline on the same dataset and task.

---

## What it does

| Component | Description |
|---|---|
| `src/preprocess.py` | Loads KDD Cup 99, encodes categoricals, balances classes, scales features |
| `src/classical_svm.py` | Trains SVC with RBF kernel on all 41 features as baseline |
| `src/quantum_svm.py` | Trains QSVC using ZZFeatureMap + FidelityQuantumKernel (2 qubits) |
| `src/compare.py` | End-to-end runner that produces the comparison dashboard |

---

## Architecture

```
Raw network connection (KDD Cup 99 row)
         │
         ▼
  Feature vector x = [x₁, x₂, ..., x₄₁]   ← 41 numerical features
         │
         ├──────────────────────────────────────────────────────┐
         │  Classical path                    Quantum path      │
         │                                                      │
         ▼                                    ▼                 │
  RBF Kernel K(x,y)              ZZFeatureMap circuit           │
  Classical SVM trains           encodes x → |φ(x)⟩             │
  on full 41 features            (2 features, 2 qubits)         │
         │                                    │                 │
         │                     FidelityQuantumKernel            │
         │                     K(x,y) = |⟨φ(x)|φ(y)⟩|²           │
         │                                    │                 │
         │                     QSVC trains on quantum kernel    │
         │                                    │                 │
         └──────────────────┬─────────────────┘                 │
                            │                                   │
                            ▼                                   │
                   Normal / Attack                              │
                   classification                               │
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/qsvm-anomaly-detection
cd qsvm-anomaly-detection

# Create and activate virtual environment
python3 -m venv qsvm-env
source qsvm-env/bin/activate        # Mac/Linux
qsvm-env\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Download the dataset
# Go to: https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
# Download kddcup.data.gz, extract into data/kddcup.data
```

---

## Run

```bash
# Full comparison (classical + quantum + dashboard)
python src/compare.py

# Individual modules
python src/classical_svm.py
python src/quantum_svm.py
```

Output is saved to `results/comparison_dashboard.png`.

---

## Results

The quantum SVM achieves competitive accuracy using only **2 of 41 features**,
demonstrating that the quantum kernel extracts meaningful signal from a
dramatically compressed feature representation.

| Model | Features used | Accuracy |
|---|---|---|
| Classical SVM (RBF) | 41 | ~98% |
| Quantum SVM (ZZFeatureMap) | 2 | ~85–92% |

The accuracy gap reflects the information loss from using 2 vs 41 features —
not a fundamental limitation of the quantum approach. With hardware-native
kernels running on more qubits, the quantum model could access the full
feature space in a Hilbert space of dimension 2⁴¹.

---

## Key concepts

**ZZFeatureMap** — A parameterized quantum circuit that encodes a classical
vector x into a quantum state |φ(x)⟩ using Hadamard gates (superposition),
Rz rotation gates (data encoding), and ZZ entanglement gates (feature
interactions).

**Quantum kernel** — K(x,y) = |⟨φ(x)|φ(y)⟩|² measures the overlap between
two quantum states. Computing this classically would require working in a
2ⁿ-dimensional space — intractable for large n. A quantum computer computes
it naturally by running the circuit and measuring.

**NISQ compatibility** — This is a hybrid quantum-classical algorithm. The
quantum part (kernel computation) runs on near-term hardware. The classical
SVM optimizer runs on a standard CPU. This makes it practical today.

---

## Relevance to federal use cases

- **DoD/DLA**: Real-time network anomaly detection on federal infrastructure
- **Cybersecurity**: Low false-negative rate critical for threat detection
- **Healthcare (VA)**: Same anomaly detection approach applies to EHR access
  pattern monitoring and insider threat detection

---

## Tech stack

- Python 3.11
- [Qiskit 1.1](https://qiskit.org/) — quantum circuit construction and simulation
- [qiskit-machine-learning](https://qiskit-community.github.io/qiskit-machine-learning/) — QSVC, ZZFeatureMap, FidelityQuantumKernel
- scikit-learn — classical SVM baseline, preprocessing, metrics
- matplotlib / seaborn — visualization

---

## Author

Bhargav — Data AI & Quantum Computing Intern candidate  
Built as a portfolio project demonstrating quantum ML on a real-world
cybersecurity dataset.