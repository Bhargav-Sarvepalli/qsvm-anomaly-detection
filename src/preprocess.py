"""
preprocess.py
-------------
Loads the KDD Cup 99 network intrusion dataset, cleans it,
encodes categorical columns, scales features, and returns
train/test splits ready for both classical and quantum SVMs.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# ── Column names ──────────────────────────────────────────────────────────────
# The KDD Cup 99 CSV has no header row, so we define the names manually.
# These 41 features describe one network connection.
COLUMN_NAMES = [
    "duration",           # how long the connection lasted (seconds)
    "protocol_type",      # tcp, udp, or icmp  <- text, needs encoding
    "service",            # destination service: http, ftp, smtp...  <- text
    "flag",               # connection status: SF, REJ, S0...  <- text
    "src_bytes",          # bytes sent from source to destination
    "dst_bytes",          # bytes sent from destination to source
    "land",               # 1 if source/dest host and port are the same
    "wrong_fragment",     # number of wrong fragments
    "urgent",             # number of urgent packets
    "hot",                # number of "hot" indicators
    "num_failed_logins",  # failed login attempts
    "logged_in",          # 1 if successfully logged in
    "num_compromised",    # number of compromised conditions
    "root_shell",         # 1 if root shell was obtained
    "su_attempted",       # 1 if su root command was attempted
    "num_root",           # number of root accesses
    "num_file_creations", # number of file creation operations
    "num_shells",         # number of shell prompts
    "num_access_files",   # number of operations on access control files
    "num_outbound_cmds",  # number of outbound commands in ftp session
    "is_host_login",      # 1 if login belongs to the host list
    "is_guest_login",     # 1 if login is a guest login
    "count",              # connections to same host in past 2 seconds
    "srv_count",          # connections to same service in past 2 seconds
    "serror_rate",        # % connections with SYN errors
    "srv_serror_rate",    # % connections with SYN errors (same service)
    "rerror_rate",        # % connections with REJ errors
    "srv_rerror_rate",    # % connections with REJ errors (same service)
    "same_srv_rate",      # % connections to same service
    "diff_srv_rate",      # % connections to different services
    "srv_diff_host_rate", # % connections to different hosts (same service)
    "dst_host_count",     # connections to same destination host
    "dst_host_srv_count", # connections to same destination service
    "dst_host_same_srv_rate",      # % connections to same service (dest host)
    "dst_host_diff_srv_rate",      # % connections to different services (dest)
    "dst_host_same_src_port_rate", # % connections from same source port
    "dst_host_srv_diff_host_rate", # % connections from different hosts (srv)
    "dst_host_serror_rate",        # % connections with SYN errors (dest host)
    "dst_host_srv_serror_rate",    # % SYN errors (dest host, same service)
    "dst_host_rerror_rate",        # % REJ errors (dest host)
    "dst_host_srv_rerror_rate",    # % REJ errors (dest host, same service)
    "label"               # the answer: "normal" or an attack name
]

# ── Features chosen for the quantum model ─────────────────────────────────────
# We pick 2 features for the quantum SVM because each feature = 1 qubit,
# and more qubits = exponentially slower simulation.
# These two are chosen because they're the strongest signals for intrusion:
#   src_bytes  - attackers often send abnormal amounts of data
#   dst_bytes  - and receive abnormal responses
QUANTUM_FEATURES = ["src_bytes", "dst_bytes"]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Read the KDD Cup 99 data into a DataFrame.
    Handles two formats automatically:
      - CSV with header: saved by sklearn's fetch_kddcup99 (our case)
      - Raw no-header format: the original kddcup.data file from UCI

    We detect which format it is by peeking at the first line.
    If it contains 'duration' it has a header row; otherwise it doesn't.
    """
    # Peek at the first line to detect format
    with open(filepath, "r") as f:
        first_line = f.readline().strip()

    if "duration" in first_line:
        # CSV format from sklearn fetch_kddcup99
        # Has a header row, and the label column is called 'labels'
        df = pd.read_csv(filepath)

        # sklearn saves label values as byte strings like b'normal.'
        # We decode them to regular strings
        df["labels"] = df["labels"].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x)
        )

        # Rename 'labels' -> 'label' to match the rest of our pipeline
        df = df.rename(columns={"labels": "label"})
    else:
        # Raw UCI format - no header row, attach column names manually
        df = pd.read_csv(
            filepath,
            names=COLUMN_NAMES,
            header=None
        )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three columns are text-based: protocol_type, service, flag.
    Machine learning models need numbers, not strings.
    LabelEncoder converts each unique string to an integer:
      e.g. "tcp" -> 0, "udp" -> 1, "icmp" -> 2
    """
    df = df.copy()  # never modify the original dataframe in place

    text_columns = ["protocol_type", "service", "flag"]

    for col in text_columns:
        encoder = LabelEncoder()
        # fit_transform: first learns all unique values, then converts them
        df[col] = encoder.fit_transform(df[col])

    return df


def binarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    The dataset has many specific attack types: "neptune", "portsweep",
    "smurf", "back", etc. We don't care about the specific attack type --
    we just want to know: normal (0) or attack (1)?

    This collapses all attack subtypes into a single class.
    """
    df = df.copy()

    # Strip whitespace and trailing periods from label values
    # sklearn sometimes saves them as "normal." or " normal."
    # Labels look like: b'normal.' — strip b' prefix, trailing quote and period
    df["label"] = df["label"].str.replace("b'", "", regex=False).str.strip("'.").str.strip()

    # Any label that is NOT "normal" becomes 1 (attack)
    df["label"] = df["label"].apply(
        lambda x: 0 if x == "normal" else 1
    )

    # Rename to make intent clear
    df = df.rename(columns={"label": "is_attack"})

    return df


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """
    MinMaxScaler scales every feature to the range [0, 1].

    Why do we need this?
    - src_bytes can range from 0 to 1,000,000+
    - logged_in is just 0 or 1
    Without scaling, src_bytes would completely dominate the SVM's
    distance calculations just because its numbers are bigger.

    IMPORTANT: we fit the scaler ONLY on training data, then apply
    it to test data. Why? Because in real life you don't know the
    test data when you're training. Fitting on test data would be
    "cheating" -- called data leakage.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # learn scale from train
    X_test_scaled = scaler.transform(X_test)         # apply same scale to test
    return X_train_scaled, X_test_scaled, scaler


def get_splits(
    filepath: str,
    n_samples: int = 2000,
    n_quantum_samples: int = 200,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Full pipeline: load -> encode -> binarize -> split -> scale.

    Returns two pairs of train/test splits:
      1. Classical split: all 41 features, 2000 samples
      2. Quantum split: 2 features only, 200 samples
    """

    # Step 1: Load raw data
    df = load_data(filepath)

    # Step 2: Encode text columns -> integers
    df = encode_categoricals(df)

    # Step 3: Collapse attack labels -> binary (0 = normal, 1 = attack)
    df = binarize_labels(df)

    # Step 4: Separate features (X) from label (y)
    X = df.drop(columns=["is_attack"]).values
    y = df["is_attack"].values

    # Step 5: Sample a balanced subset
    def balanced_sample(X, y, n):
        """Take n/2 normal rows and n/2 attack rows."""
        idx_normal = np.where(y == 0)[0]
        idx_attack = np.where(y == 1)[0]

        rng = np.random.default_rng(random_state)
        chosen_normal = rng.choice(idx_normal, size=n // 2, replace=False)
        chosen_attack = rng.choice(idx_attack, size=n // 2, replace=False)

        idx = np.concatenate([chosen_normal, chosen_attack])
        rng.shuffle(idx)
        return X[idx], y[idx]

    # Classical split -- all features, larger sample
    X_cls, y_cls = balanced_sample(X, y, n_samples)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=test_size, random_state=random_state
    )
    X_train_c, X_test_c, scaler_c = scale_features(X_train_c, X_test_c)

    # Quantum split -- only 2 features, smaller sample
    feature_indices = [COLUMN_NAMES.index(f) for f in QUANTUM_FEATURES]
    X_q = X[:, feature_indices]

    X_q_sub, y_q_sub = balanced_sample(X_q, y, n_quantum_samples)
    X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
        X_q_sub, y_q_sub, test_size=test_size, random_state=random_state
    )
    X_train_q, X_test_q, scaler_q = scale_features(X_train_q, X_test_q)

    return {
        "classical": {
            "X_train": X_train_c,
            "X_test":  X_test_c,
            "y_train": y_train_c,
            "y_test":  y_test_c,
        },
        "quantum": {
            "X_train": X_train_q,
            "X_test":  X_test_q,
            "y_train": y_train_q,
            "y_test":  y_test_q,
        }
    }


if __name__ == "__main__":
    splits = get_splits("../data/kddcup.data")
    c = splits["classical"]
    q = splits["quantum"]
    print(f"Classical  -- train: {c['X_train'].shape}, test: {c['X_test'].shape}")
    print(f"Quantum    -- train: {q['X_train'].shape}, test: {q['X_test'].shape}")
    print(f"Label balance (quantum train): {np.bincount(q['y_train'])}")
    print("Preprocessing OK")