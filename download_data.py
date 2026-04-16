"""
download_data.py
----------------
Auto-downloads the KDD Cup 99 dataset using sklearn's built-in fetcher.
Run this once before starting the app if data/kddcup.data doesn't exist:
    python download_data.py
Also called automatically by the Streamlit app on first load.
"""

import os
import pandas as pd


def ensure_data_exists(data_path: str):
    """
    If the dataset doesn't exist at data_path, download it via sklearn.
    This runs automatically on Streamlit Cloud where we can't commit large files.
    """
    if os.path.exists(data_path):
        return  # already there, nothing to do

    print("Dataset not found — downloading via sklearn (this takes ~30 seconds)...")

    from sklearn.datasets import fetch_kddcup99

    data = fetch_kddcup99(subset=None, shuffle=True, random_state=42, as_frame=True)
    df = data.frame

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)

    print(f"Dataset saved to {data_path} — shape: {df.shape}")


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "data", "kddcup.data")
    ensure_data_exists(path)
    print("Ready.")