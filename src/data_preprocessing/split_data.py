"""
Dataset Splitting & Matrix Construction
========================================
Constructs final feature matrices for the train/test paradigm:

    **Training** = Essay + Short-answer data  (transfer-learning signal)
    **Testing**  = Interview data              (generalisation evaluation)

Feature vector (per sample):
    [hand-crafted features (12-d)]  ⊕  [SBERT embedding (384-d)]  =  396-d

Outputs saved to ``data/processed/``:
    ``X_train.npy``, ``y_train.npy``,
    ``X_test.npy``,  ``y_test.npy``

Author: Research Team
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

# Feature columns produced by features.py
FEATURE_COLUMNS = [
    "word_count", "sentence_count", "avg_word_length", "lexical_diversity",
    "flesch_reading_ease", "flesch_kincaid_grade",
    "gunning_fog", "automated_readability_index",
    "vader_neg", "vader_neu", "vader_pos", "vader_compound",
]


def build_matrices(
    features_path: str,
    embeddings_path: str,
    output_dir: str,
) -> dict[str, np.ndarray]:
    """Combine features + embeddings and split by dataset source.

    Parameters
    ----------
    features_path : str
        Path to ``features.csv`` (must have a ``source`` column).
    embeddings_path : str
        Path to ``embeddings.npy``.
    output_dir : str
        Directory for output ``.npy`` files.

    Returns
    -------
    dict
        Keys: ``X_train``, ``y_train``, ``X_test``, ``y_test``.
    """
    logger.info("Loading features from %s", features_path)
    df = pd.read_csv(features_path)

    logger.info("Loading embeddings from %s", embeddings_path)
    embeddings = np.load(embeddings_path)

    assert len(df) == embeddings.shape[0], (
        f"Row mismatch: features={len(df)}, embeddings={embeddings.shape[0]}"
    )

    # ── Separate train (essay + short_answer) vs test (interview) ─────
    train_mask = df["source"].isin(["essay", "short_answer"])
    test_mask  = df["source"] == "interview"

    logger.info("Train samples: %d | Test samples: %d",
                train_mask.sum(), test_mask.sum())

    # ── Extract hand-crafted features ─────────────────────────────────
    feat_train = df.loc[train_mask, FEATURE_COLUMNS].values.astype(np.float32)
    feat_test  = df.loc[test_mask,  FEATURE_COLUMNS].values.astype(np.float32)

    # Handle NaN / Inf in hand-crafted features
    feat_train = np.nan_to_num(feat_train, nan=0.0, posinf=0.0, neginf=0.0)
    feat_test  = np.nan_to_num(feat_test,  nan=0.0, posinf=0.0, neginf=0.0)

    # ── Standardise hand-crafted features (fit on train only) ─────────
    scaler = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_test  = scaler.transform(feat_test)

    # Save scaler for reproducibility
    scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info("Saved feature scaler to %s", scaler_path)

    # ── Concatenate: [features | embeddings] ──────────────────────────
    emb_train = embeddings[train_mask.values]
    emb_test  = embeddings[test_mask.values]

    X_train = np.hstack([feat_train, emb_train]).astype(np.float32)
    X_test  = np.hstack([feat_test,  emb_test]).astype(np.float32)

    y_train = df.loc[train_mask, "score"].values.astype(np.float32)
    y_test  = df.loc[test_mask,  "score"].values.astype(np.float32)

    logger.info("X_train: %s | X_test: %s", X_train.shape, X_test.shape)
    logger.info("y_train: mean=%.3f std=%.3f | y_test: mean=%.3f std=%.3f",
                y_train.mean(), y_train.std(), y_test.mean(), y_test.std())

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    for name, arr in [("X_train", X_train), ("y_train", y_train),
                      ("X_test", X_test),   ("y_test", y_test)]:
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, arr)
        logger.info("Saved %s → %s", name, path)

    return {"X_train": X_train, "y_train": y_train,
            "X_test": X_test,   "y_test": y_test}


# ── CLI entry-point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROCESSED = os.path.join(BASE, "data", "processed")

    build_matrices(
        features_path=os.path.join(PROCESSED, "features.csv"),
        embeddings_path=os.path.join(PROCESSED, "embeddings.npy"),
        output_dir=PROCESSED,
    )
