"""
Sentence-BERT Embedding Generator
==================================
Generates dense semantic embeddings using the ``all-MiniLM-L6-v2``
Sentence-Transformer model (384-dimensional).

These embeddings capture sentence-level semantics and are concatenated
with hand-crafted features to form the final input vectors for
traditional ML models (Random Forest, SVR).

Design notes
------------
* Embeddings are computed in batches (default 64) for GPU efficiency.
* A progress bar (tqdm) is shown for long text lists.
* The function is deterministic (model uses eval mode by default).

Author: Research Team
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def generate_embeddings(
    text_list: list[str] | pd.Series,
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Generate Sentence-BERT embeddings for a list of texts.

    Parameters
    ----------
    text_list : list[str] or pd.Series
        Input texts.
    model_name : str
        Sentence-Transformers model identifier.
    batch_size : int
        Encoding batch size.
    show_progress : bool
        Show tqdm progress bar.

    Returns
    -------
    np.ndarray
        Shape ``(len(text_list), embedding_dim)``.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading Sentence-Transformer model '%s' …", model_name)
    model = SentenceTransformer(model_name)

    texts = list(text_list) if isinstance(text_list, pd.Series) else text_list
    # Replace any non-string / NaN entries
    texts = [str(t) if isinstance(t, str) and t.strip() else "" for t in texts]

    logger.info("Encoding %d texts (batch_size=%d) …", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalisation — improves cosine sim
    )

    logger.info("Embeddings shape: %s", embeddings.shape)
    return embeddings


# ── CLI entry-point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROCESSED = os.path.join(BASE, "data", "processed")

    clean_path = os.path.join(PROCESSED, "clean_dataset.csv")
    if not os.path.exists(clean_path):
        logger.error("clean_dataset.csv not found. Run clean_text.py first.")
        raise SystemExit(1)

    df = pd.read_csv(clean_path)

    # Use raw answer (not cleaned) — SBERT benefits from natural text
    embeddings = generate_embeddings(df["answer"])

    out_path = os.path.join(PROCESSED, "embeddings.npy")
    np.save(out_path, embeddings)
    logger.info("Saved embeddings to %s", out_path)
