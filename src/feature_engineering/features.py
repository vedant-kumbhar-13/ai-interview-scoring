"""
Feature Engineering Module
==========================
Extracts linguistic, readability, and sentiment features from text data.

Feature groups
--------------
**Linguistic** (surface-level):
    ``word_count``, ``sentence_count``, ``avg_word_length``,
    ``lexical_diversity`` (type-token ratio)

**Readability** (Flesch–Kincaid family via *textstat*):
    ``flesch_reading_ease``, ``flesch_kincaid_grade``,
    ``gunning_fog``, ``automated_readability_index``

**Sentiment** (VADER):
    ``vader_neg``, ``vader_neu``, ``vader_pos``, ``vader_compound``

All features are deterministic given the input text.

Author: Research Team
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Lazy imports & NLTK setup ─────────────────────────────────────────────

def _ensure_deps():
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("vader_lexicon", quiet=True)


# ── Individual feature functions ──────────────────────────────────────────

def word_count(text: str) -> int:
    """Number of whitespace-delimited tokens."""
    return len(text.split())


def sentence_count(text: str) -> int:
    """Number of sentences (NLTK sent_tokenize)."""
    from nltk.tokenize import sent_tokenize
    return max(len(sent_tokenize(text)), 1)


def avg_word_length(text: str) -> float:
    """Mean character-length of tokens."""
    words = text.split()
    return np.mean([len(w) for w in words]) if words else 0.0


def lexical_diversity(text: str) -> float:
    """Type-token ratio (unique words / total words)."""
    words = text.lower().split()
    return len(set(words)) / max(len(words), 1)


def readability_scores(text: str) -> dict[str, float]:
    """Compute multiple readability indices via *textstat*.

    Returns a dict with keys:
    ``flesch_reading_ease``, ``flesch_kincaid_grade``,
    ``gunning_fog``, ``automated_readability_index``.
    """
    import textstat
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "automated_readability_index": textstat.automated_readability_index(text),
    }


def vader_sentiment(text: str) -> dict[str, float]:
    """VADER sentiment scores.

    Returns dict with keys:
    ``vader_neg``, ``vader_neu``, ``vader_pos``, ``vader_compound``.
    """
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return {
        "vader_neg": scores["neg"],
        "vader_neu": scores["neu"],
        "vader_pos": scores["pos"],
        "vader_compound": scores["compound"],
    }


# ── Aggregate feature extraction ──────────────────────────────────────────

def extract_features(df: pd.DataFrame,
                     text_col: str = "answer") -> pd.DataFrame:
    """Compute all engineered features and append them to *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *text_col*.
    text_col : str
        Column with raw (or cleaned) text.

    Returns
    -------
    pd.DataFrame
        Original columns plus all feature columns.
    """
    _ensure_deps()

    logger.info("Extracting features from %d texts (column='%s') …",
                len(df), text_col)
    out = df.copy()
    texts = out[text_col].astype(str)

    # ── Linguistic features ───────────────────────────────────────────
    out["word_count"]        = texts.apply(word_count)
    out["sentence_count"]    = texts.apply(sentence_count)
    out["avg_word_length"]   = texts.apply(avg_word_length)
    out["lexical_diversity"] = texts.apply(lexical_diversity)
    logger.info("  ✔ Linguistic features")

    # ── Readability features ──────────────────────────────────────────
    read_df = texts.apply(readability_scores).apply(pd.Series)
    out = pd.concat([out, read_df], axis=1)
    logger.info("  ✔ Readability features")

    # ── Sentiment features ────────────────────────────────────────────
    sent_df = texts.apply(vader_sentiment).apply(pd.Series)
    out = pd.concat([out, sent_df], axis=1)
    logger.info("  ✔ Sentiment features")

    logger.info("Feature extraction complete: %d features added.",
                out.shape[1] - df.shape[1])
    return out


# ── Feature column names (for downstream use) ────────────────────────────

FEATURE_COLUMNS = [
    "word_count", "sentence_count", "avg_word_length", "lexical_diversity",
    "flesch_reading_ease", "flesch_kincaid_grade",
    "gunning_fog", "automated_readability_index",
    "vader_neg", "vader_neu", "vader_pos", "vader_compound",
]


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
    df_feat = extract_features(df, text_col="answer")

    out_path = os.path.join(PROCESSED, "features.csv")
    df_feat.to_csv(out_path, index=False)
    logger.info("Saved features to %s  (%d rows × %d cols)",
                out_path, *df_feat.shape)
