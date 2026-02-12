"""
Text Cleaning Pipeline
======================
Provides NLP-based text preprocessing for the AI Interview Scoring pipeline.

Processing steps (in order):
1. Unicode normalisation (NFKD)
2. Lower-casing
3. Contraction expansion
4. Punctuation removal
5. Stop-word removal (NLTK English)
6. Lemmatisation (WordNet via NLTK)
7. Whitespace normalisation

Design notes
-------------
* ``clean_text`` is a *pure function* operating on a single string.
* ``apply_cleaning`` operates on a DataFrame column, adding a ``clean_answer``
  column without mutating the original ``answer`` column — preserving raw data
  for later embedding models (BERT) that benefit from raw text.

Author: Research Team
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata

import pandas as pd

logger = logging.getLogger(__name__)

# ── Lazy NLTK bootstrapping ───────────────────────────────────────────────

_NLTK_READY = False


def _ensure_nltk() -> None:
    """Download required NLTK resources once per session."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk
    for resource in ("punkt", "punkt_tab", "stopwords", "wordnet",
                     "omw-1.4", "averaged_perceptron_tagger",
                     "averaged_perceptron_tagger_eng"):
        nltk.download(resource, quiet=True)
    _NLTK_READY = True


# ── Contraction map ───────────────────────────────────────────────────────

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'ve": " have", "'m": " am",
}

_CONTRACTION_RE = re.compile(
    "(" + "|".join(re.escape(k) for k in CONTRACTIONS) + ")",
    flags=re.IGNORECASE,
)


def _expand_contractions(text: str) -> str:
    def _replace(match):
        return CONTRACTIONS.get(match.group(0).lower(), match.group(0))
    return _CONTRACTION_RE.sub(_replace, text)


# ── Core cleaning function ────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Apply a full NLP cleaning pipeline to *text*.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned, lemmatised text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    _ensure_nltk()

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFKD", text)

    # 2. Lower-case
    text = text.lower()

    # 3. Expand contractions
    text = _expand_contractions(text)

    # 4. Remove punctuation & digits (keep alphabetic tokens)
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5. Tokenise
    tokens = word_tokenize(text)

    # 6. Remove stopwords & lemmatise
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 1
    ]

    # 7. Rejoin & normalise whitespace
    return " ".join(tokens)


# ── DataFrame wrapper ─────────────────────────────────────────────────────

def apply_cleaning(df: pd.DataFrame, text_col: str = "answer") -> pd.DataFrame:
    """Add a ``clean_answer`` column to *df* without mutating the original.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *text_col*.
    text_col : str
        Column name holding raw text (default ``"answer"``).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an additional ``clean_answer`` column.
    """
    logger.info("Cleaning %d texts …", len(df))
    out = df.copy()
    out["clean_answer"] = out[text_col].astype(str).apply(clean_text)

    # Report empties
    empty_mask = out["clean_answer"].str.strip() == ""
    if empty_mask.any():
        logger.warning("  %d rows produced empty cleaned text.", empty_mask.sum())

    logger.info("Cleaning complete.")
    return out


# ── CLI entry-point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROCESSED = os.path.join(BASE, "data", "processed")
    os.makedirs(PROCESSED, exist_ok=True)

    # Load all datasets
    from load_data import load_all_datasets

    datasets = load_all_datasets(
        essay_path=os.path.join(BASE, "training_data", "archive (1)", "training_set_rel3.tsv"),
        short_answer_dir=os.path.join(BASE, "training_data", "archive (3)"),
        interview_dir=os.path.join(BASE, "testing_data"),
    )

    # Clean each dataset and save combined
    cleaned_frames = []
    for name, df in datasets.items():
        logger.info("Processing %s dataset …", name)
        cleaned = apply_cleaning(df)
        cleaned["source"] = name
        cleaned_frames.append(cleaned)

    combined = pd.concat(cleaned_frames, ignore_index=True)
    out_path = os.path.join(PROCESSED, "clean_dataset.csv")
    combined.to_csv(out_path, index=False)
    logger.info("Saved cleaned dataset to %s  (%d rows)", out_path, len(combined))
