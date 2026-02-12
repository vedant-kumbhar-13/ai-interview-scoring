"""
Data Loading Module
===================
Provides reusable loader functions for the three dataset types used in this
research pipeline.  Each loader returns a standardised ``pandas.DataFrame``
with two columns: **answer** (str) and **score** (float, normalised to 1–5).

Normalisation rationale
-----------------------
* Essay dataset ``domain1_score`` varies per essay-set (0–60).  We apply
  **min-max scaling per essay-set** then linearly map to [1, 5].
* Short-answer datasets have scores in [0, 1]; mapped linearly to [1, 5].
* Interview datasets already embed human scores on a 1–5 ordinal scale;
  the final score is the mean of four criteria.

Design decisions
----------------
* Loaders are **pure functions** (no side-effects beyond reading files).
* Missing / NaN text rows are silently dropped with a log warning.
* Encoding for the essay TSV is ``latin-1`` to handle special characters.

Author: Research Team
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── helpers ────────────────────────────────────────────────────────────────

def _min_max_to_range(series: pd.Series, new_min: float = 1.0,
                       new_max: float = 5.0) -> pd.Series:
    """Scale *series* from its observed range to [new_min, new_max]."""
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series([(new_min + new_max) / 2] * len(series),
                         index=series.index)
    return (series - s_min) / (s_max - s_min) * (new_max - new_min) + new_min


# ── public loaders ─────────────────────────────────────────────────────────

def load_essay_dataset(path: str) -> pd.DataFrame:
    """Load the Hewlett Foundation Automated Essay Scoring dataset.

    Parameters
    ----------
    path : str
        Path to ``training_set_rel3.tsv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``answer`` (str), ``score`` (float in [1, 5]).
    """
    logger.info("Loading essay dataset from %s …", path)
    df = pd.read_csv(path, sep="\t", encoding="latin-1")

    # Validate required columns
    required = {"essay", "essay_set", "domain1_score"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}, got {list(df.columns)}")

    df = df[["essay_set", "essay", "domain1_score"]].copy()
    df.rename(columns={"essay": "answer", "domain1_score": "score"}, inplace=True)

    # Drop rows with missing text or score
    before = len(df)
    df.dropna(subset=["answer", "score"], inplace=True)
    if (dropped := before - len(df)) > 0:
        logger.warning("Dropped %d rows with missing text or score.", dropped)

    # Normalise score per essay_set to [1, 5]
    df["score"] = df.groupby("essay_set")["score"].transform(_min_max_to_range)
    df.drop(columns=["essay_set"], inplace=True)

    df.reset_index(drop=True, inplace=True)
    logger.info("Essay dataset loaded: %d rows.", len(df))
    return df


def load_short_answer_dataset(directory: str) -> pd.DataFrame:
    """Load and concatenate all short-answer grading CSVs in *directory*.

    Expects CSV files with columns ``jawaban`` (answer text) and ``skor``
    (score in [0, 1]).

    Parameters
    ----------
    directory : str
        Directory containing one or more CSV files (e.g. ``archive (3)/``).

    Returns
    -------
    pd.DataFrame
        Columns: ``answer`` (str), ``score`` (float in [1, 5]).
    """
    logger.info("Loading short-answer dataset(s) from %s …", directory)
    csv_files = sorted(
        f for f in os.listdir(directory) if f.endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    frames = []
    for fname in csv_files:
        fpath = os.path.join(directory, fname)
        tmp = pd.read_csv(fpath)
        if "jawaban" not in tmp.columns or "skor" not in tmp.columns:
            logger.warning("Skipping %s — missing 'jawaban' or 'skor'.", fname)
            continue
        tmp = tmp[["jawaban", "skor"]].copy()
        tmp.rename(columns={"jawaban": "answer", "skor": "score"}, inplace=True)
        frames.append(tmp)
        logger.info("  • %s — %d rows", fname, len(tmp))

    df = pd.concat(frames, ignore_index=True)

    # Drop missing
    before = len(df)
    df.dropna(subset=["answer", "score"], inplace=True)
    if (dropped := before - len(df)) > 0:
        logger.warning("Dropped %d rows with missing data.", dropped)

    # Normalise from [0, 1] → [1, 5]
    df["score"] = df["score"] * 4.0 + 1.0

    df.reset_index(drop=True, inplace=True)
    logger.info("Short-answer dataset loaded: %d rows.", len(df))
    return df


def load_interview_dataset(directory: str) -> pd.DataFrame:
    """Load interview scoring sheets and compute the average human score.

    Each XLSX file is expected to contain columns for:
    ``Answer``, ``Relevance``, ``Technical Depth``,
    ``Communication Clarity``, ``Confidence & Tone``, and ``evaluator_name``.

    The final score is the **mean across the four criteria**, averaged over
    all evaluators per (Candidate, Question) pair.  This yields a single
    consensus score suitable for model evaluation.

    Parameters
    ----------
    directory : str
        Directory containing evaluator XLSX files.

    Returns
    -------
    pd.DataFrame
        Columns: ``answer`` (str), ``score`` (float), plus evaluator-level
        metadata for inter-rater analysis.
    """
    logger.info("Loading interview dataset(s) from %s …", directory)
    xlsx_files = sorted(
        f for f in os.listdir(directory) if f.endswith(".xlsx")
    )
    if not xlsx_files:
        raise FileNotFoundError(f"No XLSX files found in {directory}")

    frames = []
    for fname in xlsx_files:
        fpath = os.path.join(directory, fname)
        tmp = pd.read_excel(fpath)
        frames.append(tmp)
        logger.info("  • %s — %d rows, evaluator(s): %s",
                     fname, len(tmp),
                     tmp["evaluator_name"].unique() if "evaluator_name" in tmp.columns else "unknown")

    df = pd.concat(frames, ignore_index=True)

    # Compute per-row average of the four scoring criteria
    # Try both naming conventions for robustness
    criteria_full = ["Relevance", "Technical Depth",
                     "Communication Clarity", "Confidence & Tone"]
    criteria_short = ["relevance_score", "technical_score",
                      "clarity_score", "confidence_score"]

    if all(c in df.columns for c in criteria_full):
        criteria = criteria_full
    elif all(c in df.columns for c in criteria_short):
        criteria = criteria_short
    else:
        raise ValueError(
            f"Missing scoring columns. Expected {criteria_full} or {criteria_short}, "
            f"got {list(df.columns)}"
        )

    df["criteria_avg"] = df[criteria].mean(axis=1)

    # Average across evaluators per (Candidate, Question)
    grouped = (
        df.groupby(["Candidate_ID", "Question"])
        .agg(
            answer=("Answer", "first"),
            score=("criteria_avg", "mean"),
        )
        .reset_index()
    )

    # Also keep evaluator-level data for inter-rater reliability analysis
    eval_cols = ["Candidate_ID", "Question", "evaluator_name", "criteria_avg"]
    eval_cols = [c for c in eval_cols if c in df.columns]
    evaluator_data = df[eval_cols].copy()
    if "criteria_avg" in evaluator_data.columns:
        evaluator_data.rename(columns={"criteria_avg": "score"}, inplace=True)

    # Drop missing
    before = len(grouped)
    grouped.dropna(subset=["answer", "score"], inplace=True)
    if (dropped := before - len(grouped)) > 0:
        logger.warning("Dropped %d rows with missing data.", dropped)

    logger.info("Interview dataset loaded: %d unique (candidate, question) pairs.",
                len(grouped))
    logger.info("Score range: [%.2f, %.2f]", grouped["score"].min(),
                grouped["score"].max())

    # Store evaluator-level data as attribute for later inter-rater analysis
    grouped.attrs["evaluator_data"] = evaluator_data

    return grouped


# ── convenience ────────────────────────────────────────────────────────────

def load_all_datasets(
    essay_path: str,
    short_answer_dir: str,
    interview_dir: str,
) -> dict[str, pd.DataFrame]:
    """Load all three datasets and return as a dict.

    Returns
    -------
    dict
        Keys: ``"essay"``, ``"short_answer"``, ``"interview"``.
    """
    return {
        "essay": load_essay_dataset(essay_path),
        "short_answer": load_short_answer_dataset(short_answer_dir),
        "interview": load_interview_dataset(interview_dir),
    }


# ── CLI entry-point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    datasets = load_all_datasets(
        essay_path=os.path.join(BASE, "training_data", "archive (1)", "training_set_rel3.tsv"),
        short_answer_dir=os.path.join(BASE, "training_data", "archive (3)"),
        interview_dir=os.path.join(BASE, "testing_data"),
    )

    for name, df in datasets.items():
        print(f"\n{'='*60}")
        print(f"  {name.upper()} DATASET")
        print(f"{'='*60}")
        print(f"  Shape : {df.shape}")
        print(f"  Score : mean={df['score'].mean():.3f}, "
              f"std={df['score'].std():.3f}, "
              f"min={df['score'].min():.3f}, max={df['score'].max():.3f}")
        print(f"  Sample:\n{df.head(2).to_string()}")
