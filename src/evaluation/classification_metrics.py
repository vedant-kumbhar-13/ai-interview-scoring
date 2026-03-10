"""
Classification Accuracy Matrix
==============================
Computes classification-style evaluation metrics for all 4 regression
models by discretizing continuous scores into integer classes (1–5).

Since the models are regressors outputting scores in [1, 5], we round
predicted and actual values to the nearest integer to obtain class labels,
then compute standard classification metrics.

Metrics produced (weighted average across classes):
    - Accuracy
    - Precision
    - Recall
    - F1 Score

Outputs
-------
* ``results/tables/classification_metrics.csv``     — CSV comparison table
* ``results/tables/classification_metrics_latex.tex``— LaTeX table for paper
* ``results/tables/classification_metrics.json``     — detailed per-model JSON

Author: Research Team
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    ("Random Forest", "random_forest_predictions.npy"),
    ("SVR",           "svr_predictions.npy"),
    ("Bi-LSTM",       "lstm_predictions.npy"),
    ("BERT",          "bert_predictions.npy"),
]

SCORE_CLASSES = [1, 2, 3, 4, 5]


# ── Helper functions ──────────────────────────────────────────────────────

def discretize_scores(scores: np.ndarray) -> np.ndarray:
    """Round continuous scores to nearest integer and clip to [1, 5].

    Parameters
    ----------
    scores : np.ndarray
        Continuous scores, typically in [1.0, 5.0].

    Returns
    -------
    np.ndarray
        Integer class labels in {1, 2, 3, 4, 5}.
    """
    return np.clip(np.round(scores), 1, 5).astype(int)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute accuracy, precision, recall, and F1 for discretized scores.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer class labels.
    y_pred : np.ndarray
        Predicted integer class labels.

    Returns
    -------
    dict
        Summary metrics (weighted avg) and per-class breakdown.
    """
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average="weighted", zero_division=0,
    )
    recall = recall_score(
        y_true, y_pred, average="weighted", zero_division=0,
    )
    f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0,
    )

    # Per-class breakdown for the detailed JSON report
    report = classification_report(
        y_true, y_pred,
        labels=SCORE_CLASSES,
        target_names=[f"Score {c}" for c in SCORE_CLASSES],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy":  round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
        "f1_score":  round(float(f1), 4),
        "per_class": report,
    }


# ── Main pipeline function ───────────────────────────────────────────────

def generate_classification_matrix(
    tables_dir: str,
    processed_dir: str,
) -> pd.DataFrame:
    """Generate classification accuracy matrix for all 4 models.

    Parameters
    ----------
    tables_dir : str
        Directory containing ``*_predictions.npy`` and for output files.
    processed_dir : str
        Directory containing ``y_test.npy``.

    Returns
    -------
    pd.DataFrame
        Classification metrics comparison table.
    """
    logger.info("═" * 60)
    logger.info("  CLASSIFICATION ACCURACY MATRIX")
    logger.info("═" * 60)

    # ── Load ground truth ─────────────────────────────────────────────
    y_test_path = os.path.join(processed_dir, "y_test.npy")
    if not os.path.exists(y_test_path):
        raise FileNotFoundError(f"Ground truth not found: {y_test_path}")

    y_test_raw = np.load(y_test_path)
    y_test_classes = discretize_scores(y_test_raw)
    logger.info(
        "Ground truth: %d samples, class distribution: %s",
        len(y_test_classes),
        dict(zip(*np.unique(y_test_classes, return_counts=True))),
    )

    # ── Compute metrics for each model ────────────────────────────────
    summary_records = []
    detailed_results = {}

    for model_name, pred_file in MODEL_CONFIGS:
        pred_path = os.path.join(tables_dir, pred_file)
        if not os.path.exists(pred_path):
            logger.warning("Missing predictions: %s — skipping %s.",
                           pred_path, model_name)
            continue

        y_pred_raw = np.load(pred_path)
        # Ensure matching lengths (BERT test set may differ slightly)
        n = min(len(y_test_raw), len(y_pred_raw))
        y_true = y_test_classes[:n]
        y_pred = discretize_scores(y_pred_raw[:n])

        metrics = compute_classification_metrics(y_true, y_pred)

        summary_records.append({
            "Model":     model_name,
            "Accuracy":  metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall":    metrics["recall"],
            "F1 Score":  metrics["f1_score"],
        })

        detailed_results[model_name] = metrics

        logger.info(
            "  %-14s | Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f",
            model_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
        )

    if not summary_records:
        raise FileNotFoundError(
            "No prediction files found. Train models first."
        )

    comparison_df = pd.DataFrame(summary_records)

    # ── Save CSV ──────────────────────────────────────────────────────
    os.makedirs(tables_dir, exist_ok=True)

    csv_path = os.path.join(tables_dir, "classification_metrics.csv")
    comparison_df.to_csv(csv_path, index=False)
    logger.info("CSV saved to %s", csv_path)

    # ── Save detailed JSON ────────────────────────────────────────────
    json_path = os.path.join(tables_dir, "classification_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, default=str)
    logger.info("JSON saved to %s", json_path)

    # ── Save LaTeX table ──────────────────────────────────────────────
    latex_path = os.path.join(tables_dir, "classification_metrics_latex.tex")
    _save_latex_table(comparison_df, latex_path)
    logger.info("LaTeX table saved to %s", latex_path)

    return comparison_df


def _save_latex_table(df: pd.DataFrame, output_path: str) -> None:
    """Generate a publication-ready LaTeX table.

    Parameters
    ----------
    df : pd.DataFrame
        Classification metrics comparison table.
    output_path : str
        Path to save the ``.tex`` file.
    """
    # Find the best value in each metric column for bold highlighting
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    best_indices = {col: df[col].idxmax() for col in metric_cols}

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Classification Accuracy Matrix — Model Evaluation}",
        r"  \label{tab:classification_metrics}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Accuracy} & \textbf{Precision} "
        r"& \textbf{Recall} & \textbf{F1 Score} \\",
        r"    \midrule",
    ]

    for idx, row in df.iterrows():
        cells = [f"    {row['Model']}"]
        for col in metric_cols:
            val = f"{row[col]:.4f}"
            if idx == best_indices[col]:
                val = r"\textbf{" + val + "}"
            cells.append(val)
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \vspace{4pt}",
        r"  \parbox{0.9\linewidth}{\footnotesize\textit{Note:} "
        r"Continuous regression scores were discretized to integer "
        r"classes (1--5) via rounding. Precision, Recall, and F1 are "
        r"weighted averages across all classes. Best values per metric "
        r"are shown in \textbf{bold}.}",
        r"\end{table}",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── CLI entry-point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    BASE = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    PROCESSED  = os.path.join(BASE, "data", "processed")
    TABLES_DIR = os.path.join(BASE, "results", "tables")

    df = generate_classification_matrix(TABLES_DIR, PROCESSED)
    print("\nClassification Accuracy Matrix:")
    print(df.to_string(index=False))
