"""
Model Comparison & Visualisation
=================================
Aggregates results from all trained models and produces:
1. A comparison table (``comparison.csv``)
2. A grouped bar chart (``model_comparison.png``)
3. Inter-rater reliability analysis (Human A vs Human B)
4. Human vs AI correlation analysis

This module computes the **key claim** of the research:
    "AI can evaluate interview answers with reliability
     comparable to human interviewers."

Author: Research Team
"""

from __future__ import annotations

import json
import logging
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

# Consistent styling
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
MODEL_NAMES = ["Random Forest", "SVR", "LSTM", "BERT"]
METRIC_FILES = [
    "random_forest_metrics.json",
    "svr_metrics.json",
    "lstm_metrics.json",
    "bert_metrics.json",
]


def load_all_metrics(tables_dir: str) -> pd.DataFrame:
    """Load metrics from all model JSON files into a DataFrame.

    Parameters
    ----------
    tables_dir : str
        Directory containing ``*_metrics.json`` files.

    Returns
    -------
    pd.DataFrame
        Columns: Model, Pearson, MAE, RMSE.
    """
    records = []
    for fname in METRIC_FILES:
        fpath = os.path.join(tables_dir, fname)
        if not os.path.exists(fpath):
            logger.warning("Missing: %s — skipping.", fpath)
            continue
        with open(fpath) as f:
            m = json.load(f)
        records.append({
            "Model":   m.get("model", fname.replace("_metrics.json", "")),
            "Pearson": m["pearson_r"],
            "MAE":     m["mae"],
            "RMSE":    m["rmse"],
        })

    if not records:
        raise FileNotFoundError(f"No metric files found in {tables_dir}")

    return pd.DataFrame(records)


def compute_inter_rater_reliability(interview_dir: str) -> dict:
    """Compute Pearson correlation between two human evaluators.

    Parameters
    ----------
    interview_dir : str
        Directory containing evaluator XLSX files.

    Returns
    -------
    dict
        Keys: ``evaluator_a``, ``evaluator_b``, ``pearson_r``, ``p_value``.
    """
    xlsx_files = sorted(
        f for f in os.listdir(interview_dir) if f.endswith(".xlsx")
    )
    if len(xlsx_files) < 2:
        logger.warning("Need ≥2 evaluator files for inter-rater reliability.")
        return {}

    frames = {}
    for fname in xlsx_files[:2]:  # first two evaluators
        fpath = os.path.join(interview_dir, fname)
        df = pd.read_excel(fpath)
        criteria_full = ["Relevance", "Technical Depth",
                         "Communication Clarity", "Confidence & Tone"]
        criteria_short = ["relevance_score", "technical_score",
                          "clarity_score", "confidence_score"]
        if all(c in df.columns for c in criteria_full):
            criteria = criteria_full
        elif all(c in df.columns for c in criteria_short):
            criteria = criteria_short
        else:
            logger.warning("Cannot find scoring columns in %s", fname)
            continue
        df["avg_score"] = df[criteria].mean(axis=1)
        evaluator_name = df["evaluator_name"].iloc[0] if "evaluator_name" in df.columns else fname
        frames[evaluator_name] = df[["Candidate_ID", "Question", "avg_score"]].copy()

    names = list(frames.keys())
    merged = frames[names[0]].merge(
        frames[names[1]],
        on=["Candidate_ID", "Question"],
        suffixes=("_A", "_B"),
    )

    if merged.empty:
        logger.warning("No matching (Candidate, Question) pairs between evaluators.")
        return {}

    r, p = pearsonr(merged["avg_score_A"], merged["avg_score_B"])

    result = {
        "evaluator_a": names[0],
        "evaluator_b": names[1],
        "n_pairs": len(merged),
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 6),
    }

    logger.info("Inter-rater reliability: r=%.4f (p=%.6f), n=%d",
                r, p, len(merged))
    return result


def plot_comparison(
    df: pd.DataFrame,
    figures_dir: str,
    inter_rater: dict | None = None,
) -> str:
    """Create a publication-quality grouped bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Comparison table with Model, Pearson, MAE, RMSE.
    figures_dir : str
        Output directory.
    inter_rater : dict, optional
        Inter-rater stats (plotted as reference line on Pearson subplot).

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "AI Interview Scoring — Model Performance Comparison",
        fontsize=16, fontweight="bold", y=1.02,
    )

    metrics = ["Pearson", "MAE", "RMSE"]
    ylabels = ["Pearson Correlation (r)", "Mean Absolute Error", "Root Mean Squared Error"]
    higher_better = [True, False, False]

    for ax, metric, ylabel, hb in zip(axes, metrics, ylabels, higher_better):
        values = df[metric].values
        bars = ax.bar(
            df["Model"], values,
            color=COLORS[:len(df)],
            edgecolor="white", linewidth=1.2,
            width=0.6, zorder=3,
        )

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        # Inter-rater reference line on Pearson plot
        if metric == "Pearson" and inter_rater and "pearson_r" in inter_rater:
            ax.axhline(
                inter_rater["pearson_r"], color="red", linestyle="--",
                linewidth=2, label=f"Human inter-rater (r={inter_rater['pearson_r']:.4f})",
                zorder=4,
            )
            ax.legend(loc="lower right", fontsize=9)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3, zorder=0)

        # Better y-axis bounds
        if hb:
            ax.set_ylim(0, min(max(values) * 1.3, 1.0))
        else:
            ax.set_ylim(0, max(values) * 1.3)

    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "model_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison chart saved to %s", out_path)
    return out_path


def plot_predictions_scatter(
    tables_dir: str,
    figures_dir: str,
    y_test: np.ndarray,
) -> str:
    """Scatter plots: predicted vs actual for each model.

    Returns
    -------
    str
        Path to saved figure.
    """
    pred_files = {
        "Random Forest": "random_forest_predictions.npy",
        "SVR": "svr_predictions.npy",
        "LSTM": "lstm_predictions.npy",
        "BERT": "bert_predictions.npy",
    }

    available = {name: np.load(os.path.join(tables_dir, f))
                 for name, f in pred_files.items()
                 if os.path.exists(os.path.join(tables_dir, f))}

    if not available:
        logger.warning("No prediction files found for scatter plots.")
        return ""

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle("Predicted vs Actual Scores", fontsize=16,
                 fontweight="bold", y=1.02)

    for ax, (name, y_pred), color in zip(axes, available.items(), COLORS):
        ax.scatter(y_test[:len(y_pred)], y_pred, alpha=0.5, s=30,
                   color=color, edgecolor="white", linewidth=0.5)
        ax.plot([1, 5], [1, 5], "k--", alpha=0.5, label="Perfect prediction")
        ax.set_xlabel("Actual Score", fontsize=11)
        ax.set_ylabel("Predicted Score", fontsize=11)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0.8, 5.2)
        ax.legend(fontsize=9)
        ax.set_aspect("equal")

    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "predictions_scatter.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Scatter plot saved to %s", out_path)
    return out_path


def generate_report(
    comparison_df: pd.DataFrame,
    inter_rater: dict,
    tables_dir: str,
) -> str:
    """Generate a textual summary of findings.

    Returns
    -------
    str
        Path to the summary report.
    """
    best = comparison_df.loc[comparison_df["Pearson"].idxmax()]

    lines = [
        "=" * 70,
        "  RESEARCH FINDINGS — AI Interview Answer Scoring",
        "=" * 70,
        "",
        "1. MODEL PERFORMANCE COMPARISON",
        "-" * 40,
        comparison_df.to_string(index=False),
        "",
        f"  Best model: {best['Model']} (Pearson r = {best['Pearson']:.4f})",
        "",
        "2. INTER-RATER RELIABILITY (Human Baseline)",
        "-" * 40,
    ]

    if inter_rater:
        lines += [
            f"  Evaluator A : {inter_rater.get('evaluator_a', 'N/A')}",
            f"  Evaluator B : {inter_rater.get('evaluator_b', 'N/A')}",
            f"  Pearson r   : {inter_rater.get('pearson_r', 'N/A')}",
            f"  p-value     : {inter_rater.get('p_value', 'N/A')}",
            f"  N pairs     : {inter_rater.get('n_pairs', 'N/A')}",
            "",
        ]

        if "pearson_r" in inter_rater and best["Pearson"] >= inter_rater["pearson_r"] * 0.9:
            lines.append(
                "  [SUPPORTED] CLAIM: AI achieves >=90% of human inter-rater reliability."
            )
        else:
            lines.append(
                "  [NOTE] AI has not yet reached human-level inter-rater reliability."
            )
    else:
        lines.append("  (Inter-rater data unavailable)")

    lines += ["", "=" * 70]

    report_text = "\n".join(lines)
    logger.info("\n%s", report_text)

    report_path = os.path.join(tables_dir, "research_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_path


# ── CLI entry-point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROCESSED    = os.path.join(BASE, "data", "processed")
    TABLES_DIR   = os.path.join(BASE, "results", "tables")
    FIGURES_DIR  = os.path.join(BASE, "results", "figures")
    INTERVIEW_DIR = os.path.join(BASE, "testing_data")

    # 1. Load all model metrics
    comparison_df = load_all_metrics(TABLES_DIR)
    logger.info("\n%s", comparison_df.to_string(index=False))

    # 2. Save comparison CSV
    csv_path = os.path.join(TABLES_DIR, "comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    logger.info("Saved comparison to %s", csv_path)

    # 3. Inter-rater reliability
    inter_rater = compute_inter_rater_reliability(INTERVIEW_DIR)

    # 4. Save inter-rater results
    if inter_rater:
        ir_path = os.path.join(TABLES_DIR, "inter_rater_reliability.json")
        with open(ir_path, "w") as f:
            json.dump(inter_rater, f, indent=2)

    # 5. Visualisations
    plot_comparison(comparison_df, FIGURES_DIR, inter_rater)

    # Scatter plots
    y_test_path = os.path.join(PROCESSED, "y_test.npy")
    if os.path.exists(y_test_path):
        y_test = np.load(y_test_path)
        plot_predictions_scatter(TABLES_DIR, FIGURES_DIR, y_test)

    # 6. Text report
    generate_report(comparison_df, inter_rater, TABLES_DIR)
