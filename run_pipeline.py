"""
Master Pipeline Runner
======================
Orchestrates the full AI Interview Scoring pipeline end-to-end.

Usage:
    python run_pipeline.py                     # Run all steps
    python run_pipeline.py --step 3            # Run from step 3 onwards
    python run_pipeline.py --step 7 --only     # Run only step 7

Steps:
    1. Project setup
    2. Data loading
    3. Text cleaning
    4. Feature engineering
    5. Embedding generation
    6. Dataset splitting
    7. Random Forest training
    8. SVR training
    9. LSTM training
    10. BERT training
    11. Model comparison
    12. Classification accuracy matrix

Author: Research Team
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# ── Logger ────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log"), mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
ESSAY_PATH       = os.path.join(BASE_DIR, "training_data", "archive (1)", "training_set_rel3.tsv")
SHORT_ANSWER_DIR = os.path.join(BASE_DIR, "training_data", "archive (3)")
INTERVIEW_DIR    = os.path.join(BASE_DIR, "testing_data")
PROCESSED_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR        = os.path.join(BASE_DIR, "results", "models")
TABLES_DIR       = os.path.join(BASE_DIR, "results", "tables")
FIGURES_DIR      = os.path.join(BASE_DIR, "results", "figures")


def _timer(step_name: str):
    """Context-manager to time a pipeline step."""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            logger.info("> STEP: %s", step_name)
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            logger.info("v %s completed in %.1f s", step_name, elapsed)
    return Timer()


# ── Step functions ────────────────────────────────────────────────────────

def step_1_setup():
    with _timer("1 — Project Setup"):
        from setup_project import setup_project
        setup_project(BASE_DIR)


def step_2_load():
    with _timer("2 — Data Loading"):
        from data_preprocessing.load_data import load_all_datasets
        datasets = load_all_datasets(ESSAY_PATH, SHORT_ANSWER_DIR, INTERVIEW_DIR)
        for name, df in datasets.items():
            logger.info("  %s: %d rows, score range [%.2f, %.2f]",
                        name, len(df), df["score"].min(), df["score"].max())
        return datasets


def step_3_clean(datasets: dict):
    with _timer("3 — Text Cleaning"):
        from data_preprocessing.clean_text import apply_cleaning
        cleaned_frames = []
        for name, df in datasets.items():
            cleaned = apply_cleaning(df)
            cleaned["source"] = name
            cleaned_frames.append(cleaned)
        combined = pd.concat(cleaned_frames, ignore_index=True)
        out_path = os.path.join(PROCESSED_DIR, "clean_dataset.csv")
        combined.to_csv(out_path, index=False)
        logger.info("Saved: %s (%d rows)", out_path, len(combined))
        return combined


def step_4_features():
    with _timer("4 — Feature Engineering"):
        from feature_engineering.features import extract_features
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "clean_dataset.csv"))
        df_feat = extract_features(df, text_col="answer")
        out_path = os.path.join(PROCESSED_DIR, "features.csv")
        df_feat.to_csv(out_path, index=False)
        logger.info("Saved: %s (%d × %d)", out_path, *df_feat.shape)


def step_5_embeddings():
    with _timer("5 — Embedding Generation"):
        from feature_engineering.embeddings import generate_embeddings
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "clean_dataset.csv"))
        embeddings = generate_embeddings(df["answer"])
        out_path = os.path.join(PROCESSED_DIR, "embeddings.npy")
        np.save(out_path, embeddings)
        logger.info("Saved: %s  shape=%s", out_path, embeddings.shape)


def step_6_split():
    with _timer("6 — Dataset Splitting"):
        from data_preprocessing.split_data import build_matrices
        build_matrices(
            features_path=os.path.join(PROCESSED_DIR, "features.csv"),
            embeddings_path=os.path.join(PROCESSED_DIR, "embeddings.npy"),
            output_dir=PROCESSED_DIR,
        )


def step_7_random_forest():
    with _timer("7 — Random Forest"):
        from models.random_forest import train_random_forest
        X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
        X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
        y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
        train_random_forest(X_train, y_train, X_test, y_test, MODEL_DIR, TABLES_DIR)


def step_8_svr():
    with _timer("8 — SVR"):
        from models.svr_model import train_svr
        X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
        X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
        y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
        train_svr(X_train, y_train, X_test, y_test, MODEL_DIR, TABLES_DIR)


def step_9_lstm():
    with _timer("9 — LSTM"):
        from models.lstm_model import train_lstm
        X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
        X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
        y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
        train_lstm(X_train, y_train, X_test, y_test, MODEL_DIR, TABLES_DIR)


def step_10_bert():
    with _timer("10 — BERT"):
        from models.bert_model import train_bert
        train_bert(
            clean_csv_path=os.path.join(PROCESSED_DIR, "clean_dataset.csv"),
            model_dir=MODEL_DIR,
            tables_dir=TABLES_DIR,
        )


def step_11_compare():
    with _timer("11 — Model Comparison"):
        from evaluation.compare_models import (
            load_all_metrics,
            compute_inter_rater_reliability,
            plot_comparison,
            plot_predictions_scatter,
            generate_report,
        )
        comparison_df = load_all_metrics(TABLES_DIR)
        comparison_df.to_csv(os.path.join(TABLES_DIR, "comparison.csv"), index=False)

        inter_rater = compute_inter_rater_reliability(INTERVIEW_DIR)
        if inter_rater:
            import json
            with open(os.path.join(TABLES_DIR, "inter_rater_reliability.json"), "w") as f:
                json.dump(inter_rater, f, indent=2)

        plot_comparison(comparison_df, FIGURES_DIR, inter_rater)

        y_test_path = os.path.join(PROCESSED_DIR, "y_test.npy")
        if os.path.exists(y_test_path):
            y_test = np.load(y_test_path)
            plot_predictions_scatter(TABLES_DIR, FIGURES_DIR, y_test)

        generate_report(comparison_df, inter_rater, TABLES_DIR)


def step_12_classification_metrics():
    with _timer("12 — Classification Accuracy Matrix"):
        from evaluation.classification_metrics import generate_classification_matrix
        generate_classification_matrix(TABLES_DIR, PROCESSED_DIR)


# ── Step registry ─────────────────────────────────────────────────────────

STEPS = {
    1:  ("Project Setup",          step_1_setup),
    2:  ("Data Loading",           step_2_load),
    3:  ("Text Cleaning",          step_3_clean),
    4:  ("Feature Engineering",    step_4_features),
    5:  ("Embedding Generation",   step_5_embeddings),
    6:  ("Dataset Splitting",      step_6_split),
    7:  ("Random Forest",          step_7_random_forest),
    8:  ("SVR",                    step_8_svr),
    9:  ("LSTM",                   step_9_lstm),
    10: ("BERT",                   step_10_bert),
    11: ("Model Comparison",       step_11_compare),
    12: ("Classification Metrics", step_12_classification_metrics),
}


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Interview Scoring Pipeline")
    parser.add_argument("--step", type=int, default=1,
                        help="Step number to start from (default: 1)")
    parser.add_argument("--only", action="store_true",
                        help="Run ONLY the specified step")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("  AI Interview Scoring — Research Pipeline")
    logger.info("=" * 70)

    datasets = None
    combined = None

    steps_to_run = [args.step] if args.only else range(args.step, 13)

    for step_num in steps_to_run:
        if step_num not in STEPS:
            logger.error("Unknown step: %d", step_num)
            continue

        name, func = STEPS[step_num]
        logger.info("\n" + "-" * 70)

        try:
            if step_num == 2:
                datasets = func()
            elif step_num == 3:
                if datasets is None:
                    from data_preprocessing.load_data import load_all_datasets
                    datasets = load_all_datasets(ESSAY_PATH, SHORT_ANSWER_DIR, INTERVIEW_DIR)
                combined = func(datasets)
            else:
                func()
        except Exception:
            logger.exception("Step %d (%s) FAILED", step_num, name)
            if not args.only:
                logger.info("Stopping pipeline. Fix the error and re-run with --step %d", step_num)
            raise

    logger.info("\n" + "=" * 70)
    logger.info("  Pipeline finished successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
