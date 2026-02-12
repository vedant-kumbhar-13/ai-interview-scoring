"""
Random Forest Regressor — Baseline Model
=========================================
Trains a ``RandomForestRegressor`` on the concatenated feature+embedding
vectors (396-d) and evaluates on the interview test set.

Hyper-parameter search uses 5-fold cross-validation on the training set
via ``RandomizedSearchCV`` with a fixed random seed for reproducibility.

Outputs
-------
* Trained model  → ``results/models/random_forest.pkl``
* Metrics JSON   → ``results/tables/random_forest_metrics.json``

Evaluation metrics: Pearson r, MAE, RMSE.

Author: Research Team
"""

from __future__ import annotations

import json
import logging
import os

import joblib
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger(__name__)

SEED = 42


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute Pearson r, MAE, RMSE."""
    r, p_value = pearsonr(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "pearson_r": round(float(r), 4),
        "pearson_p": round(float(p_value), 6),
        "mae":       round(mae, 4),
        "rmse":      round(rmse, 4),
    }


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_dir: str,
    tables_dir: str,
    n_iter: int = 30,
) -> dict:
    """Train, tune, evaluate, and persist a Random Forest regressor.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data.
    X_test, y_test : np.ndarray
        Test data (interview set).
    model_dir : str
        Directory for saved model.
    tables_dir : str
        Directory for metrics JSON.
    n_iter : int
        Number of random search iterations.

    Returns
    -------
    dict
        Evaluation metrics on the test set.
    """
    logger.info("═" * 60)
    logger.info("  RANDOM FOREST REGRESSOR")
    logger.info("═" * 60)

    # ── Hyper-parameter search space ──────────────────────────────────
    param_dist = {
        "n_estimators":      [100, 200, 300, 500],
        "max_depth":         [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2", 0.3, 0.5],
    }

    rf = RandomForestRegressor(random_state=SEED, n_jobs=-1)

    logger.info("Running RandomizedSearchCV (n_iter=%d, cv=5) …", n_iter)
    search = RandomizedSearchCV(
        rf, param_dist,
        n_iter=n_iter,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    logger.info("Best params: %s", search.best_params_)
    logger.info("Best CV MAE: %.4f", -search.best_score_)

    # ── Evaluate on test set ──────────────────────────────────────────
    y_pred = best_model.predict(X_test)
    # Clip predictions to [1, 5]
    y_pred = np.clip(y_pred, 1.0, 5.0)

    metrics = _compute_metrics(y_test, y_pred)
    metrics["model"] = "Random Forest"
    metrics["best_params"] = search.best_params_

    logger.info("Test Metrics:")
    logger.info("  Pearson r : %.4f  (p=%.6f)", metrics["pearson_r"], metrics["pearson_p"])
    logger.info("  MAE       : %.4f", metrics["mae"])
    logger.info("  RMSE      : %.4f", metrics["rmse"])

    # ── Save model & metrics ──────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "random_forest.pkl")
    joblib.dump(best_model, model_path)
    logger.info("Model saved to %s", model_path)

    metrics_path = os.path.join(tables_dir, "random_forest_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)

    # Save predictions for later analysis
    preds_path = os.path.join(tables_dir, "random_forest_predictions.npy")
    np.save(preds_path, y_pred)

    return metrics


# ── CLI entry-point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROCESSED  = os.path.join(BASE, "data", "processed")
    MODEL_DIR  = os.path.join(BASE, "results", "models")
    TABLES_DIR = os.path.join(BASE, "results", "tables")

    X_train = np.load(os.path.join(PROCESSED, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED, "y_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED, "X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED, "y_test.npy"))

    train_random_forest(X_train, y_train, X_test, y_test, MODEL_DIR, TABLES_DIR)
