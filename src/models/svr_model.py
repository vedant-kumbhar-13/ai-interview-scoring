"""
Support Vector Regressor (SVR)
==============================
Trains an SVR with RBF kernel on the 396-d feature+embedding vectors.

SVR is sensitive to feature scale, but our features are already
standardised (see ``split_data.py``), which is optimal for the RBF kernel.

Hyper-parameter tuning via ``RandomizedSearchCV`` (5-fold CV).

Outputs
-------
* Trained model  → ``results/models/svr_model.pkl``
* Metrics JSON   → ``results/tables/svr_metrics.json``

Author: Research Team
"""

from __future__ import annotations

import json
import logging
import os

import joblib
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR

logger = logging.getLogger(__name__)

SEED = 42


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r, p_value = pearsonr(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "pearson_r": round(float(r), 4),
        "pearson_p": round(float(p_value), 6),
        "mae":       round(mae, 4),
        "rmse":      round(rmse, 4),
    }


def train_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_dir: str,
    tables_dir: str,
    n_iter: int = 30,
) -> dict:
    """Train, tune, evaluate, and persist an SVR model.

    Returns
    -------
    dict
        Evaluation metrics on the test set.
    """
    logger.info("═" * 60)
    logger.info("  SUPPORT VECTOR REGRESSOR")
    logger.info("═" * 60)

    param_dist = {
        "C":       [0.1, 1, 10, 50, 100],
        "gamma":   ["scale", "auto", 0.001, 0.01, 0.1],
        "epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
        "kernel":  ["rbf"],
    }

    svr = SVR()

    logger.info("Running RandomizedSearchCV (n_iter=%d, cv=5) …", n_iter)
    search = RandomizedSearchCV(
        svr, param_dist,
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

    # Evaluate
    y_pred = best_model.predict(X_test)
    y_pred = np.clip(y_pred, 1.0, 5.0)

    metrics = _compute_metrics(y_test, y_pred)
    metrics["model"] = "SVR"
    metrics["best_params"] = {k: str(v) for k, v in search.best_params_.items()}

    logger.info("Test Metrics:")
    logger.info("  Pearson r : %.4f  (p=%.6f)", metrics["pearson_r"], metrics["pearson_p"])
    logger.info("  MAE       : %.4f", metrics["mae"])
    logger.info("  RMSE      : %.4f", metrics["rmse"])

    # Save
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "svr_model.pkl")
    joblib.dump(best_model, model_path)
    logger.info("Model saved to %s", model_path)

    metrics_path = os.path.join(tables_dir, "svr_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    preds_path = os.path.join(tables_dir, "svr_predictions.npy")
    np.save(preds_path, y_pred)

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────

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

    train_svr(X_train, y_train, X_test, y_test, MODEL_DIR, TABLES_DIR)
