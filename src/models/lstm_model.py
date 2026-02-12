"""
LSTM Regression Model
=====================
Trains a bidirectional LSTM neural network for score prediction using
TensorFlow / Keras.

Architecture
------------
Input (396-d) → Reshape → BiLSTM(128) → Dropout(0.3) →
BiLSTM(64) → Dropout(0.3) → Dense(64, ReLU) → Dense(1, sigmoid→[1,5])

The final Dense layer uses sigmoid activation scaled to [1, 5] to
naturally constrain output to the valid score range.

Training
--------
* Optimizer: Adam with learning rate warm-up via ReduceLROnPlateau
* Loss: Huber loss (robust to outliers)
* Early stopping on validation MAE (patience=15)
* 20% of training data used for validation

Outputs
-------
* Model           → ``results/models/lstm_model.h5``
* Training history → ``results/tables/lstm_history.json``
* Metrics JSON    → ``results/tables/lstm_metrics.json``

Author: Research Team
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

SEED = 42


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r, p_value = pearsonr(y_true.ravel(), y_pred.ravel())
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "pearson_r": round(float(r), 4),
        "pearson_p": round(float(p_value), 6),
        "mae":       round(mae, 4),
        "rmse":      round(rmse, 4),
    }


def build_lstm_model(input_dim: int):
    """Construct a bidirectional LSTM regression model.

    Parameters
    ----------
    input_dim : int
        Number of input features (e.g. 396).

    Returns
    -------
    tf.keras.Model
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.random.set_seed(SEED)

    inp = layers.Input(shape=(input_dim,))
    # Reshape to (timesteps=1, features) for LSTM — treating whole vector as one step
    # We split features into pseudo-timesteps for the LSTM to process sequentially
    # Reshape to (n_steps, step_features) where n_steps = reasonable chunk count
    n_steps = 12  # one per hand-crafted feature "group"
    step_features = input_dim  # use RepeatVector approach instead

    x = layers.Reshape((1, input_dim))(inp)
    # Tile to give LSTM temporal steps to process
    x = layers.RepeatVector(n_steps)(layers.Flatten()(x))
    # Reshape back: (batch, n_steps, input_dim)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)

    # Output: sigmoid scaled to [1, 5]
    out = layers.Dense(1, activation="sigmoid")(x)
    out = layers.Lambda(lambda t: t * 4.0 + 1.0)(out)

    model = models.Model(inputs=inp, outputs=out)
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_dir: str,
    tables_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
) -> dict:
    """Train, evaluate, and persist an LSTM regression model.

    Returns
    -------
    dict
        Evaluation metrics on the test set.
    """
    import tensorflow as tf

    logger.info("═" * 60)
    logger.info("  LSTM REGRESSION MODEL")
    logger.info("═" * 60)

    # Suppress excessive TF logging
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    model = build_lstm_model(X_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    model.summary(print_fn=logger.info)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae", patience=15, restore_best_weights=True,
            mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae", factor=0.5, patience=5, min_lr=1e-6,
            mode="min",
        ),
    ]

    logger.info("Training for up to %d epochs …", epochs)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    y_pred = model.predict(X_test, verbose=0).ravel()
    y_pred = np.clip(y_pred, 1.0, 5.0)

    metrics = _compute_metrics(y_test, y_pred)
    metrics["model"] = "LSTM"
    metrics["epochs_trained"] = len(history.history["loss"])

    logger.info("Test Metrics:")
    logger.info("  Pearson r : %.4f  (p=%.6f)", metrics["pearson_r"], metrics["pearson_p"])
    logger.info("  MAE       : %.4f", metrics["mae"])
    logger.info("  RMSE      : %.4f", metrics["rmse"])

    # Save
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "lstm_model.h5")
    model.save(model_path)
    logger.info("Model saved to %s", model_path)

    # Save history
    hist_path = os.path.join(tables_dir, "lstm_history.json")
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(hist_path, "w") as f:
        json.dump(hist_data, f, indent=2)

    metrics_path = os.path.join(tables_dir, "lstm_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    preds_path = os.path.join(tables_dir, "lstm_predictions.npy")
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

    train_lstm(X_train, y_train, X_test, y_test, MODEL_DIR, TABLES_DIR)
