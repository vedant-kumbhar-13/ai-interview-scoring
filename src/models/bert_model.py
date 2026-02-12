"""
BERT Regression Model
=====================
Fine-tunes a pre-trained BERT model (``bert-base-uncased``) for interview
answer score regression.

Unlike the traditional ML models (RF, SVR) and the LSTM which use the
pre-computed feature+embedding vectors, this model operates directly on
**raw text** — leveraging BERT's contextualised token representations.

Architecture
------------
BERT encoder → [CLS] pooler output (768-d) →
Dropout(0.3) → Dense(256, ReLU) → Dropout(0.2) → Dense(1)

Training
--------
* Optimizer : AdamW (weight decay = 0.01)
* Scheduler : Linear warm-up (10% of steps) → linear decay
* Loss      : MSE
* Epochs    : 4 (standard for BERT fine-tuning)
* Batch     : 16

The model reads from ``clean_dataset.csv`` (raw ``answer`` column) rather
than pre-computed matrices, as BERT tokenises text internally.

Outputs
-------
* Model weights → ``results/models/bert_model.pt``
* Metrics JSON  → ``results/tables/bert_metrics.json``

Author: Research Team
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

SEED = 42
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128  # Reduced from 256 for 4GB VRAM


# ── Dataset class ─────────────────────────────────────────────────────────

class InterviewDataset(Dataset):
    """PyTorch Dataset for text → score pairs."""

    def __init__(self, texts: list[str], scores: np.ndarray,
                 tokenizer: BertTokenizer, max_len: int = MAX_LEN):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.scores[idx], dtype=torch.float),
        }


# ── Metrics ───────────────────────────────────────────────────────────────

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


# ── Training loop ────────────────────────────────────────────────────────

def train_bert(
    clean_csv_path: str,
    model_dir: str,
    tables_dir: str,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    accumulation_steps: int = 2,  # Effective batch = 16
) -> dict:
    """Fine-tune BERT for regression on interview answer scoring.

    Parameters
    ----------
    clean_csv_path : str
        Path to ``clean_dataset.csv`` with columns ``answer``, ``score``, ``source``.
    model_dir, tables_dir : str
        Output directories.
    epochs : int
        Training epochs (default 4).
    batch_size : int
        Mini-batch size.
    lr : float
        Peak learning rate.

    Returns
    -------
    dict
        Test-set evaluation metrics.
    """
    logger.info("═" * 60)
    logger.info("  BERT REGRESSION MODEL")
    logger.info("═" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load data ─────────────────────────────────────────────────────
    df = pd.read_csv(clean_csv_path)
    train_df = df[df["source"].isin(["essay", "short_answer"])].reset_index(drop=True)
    test_df  = df[df["source"] == "interview"].reset_index(drop=True)

    logger.info("Train: %d samples | Test: %d samples", len(train_df), len(test_df))

    # ── Tokenizer & datasets ─────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = InterviewDataset(
        train_df["answer"].astype(str).tolist(),
        train_df["score"].values,
        tokenizer,
    )
    test_dataset = InterviewDataset(
        test_df["answer"].astype(str).tolist(),
        test_df["score"].values,
        tokenizer,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, problem_type="regression",
    )
    model.to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    loss_fn = torch.nn.MSELoss()

    # ── Training with gradient accumulation for low VRAM ─────────────
    logger.info("Training for %d epochs (%d total steps, grad accum=%d) …",
                epochs, total_steps, accumulation_steps)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            labels          = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits.squeeze(-1)
            loss    = loss_fn(logits, labels) / accumulation_steps

            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps

            if (batch_idx + 1) % 100 == 0:
                logger.info("  Epoch %d | Batch %d/%d | Loss: %.4f",
                            epoch + 1, batch_idx + 1, len(train_loader),
                            loss.item() * accumulation_steps)

        avg_loss = epoch_loss / len(train_loader)
        logger.info("Epoch %d/%d — Avg Loss: %.4f", epoch + 1, epochs, avg_loss)

    # ── Evaluation ────────────────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels        = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = outputs.logits.squeeze(-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_pred = np.clip(np.array(all_preds), 1.0, 5.0)
    y_true = np.array(all_labels)

    metrics = _compute_metrics(y_true, y_pred)
    metrics["model"] = "BERT"
    metrics["epochs"] = epochs

    logger.info("Test Metrics:")
    logger.info("  Pearson r : %.4f  (p=%.6f)", metrics["pearson_r"], metrics["pearson_p"])
    logger.info("  MAE       : %.4f", metrics["mae"])
    logger.info("  RMSE      : %.4f", metrics["rmse"])

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "bert_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info("Model saved to %s", model_path)

    metrics_path = os.path.join(tables_dir, "bert_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    preds_path = os.path.join(tables_dir, "bert_predictions.npy")
    np.save(preds_path, y_pred)

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROCESSED  = os.path.join(BASE, "data", "processed")
    MODEL_DIR  = os.path.join(BASE, "results", "models")
    TABLES_DIR = os.path.join(BASE, "results", "tables")

    train_bert(
        clean_csv_path=os.path.join(PROCESSED, "clean_dataset.csv"),
        model_dir=MODEL_DIR,
        tables_dir=TABLES_DIR,
    )
