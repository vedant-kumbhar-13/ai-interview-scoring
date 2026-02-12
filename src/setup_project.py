"""
Project Directory Setup Script
==============================
Creates the standardized directory structure for the AI Interview Scoring
research pipeline. Idempotent — safe to run multiple times.

Reference Architecture:
    data/raw/           — Original, immutable datasets
    data/processed/     — Cleaned and transformed datasets
    src/                — Source code modules
    results/figures/    — Visualization outputs
    results/tables/     — Tabular results (CSV)
    results/models/     — Serialized trained models
    logs/               — Experiment logs

Author: Research Team
"""

import os
import logging

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIRECTORIES = [
    "data/raw",
    "data/processed",
    "src/data_preprocessing",
    "src/feature_engineering",
    "src/models",
    "src/evaluation",
    "results/figures",
    "results/tables",
    "results/models",
    "logs",
]

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_project(base_dir: str = BASE_DIR) -> None:
    """Create all required project directories if they do not exist."""
    logger.info("Setting up project directory structure …")
    for rel_path in DIRECTORIES:
        full_path = os.path.join(base_dir, rel_path)
        os.makedirs(full_path, exist_ok=True)
        logger.info("  ✔  %s", full_path)
    logger.info("Project structure ready.")


if __name__ == "__main__":
    setup_project()
