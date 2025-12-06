"""
Global project configuration.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

RANDOM_STATE = 42
N_FOLDS = 5
N_JOBS = -1
