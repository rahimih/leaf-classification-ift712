"""
Load Leaf Classification dataset.
"""

import pandas as pd
from typing import Tuple
from .config import TRAIN_CSV, TEST_CSV


def load_train_data() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(TRAIN_CSV)
    y = df["species"]
    X = df.drop(columns=["species", "id"])
    return X, y


def load_test_data() -> pd.DataFrame:
    df = pd.read_csv(TEST_CSV)
    return df.drop(columns=["id"])
