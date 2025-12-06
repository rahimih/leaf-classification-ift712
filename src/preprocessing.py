"""
Data preprocessing helpers.
"""

from sklearn.model_selection import train_test_split
from .config import RANDOM_STATE


def train_validation_split(X, y, test_size=0.2):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )
