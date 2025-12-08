"""
Model registry (BASE version for main branch).
Individual models must be added in feature branches.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelConfig:
    name: str
    estimator: Any
    param_grid: Dict[str, Any]


def get_hadi_models() -> Dict[str, ModelConfig]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier

    return {
        "logistic_regression": ModelConfig(
            name="Logistic Regression (Hadi)",
            estimator=LogisticRegression(
                max_iter=2000,
                solver="saga",
                multi_class="multinomial",
            ),
            param_grid={
                "clf__C": [0.1, 1.0, 10.0],
                "clf__penalty": ["l1", "l2"],
            },
        ),
        "knn": ModelConfig(
            name="K-Nearest Neighbors (Hadi)",
            estimator=KNeighborsClassifier(),
            param_grid={
                "clf__n_neighbors": [3, 5, 7, 9],
                "clf__weights": ["uniform", "distance"],
            },
        ),
        "random_forest": ModelConfig(
            name="Random Forest (Hadi)",
            estimator=RandomForestClassifier(),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5],
            },
        ),
    }


# ===== Sara  =====

def get_sara_models() -> Dict[str, ModelConfig]:
    return {}


def get_all_models() -> Dict[str, ModelConfig]:
    models: Dict[str, ModelConfig] = {}
    models.update(get_hadi_models())
    models.update(get_sara_models())
    return models
