"""
Model registry (BASE version for main branch).
Individual models must be added in feature branches.
"""

from dataclasses import dataclass
from typing import Dict, Any
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

@dataclass
class ModelConfig:
    name: str
    estimator: Any
    param_grid: Dict[str, Any]


# ===== Hadi will fill this =====

def get_hadi_models() -> Dict[str, ModelConfig]:
    return {}



def get_sara_models() -> Dict[str, ModelConfig]:
    

    return {
        "gradient_boosting": ModelConfig(
            name="Gradient Boosting (Sara)",
            estimator=GradientBoostingClassifier(),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 5],
            },
        ),
        "svm": ModelConfig(
            name="SVM RBF (Sara)",
            estimator=SVC(),
            param_grid={
                "clf__C": [0.1, 1.0, 10.0],
                "clf__gamma": ["scale", 0.01, 0.001],
                "clf__kernel": ["rbf"],
            },
        ),
        "gaussian_nb": ModelConfig(
            name="Gaussian Naive Bayes (Sara)",
            estimator=GaussianNB(),
            param_grid={
                "clf__var_smoothing": [1e-9, 1e-8, 1e-7],
            },
        ),
    }


def get_all_models() -> Dict[str, ModelConfig]:
    models = {}
    models.update(get_hadi_models())
    models.update(get_sara_models())
    return models
