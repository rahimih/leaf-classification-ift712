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


# ===== Hadi will fill this =====

def get_hadi_models() -> Dict[str, ModelConfig]:
    return {}


# ===== Sara will fill this =====

def get_sara_models() -> Dict[str, ModelConfig]:
    return {}


def get_all_models() -> Dict[str, ModelConfig]:
    models = {}
    models.update(get_hadi_models())
    models.update(get_sara_models())
    return models
