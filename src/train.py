"""
Training runner (BASE version).
Models will be added from branches.
"""

import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .config import RESULTS_DIR, N_FOLDS, N_JOBS, RANDOM_STATE
from .data_loader import load_train_data
from .preprocessing import train_validation_split
from .evaluation import compute_metrics
from .models import get_all_models


def run():
    X, y = load_train_data()
    X_train, X_val, y_train, y_val = train_validation_split(X, y)

    models = get_all_models()
    if not models:
        raise RuntimeError("No models found. Implement models in feature branches.")

    RESULTS_DIR.mkdir(exist_ok=True)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for key, m in models.items():
        print(f"Training {m.name}")

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", m.estimator),
        ])

        grid = GridSearchCV(pipe, m.param_grid, cv=cv, n_jobs=N_JOBS, scoring="accuracy")
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)

        rows.append({
            "model": m.name,
            "best_params": json.dumps(grid.best_params_),
            **metrics
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    print(df)


if __name__ == "__main__":
    run()
