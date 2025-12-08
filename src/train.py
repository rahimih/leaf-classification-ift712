"""
Training runner (BASE version).
Models will be added from branches.
"""

import json
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import N_FOLDS, N_JOBS, RANDOM_STATE, RESULTS_DIR
from .data_loader import load_train_data
from .preprocessing import train_validation_split
from .evaluation import compute_metrics
from .models import get_all_models


def run_experiment():
    X, y = load_train_data()
    X_train, X_val, y_train, y_val = train_validation_split(X, y)

    models = get_all_models()
    if not models:
        raise RuntimeError("No models registered in get_all_models().")

    RESULTS_DIR.mkdir(exist_ok=True)

    cv = StratifiedKFold(
        n_splits=N_FOLDS, 
        shuffle=True, 
        random_state=RANDOM_STATE
        )
    rows = []

    for key, cfg in models.items():
        print(f"\n=== Training {cfg.name} ({key}) ===")

        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", cfg.estimator),
            ]
        )

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=cfg.param_grid,
            cv=cv,
            n_jobs=N_JOBS,
            scoring="accuracy",
            verbose=1,
        )
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)

        rows.append({
            "model_key": key,
            "model_name": cfg.name,
            "best_params": json.dumps(grid.best_params_),
            **metrics,
        })

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "model_metrics.csv"
    df.to_csv(out_path, index=False)
    print(df)

    return df

if __name__ == "__main__":
    run_experiment()
