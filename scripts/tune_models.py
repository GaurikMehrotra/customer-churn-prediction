"""Tune Logistic Regression and Random Forest using cross-validation."""

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

DATA_DIR = Path("data/processed/splits")
RANDOM_STATE = 42
N_SPLITS = 5


def main() -> None:
    """Search small hyperparameter grids using training data only."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["Churn"]

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search_spaces = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 12],
                "model__min_samples_leaf": [5],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }

    summaries = []
    best_parameters = {}

    for name, (model, param_grid) in search_spaces.items():
        print(f"\nTuning {name}...", flush=True)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        search.fit(X_train, y_train)

        summaries.append(
            {
                "model": name,
                "best_cv_pr_auc": search.best_score_,
                "best_params": json.dumps(search.best_params_),
            }
        )
        best_parameters[name] = search.best_params_

        print(f"Best cross-validation PR-AUC: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")

    results = pd.DataFrame(summaries).sort_values(
        "best_cv_pr_auc", ascending=False
    )

    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / "tuning_results.csv", index=False)
    with (output_dir / "best_parameters.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(best_parameters, file, indent=2)

    print("\nTUNING SUMMARY")
    print(results.to_string(index=False))
    print("\nSaved results to reports/tuning_results.csv")
    print("Saved parameters to reports/best_parameters.json")


if __name__ == "__main__":
    main()
