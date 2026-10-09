"""Compare baseline and class-weighted classifiers."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

DATA_DIR = Path("data/processed/splits")
RANDOM_STATE = 42
N_SPLITS = 5


def main() -> None:
    """Compare class weighting with out-of-fold predictions."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["Churn"]

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
    }

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for name, model in models.items():
        print(f"Evaluating {name}...", flush=True)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model),
            ]
        )

        probabilities = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        predictions = (probabilities >= 0.5).astype("int8")

        results.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(y_train, probabilities),
                "pr_auc": average_precision_score(y_train, probabilities),
                "precision": precision_score(
                    y_train, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_train, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_train, predictions, zero_division=0
                ),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        "pr_auc", ascending=False
    )

    print("\nCLASS-WEIGHT COMPARISON — 5-FOLD OOF RESULTS")
    print(results_df.round(4).to_string(index=False))

    output_path = Path("reports/class_weight_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
