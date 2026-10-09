"""Compare classification thresholds using out-of-fold predictions."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

DATA_DIR = Path("data/processed/splits")
RANDOM_STATE = 42
N_SPLITS = 5
FN_COST = 5
FP_COST = 1


def main() -> None:
    """Evaluate thresholds without using the held-out test set."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["Churn"]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    probabilities = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    results = []

    for threshold in [i / 100 for i in range(10, 91, 5)]:
        predictions = (probabilities >= threshold).astype("int8")

        false_negatives = int(
            ((y_train == 1) & (predictions == 0)).sum()
        )
        false_positives = int(
            ((y_train == 0) & (predictions == 1)).sum()
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_train, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_train, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_train, predictions, zero_division=0
                ),
                "false_negatives": false_negatives,
                "false_positives": false_positives,
                "business_cost": (
                    FN_COST * false_negatives
                    + FP_COST * false_positives
                ),
            }
        )

    results_df = pd.DataFrame(results).sort_values("threshold")
    print("\nTHRESHOLD COMPARISON — TRAINING OOF PREDICTIONS")
    print(results_df.round(4).to_string(index=False))

    best = results_df.loc[results_df["business_cost"].idxmin()]
    print("\nLowest estimated cost in this threshold sweep:")
    print(best.round(4).to_string())


if __name__ == "__main__":
    main()
