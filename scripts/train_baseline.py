"""Evaluate a Logistic Regression baseline using cross-validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
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
    """Train fold-specific pipelines and report out-of-fold metrics."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["Churn"]

    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Training target must contain both classes.")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

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

    # Every prediction is made by a model that excluded that fold.
    probabilities = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    # The 0.5 threshold is our initial baseline, not necessarily optimal.
    predictions = (probabilities >= 0.5).astype("int8")

    print("\nLOGISTIC REGRESSION — 5-FOLD OUT-OF-FOLD RESULTS")
    print("-" * 52)
    print(f"ROC-AUC:  {roc_auc_score(y_train, probabilities):.4f}")
    print(
        "PR-AUC:   "
        f"{average_precision_score(y_train, probabilities):.4f}"
    )
    print(f"Precision: {precision_score(y_train, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_train, predictions, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_train, predictions, zero_division=0):.4f}")

    matrix = confusion_matrix(y_train, predictions, labels=[0, 1])
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print("                 Predicted stay  Predicted churn")
    print(f"Actual stay      {matrix[0, 0]:14d}  {matrix[0, 1]:15d}")
    print(f"Actual churn     {matrix[1, 0]:14d}  {matrix[1, 1]:15d}")


if __name__ == "__main__":
    main()
