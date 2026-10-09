"""Evaluate the selected churn model once on the held-out test set."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

DATA_DIR = Path("data/processed/splits")
RANDOM_STATE = 42
THRESHOLD = 0.5


def main() -> None:
    """Fit on training data and evaluate on the held-out test set."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["Churn"]
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["Churn"]

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype("int8")

    metrics = {
        "model": "Logistic Regression",
        "threshold": THRESHOLD,
        "test_rows": len(y_test),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics["confusion_matrix"] = matrix.tolist()

    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "final_test_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "logistic_regression.joblib")

    print("\nFINAL HELD-OUT TEST RESULTS")
    print("-" * 38)
    for key, value in metrics.items():
        if key != "confusion_matrix":
            print(
                f"{key}: {value:.4f}"
                if isinstance(value, float)
                else f"{key}: {value}"
            )

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print("                 Predicted stay  Predicted churn")
    print(f"Actual stay      {matrix[0, 0]:14d}  {matrix[0, 1]:15d}")
    print(f"Actual churn     {matrix[1, 0]:14d}  {matrix[1, 1]:15d}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Metrics saved to reports/final_test_metrics.json")
    print("Model saved to models/logistic_regression.joblib")


if __name__ == "__main__":
    main()
