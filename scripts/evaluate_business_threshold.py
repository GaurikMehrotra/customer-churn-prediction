"""Evaluate preselected churn thresholds on the held-out test set."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATA_DIR = Path("data/processed/splits")
MODEL_PATH = Path("models/logistic_regression.joblib")
FN_COST = 5
FP_COST = 1
THRESHOLDS = [0.50, 0.15]


def main() -> None:
    """Compare fixed thresholds without refitting the model."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["Churn"]

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype("int8")
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

        tn, fp, fn, tp = matrix.ravel()
        business_cost = FN_COST * int(fn) + FP_COST * int(fp)

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "business_cost": business_cost,
            }
        )

    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "business_threshold_evaluation.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    print("\nHELD-OUT TEST: FIXED THRESHOLD COMPARISON")
    print("Illustrative cost = 5 × false negatives + false positives")
    print(pd.DataFrame(results).round(4).to_string(index=False))
    print("\nSaved to reports/business_threshold_evaluation.json")


if __name__ == "__main__":
    main()
