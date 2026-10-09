"""Make customer churn predictions using the saved model pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "logistic_regression.joblib"
DEFAULT_THRESHOLD = 0.50


def load_model(model_path: str | Path = MODEL_PATH):
    """Load the saved preprocessing and classification pipeline."""
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {path}")

    return joblib.load(path)


def predict_churn(
    customers: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    model=None,
) -> pd.DataFrame:
    """Return churn probabilities and labels for customer records.

    Args:
        customers: Raw customer features, without the target or customer ID.
        threshold: Probability cutoff used to classify a customer as churn.
        model: Optional preloaded model pipeline, useful for testing.
    """
    if not isinstance(customers, pd.DataFrame):
        raise TypeError("customers must be a pandas DataFrame.")

    if customers.empty:
        raise ValueError("customers must contain at least one row.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    if "Churn" in customers.columns:
        raise ValueError("Input must not contain the target column 'Churn'.")

    if "customerID" in customers.columns:
        raise ValueError(
            "Input must not contain 'customerID'; provide customer features only."
        )

    if customers.columns.duplicated().any():
        raise ValueError("Input contains duplicate column names.")

    pipeline = model if model is not None else load_model()

    expected_columns = list(pipeline.feature_names_in_)
    missing_columns = sorted(set(expected_columns) - set(customers.columns))
    extra_columns = sorted(set(customers.columns) - set(expected_columns))

    if missing_columns or extra_columns:
        details = []
        if missing_columns:
            details.append(f"missing columns: {missing_columns}")
        if extra_columns:
            details.append(f"unexpected columns: {extra_columns}")
        raise ValueError("Invalid customer features; " + "; ".join(details))

    customers = customers[expected_columns].copy()
    probabilities = pipeline.predict_proba(customers)[:, 1]

    return pd.DataFrame(
        {
            "churn_probability": probabilities,
            "predicted_churn": (probabilities >= threshold).astype(int),
        },
        index=customers.index,
    )


if __name__ == "__main__":
    data_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "splits"
        / "X_test.csv"
    )
    sample = pd.read_csv(data_path).head(5)
    results = predict_churn(sample)

    print("Sample churn predictions")
    print(results.round(4).to_string())
