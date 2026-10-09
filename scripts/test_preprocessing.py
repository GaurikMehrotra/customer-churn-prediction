"""Smoke tests for the churn preprocessing pipeline."""

from pathlib import Path

import pandas as pd

from src.preprocessing import build_preprocessor

DATA_DIR = Path("data/processed/splits")


def main() -> None:
    """Fit preprocessing on training data and validate its output."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    preprocessor = build_preprocessor(X_train)

    # Fit only on training data to avoid leakage.
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    assert X_train_transformed.shape[0] == len(X_train)
    assert X_test_transformed.shape[0] == len(X_test)
    assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
    assert X_train_transformed.shape[1] > X_train.shape[1]

    assert not pd.isna(X_train_transformed).any()
    assert not pd.isna(X_test_transformed).any()

    print("Preprocessing smoke tests passed.")
    print(f"Raw training shape: {X_train.shape}")
    print(f"Transformed training shape: {X_train_transformed.shape}")
    print(f"Transformed test shape: {X_test_transformed.shape}")
    print("Missing values after preprocessing: 0")
    print("Training/test feature dimensions match.")


if __name__ == "__main__":
    main()
