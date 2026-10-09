"""Create a reproducible, stratified train/test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(
        description="Split cleaned churn data into train and test sets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/telco_churn_clean.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/splits"),
    )
    return parser.parse_args()


def main() -> None:
    """Validate data, split it, and save reproducible outputs."""
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    data = pd.read_csv(args.input)

    required_columns = {TARGET_COLUMN, ID_COLUMN}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values.")

    if data[ID_COLUMN].isna().any() or data[ID_COLUMN].duplicated().any():
        raise ValueError("Customer IDs must be present and unique.")

    if set(data[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Churn target must contain both labels 0 and 1.")

    # Keep IDs separate for auditing; never pass them to a model.
    customer_ids = data[ID_COLUMN].copy()
    X = data.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = data[TARGET_COLUMN].astype("int8")

    X_train, X_test, y_train, y_test, ids_train, ids_test = (
        train_test_split(
            X,
            y,
            customer_ids,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save feature matrices and targets separately.
    X_train.to_csv(args.output_dir / "X_train.csv", index=False)
    X_test.to_csv(args.output_dir / "X_test.csv", index=False)
    y_train.to_frame(TARGET_COLUMN).to_csv(
        args.output_dir / "y_train.csv", index=False
    )
    y_test.to_frame(TARGET_COLUMN).to_csv(
        args.output_dir / "y_test.csv", index=False
    )

    # Keep IDs only in audit files, not in model feature matrices.
    ids_train.to_frame(ID_COLUMN).to_csv(
        args.output_dir / "train_ids.csv", index=False
    )
    ids_test.to_frame(ID_COLUMN).to_csv(
        args.output_dir / "test_ids.csv", index=False
    )

    overlap = set(ids_train) & set(ids_test)
    if overlap:
        raise RuntimeError("Customer IDs overlap between train and test.")

    print("Train/test split created successfully.")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Training churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate: {y_test.mean():.2%}")
    print(f"Feature count: {X_train.shape[1]}")
    print(f"Customer ID overlap: {len(overlap)}")
    print(f"Saved split files to: {args.output_dir}")


if __name__ == "__main__":
    main()
