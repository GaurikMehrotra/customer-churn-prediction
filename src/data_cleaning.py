"""Data loading, validation, and cleaning for Telco Customer Churn."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
REQUIRED_COLUMNS = {
    ID_COLUMN,
    TARGET_COLUMN,
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
}


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV, preserving customer IDs as strings."""
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path, dtype={ID_COLUMN: "string"})


def clean_telco_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and clean customer data, returning a quality report."""
    if data.empty:
        raise ValueError("The input dataset is empty.")

    cleaned = data.copy()
    original_rows = len(cleaned)

    # Normalize column names and string values.
    cleaned.columns = cleaned.columns.str.strip()

    if cleaned.columns.duplicated().any():
        raise ValueError("Duplicate column names after normalization.")

    missing_columns = REQUIRED_COLUMNS - set(cleaned.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].str.strip()

    # Convert blank strings to missing values.
    cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)

    if cleaned[ID_COLUMN].isna().any():
        raise ValueError("Customer IDs cannot be missing.")

    # Remove exact duplicate records only.
    exact_duplicates = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()

    # Do not silently discard conflicting customer records.
    repeated_ids = cleaned[ID_COLUMN].duplicated(keep=False)
    if repeated_ids.any():
        examples = (
            cleaned.loc[repeated_ids, ID_COLUMN]
            .drop_duplicates()
            .astype(str)
            .tolist()[:10]
        )
        raise ValueError(
            f"Repeated customer IDs remain after deduplication: {examples}"
        )

    # Convert required numeric columns.
    for column in ["tenure", "MonthlyCharges"]:
        try:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric values in {column!r}.") from exc

    # Blank TotalCharges is accepted only for zero-tenure customers.
    original_total_charges = cleaned["TotalCharges"].copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    invalid_charges = original_total_charges.notna() & cleaned["TotalCharges"].isna()
    if invalid_charges.any():
        raise ValueError("TotalCharges contains non-numeric values.")

    if cleaned[["tenure", "MonthlyCharges"]].isna().any().any():
        raise ValueError("Missing tenure or MonthlyCharges values.")

    if (cleaned["tenure"] < 0).any():
        raise ValueError("Tenure cannot be negative.")

    if (cleaned["MonthlyCharges"] < 0).any():
        raise ValueError("MonthlyCharges cannot be negative.")

    blank_charges = cleaned["TotalCharges"].isna()
    zero_tenure = cleaned["tenure"].eq(0)

    if (blank_charges & ~zero_tenure).any():
        raise ValueError("Blank TotalCharges found for customers with nonzero tenure.")

    zero_tenure_with_charges = zero_tenure & cleaned["TotalCharges"].notna()
    if (cleaned.loc[zero_tenure_with_charges, "TotalCharges"] != 0).any():
        raise ValueError("Zero-tenure customers have nonzero TotalCharges.")

    imputed_charges = int(blank_charges.sum())
    cleaned.loc[blank_charges, "TotalCharges"] = 0.0

    if (cleaned["TotalCharges"] < 0).any():
        raise ValueError("TotalCharges cannot be negative.")

    # Encode the target: No -> 0, Yes -> 1.
    target = cleaned[TARGET_COLUMN]
    allowed_targets = {"Yes", "No"}
    invalid_targets = sorted(set(target.dropna().unique()) - allowed_targets)

    if target.isna().any() or invalid_targets:
        raise ValueError(
            f"Invalid Churn labels: {invalid_targets}; "
            f"missing labels: {int(target.isna().sum())}"
        )

    cleaned[TARGET_COLUMN] = target.map({"No": 0, "Yes": 1}).astype("int8")

    report: dict[str, Any] = {
        "input_rows": original_rows,
        "output_rows": len(cleaned),
        "exact_duplicate_rows_removed": exact_duplicates,
        "missing_total_charges_imputed": imputed_charges,
        "target_mapping": {"No": 0, "Yes": 1},
        "target_counts": {
            str(int(label)): int(count)
            for label, count in cleaned[TARGET_COLUMN]
            .value_counts()
            .sort_index()
            .items()
        },
        "remaining_missing_values": {
            str(column): int(count)
            for column, count in cleaned.isna().sum().items()
            if count > 0
        },
        "dtypes": {str(column): str(dtype) for column, dtype in cleaned.dtypes.items()},
    }

    return cleaned.reset_index(drop=True), report
