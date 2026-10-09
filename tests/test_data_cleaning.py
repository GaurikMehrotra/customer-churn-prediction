"""Tests for Telco dataset cleaning."""

import pandas as pd
import pytest

from src.data_cleaning import clean_telco_data


def sample_data() -> pd.DataFrame:
    """Return a small valid dataset for unit tests."""
    return pd.DataFrame(
        {
            "customerID": ["A001", "A002", "A003"],
            "tenure": [0, 12, 24],
            "MonthlyCharges": [25.0, 60.0, 80.0],
            "TotalCharges": [" ", "720.0", "1920.0"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "Churn": ["No", "Yes", "No"],
        }
    )


def test_target_is_encoded_and_blank_zero_tenure_charges_imputed():
    cleaned, report = clean_telco_data(sample_data())

    assert cleaned["Churn"].tolist() == [0, 1, 0]
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert report["missing_total_charges_imputed"] == 1


def test_exact_duplicate_rows_are_removed():
    data = pd.concat([sample_data(), sample_data().iloc[[1]]], ignore_index=True)

    cleaned, report = clean_telco_data(data)

    assert len(cleaned) == 3
    assert report["exact_duplicate_rows_removed"] == 1


def test_repeated_customer_id_with_conflicting_record_is_rejected():
    data = sample_data()
    duplicate = data.iloc[[1]].copy()
    duplicate["MonthlyCharges"] = 99.0
    data = pd.concat([data, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="Repeated customer IDs"):
        clean_telco_data(data)


def test_blank_total_charges_for_nonzero_tenure_is_rejected():
    data = sample_data()
    data.loc[1, "TotalCharges"] = " "

    with pytest.raises(ValueError, match="Blank TotalCharges"):
        clean_telco_data(data)
