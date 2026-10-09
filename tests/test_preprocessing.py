"""Tests for the churn feature-preprocessing pipeline."""

import numpy as np
import pandas as pd

from src.preprocessing import build_preprocessor


def sample_features() -> pd.DataFrame:
    """Return a small dataset containing numeric and categorical features."""
    return pd.DataFrame(
        {
            "tenure": [1, 12, 24, 36],
            "MonthlyCharges": [25.0, 60.0, 80.0, 90.0],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
            ],
            "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
        }
    )


def test_preprocessor_expands_categorical_features():
    data = sample_features()
    preprocessor = build_preprocessor(data)

    transformed = preprocessor.fit_transform(data)

    assert transformed.shape[0] == len(data)
    assert transformed.shape[1] > data.shape[1]


def test_preprocessor_handles_missing_values():
    data = sample_features()
    data.loc[0, "MonthlyCharges"] = np.nan
    data.loc[1, "Contract"] = None

    preprocessor = build_preprocessor(data)
    transformed = preprocessor.fit_transform(data)

    assert not np.isnan(transformed).any()


def test_preprocessor_handles_unseen_categories():
    train = sample_features()
    test = pd.DataFrame(
        {
            "tenure": [8],
            "MonthlyCharges": [45.0],
            "Contract": ["Special contract"],
            "InternetService": ["Satellite"],
        }
    )

    preprocessor = build_preprocessor(train)
    preprocessor.fit(train)
    transformed = preprocessor.transform(test)

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == preprocessor.transform(train).shape[1]
    assert not np.isnan(transformed).any()


def test_preprocessor_rejects_identifier_only_features():
    import pytest

    data = pd.DataFrame({"customerID": ["A001", "A002"]})

    with pytest.raises(ValueError, match="identifier"):
        build_preprocessor(data)
