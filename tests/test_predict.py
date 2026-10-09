"""Tests for the saved-model prediction interface."""

import pandas as pd
import pytest

from src.predict import predict_churn


@pytest.fixture
def sample_customers():
    return pd.read_csv(
        "data/processed/splits/X_test.csv"
    ).head(5)


def test_predictions_have_expected_columns_and_valid_probabilities(
    sample_customers,
):
    results = predict_churn(sample_customers)

    assert list(results.columns) == [
        "churn_probability",
        "predicted_churn",
    ]
    assert len(results) == len(sample_customers)
    assert results["churn_probability"].between(0, 1).all()
    assert results["predicted_churn"].isin([0, 1]).all()


def test_threshold_controls_predicted_labels(sample_customers):
    results = predict_churn(sample_customers, threshold=0.50)

    expected = (
        results["churn_probability"] >= 0.50
    ).astype(int).rename("predicted_churn")

    pd.testing.assert_series_equal(
        results["predicted_churn"],
        expected,
    )


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_invalid_threshold_is_rejected(sample_customers, threshold):
    with pytest.raises(ValueError, match="threshold"):
        predict_churn(sample_customers, threshold=threshold)


def test_empty_input_is_rejected(sample_customers):
    with pytest.raises(ValueError, match="at least one row"):
        predict_churn(sample_customers.iloc[0:0])


def test_missing_feature_is_rejected(sample_customers):
    incomplete = sample_customers.drop(columns=["tenure"])

    with pytest.raises(ValueError, match="missing columns"):
        predict_churn(incomplete)


def test_unexpected_feature_is_rejected(sample_customers):
    invalid = sample_customers.assign(unexpected_feature=1)

    with pytest.raises(ValueError, match="unexpected columns"):
        predict_churn(invalid)


def test_target_column_is_rejected(sample_customers):
    invalid = sample_customers.assign(Churn=0)

    with pytest.raises(ValueError, match="target column"):
        predict_churn(invalid)


def test_customer_id_is_rejected(sample_customers):
    invalid = sample_customers.assign(customerID="TEST-001")

    with pytest.raises(ValueError, match="customerID"):
        predict_churn(invalid)


def test_non_dataframe_input_is_rejected():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        predict_churn([{"tenure": 12}])


@pytest.mark.parametrize(
    ("threshold", "expected_label"),
    [(0.0, 1), (1.0, 0)],
)
def test_threshold_boundaries(sample_customers, threshold, expected_label):
    results = predict_churn(sample_customers, threshold=threshold)

    assert results["predicted_churn"].eq(expected_label).all()


def test_missing_model_file_is_rejected(tmp_path):
    from src.predict import load_model

    missing_path = tmp_path / "missing_model.joblib"

    with pytest.raises(FileNotFoundError, match="Saved model not found"):
        load_model(missing_path)
