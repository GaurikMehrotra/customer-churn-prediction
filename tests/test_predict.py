"""Tests for the saved-model prediction interface."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.predict import predict_churn
from src.preprocessing import build_preprocessor


@pytest.fixture
def customer_data():
    """Create deterministic synthetic Telco-style customer records."""
    rows = []
    for i in range(12):
        rows.append(
            {
                "gender": "Female" if i % 2 == 0 else "Male",
                "SeniorCitizen": i % 2,
                "Partner": "Yes" if i % 3 == 0 else "No",
                "Dependents": "No" if i % 3 else "Yes",
                "tenure": i * 5 + 1,
                "PhoneService": "Yes",
                "MultipleLines": "No" if i % 2 == 0 else "Yes",
                "InternetService": ["DSL", "Fiber optic", "No"][i % 3],
                "OnlineSecurity": "Yes" if i % 2 else "No",
                "OnlineBackup": "No" if i % 2 else "Yes",
                "DeviceProtection": "No" if i % 3 else "Yes",
                "TechSupport": "Yes" if i % 2 else "No",
                "StreamingTV": "No" if i % 2 else "Yes",
                "StreamingMovies": "Yes" if i % 2 else "No",
                "Contract": ["Month-to-month", "One year", "Two year"][i % 3],
                "PaperlessBilling": "Yes" if i % 2 else "No",
                "PaymentMethod": [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ][i % 4],
                "MonthlyCharges": 25.0 + i * 7.5,
                "TotalCharges": 25.0 + i * 30.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_customers(customer_data):
    return customer_data.head(5).copy()


@pytest.fixture
def fitted_model(customer_data):
    """Fit a small pipeline so CI needs no local dataset or model artifact."""
    labels = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    model = Pipeline(
        [
            ("preprocessor", build_preprocessor(customer_data)),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )
    return model.fit(customer_data, labels)


def test_predictions_have_expected_columns_and_valid_probabilities(
    sample_customers, fitted_model
):
    results = predict_churn(sample_customers, model=fitted_model)

    assert list(results.columns) == [
        "churn_probability",
        "predicted_churn",
    ]
    assert len(results) == len(sample_customers)
    assert results["churn_probability"].between(0, 1).all()
    assert results["predicted_churn"].isin([0, 1]).all()


def test_threshold_controls_predicted_labels(sample_customers, fitted_model):
    results = predict_churn(
        sample_customers, threshold=0.50, model=fitted_model
    )

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


def test_missing_feature_is_rejected(sample_customers, fitted_model):
    incomplete = sample_customers.drop(columns=["tenure"])

    with pytest.raises(ValueError, match="missing columns"):
        predict_churn(incomplete, model=fitted_model)


def test_unexpected_feature_is_rejected(sample_customers, fitted_model):
    invalid = sample_customers.assign(unexpected_feature=1)

    with pytest.raises(ValueError, match="unexpected columns"):
        predict_churn(invalid, model=fitted_model)


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
def test_threshold_boundaries(
    sample_customers, fitted_model, threshold, expected_label
):
    results = predict_churn(
        sample_customers, threshold=threshold, model=fitted_model
    )

    assert results["predicted_churn"].eq(expected_label).all()


def test_missing_model_file_is_rejected(tmp_path):
    from src.predict import load_model

    missing_path = tmp_path / "missing_model.joblib"

    with pytest.raises(FileNotFoundError, match="Saved model not found"):
        load_model(missing_path)
