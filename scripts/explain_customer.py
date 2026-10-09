"""Explain an individual customer churn prediction with SHAP."""

from pathlib import Path

import joblib
import pandas as pd
import shap

DATA_DIR = Path("data/processed/splits")
MODEL_PATH = Path("models/logistic_regression.joblib")
OUTPUT_DIR = Path("reports")
RANDOM_STATE = 42


def main() -> None:
    """Explain the highest-risk customer in the held-out test set."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["Churn"]

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Select the highest predicted-risk test customer for illustration.
    customer_position = int(probabilities.argmax())
    customer = X_test.iloc[[customer_position]]
    actual_churn = int(y_test.iloc[customer_position])
    probability = float(probabilities[customer_position])

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    background = X_train.sample(
        n=min(50, len(X_train)),
        random_state=RANDOM_STATE,
    )

    background_transformed = preprocessor.transform(background)
    customer_transformed = preprocessor.transform(customer)
    feature_names = preprocessor.get_feature_names_out()

    def predict_churn_probability(transformed_data):
        return classifier.predict_proba(transformed_data)[:, 1]

    explainer = shap.KernelExplainer(
        predict_churn_probability,
        background_transformed,
    )
    shap_values = explainer.shap_values(
        customer_transformed,
        nsamples=100,
    )

    contributions = pd.DataFrame(
        {
            "feature": feature_names,
            "transformed_value": customer_transformed[0],
            "shap_value": shap_values[0],
        }
    )
    contributions["absolute_shap"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values(
        "absolute_shap", ascending=False
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contributions.to_csv(
        OUTPUT_DIR / "individual_shap_explanation.csv",
        index=False,
    )

    print("INDIVIDUAL CUSTOMER EXPLANATION")
    print(f"Test-set row position: {customer_position}")
    print(f"Actual churn label: {actual_churn} (1=churn, 0=stay)")
    print(f"Predicted churn probability: {probability:.4f}")
    print("\nLargest feature contributions:")
    print(
        contributions[
            ["feature", "transformed_value", "shap_value"]
        ].head(10).round(4).to_string(index=False)
    )
    print("\nPositive SHAP values push the prediction toward churn.")
    print("Negative SHAP values push the prediction away from churn.")
    print(
        "\nSaved explanation to "
        "reports/individual_shap_explanation.csv"
    )


if __name__ == "__main__":
    main()
