"""Explain the highest-risk test customer using linear SHAP values."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

DATA_DIR = Path("data/processed/splits")
MODEL_PATH = Path("models/logistic_regression.joblib")
OUTPUT_PATH = Path("reports/individual_shap_explanation.csv")
RANDOM_STATE = 42


def main():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(X_test)[:, 1]
    position = int(probabilities.argmax())
    customer = X_test.iloc[[position]]
    probability = float(probabilities[position])
    actual_churn = int(y_test.iloc[position])

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    background = X_train.sample(
        n=min(100, len(X_train)), random_state=RANDOM_STATE
    )
    background_transformed = preprocessor.transform(background)
    customer_transformed = preprocessor.transform(customer)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.LinearExplainer(classifier, background_transformed)
    shap_values = np.asarray(explainer.shap_values(customer_transformed))

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    values = shap_values.reshape(-1)
    if len(values) != len(feature_names):
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

    contributions = pd.DataFrame(
        {
            "feature": feature_names,
            "transformed_value": np.asarray(customer_transformed).reshape(-1),
            "shap_value": values,
        }
    )
    contributions["absolute_shap"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values(
        "absolute_shap", ascending=False
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    contributions.to_csv(OUTPUT_PATH, index=False)

    print(f"Test row position: {position}")
    print(f"Actual churn label: {actual_churn}")
    print(f"Predicted churn probability: {probability:.4f}")
    print("SHAP contributions are in log-odds units.")
    print("\nTop 10 contributing features:")
    print(
        contributions[
            ["feature", "transformed_value", "shap_value"]
        ].head(10).round(6).to_string(index=False)
    )
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
