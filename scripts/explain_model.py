"""Explain the churn model using linear SHAP values."""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

DATA_DIR = Path("data/processed/splits")
MODEL_PATH = Path("models/logistic_regression.joblib")
OUTPUT_DIR = Path("reports")
RANDOM_STATE = 42


def main():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    model = joblib.load(MODEL_PATH)

    background = X_train.sample(
        n=min(100, len(X_train)), random_state=RANDOM_STATE
    )
    explain_rows = X_test.sample(
        n=min(200, len(X_test)), random_state=RANDOM_STATE
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    background_transformed = preprocessor.transform(background)
    explain_transformed = preprocessor.transform(explain_rows)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.LinearExplainer(classifier, background_transformed)
    shap_values = np.asarray(explainer.shap_values(explain_transformed))

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    if shap_values.shape != (len(explain_rows), len(feature_names)):
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(OUTPUT_DIR / "shap_global_importance.csv", index=False)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        explain_transformed,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.title("Global SHAP importance (log-odds scale)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    print("Global SHAP importance (log-odds scale)")
    print(summary.head(15).to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR / 'shap_global_importance.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'shap_summary.png'}")


if __name__ == "__main__":
    main()
