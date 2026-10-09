
"""Run the complete customer churn project pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

STEPS = [
    (
        "Clean and validate dataset",
        [
            "scripts.clean_data",
            "--input",
            "data/raw/telco_churn.csv",
            "--output",
            "data/processed/telco_churn_clean.csv",
            "--report",
            "reports/data_quality_report.json",
        ],
    ),
    ("Create train/test split", ["scripts.split_data"]),
    ("Check preprocessing", ["scripts.test_preprocessing"]),
    ("Train Logistic Regression baseline", ["scripts.train_baseline"]),
    ("Compare models", ["scripts.compare_models"]),
    ("Compare class weights", ["scripts.compare_class_weights"]),
    ("Analyze classification thresholds", ["scripts.analyze_thresholds"]),
    ("Tune model hyperparameters", ["scripts.tune_models"]),
    ("Evaluate final model", ["scripts.evaluate_final"]),
    ("Evaluate business threshold", ["scripts.evaluate_business_threshold"]),
    ("Generate SHAP explanations", ["scripts.explain_model"]),
    ("Explain an individual customer", ["scripts.explain_customer"]),
]


def main() -> int:
    """Run each pipeline step sequentially and stop on failure."""
    print("CUSTOMER CHURN PREDICTION PIPELINE", flush=True)
    print("=" * 40, flush=True)

    for index, (description, command_parts) in enumerate(STEPS, start=1):
        command = [sys.executable, "-m", *command_parts]

        print(
            f"\n[{index}/{len(STEPS)}] {description}",
            flush=True,
        )
        print(f"$ {' '.join(command)}", flush=True)

        result = subprocess.run(command, cwd=ROOT_DIR, check=False)

        if result.returncode != 0:
            print(
                f"\nFAILED: {description} "
                f"(exit code {result.returncode}).",
                file=sys.stderr,
                flush=True,
            )
            print("Pipeline stopped; later steps were not run.", flush=True)
            return result.returncode

        print(f"COMPLETED: {description}", flush=True)

    print("\nPipeline completed successfully!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
