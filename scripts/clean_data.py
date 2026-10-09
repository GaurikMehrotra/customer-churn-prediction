"""Run the Telco data-cleaning workflow from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_cleaning import clean_telco_data, load_raw_data


def parse_args() -> argparse.Namespace:
    """Parse the input and output file paths."""
    parser = argparse.ArgumentParser(
        description="Clean and validate the Telco churn dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw CSV dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the cleaned CSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path for the JSON quality report.",
    )
    return parser.parse_args()


def main() -> None:
    """Load, clean, and save the dataset and quality report."""
    args = parse_args()

    raw_data = load_raw_data(args.input)
    cleaned_data, report = clean_telco_data(raw_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    cleaned_data.to_csv(args.output, index=False)

    with args.report.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    print("Data cleaning completed successfully.")
    print(f"Input rows: {report['input_rows']}")
    print(f"Output rows: {report['output_rows']}")
    print(f"Exact duplicate rows removed: {report['exact_duplicate_rows_removed']}")
    print(
        f"Blank TotalCharges values imputed: {report['missing_total_charges_imputed']}"
    )
    print(f"Target counts (0=No, 1=Yes): {report['target_counts']}")
    print(f"Cleaned data: {args.output}")
    print(f"Quality report: {args.report}")


if __name__ == "__main__":
    main()
