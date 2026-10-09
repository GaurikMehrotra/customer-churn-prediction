"""Leakage-safe feature preprocessing for customer churn prediction."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"


def build_preprocessor(
    X,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build preprocessing that can be fitted on training data only."""
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    if ID_COLUMN in numeric_features or ID_COLUMN in categorical_features:
        raise ValueError(
            f"{ID_COLUMN} is an identifier and must not be a model feature."
        )

    if not numeric_features and not categorical_features:
        raise ValueError("No supported numeric or categorical features found.")

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
