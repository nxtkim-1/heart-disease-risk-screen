"""Impute, scale and one-hot encode the Cleveland features."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartscreen.schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


def preprocessor() -> ColumnTransformer:
    """Return an unfitted transformer for the Cleveland feature columns.

    Numeric columns get the median for gaps and are standardized. Binary
    columns get the most common value and pass through as 0/1. Categorical
    columns (ca and thal hold the only '?' in the file) get the most common
    value and are one-hot encoded.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("binary", binary, BINARY_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def with_preprocessing(model) -> Pipeline:
    """Put the preprocessor in front of a model so both fit on the same rows."""
    return Pipeline([("preprocess", preprocessor()), ("model", model)])
