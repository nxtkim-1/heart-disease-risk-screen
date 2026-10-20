"""Stratified train/test split with a fixed seed."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

# Fixed seed and fraction so the held-out rows stay the same.
SEED = 42
TEST_SIZE = 0.2


def train_test(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test, stratified on the target."""
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=target,
    )
