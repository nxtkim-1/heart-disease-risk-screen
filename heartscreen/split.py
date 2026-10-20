"""Split features and the binary target into train and test sets."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

# Fixed so every model trains and scores on the same rows.
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
