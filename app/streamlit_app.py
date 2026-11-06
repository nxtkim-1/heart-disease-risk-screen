"""Local screen: enter one patient's Cleveland features, see the random forest's probability.

Run from the repo root: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# streamlit puts app/ on the import path, not the repo root, so heartscreen would not be found.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heartscreen.data import features_and_target, load_cleveland  # noqa: E402
from heartscreen.schema import BY_NAME, FEATURE_NAMES  # noqa: E402
from heartscreen.split import train_test  # noqa: E402
from heartscreen.train import models  # noqa: E402

BANNER = (
    "Educational only, not medical advice. This screen is not a medical device "
    "and must not be used to make decisions about anyone's health."
)


def fit_model():
    """Fit the random forest on the training rows only, the same split train.py uses."""
    features, target = features_and_target(load_cleveland())
    X_train, _, y_train, _ = train_test(features, target)
    model = models()["random_forest"]
    model.fit(X_train, y_train)
    return model, X_train


def default_values(X_train: pd.DataFrame) -> dict[str, float]:
    """Training median for numeric columns, most common training code for the rest."""
    values = {}
    for name in FEATURE_NAMES:
        if BY_NAME[name].kind == "numeric":
            values[name] = float(X_train[name].median())
        else:
            values[name] = float(X_train[name].mode().iloc[0])
    return values


def patient_frame(values: dict[str, float]) -> pd.DataFrame:
    """One-row frame with columns in FEATURE_NAMES order, whatever order the dict has."""
    if set(values) != set(FEATURE_NAMES):
        missing = sorted(set(FEATURE_NAMES) - set(values))
        extra = sorted(set(values) - set(FEATURE_NAMES))
        raise ValueError(f"missing features {missing}, unexpected features {extra}")
    return pd.DataFrame([[float(values[name]) for name in FEATURE_NAMES]], columns=FEATURE_NAMES)


def predict_proba_one(model, values: dict[str, float]) -> float:
    """Probability of the positive class (any narrowing) for one patient."""
    return float(model.predict_proba(patient_frame(values))[0, 1])


def feature_input(name: str, default: float) -> float:
    column = BY_NAME[name]
    label = f"{name}: {column.description}"
    if column.levels is not None:
        options = list(column.levels)
        return float(st.selectbox(label, options, index=options.index(int(default))))
    low, high = column.bounds
    return float(st.number_input(label, min_value=float(low), max_value=float(high), value=default))


def main() -> None:
    st.set_page_config(page_title="Heart disease risk screen")
    st.title("Heart disease risk screen")
    st.warning(BANNER)

    # Cache so the forest is fitted once per server, not on every widget change.
    model, X_train = st.cache_resource(fit_model)()
    defaults = default_values(X_train)

    st.caption("Defaults are training-set medians (numeric) or the most common code (categorical).")
    values = {name: feature_input(name, defaults[name]) for name in FEATURE_NAMES}

    probability = predict_proba_one(model, values)
    st.metric("Predicted probability of heart disease", f"{probability:.0%}")
    st.caption(
        f"Random forest trained on {len(X_train)} Cleveland patients from 1989. "
        "See the README limitations before reading anything into this number."
    )


if __name__ == "__main__":
    main()
