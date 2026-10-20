import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from heartscreen.data import features_and_target, load_cleveland
from heartscreen.preprocess import preprocessor, with_preprocessing
from heartscreen.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from heartscreen.split import train_test


@pytest.fixture(scope="module")
def split():
    features, target = features_and_target(load_cleveland())
    return train_test(features, target)


def test_output_has_no_gaps(split):
    X_train, X_test, _, _ = split
    transform = preprocessor().fit(X_train)
    assert not np.isnan(transform.transform(X_train)).any()
    assert not np.isnan(transform.transform(X_test)).any()


def test_scaler_and_imputers_use_train_rows_only(split):
    X_train, _, _, _ = split
    transform = preprocessor().fit(X_train)
    numeric = transform.named_transformers_["numeric"]
    assert numeric.named_steps["impute"].statistics_ == pytest.approx(
        X_train[NUMERIC_FEATURES].median().to_numpy()
    )
    assert numeric.named_steps["scale"].mean_ == pytest.approx(
        X_train[NUMERIC_FEATURES].mean().to_numpy()
    )
    categorical = transform.named_transformers_["categorical"]
    modes = [X_train[name].mode().min() for name in CATEGORICAL_FEATURES]
    assert categorical.named_steps["impute"].statistics_ == pytest.approx(modes)


def test_pipeline_fit_ignores_test_rows(split):
    X_train, X_test, y_train, _ = split
    model = with_preprocessing(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    scaler = model.named_steps["preprocess"].named_transformers_["numeric"]
    mean = scaler.named_steps["scale"].mean_
    all_rows = np.concatenate([X_train[NUMERIC_FEATURES], X_test[NUMERIC_FEATURES]])
    assert mean == pytest.approx(X_train[NUMERIC_FEATURES].mean().to_numpy())
    assert mean != pytest.approx(all_rows.mean(axis=0))


def test_one_hot_columns_come_from_train_levels(split):
    X_train, _, _, _ = split
    transform = preprocessor().fit(X_train)
    onehot = transform.named_transformers_["categorical"].named_steps["onehot"]
    for name, levels in zip(CATEGORICAL_FEATURES, onehot.categories_):
        assert set(levels) == set(X_train[name].dropna().unique())


def test_pipeline_fits_a_model(split):
    X_train, X_test, y_train, _ = split
    model = with_preprocessing(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    assert proba.shape == (len(X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()
