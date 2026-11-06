import importlib.util

import numpy as np
import pytest

from heartscreen.data import REPO_ROOT
from heartscreen.schema import FEATURE_NAMES

APP_PATH = REPO_ROOT / "app" / "streamlit_app.py"


def load_app():
    # app/ is a script folder, not a package, so load it by path.
    spec = importlib.util.spec_from_file_location("streamlit_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = load_app()


class RecordingModel:
    """Stands in for the pipeline and keeps whatever frame it was given."""

    def predict_proba(self, X):
        self.seen = X
        return np.array([[0.3, 0.7]])


def sample_values():
    return {name: float(i) for i, name in enumerate(FEATURE_NAMES)}


def test_app_imports_without_starting_ui():
    assert callable(app.main)
    assert "not medical advice" in app.BANNER


def test_prediction_gets_columns_in_feature_order():
    values = sample_values()
    shuffled = dict(reversed(list(values.items())))
    assert list(shuffled) != FEATURE_NAMES

    model = RecordingModel()
    assert app.predict_proba_one(model, shuffled) == 0.7
    assert list(model.seen.columns) == FEATURE_NAMES
    assert model.seen.shape == (1, len(FEATURE_NAMES))
    # Each value must land under its own name, not just in a correctly named column.
    assert model.seen.iloc[0].to_dict() == values


@pytest.mark.parametrize("change", ["drop", "add"])
def test_patient_frame_rejects_wrong_features(change):
    values = sample_values()
    if change == "drop":
        values.pop("thal")
    else:
        values["height"] = 170.0
    with pytest.raises(ValueError):
        app.patient_frame(values)


def test_fitted_model_gives_a_probability_for_defaults():
    model, X_train = app.fit_model()
    assert len(X_train) == 242
    defaults = app.default_values(X_train)
    probability = app.predict_proba_one(model, defaults)
    assert 0.0 <= probability <= 1.0
