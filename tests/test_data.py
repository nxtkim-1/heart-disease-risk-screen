import pandas as pd
import pytest

from heartscreen.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_NOTICE_PATH,
    DataError,
    features_and_target,
    load_cleveland,
    parse_cleveland,
)
from heartscreen.schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    COLUMNS,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
    TARGET_NAME,
    binary_target,
)


def _line(**overrides) -> str:
    values = {
        "age": "63.0",
        "sex": "1.0",
        "cp": "4.0",
        "trestbps": "130.0",
        "chol": "250.0",
        "fbs": "0.0",
        "restecg": "2.0",
        "thalach": "150.0",
        "exang": "0.0",
        "oldpeak": "1.4",
        "slope": "2.0",
        "ca": "0.0",
        "thal": "3.0",
        "num": "0",
    }
    values.update(overrides)
    return ",".join(values[column.name] for column in COLUMNS)


def test_schema_has_thirteen_features_and_a_target():
    assert FEATURE_NAMES == [column.name for column in COLUMNS if column.name != TARGET_NAME]
    assert len(FEATURE_NAMES) == 13
    assert TARGET_NAME == "num"
    kinds = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    assert sorted(kinds) == sorted(FEATURE_NAMES)
    assert [column.name for column in COLUMNS if column.nullable] == ["ca", "thal"]


def test_grades_one_through_four_are_one_positive_class():
    text = "\n".join(_line(num=str(num)) for num in range(5)) + "\n"
    _, target = features_and_target(parse_cleveland(text))
    assert list(target) == [0, 1, 1, 1, 1]


def test_binary_target_rejects_an_unknown_grade():
    frame = parse_cleveland(_line() + "\n")
    frame.loc[0, "num"] = 5
    with pytest.raises(ValueError, match="num"):
        features_and_target(frame)
    with pytest.raises(ValueError, match="num"):
        binary_target(5)


def test_missing_ca_and_thal_stay_missing():
    frame = parse_cleveland(_line(ca="?", thal="?", num="4") + "\n")
    assert pd.isna(frame.loc[0, "ca"])
    assert pd.isna(frame.loc[0, "thal"])
    features, target = features_and_target(frame)
    assert pd.isna(features.loc[0, "ca"])
    assert pd.isna(features.loc[0, "thal"])
    assert list(target) == [1]


def test_blank_lines_are_skipped():
    frame = parse_cleveland("\n" + _line(num="1") + "\n\n")
    assert len(frame) == 1
    assert list(features_and_target(frame)[1]) == [1]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"chol": "?"}, "chol"),
        ({"age": "old"}, "age"),
        ({"cp": "0.0"}, "cp"),
        ({"thal": "2.0"}, "thal"),
        ({"ca": "4.0"}, "ca"),
        ({"num": "5"}, "num"),
        ({"oldpeak": "-1.0"}, "oldpeak"),
        ({"sex": "2"}, "sex"),
        ({"restecg": "3.0"}, "restecg"),
    ],
)
def test_bad_rows_are_rejected(overrides, match):
    with pytest.raises(DataError, match=match):
        parse_cleveland(_line(**overrides) + "\n")


def test_short_row_is_rejected():
    with pytest.raises(DataError, match="fields"):
        parse_cleveland("1,2,3\n")


def test_empty_input_is_rejected():
    with pytest.raises(DataError):
        parse_cleveland("")
    with pytest.raises(DataError):
        parse_cleveland("\n\n")


def test_cleveland_file_loads_and_binarizes():
    frame = load_cleveland()
    assert len(frame) == 303
    assert int(frame["ca"].isna().sum()) == 4
    assert int(frame["thal"].isna().sum()) == 2
    others = [name for name in FEATURE_NAMES if name not in ("ca", "thal")]
    assert int(frame[others].isna().sum().sum()) == 0
    features, target = features_and_target(frame)
    assert list(features.columns) == FEATURE_NAMES
    assert list(target) == [binary_target(int(value)) for value in frame["num"]]
    assert set(target.tolist()) == {0, 1}
    assert int((target == 0).sum()) == int((frame["num"] == 0).sum())
    assert int((target == 1).sum()) == int((frame["num"] > 0).sum())


def test_tampered_file_is_refused(tmp_path):
    flipped = bytearray(DEFAULT_DATA_PATH.read_bytes())
    flipped[0] ^= 0x01
    path = tmp_path / "processed.cleveland.data"
    path.write_bytes(flipped)
    with pytest.raises(DataError, match="does not match"):
        load_cleveland(path, DEFAULT_NOTICE_PATH)


def test_notice_must_carry_the_cc_by_credit(tmp_path):
    data = tmp_path / "processed.cleveland.data"
    data.write_bytes(DEFAULT_DATA_PATH.read_bytes())
    notice = tmp_path / "SOURCES.md"
    notice.write_text("Heart Disease dataset, no terms recorded.\n")
    with pytest.raises(DataError, match="CC BY 4.0"):
        load_cleveland(data, notice)


def test_missing_notice_is_refused(tmp_path):
    data = tmp_path / "processed.cleveland.data"
    data.write_bytes(DEFAULT_DATA_PATH.read_bytes())
    with pytest.raises(DataError, match="attribution"):
        load_cleveland(data, tmp_path / "SOURCES.md")
