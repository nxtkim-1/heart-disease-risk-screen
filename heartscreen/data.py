"""Read the Cleveland file and reject rows that break the schema."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd

from heartscreen.schema import (
    BY_NAME,
    COLUMNS,
    FEATURE_NAMES,
    MISSING_TOKEN,
    TARGET_NAME,
    binary_target,
)

# SHA-256 of processed.cleveland.data from the UCI archive.
# That file is redistributed under CC BY 4.0; see data/SOURCES.md.
CLEVELAND_SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "raw" / "processed.cleveland.data"
DEFAULT_NOTICE_PATH = REPO_ROOT / "data" / "SOURCES.md"

_NOTICE_MARKERS = ("CC BY 4.0", "10.24432/C52P4X")


class DataError(ValueError):
    """A Cleveland file, notice, or row that this project will not use."""


def check_attribution(data_path: Path, notice_path: Path) -> None:
    """Refuse a file that is not the Cleveland copy named in the notice."""
    data_path = Path(data_path)
    notice_path = Path(notice_path)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != CLEVELAND_SHA256:
        raise DataError(
            f"{data_path.name} does not match the UCI Cleveland file named in the attribution"
        )
    if not notice_path.is_file():
        raise DataError(f"missing attribution notice: {notice_path}")
    notice = notice_path.read_text(encoding="utf-8")
    if any(marker not in notice for marker in _NOTICE_MARKERS):
        raise DataError("attribution notice must name CC BY 4.0 and the dataset DOI")


def load_cleveland(
    path: Path = DEFAULT_DATA_PATH,
    notice_path: Path = DEFAULT_NOTICE_PATH,
) -> pd.DataFrame:
    """Load the attributed Cleveland file, then check every row."""
    check_attribution(path, notice_path)
    return parse_cleveland(Path(path).read_text(encoding="utf-8"))


def parse_cleveland(text: str) -> pd.DataFrame:
    """Turn Cleveland CSV text into a frame. '?' is allowed only in ca and thal."""
    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != len(COLUMNS):
            raise DataError(
                f"line {line_no}: expected {len(COLUMNS)} fields, got {len(fields)}"
            )
        records.append(_parse_fields(line_no, fields))
    if not records:
        raise DataError("no data rows")
    frame = pd.DataFrame.from_records(
        records, columns=[column.name for column in COLUMNS]
    )
    for column in COLUMNS:
        if column.kind == "target":
            frame[column.name] = frame[column.name].astype("int64")
        else:
            frame[column.name] = pd.to_numeric(frame[column.name])
    return frame


def features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the 13 features and the binary label."""
    required = FEATURE_NAMES + [TARGET_NAME]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise DataError("missing columns: " + ", ".join(missing))
    features = frame.loc[:, FEATURE_NAMES].copy()
    label = [
        binary_target(int(value)) for value in frame[TARGET_NAME].tolist()
    ]
    target = pd.Series(label, index=frame.index, name="heart_disease", dtype="int64")
    return features, target


def _parse_fields(line_no: int, fields: list[str]) -> dict:
    return {
        column.name: _parse_value(line_no, column.name, raw)
        for column, raw in zip(COLUMNS, fields)
    }


def _parse_value(line_no: int, name: str, raw: str):
    column = BY_NAME[name]
    if raw == MISSING_TOKEN:
        if column.nullable:
            return math.nan
        raise DataError(f"line {line_no}: unexpected missing value in {name}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise DataError(f"line {line_no}: {name} is not a number") from exc
    if not math.isfinite(value):
        raise DataError(f"line {line_no}: {name} is not finite")
    if column.levels is not None:
        if value != int(value) or int(value) not in column.levels:
            raise DataError(f"line {line_no}: {name} has value {raw}")
        if column.kind == "target":
            return int(value)
        return float(int(value))
    low, high = column.bounds
    if value < low or value > high:
        raise DataError(f"line {line_no}: {name} is outside {low}..{high}")
    return value
