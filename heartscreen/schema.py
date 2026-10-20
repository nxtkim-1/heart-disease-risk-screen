"""Cleveland columns and the binary label."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    name: str
    kind: str
    description: str
    levels: tuple[int, ...] | None = None
    bounds: tuple[float, float] | None = None
    nullable: bool = False


# Bounds catch damaged rows. They are not reference ranges.
# Category codes are the ones in the Cleveland file, including chest pain 1-4.
COLUMNS: tuple[Column, ...] = (
    Column("age", "numeric", "Age in years", bounds=(1, 120)),
    Column("sex", "binary", "Sex: 1 = male, 0 = female", levels=(0, 1)),
    Column(
        "cp",
        "categorical",
        "Chest pain type: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic",
        levels=(1, 2, 3, 4),
    ),
    Column(
        "trestbps",
        "numeric",
        "Resting blood pressure on admission, mm Hg",
        bounds=(1, 300),
    ),
    Column("chol", "numeric", "Serum cholesterol, mg/dl", bounds=(1, 1000)),
    Column(
        "fbs",
        "binary",
        "Fasting blood sugar over 120 mg/dl: 1 = true, 0 = false",
        levels=(0, 1),
    ),
    Column(
        "restecg",
        "categorical",
        "Resting ECG: 0 normal, 1 ST-T abnormality, 2 probable or definite left ventricular hypertrophy",
        levels=(0, 1, 2),
    ),
    Column("thalach", "numeric", "Maximum heart rate achieved", bounds=(1, 300)),
    Column(
        "exang",
        "binary",
        "Exercise-induced angina: 1 = yes, 0 = no",
        levels=(0, 1),
    ),
    Column(
        "oldpeak",
        "numeric",
        "ST depression induced by exercise relative to rest",
        bounds=(0, 20),
    ),
    Column(
        "slope",
        "categorical",
        "Slope of the peak exercise ST segment: 1 upsloping, 2 flat, 3 downsloping",
        levels=(1, 2, 3),
    ),
    Column(
        "ca",
        "categorical",
        "Number of major vessels (0-3) colored by fluoroscopy",
        levels=(0, 1, 2, 3),
        nullable=True,
    ),
    Column(
        "thal",
        "categorical",
        "Thallium stress test: 3 = normal, 6 = fixed defect, 7 = reversible defect",
        levels=(3, 6, 7),
        nullable=True,
    ),
    Column(
        "num",
        "target",
        "0 means no significant narrowing was recorded; 1-4 are grades of narrowing",
        levels=(0, 1, 2, 3, 4),
    ),
)

BY_NAME = {column.name: column for column in COLUMNS}
FEATURE_NAMES = [column.name for column in COLUMNS if column.kind != "target"]
NUMERIC_FEATURES = [column.name for column in COLUMNS if column.kind == "numeric"]
BINARY_FEATURES = [column.name for column in COLUMNS if column.kind == "binary"]
CATEGORICAL_FEATURES = [column.name for column in COLUMNS if column.kind == "categorical"]
TARGET_NAME = "num"
MISSING_TOKEN = "?"


def binary_target(num: int) -> int:
    """Keep 0 as negative and collapse grades 1-4 into one positive class."""
    if num == 0:
        return 0
    if num in (1, 2, 3, 4):
        return 1
    raise ValueError(f"num must be 0, 1, 2, 3, or 4, got {num}")
