import json
import re

import pytest

from heartscreen.data import REPO_ROOT
from heartscreen.train import METRICS_PATH

README = REPO_ROOT / "README.md"

MODEL_KEYS = {
    "Majority class": "majority",
    "Logistic regression": "logistic",
    "Random forest": "random_forest",
}

# Path into a model's entry in metrics.json for each table column.
# The CV column has no key until a model writes CV scores.
COLUMN_KEYS = {
    "ROC AUC": ("test", "roc_auc"),
    "Sensitivity": ("test", "sensitivity"),
    "Specificity": ("test", "specificity"),
    "5-fold CV ROC AUC": None,
}


def results_table(text: str) -> list[dict[str, str]]:
    block = re.search(r"<!-- results:start -->\n(.*?)\n<!-- results:end -->", text, re.S)
    assert block, "README results table markers are missing"
    lines = block.group(1).splitlines()
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(dict(zip(header, cells)))
    return rows


def check_readme(text: str, metrics: dict) -> None:
    for row in results_table(text):
        model = row.pop("Model")
        for column, cell in row.items():
            if cell == "TBD":
                continue
            assert model in MODEL_KEYS, f"unknown model row {model!r}"
            assert COLUMN_KEYS[column], f"{column!r} has no metrics.json key yet"
            value = metrics["models"][MODEL_KEYS[model]]
            for key in COLUMN_KEYS[column]:
                value = value[key]
            assert cell == f"{value:.2f}", f"{model} {column}: README {cell}, metrics.json {value:.2f}"

    counts = re.search(r"Test set: (\d+) patients \((\d+) negative, (\d+) positive\)", text)
    assert counts, "README test-set line is missing"
    n_test, negative, positive = map(int, counts.groups())
    matrix = metrics["models"]["majority"]["test"]["confusion_matrix"]
    assert n_test == metrics["split"]["n_test"]
    assert negative == matrix["tn"] + matrix["fp"]
    assert positive == matrix["tp"] + matrix["fn"]


def committed() -> tuple[str, dict]:
    return README.read_text(encoding="utf-8"), json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def test_readme_numbers_match_metrics():
    check_readme(*committed())


def test_a_changed_readme_number_fails():
    text, metrics = committed()
    changed = text.replace("| Majority class | 0.50 |", "| Majority class | 0.51 |")
    assert changed != text
    with pytest.raises(AssertionError, match="Majority class ROC AUC"):
        check_readme(changed, metrics)
