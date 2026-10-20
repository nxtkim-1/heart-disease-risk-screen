"""Score predicted probabilities and write them to a metrics file."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, roc_auc_score

# A row counts as positive when its predicted probability reaches this value.
THRESHOLD = 0.5

SCORE_KEYS = ("roc_auc", "sensitivity", "specificity", "threshold", "confusion_matrix")
CONFUSION_KEYS = ("tn", "fp", "fn", "tp")


def score(y_true, y_prob, threshold: float = THRESHOLD) -> dict:
    """ROC AUC on the probabilities, then counts and rates at the threshold."""
    y_pred = [int(p >= threshold) for p in y_prob]
    tn, fp, fn, tp = (
        int(n) for n in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    )
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "threshold": threshold,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def write_metrics(metrics: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
