"""Fit each model on the training rows, score it on the test rows, write reports/metrics.json.

Run from the repo root: python -m heartscreen.train
"""

from __future__ import annotations

from pathlib import Path

from sklearn.dummy import DummyClassifier

from heartscreen.data import REPO_ROOT, features_and_target, load_cleveland
from heartscreen.metrics import score, write_metrics
from heartscreen.split import SEED, TEST_SIZE, train_test

METRICS_PATH = REPO_ROOT / "reports" / "metrics.json"


def models() -> dict:
    # The majority-class model ignores the features. Any real model has to beat it.
    return {"majority": DummyClassifier(strategy="most_frequent")}


def run() -> dict:
    features, target = features_and_target(load_cleveland())
    X_train, X_test, y_train, y_test = train_test(features, target)
    results = {}
    for name, model in models().items():
        model.fit(X_train, y_train)
        results[name] = {"test": score(y_test, model.predict_proba(X_test)[:, 1])}
    return {
        "split": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        "models": results,
    }


def main(path: Path = METRICS_PATH) -> None:
    write_metrics(run(), path)


if __name__ == "__main__":
    main()
