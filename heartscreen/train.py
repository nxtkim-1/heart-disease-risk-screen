"""Fit each model on the training rows, score it on the test rows, write reports/metrics.json.

Each model also gets a stratified 5-fold cross-validated ROC AUC on the training rows.

Run from the repo root: python -m heartscreen.train
"""

from __future__ import annotations

from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from heartscreen.data import REPO_ROOT, features_and_target, load_cleveland
from heartscreen.metrics import score, write_metrics
from heartscreen.preprocess import with_preprocessing
from heartscreen.split import SEED, TEST_SIZE, train_test

METRICS_PATH = REPO_ROOT / "reports" / "metrics.json"
CV_FOLDS = 5


def models() -> dict:
    # The majority-class model ignores the features. Any real model has to beat it.
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "random_forest": with_preprocessing(
            RandomForestClassifier(n_estimators=100, random_state=SEED)
        ),
    }


def cross_validate(model, X_train, y_train) -> dict:
    # Training rows only, so the test rows stay unseen until the final score.
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    aucs = cross_val_score(model, X_train, y_train, cv=folds, scoring="roc_auc")
    return {"roc_auc_mean": float(aucs.mean()), "roc_auc_std": float(aucs.std())}


def run() -> dict:
    features, target = features_and_target(load_cleveland())
    X_train, X_test, y_train, y_test = train_test(features, target)
    results = {}
    for name, model in models().items():
        cv = cross_validate(model, X_train, y_train)
        model.fit(X_train, y_train)
        results[name] = {
            "test": score(y_test, model.predict_proba(X_test)[:, 1]),
            "cv": cv,
        }
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
