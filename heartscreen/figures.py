"""Draw a ROC curve and a confusion matrix for every model in models().

Run from the repo root: python -m heartscreen.figures
Figures go to reports/figures/ as <model>_roc.png and <model>_confusion.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from heartscreen.data import REPO_ROOT, features_and_target, load_cleveland
from heartscreen.metrics import THRESHOLD
from heartscreen.split import train_test
from heartscreen.train import models

FIGURES_DIR = REPO_ROOT / "reports" / "figures"


def figure_names(name: str) -> tuple[str, str]:
    return f"{name}_roc.png", f"{name}_confusion.png"


def make_figures(out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Fit each model on the training rows and draw it on the test rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features, target = features_and_target(load_cleveland())
    X_train, X_test, y_train, y_test = train_test(features, target)
    written = []
    for name, model in models().items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= THRESHOLD).astype(int)
        roc_name, confusion_name = figure_names(name)

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
        ax.set_title(f"{name}: ROC curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / roc_name, dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            labels=[0, 1],
            display_labels=["no disease", "disease"],
            ax=ax,
            colorbar=False,
        )
        ax.set_title(f"{name}: confusion matrix at {THRESHOLD}")
        fig.tight_layout()
        fig.savefig(out_dir / confusion_name, dpi=150)
        plt.close(fig)

        written += [out_dir / roc_name, out_dir / confusion_name]
    return written


if __name__ == "__main__":
    for path in make_figures():
        print(path)
