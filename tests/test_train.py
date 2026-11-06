import json

from heartscreen.metrics import CONFUSION_KEYS, SCORE_KEYS
from heartscreen.train import METRICS_PATH, main, models, run

CV_KEYS = {"roc_auc_mean", "roc_auc_std"}


def test_metrics_file_has_every_key(tmp_path):
    path = tmp_path / "metrics.json"
    main(path)
    metrics = json.loads(path.read_text(encoding="utf-8"))
    assert set(metrics) == {"split", "models"}
    assert set(metrics["split"]) == {"seed", "test_size", "n_train", "n_test"}
    assert set(metrics["models"]) == set(models())
    assert {"majority", "random_forest"} <= set(metrics["models"])
    for result in metrics["models"].values():
        assert set(result) == {"test", "cv"}
        test = result["test"]
        assert set(test) == set(SCORE_KEYS)
        assert set(test["confusion_matrix"]) == set(CONFUSION_KEYS)
        for key in ("roc_auc", "sensitivity", "specificity"):
            assert 0.0 <= test[key] <= 1.0
        assert set(result["cv"]) == CV_KEYS
        assert 0.0 <= result["cv"]["roc_auc_mean"] <= 1.0
        assert result["cv"]["roc_auc_std"] >= 0.0


def test_confusion_matrix_covers_the_test_rows():
    metrics = run()
    n_test = metrics["split"]["n_test"]
    for result in metrics["models"].values():
        assert sum(result["test"]["confusion_matrix"].values()) == n_test


def test_majority_baseline_calls_every_row_negative():
    test = run()["models"]["majority"]["test"]
    assert test["sensitivity"] == 0.0
    assert test["specificity"] == 1.0
    assert test["roc_auc"] == 0.5


def test_majority_cv_scores_chance_on_every_fold():
    cv = run()["models"]["majority"]["cv"]
    assert cv == {"roc_auc_mean": 0.5, "roc_auc_std": 0.0}


def test_committed_metrics_match_a_fresh_run():
    committed = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(run()))
