import json

from heartscreen.metrics import CONFUSION_KEYS, SCORE_KEYS
from heartscreen.train import METRICS_PATH, main, run


def test_metrics_file_has_every_key(tmp_path):
    path = tmp_path / "metrics.json"
    main(path)
    metrics = json.loads(path.read_text(encoding="utf-8"))
    assert set(metrics) == {"split", "models"}
    assert set(metrics["split"]) == {"seed", "test_size", "n_train", "n_test"}
    assert "majority" in metrics["models"]
    for result in metrics["models"].values():
        test = result["test"]
        assert set(test) == set(SCORE_KEYS)
        assert set(test["confusion_matrix"]) == set(CONFUSION_KEYS)
        for key in ("roc_auc", "sensitivity", "specificity"):
            assert 0.0 <= test[key] <= 1.0


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


def test_committed_metrics_match_a_fresh_run():
    committed = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(run()))
