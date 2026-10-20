import json

from heartscreen.metrics import CONFUSION_KEYS, SCORE_KEYS, score, write_metrics


def test_score_on_a_hand_checked_case():
    result = score([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["sensitivity"] == 0.5
    assert result["specificity"] == 0.5
    assert result["roc_auc"] == 0.75


def test_threshold_is_inclusive():
    result = score([0, 1], [0.2, 0.5])
    assert result["confusion_matrix"]["tp"] == 1


def test_score_has_every_key():
    result = score([0, 1, 1], [0.3, 0.7, 0.2])
    assert set(result) == set(SCORE_KEYS)
    assert set(result["confusion_matrix"]) == set(CONFUSION_KEYS)


def test_write_metrics_round_trips(tmp_path):
    path = tmp_path / "reports" / "metrics.json"
    metrics = {"models": {"m": score([0, 1], [0.4, 0.6])}}
    write_metrics(metrics, path)
    assert json.loads(path.read_text(encoding="utf-8")) == metrics
