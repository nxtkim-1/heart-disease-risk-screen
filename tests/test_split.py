from heartscreen.data import features_and_target, load_cleveland
from heartscreen.split import TEST_SIZE, train_test


def _split():
    features, target = features_and_target(load_cleveland())
    return features, target, train_test(features, target)


def test_split_is_deterministic():
    _, _, (X_train_a, X_test_a, _, _) = _split()
    _, _, (X_train_b, X_test_b, _, _) = _split()
    assert list(X_train_a.index) == list(X_train_b.index)
    assert list(X_test_a.index) == list(X_test_b.index)


def test_train_and_test_cover_every_row_once():
    features, _, (X_train, X_test, _, _) = _split()
    train, test = set(X_train.index), set(X_test.index)
    assert train.isdisjoint(test)
    assert train | test == set(features.index)


def test_held_out_count_matches_test_size():
    features, _, (_, X_test, _, _) = _split()
    assert len(X_test) == round(len(features) * TEST_SIZE)


def test_positive_rate_is_preserved():
    _, target, (_, _, y_train, y_test) = _split()
    full = target.mean()
    assert abs(y_train.mean() - full) < 0.02
    assert abs(y_test.mean() - full) < 0.02
