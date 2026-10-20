from heartscreen.data import features_and_target, load_cleveland
from heartscreen.split import TEST_SIZE, train_test


def _split():
    features, target = features_and_target(load_cleveland())
    return features, target, train_test(features, target)


def test_split_is_deterministic():
    _, _, (X1_tr, X1_te, _, _) = _split()
    _, _, (X2_tr, X2_te, _, _) = _split()
    assert list(X1_tr.index) == list(X2_tr.index)
    assert list(X1_te.index) == list(X2_te.index)


def test_train_and_test_cover_every_row_once():
    features, _, (X_tr, X_te, _, _) = _split()
    train, test = set(X_tr.index), set(X_te.index)
    assert train.isdisjoint(test)
    assert train | test == set(features.index)


def test_test_fraction_matches_the_constant():
    features, _, (_, X_te, _, _) = _split()
    assert len(X_te) == round(len(features) * TEST_SIZE)


def test_positive_rate_is_preserved():
    _, target, (_, _, y_tr, y_te) = _split()
    full = target.mean()
    assert abs(y_tr.mean() - full) < 0.02
    assert abs(y_te.mean() - full) < 0.02
