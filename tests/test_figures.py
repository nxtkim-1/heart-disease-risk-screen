from heartscreen.figures import figure_names, make_figures
from heartscreen.train import models

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_every_model_gets_a_roc_and_a_confusion_figure(tmp_path):
    make_figures(tmp_path)
    for name in models():
        for filename in figure_names(name):
            path = tmp_path / filename
            assert path.is_file(), filename
            assert path.read_bytes().startswith(PNG_SIGNATURE), filename


def test_figures_are_not_empty(tmp_path):
    written = make_figures(tmp_path)
    assert written
    assert all(path.stat().st_size > 1000 for path in written)
