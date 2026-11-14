import shutil

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_loads_and_scores(trained, tmp_path, monkeypatch):
    root, _ = trained
    shutil.copytree(root / "models", tmp_path / "models")
    shutil.copytree("src", tmp_path / "src")
    shutil.copy("app.py", tmp_path / "app.py")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(tmp_path / "app.py"), default_timeout=120).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert [m.label for m in at.metric][:1] == ["Fraud ehtimalı (calibrated)"]
