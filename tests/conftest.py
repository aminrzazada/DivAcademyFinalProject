import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_synthetic_baf import make_synthetic_baf  # noqa: E402
from src.train import train_and_save  # noqa: E402


@pytest.fixture(scope="session")
def df():
    """Sintetik BAF-a oxşar data (real deyil; yalnız kod davranışını yoxlamaq üçün)."""
    return make_synthetic_baf(30_000, seed=1)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, df):
    """Bir dəfə təlim: (root, meta). root/models/ altında model.joblib + meta.json."""
    root = tmp_path_factory.mktemp("proj")
    meta = train_and_save(df, root / "models")
    return root, meta
