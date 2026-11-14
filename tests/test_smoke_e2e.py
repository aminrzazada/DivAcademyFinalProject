import numpy as np

from src.inference import load_artifacts, score_frame
from src.train import train_and_save


def test_end_to_end_train_predict(trained, df):
    root, meta = trained
    for k in ("pr_auc", "roc_auc", "tpr_at_5pct_fpr"):
        assert 0.0 <= meta["test_metrics"]["calibrated"][k] <= 1.0
    assert 0 < meta["threshold"] < 1
    sp = meta["split_rows"]
    assert sp["train"] + sp["valid"] + sp["test"] == len(df)
    model, m = load_artifacts(root)
    out = score_frame(model, m, df.drop(columns=["fraud_bool"]).head(200))
    assert len(out) == 200 and out["flagged"].dtype == bool


def test_training_is_reproducible(df, tmp_path):
    a = train_and_save(df, tmp_path / "a")
    b = train_and_save(df, tmp_path / "b")
    assert a["threshold"] == b["threshold"]
    assert a["test_metrics"] == b["test_metrics"]
