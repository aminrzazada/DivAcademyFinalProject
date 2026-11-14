import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET
from src.inference import flag_by_capacity, load_artifacts, score_frame, validate_input, values_to_frame


def test_reload_predictions_match_and_schema_order(trained, df):
    root, meta = trained
    model, meta2 = load_artifacts(root)
    X = df.drop(columns=[TARGET]).tail(500)
    res = score_frame(model, meta2, X)
    assert list(validate_input(X, meta2).columns) == list(meta2["input_schema"])
    assert np.allclose(res["fraud_probability"], model.predict_proba(X[list(meta2["input_schema"])])[:, 1])
    assert ((res["fraud_probability"] >= 0) & (res["fraud_probability"] <= 1)).all()
    assert (res["flagged"] == (res["fraud_probability"] >= meta2["threshold"])).all()


def test_extra_columns_ignored_and_input_not_mutated(trained, df):
    root, _ = trained
    model, meta = load_artifacts(root)
    X = df.tail(50).copy()          # fraud_bool + month artıq sütun kimi qalır
    snap = X.copy()
    a = score_frame(model, meta, X)["fraud_probability"]
    b = score_frame(model, meta, X.drop(columns=[TARGET, "month"]))["fraud_probability"]
    assert np.allclose(a, b)
    pd.testing.assert_frame_equal(X, snap)


def test_schema_violations_raise(trained, df):
    root, _ = trained
    model, meta = load_artifacts(root)
    X = df.drop(columns=[TARGET]).tail(10)
    with pytest.raises(ValueError, match="Çatışmayan"):
        score_frame(model, meta, X.drop(columns=["income"]))
    bad = X.copy(); bad["income"] = bad["income"].astype(object); bad.iloc[0, bad.columns.get_loc("income")] = "abc"
    with pytest.raises(ValueError, match="rəqəm"):
        score_frame(model, meta, bad)
    with pytest.raises(ValueError):
        score_frame(model, meta, X.iloc[0:0])


def test_unseen_category_and_nan_do_not_crash(trained, df):
    root, _ = trained
    model, meta = load_artifacts(root)
    X = df.drop(columns=[TARGET]).tail(20).copy()
    X["payment_type"] = "ZZZ"
    X.loc[X.index[0], "prev_address_months_count"] = np.nan
    assert score_frame(model, meta, X)["fraud_probability"].notna().all()


def test_values_to_frame_types(trained):
    _, meta = trained
    vals = {c: s.get("default") for c, s in meta["input_schema"].items()}
    f = values_to_frame(vals, meta)
    assert list(f.columns) == list(meta["input_schema"]) and len(f) == 1


def test_tampered_model_rejected(trained):
    root, meta = trained
    assert meta["model_sha256"]
    p = root / "models" / "model.joblib"
    orig = p.read_bytes()
    try:
        p.write_bytes(orig + b"0")
        with pytest.raises(ValueError, match="hash"):
            load_artifacts(root)
    finally:
        p.write_bytes(orig)
    load_artifacts(root)


def test_flag_by_capacity_flags_requested_share():
    p = pd.Series(np.random.default_rng(0).random(1000))
    assert abs(flag_by_capacity(p, 10).mean() - 0.10) < 0.01
