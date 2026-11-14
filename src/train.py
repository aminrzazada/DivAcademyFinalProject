"""Təkrarlana bilən end-to-end təlim: split → fit (yalnız train) → calibration (valid) → threshold (valid) → test (bir dəfə).

02_modeling.ipynb-in məntiqinin skript variantıdır (smoke test və CLI üçün).
İstifadə: python -m src.train data/Base.csv models
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator

from .config import COST_FN, COST_FP, RANDOM_STATE, TARGET, TIME_COL, CATEGORICAL
from .evaluation import choose_cost_threshold, operating_point, score_report
from .features import build_pipeline
from .inference import sha256_file
from .splits import time_split

# Notebook 02-də valid-da seçilmiş HGB parametrləri
DEFAULT_HGB = dict(learning_rate=0.1, max_depth=3, min_samples_leaf=20, l2_regularization=0.0,
                   max_iter=200, class_weight="balanced", early_stopping=False, random_state=RANDOM_STATE)


def build_input_schema(X_tr: pd.DataFrame, drop_cols) -> dict:
    """Streamlit/inference üçün giriş sxemi: hər xam feature üçün tip, aralıq, default (yalnız train-dən)."""
    schema = {}
    for col in [c for c in X_tr.columns if c not in drop_cols]:
        if col in CATEGORICAL:
            schema[col] = {"type": "cat", "choices": sorted(X_tr[col].unique().tolist()), "default": X_tr[col].mode()[0]}
        else:
            is_int = bool((X_tr[col].dropna() % 1 == 0).all())
            schema[col] = {"type": "int" if is_int else "float", "min": float(X_tr[col].min()),
                           "max": float(X_tr[col].max()), "default": float(X_tr[col].median())}
    return schema


def train_and_save(df: pd.DataFrame, out_dir, hgb_params=None, c_fn=COST_FN, c_fp=COST_FP) -> dict:
    """Tam təlim. Test yalnız sonda, threshold/calibration seçimi bitdikdən sonra bir dəfə hesablanır."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = (time_split(df)[k] for k in ("train", "valid", "test"))

    # Sabit sütunlar YALNIZ train-dən müəyyən edilir; month split açarıdır
    drop_cols = [c for c in X_tr.columns if X_tr[c].nunique(dropna=False) <= 1] + [TIME_COL]
    pipe = build_pipeline(HistGradientBoostingClassifier(**(hgb_params or DEFAULT_HGB)), drop_cols).fit(X_tr, y_tr)

    # Calibration və threshold: yalnız valid
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid").fit(X_va, y_va)
    threshold = choose_cost_threshold(y_va, calibrated.predict_proba(X_va)[:, 1], c_fn, c_fp)

    # Final test (bir dəfə)
    p_te = calibrated.predict_proba(X_te)[:, 1]
    op = operating_point(y_te, p_te, threshold, c_fn, c_fp)
    schema = build_input_schema(X_tr, drop_cols)
    model_path = out_dir / "model.joblib"
    joblib.dump(calibrated, model_path, compress=3)
    meta = {
        "model": "HistGradientBoosting", "calibration": "sigmoid", "threshold": threshold,
        "cost_fn": c_fn, "cost_fp": c_fp, "n_features_used": None, "features_used": None,
        "drop_cols": drop_cols, "random_state": RANDOM_STATE, "sklearn_version": sklearn.__version__,
        "model_sha256": sha256_file(model_path),
        "split_rows": {"train": len(X_tr), "valid": len(X_va), "test": len(X_te)},
        "test_metrics": {"calibrated": score_report(y_te, p_te)},
        "operating_point_test": {k: (v.item() if hasattr(v, "item") else v) for k, v in op.items()},
        "input_schema": schema,
        "examples": {},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return meta


if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1])
    m = train_and_save(data, sys.argv[2] if len(sys.argv) > 2 else "models")
    print(json.dumps({k: m[k] for k in ("threshold", "test_metrics", "split_rows")}, indent=2))
