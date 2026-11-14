import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import TARGET, TIME_COL
from src.evaluation import choose_cost_threshold
from src.features import BAFCleaner, build_pipeline
from src.splits import time_split
from src.train import train_and_save


def _pipe(df):
    (Xtr, ytr), (Xva, _), (Xte, _) = (time_split(df)[k] for k in ("train", "valid", "test"))
    drop = [TIME_COL]
    pipe = build_pipeline(HistGradientBoostingClassifier(max_iter=20, random_state=0), drop).fit(Xtr, ytr)
    return pipe, Xtr, Xva, Xte


def test_preprocessing_fitted_on_train_only(df):
    pipe, Xtr, Xva, Xte = _pipe(df)
    scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scaler"]
    imputer = pipe.named_steps["prep"].named_transformers_["num"].named_steps["imputer"]
    before = (scaler.mean_.copy(), imputer.statistics_.copy())
    # valid/test üzərində transform/predict öyrənilən statistikanı dəyişməməlidir
    shifted = Xte.copy()
    shifted["income"] = shifted["income"] * 100
    pipe.predict_proba(Xva); pipe.predict_proba(shifted)
    assert np.array_equal(before[0], scaler.mean_) and np.array_equal(before[1], imputer.statistics_)
    # statistika train-dəki ilə eyni olmalı (test/valid qarışmayıb)
    ref = BAFCleaner([TIME_COL]).transform(Xtr)
    fe_cols = pipe.named_steps["fe"].transform(pipe.named_steps["clean"].transform(Xtr))
    i = list(pipe.named_steps["prep"].named_transformers_["num"].feature_names_in_).index("income")
    assert np.isclose(scaler.mean_[i], fe_cols["income"].mean())
    assert ref is not None


def test_test_data_does_not_influence_fit(df):
    """Test aylarının label-ları və dəyərləri dəyişsə də train-də öyrənilmiş model eyni qalmalıdır."""
    pipe1, Xtr, _, Xte = _pipe(df)
    df2 = df.copy()
    m = df2[TIME_COL].isin([6, 7])
    df2.loc[m, TARGET] = 1 - df2.loc[m, TARGET]
    df2.loc[m, "income"] = 0.99
    pipe2, _, _, _ = _pipe(df2)
    assert np.allclose(pipe1.predict_proba(Xtr)[:, 1], pipe2.predict_proba(Xtr)[:, 1])


def test_threshold_and_calibration_independent_of_test(df, tmp_path):
    """Threshold yalnız valid-dən seçilir: test label-ları dəyişəndə threshold eyni qalmalıdır."""
    m1 = train_and_save(df, tmp_path / "a")
    df2 = df.copy()
    mask = df2[TIME_COL].isin([6, 7])
    df2.loc[mask, TARGET] = np.random.default_rng(0).integers(0, 2, mask.sum())
    m2 = train_and_save(df2, tmp_path / "b")
    assert m1["threshold"] == m2["threshold"]
    assert m1["test_metrics"] != m2["test_metrics"]  # test metrikası isə dəyişir (yalnız hesabat üçün)


def test_choose_cost_threshold_uses_only_given_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    p = np.clip(rng.random(5000) * 0.5 + 0.4 * y, 1e-4, 1)
    assert choose_cost_threshold(y, p, 500, 10) == choose_cost_threshold(y.copy(), p.copy(), 500, 10)


def test_no_future_or_target_columns_in_input_schema(trained):
    _, meta = trained
    assert TARGET not in meta["input_schema"] and TIME_COL not in meta["input_schema"]
    assert TIME_COL in meta["drop_cols"]


def test_cleaner_and_fe_are_stateless(df):
    X = df.drop(columns=[TARGET])
    a = BAFCleaner([TIME_COL]).fit(X.head(100)).transform(X.tail(50))
    b = BAFCleaner([TIME_COL]).fit(X.tail(100)).transform(X.tail(50))
    pd.testing.assert_frame_equal(a, b)
