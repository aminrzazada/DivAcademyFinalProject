import numpy as np
import pandas as pd

from src.config import TARGET, TIME_COL
from src.features import BAFCleaner, FeatureEngineer


def test_minus_one_and_nan_become_missing_with_indicator(df):
    X = df.drop(columns=[TARGET]).head(5).copy()
    X.loc[X.index[0], "prev_address_months_count"] = -1
    X.loc[X.index[1], "prev_address_months_count"] = np.nan
    X.loc[X.index[2], "intended_balcon_amount"] = -0.5
    out = BAFCleaner([TIME_COL]).transform(X)
    assert out["prev_address_months_count"].isna().iloc[:2].all()
    assert out["prev_address_months_count_missing"].iloc[:2].tolist() == [1, 1]
    assert out["intended_balcon_amount_missing"].iloc[2] == 1
    assert TIME_COL not in out.columns


def test_feature_engineering_no_inf_and_22_new(df):
    X = df.drop(columns=[TARGET])
    c = BAFCleaner([TIME_COL, "device_fraud_count"]).transform(X)
    out = FeatureEngineer().transform(c)
    assert not np.isinf(out.select_dtypes("number").to_numpy(dtype=float)).any()
    # 6 indikator (cleaner) + 16 FE feature = 22; X-də cleaner-dən əvvəlki sütunlar var
    assert len([c for c in out.columns if c not in X.columns]) == 22
