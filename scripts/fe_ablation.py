"""FE ablation (HGB, final hiperparametrlər): yalnız clean (56 feature) vs clean + FE (72 feature).

Repo-da bu skript yox idi (yalnız results/fe_ablation.csv); eyni tələb üçün yenidən yazılıb.
İşə salma (repo kökündən): python scripts/fe_ablation.py     Nəticə: results/fe_ablation.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.config import *
from src.evaluation import score_report
from src.features import BAFCleaner, FeatureEngineer, FeatureSelector, build_preprocessor
from src.splits import time_split
from src.train import DEFAULT_HGB

df = pd.read_csv(DATA_PATH)
(Xtr, ytr), (Xva, yva), (Xte, yte) = (time_split(df)[k] for k in ("train", "valid", "test"))
DROP = [c for c in Xtr.columns if Xtr[c].nunique(dropna=False) <= 1] + [TIME_COL]


def run(use_fe):
    """FE ilə/siz pipeline qurub valid/test metrikalarını qaytarır."""
    steps = [("clean", BAFCleaner(DROP))] + ([("fe", FeatureEngineer())] if use_fe else []) + [
        ("prep", build_preprocessor()), ("select", FeatureSelector()),
        ("model", HistGradientBoostingClassifier(**DEFAULT_HGB))]
    p = Pipeline(steps).fit(Xtr, ytr)
    n = len(p.named_steps["select"].cols_)
    v, t = score_report(yva, p.predict_proba(Xva)[:, 1]), score_report(yte, p.predict_proba(Xte)[:, 1])
    return {"n_features": n, **{f"valid_{k}": x for k, x in v.items()}, **{f"test_{k}": x for k, x in t.items()}}


out = pd.DataFrame({"clean only (raw + missing indikatorlar)": run(False), "clean + feature engineering (tam)": run(True)}).T
out.to_csv("results/fe_ablation.csv")
print(out.round(4).T)
