"""Streamlit app üçün model yükləmə və skorinq məntiqi (UI-dan ayrıdır ki, test oluna bilsin)."""
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd


def sha256_file(path: Path) -> str:
    """Faylın SHA-256 hash-i (joblib/pickle yüklənməzdən əvvəl bütövlük yoxlaması üçün)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_artifacts(root: Path):
    """models/ qovluğundan kalibrə olunmuş modeli və meta məlumatı yükləyir.

    joblib/pickle etibarsız mənbədən yüklənməməlidir: meta.json-da `model_sha256` varsa, fayl yüklənməzdən
    ƏVVƏL hash ilə yoxlanılır (uyğun gəlmirsə ValueError). Sklearn versiyası fərqlidirsə xəbərdarlıq verilir.
    """
    model_path = root / "models" / "model.joblib"
    meta = json.loads((root / "models" / "meta.json").read_text(encoding="utf-8"))
    expected = meta.get("model_sha256")
    if expected:
        if sha256_file(model_path) != expected:
            raise ValueError("model.joblib hash-i meta.json ilə uyğun gəlmir: fayl dəyişdirilib və ya fərqli təlimdəndir.")
    else:
        warnings.warn("meta.json-da model_sha256 yoxdur: model faylının bütövlüyü yoxlanılmadı (02_modeling.ipynb-ni yenidən işlədin).")
    import sklearn
    trained_with = meta.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        warnings.warn(f"Model scikit-learn {trained_with} ilə hazırlanıb, indiki versiya {sklearn.__version__}.")
    model = joblib.load(model_path)
    return model, meta


def values_to_frame(values: dict, meta: dict) -> pd.DataFrame:
    """Tək müraciətin dəyərlərini modelin gözlədiyi sütun sırası və tiplə DataFrame edir."""
    schema = meta["input_schema"]
    row = {}
    for col, spec in schema.items():
        v = values[col]
        row[col] = int(v) if spec["type"] == "int" else (float(v) if spec["type"] == "float" else v)
    return pd.DataFrame([row], columns=list(schema))


def validate_input(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Giriş cədvəlini schema-ya uyğunlaşdırır: sütun sırası, rəqəmsal tip, kateqorik ədəd.

    Çatışan sütun və ya rəqəmə çevrilməyən dəyər ValueError verir. Artıq sütunlar (məs. fraud_bool, month)
    atılır ki, modelə düşməsin. Boş (NaN) xanalar pipeline-da "missing" sayılır.
    """
    schema = meta["input_schema"]
    cols = list(schema)
    if len(df) == 0:
        raise ValueError("Boş cədvəl.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Çatışmayan sütunlar: {missing}")
    out = df[cols].copy()
    for c, spec in schema.items():
        if spec["type"] == "cat":
            out[c] = out[c].astype(object)
        else:
            conv = pd.to_numeric(out[c], errors="coerce")
            bad = conv.isna() & out[c].notna()
            if bad.any():
                raise ValueError(f"'{c}' sütununda rəqəm olmayan dəyərlər var (məs. sətir {int(bad.idxmax())}).")
            out[c] = conv
    return out


def score_frame(model, meta: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Cədvəli skorlayır: fraud_probability və flagged (threshold-a görə) sütunları əlavə edir."""
    X = validate_input(df, meta)
    p = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["fraud_probability"] = p
    out["flagged"] = p >= meta["threshold"]
    return out


def flag_by_capacity(prob: pd.Series, capacity_pct: float) -> pd.Series:
    """Yoxlama komandasının tutumuna görə: ən yüksək riskli capacity_pct% müraciəti işarələyir."""
    cut = prob.quantile(1 - capacity_pct / 100)
    return prob >= cut
