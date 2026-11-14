"""Zamana görə train/valid/test bölgüsü və sızıntı yoxlaması (notebook, train.py və testlər üçün vahid mənbə)."""
import pandas as pd

from .config import TARGET, TEST_MONTHS, TIME_COL, TRAIN_MONTHS, VALID_MONTHS


def assert_chronological(train_months, valid_months, test_months):
    """Aylar kəsişməməli və train < valid < test sırası pozulmamalıdır (gələcək → keçmiş sızıntısı yoxdur)."""
    t, v, e = set(train_months), set(valid_months), set(test_months)
    if (t & v) or (t & e) or (v & e):
        raise ValueError("Split-lərdə ortaq ay var (sızıntı).")
    if not (max(t) < min(v) and max(v) < min(e)):
        raise ValueError("Split-lər xronoloji deyil: train < valid < test olmalıdır.")


def time_split(df: pd.DataFrame, train_months=TRAIN_MONTHS, valid_months=VALID_MONTHS, test_months=TEST_MONTHS):
    """month sütununa görə {'train','valid','test': (X, y)} qaytarır; xronologiyanı yoxlayır."""
    assert_chronological(train_months, valid_months, test_months)
    parts = {}
    for name, months in [("train", train_months), ("valid", valid_months), ("test", test_months)]:
        part = df[df[TIME_COL].isin(months)]
        parts[name] = (part.drop(columns=[TARGET]), part[TARGET])
    return parts
