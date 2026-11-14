"""Yalnız kodu/testləri yoxlamaq üçün BAF-a oxşar SİNTETİK data (nəticələr real deyil).

İstifadə: python scripts/make_synthetic_baf.py [n_rows] [out.csv]
"""
import sys

import numpy as np
import pandas as pd


def make_synthetic_baf(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """BAF sxemini (32 sütun, -1 missing kodları, month 0-7, ~1-2% fraud, zaman driftı) təqlid edir."""
    rng = np.random.default_rng(seed)
    month = rng.integers(0, 8, n)
    age = rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90], n, p=[.01, .25, .28, .2, .13, .08, .03, .01, .01])
    income = rng.choice(np.round(np.arange(0.1, 1.0, 0.1), 1), n)
    prev = np.where(rng.random(n) < 0.7, -1, rng.integers(0, 200, n))
    cur = np.where(rng.random(n) < 0.005, -1, rng.integers(0, 400, n))
    bank_m = np.where(rng.random(n) < 0.25, -1, rng.integers(1, 30, n))
    balcon = np.where(rng.random(n) < 0.74, -rng.random(n) * 2, rng.gamma(1.2, 8, n))
    sess = np.where(rng.random(n) < 0.002, -1, rng.gamma(1.5, 5, n))
    dev_em = np.where(rng.random(n) < 0.0004, -1, rng.integers(0, 3, n))
    name_sim = rng.random(n)
    credit = rng.integers(-50, 300, n)
    limit = rng.choice([200, 500, 1000, 1500, 2000], n, p=[.45, .25, .15, .1, .05]).astype(float)
    vel6 = rng.normal(5500 + 200 * month, 1200, n)  # zaman driftı
    z = (-5.2 + 0.0035 * (credit - 100) + 0.7 * (age >= 50) + 1.0 * (prev == -1) - 1.4 * name_sim
         + 0.0004 * (limit - 500) + 0.08 * month + rng.normal(0, 0.8, n))
    fraud = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    df = pd.DataFrame({
        "fraud_bool": fraud, "income": income, "name_email_similarity": name_sim,
        "prev_address_months_count": prev, "current_address_months_count": cur, "customer_age": age,
        "days_since_request": rng.exponential(1.0, n), "intended_balcon_amount": balcon,
        "payment_type": rng.choice(list("ABCDE"), n), "zip_count_4w": rng.integers(1, 6000, n),
        "velocity_6h": vel6, "velocity_24h": vel6 + rng.normal(0, 800, n), "velocity_4w": vel6 + rng.normal(0, 600, n),
        "bank_branch_count_8w": rng.integers(0, 2400, n), "date_of_birth_distinct_emails_4w": rng.integers(0, 40, n),
        "employment_status": rng.choice(list("ABCDEFG"), n), "credit_risk_score": credit,
        "email_is_free": rng.integers(0, 2, n), "housing_status": rng.choice(["BA", "BB", "BC", "BD", "BE", "BF", "BG"], n),
        "phone_home_valid": rng.integers(0, 2, n), "phone_mobile_valid": rng.integers(0, 2, n),
        "bank_months_count": bank_m, "has_other_cards": rng.integers(0, 2, n), "proposed_credit_limit": limit,
        "foreign_request": rng.integers(0, 2, n), "source": rng.choice(["INTERNET", "TELEAPP"], n, p=[.99, .01]),
        "session_length_in_minutes": sess, "device_os": rng.choice(["windows", "linux", "macintosh", "x11", "other"], n),
        "keep_alive_session": rng.integers(0, 2, n), "device_distinct_emails_8w": dev_em,
        "device_fraud_count": 0, "month": month,
    })
    return df


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    out = sys.argv[2] if len(sys.argv) > 2 else "data/Base_synthetic.csv"
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    d = make_synthetic_baf(rows)
    d.to_csv(out, index=False)
    print(f"{out}: {d.shape}, fraud rate {d.fraud_bool.mean():.3%}")
