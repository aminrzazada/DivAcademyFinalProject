import numpy as np
import pytest

from src.evaluation import expected_calibration_error, operating_point, threshold_at_fpr, tpr_at_fpr


def test_operating_point_confusion_and_cost():
    y = np.array([1, 1, 0, 0, 0, 0]); p = np.array([.9, .2, .8, .1, .1, .1])
    o = operating_point(y, p, 0.5, c_fn=100, c_fp=10)
    assert (o["tp"], o["fp"], o["fn"], o["tn"]) == (1, 1, 1, 3)
    assert o["cost_per_1000"] == pytest.approx((100 * 1 + 10 * 1) / 6 * 1000)


def test_perfect_ranker_metrics():
    y = np.r_[np.ones(10), np.zeros(990)]; p = np.r_[np.ones(10), np.zeros(990)] + np.random.default_rng(0).random(1000) * 1e-3
    assert tpr_at_fpr(y, p, 0.05) == 1.0
    assert threshold_at_fpr(y, p, 0.05) is not None


def test_ece_zero_when_calibrated():
    rng = np.random.default_rng(0); p = rng.random(200_000) * 0.1; y = (rng.random(200_000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.005
