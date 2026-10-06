"""Test dang ky truoc cho G_P (PREREG_A74_bench_v2.yaml: tests). Chay: pytest audits/paper_extras/test_benchmark_v2.py -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_v2 as b2  # noqa: E402


def test_table3_raffinetti_2015():
    assert abs(b2.gini_p([-5.0] * 9 + [45.01]) - 0.5556) < 1e-4
    assert abs(b2.gini_p([-45.0] + [0.0] * 8 + [45.01]) - 1.0) < 1e-4
    assert abs(b2.gini_p([-15.0, -10.0, -8.0, -7.0, -5.0, 0.0, 0.0, 0.0, 0.0, 45.01]) - 0.8346) < 1e-4


def test_nonneg_equals_model_gini_times_n_over_n_minus_1():
    rng = np.random.default_rng(7)
    for n in (2, 10, 50, 97):
        x = rng.lognormal(5.0, 0.8, n)
        assert abs(b2.gini_p(x) - b2.gini_model(x) * n / (n - 1)) < 1e-9


def test_edge_cases_and_bounds():
    assert b2.gini_p([0.0] * 7) == 0.0 and b2.gini_p([3.0]) == 0.0 and b2.gini_p([]) == 0.0
    rng = np.random.default_rng(8)
    for _ in range(200):
        g = b2.gini_p(rng.normal(0.0, 100.0, int(rng.integers(2, 80))))
        assert 0.0 <= g <= 1.0 + 1e-12


def test_bias_condition_matches_direct_comparison():
    """Dieu kien T+ + N^2 c > (2N-1) T- (PREREG directional_prediction) <=> Gini tinh tien < G_P, kiem tren du lieu ngau nhien."""
    rng = np.random.default_rng(9)
    for _ in range(500):
        n = int(rng.integers(3, 80))
        y = rng.normal(rng.uniform(-50, 300), 150.0, n)
        if y.min() >= 0:
            continue
        tp, tm, c = y[y > 0].sum(), -y[y < 0].sum(), -y.min() + 1e-5
        cond = tp + n * n * c > (2 * n - 1) * tm
        lhs, rhs = b2.gini_model(y), b2.gini_p(y)
        if abs(lhs - rhs) > 1e-9:
            assert cond == (lhs < rhs)
    assert b2.self_test()
