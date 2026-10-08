"""P0.6 V2 (YEU_CAU v4): so Gini tinh tien cua MO HINH (goi dung ham dong bang RuleEngine._compute_gini_with_negatives) voi
G_P cua Raffinetti, Siletti & Vernizzi (2015), Muc 3.2 Eq. (5): G_P = S / [2(N−1)(T⁺+T⁻)], S = Σ_iΣ_j |Y_i − Y_j|.
Du lieu: np.random.default_rng(0); 49 gia tri lognormal(mean 0, sigma 1) × 100 + mot gia tri d ∈ {−10, −300}.
Kem tai tao 3 vi du Bang 3 cua bai (0.5556 / 1 / 0.8346). Khong dung ma dong bang ngoai viec GOI ham Gini.
Dung: python thesis_support/v4/gini_negative_check.py"""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from be.rule_engine import RuleEngine  # noqa: E402


def g_p(y):
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    t_plus, t_minus = y[y > 0].sum(), -y[y < 0].sum()
    if n < 2 or t_plus + t_minus <= 0:
        return 0.0
    s = np.abs(y[:, None] - y[None, :]).sum()
    return float(s / (2.0 * (n - 1) * (t_plus + t_minus)))


rng = np.random.default_rng(0)
base = rng.lognormal(mean=0.0, sigma=1.0, size=49) * 100.0
print("N = 50; 49 gia tri lognormal(0,1)×100 (rng(0)), TB = %.2f" % base.mean())
print(f"{'d':>6} {'Gini tinh tien (ma)':>20} {'G_P Raffinetti':>16} {'Gini ma, khong co d (N=49)':>28}")
g49 = RuleEngine._compute_gini_with_negatives(list(base))
for d in (-10.0, -300.0):
    y = np.append(base, d)
    print(f"{d:>6.0f} {RuleEngine._compute_gini_with_negatives(list(y)):>20.4f} {g_p(y):>16.4f} {g49:>28.4f}")
print("\nBang 3 Raffinetti et al. (2015) — tai tao G_P:")
for name, y, want in [("(a)", [-5] * 9 + [45.01], 0.5556), ("(b)", [-45] + [0] * 8 + [45.01], 1.0),
                      ("(c)", [-15, -10, -8, -7, -5, 0, 0, 0, 0, 45.01], 0.8346)]:
    print(f"  {name} G_P = {g_p(y):.4f} (bai: {want}) | Gini tinh tien cua ma = {RuleEngine._compute_gini_with_negatives(list(map(float, y))):.4f}")
