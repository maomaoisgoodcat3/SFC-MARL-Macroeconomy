"""Tong hop quet thue phang + rl_mean_fixed (dang ky truoc: audits/final_run/PREREG_flat_sweep.md, commit 4e0327d).

CACH KIEM H1-H4 DUOC CO DINH O DAY VA COMMIT TRUOC KHI CO KET QUA (viet trong luc may dang chay, chua doc file ket qua nao):
- So lieu moi episode: Eq×Prod = TB_buoc(GDP thuc) × (1 − TB_buoc(Gini)) (giong aggregate_benchmark.py). "TB" = trung binh gop
  moi episode qua 3 seed; CI95 = bootstrap 2 cap seed -> episode, 10 000 lan, rng(0). Hieu ghep cap theo episode (cung seed reset).
- H1 DUNG neu CI95 cua (rl_learned − rl_mean_fixed) CHUA 0 o CA HAI che do.
- H2 DUNG (theo tung che do) neu TB Eq×Prod GIAM NGHIEM NGAT doc lat cat thue DN = 0: (0,0) > pit0.10 > pit0.20 > pit0.30 > pit0.50.
  Bao kem CI95 cua tung hieu ghep cap ke tiep (khong dung de phan xu).
- H3 DUNG (theo tung che do) neu ca hai: (a) voi moi cit trong {0.10, 0.20, 0.30}: can tren CI95 cua (cit − (0,0)) >= 0 ("khong thap hon");
  (b) diem TB cao nhat cua lat cat {0, 0.10, 0.20, 0.30, 0.50} la cit 0.50 VA voi moi cit <= 0.30: can tren CI95 cua (cit − cit0.50) < 0.
- H4 DUNG (theo tung che do) neu ca hai, voi moi cit trong {0.10, 0.20, 0.30, 0.50}: (a) |TB(1−Gini)_cit − TB(1−Gini)_(0,0)| < 0.02;
  (b) phan GDP chiem > 50% trong phan ra |dGDP·E0| / (|dGDP·E0| + |GDP0·dE|) (E = TB 1−Gini, GDP = TB GDP, moc (0,0)).
- Kiem nhat quan: rl_learned va free_market+rl_aux phai TRUNG tung episode voi file benchmark cu bench_seed*_{det,exp}.json (cung
  ma duong chay, cung seed episode) — bao do lech lon nhat.
Dung: python audits/final_run/aggregate_flat_sweep.py [--boot 10000] [--md-out <file>]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aggregate_benchmark import iqm, hier_boot  # noqa: E402

LOG = os.path.join(HERE, "logs")
SWEEP = os.path.join(LOG, "flat_sweep")
SEEDS = (42, 202, 303)
PIT_SLICE = ["free_market+rl_aux", "flat_pit0.10_cit0", "flat_pit0.20_cit0", "flat_pit0.30_cit0", "flat_pit0.50_cit0"]
CIT_SLICE = ["free_market+rl_aux", "flat_pit0_cit0.10", "flat_pit0_cit0.20", "flat_pit0_cit0.30", "flat_pit0_cit0.50"]
EXTRA = ["tax", "fines", "purchases", "relief_paid", "injection", "newborn_outflow", "neg_wealth_step_share"]


def ep_eqp(a):
    return np.asarray(a["gdp"]) * (1.0 - np.asarray(a["gini"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--md-out", default=os.path.join(SWEEP, "flat_sweep_table.md"))
    a = ap.parse_args()
    out = []
    P = out.append
    for mode in ("det", "exp"):
        rng = np.random.default_rng(0)
        runs = {s: json.load(open(os.path.join(SWEEP, f"sweep_seed{s}_{mode}.json"), encoding="utf-8")) for s in SEEDS}
        arms = list(runs[SEEDS[0]]["arms"])
        P(f"\n## Che do {'tat dinh' if mode == 'det' else 'lay mau'} — 3 seed x 10 episode ghep cap")
        prov = {s: (r["provenance"]["git_head"][:8], len(r["provenance"]["git_dirty"])) for s, r in runs.items()}
        P(f"Provenance (commit, so file chua commit): {prov}; TB 5 cong cu RL theo seed: "
          + "; ".join(f"{s}: {np.round(runs[s]['rl_full_mean'], 3).tolist()}" for s in SEEDS))
        # kiem nhat quan voi benchmark cu
        diffs = []
        for s in SEEDS:
            old = json.load(open(os.path.join(LOG, f"bench_seed{s}_{mode}.json"), encoding="utf-8"))["arms"]
            for arm in ("rl_learned", "free_market+rl_aux"):
                diffs.append(float(np.max(np.abs(ep_eqp(runs[s]["arms"][arm]) - ep_eqp(old[arm])))))
        P(f"Kiem nhat quan voi bench_seed*_{mode}.json (rl_learned, free_market+rl_aux): do lech Eq×Prod lon nhat theo episode = {max(diffs):.3e}")
        E = {arm: [ep_eqp(runs[s]["arms"][arm]) for s in SEEDS] for arm in arms}
        G = {arm: [np.asarray(runs[s]["arms"][arm]["gdp"]) for s in SEEDS] for arm in arms}
        Q = {arm: [1.0 - np.asarray(runs[s]["arms"][arm]["gini"]) for s in SEEDS] for arm in arms}
        P("\n| Nhanh | Eq×Prod IQM [CI95] | TB | TB theo seed | GDP TB | 1−Gini TB | Chet/ep | Thue | Phat | Chi G | Tro cap | Bom cau | So sinh | % buoc tai san min<0 |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in arms:
            ci = hier_boot(E[arm], iqm, a.boot, rng)
            ex = {k: np.mean(np.concatenate([runs[s]["arms"][arm][k] for s in SEEDS])) for k in EXTRA}
            dth = np.mean(np.concatenate([runs[s]["arms"][arm]["deaths"] for s in SEEDS]))
            P(f"| {arm} | {iqm(np.concatenate(E[arm])):.1f} [{ci[0]:.1f}, {ci[1]:.1f}] | {np.mean(np.concatenate(E[arm])):.1f} | "
              + ", ".join(f"{np.mean(x):.1f}" for x in E[arm])
              + f" | {np.mean(np.concatenate(G[arm])):.1f} | {np.mean(np.concatenate(Q[arm])):.3f} | {dth:.2f} | {ex['tax']:.0f} | {ex['fines']:.0f} | "
                f"{ex['purchases']:.0f} | {ex['relief_paid']:.0f} | {ex['injection']:.0f} | {ex['newborn_outflow']:.0f} | {ex['neg_wealth_step_share']*100:.1f}% |")

        def pdiff(x, y):
            d = [E[x][i] - E[y][i] for i in range(len(SEEDS))]
            return np.mean(np.concatenate(d)), hier_boot(d, np.mean, a.boot, rng), [float(np.mean(v)) for v in d]

        P("\nHieu ghep cap rl_learned − nhanh (TB [CI95]; theo seed 42 / 202 / 303):")
        P("\n| So voi | Hieu TB [CI95] | Theo seed |")
        P("|---|---|---|")
        for arm in arms:
            if arm == "rl_learned":
                continue
            m, ci, ps = pdiff("rl_learned", arm)
            P(f"| {arm} | {m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | " + " / ".join(f"{v:+.2f}" for v in ps) + " |")

        mean = {arm: float(np.mean(np.concatenate(E[arm]))) for arm in arms}
        P("\n### Kiem gia thuyet (cach kiem co dinh truoc, xem docstring)")
        m, ci, ps = pdiff("rl_learned", "rl_mean_fixed")
        h1 = ci[0] <= 0.0 <= ci[1]
        P(f"- H1 [{mode}]: rl_learned − rl_mean_fixed = {m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] -> CI {'CHUA' if h1 else 'KHONG chua'} 0 (H1 can CA HAI che do)")
        seq = [mean[x] for x in PIT_SLICE]
        h2 = all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
        cons = [pdiff(PIT_SLICE[i + 1], PIT_SLICE[i]) for i in range(len(PIT_SLICE) - 1)]
        P(f"- H2 [{mode}]: TB doc pit 0 -> 0.10 -> 0.20 -> 0.30 -> 0.50 (cit 0) = " + " -> ".join(f"{v:.1f}" for v in seq)
          + f"; hieu ke tiep: " + "; ".join(f"{c[0]:+.2f} [{c[1][0]:+.2f}, {c[1][1]:+.2f}]" for c in cons) + f" -> H2 {'DUNG' if h2 else 'SAI'}")
        a_ok, a_txt = True, []
        for x in CIT_SLICE[1:4]:
            mm, cc, _ = pdiff(x, "free_market+rl_aux")
            a_ok &= cc[1] >= 0.0
            a_txt.append(f"{x}−(0,0) {mm:+.2f} [{cc[0]:+.2f}, {cc[1]:+.2f}]")
        cit_means = [mean[x] for x in CIT_SLICE]
        b_ok = int(np.argmax(cit_means)) == len(CIT_SLICE) - 1
        b_txt = []
        for x in CIT_SLICE[:4]:
            mm, cc, _ = pdiff(x, "flat_pit0_cit0.50")
            b_ok &= cc[1] < 0.0
            b_txt.append(f"{x}−cit0.50 {mm:+.2f} [{cc[0]:+.2f}, {cc[1]:+.2f}]")
        P(f"- H3 [{mode}]: TB lat cat cit 0/0.10/0.20/0.30/0.50 (pit 0) = " + " / ".join(f"{v:.1f}" for v in cit_means)
          + f"; (a) {'; '.join(a_txt)} -> {'DAT' if a_ok else 'KHONG DAT'}; (b) {'; '.join(b_txt)} -> {'DAT' if b_ok else 'KHONG DAT'}"
          + f" -> H3 {'DUNG' if (a_ok and b_ok) else 'SAI'}")
        g0 = np.mean(np.concatenate(G["free_market+rl_aux"])); e0 = np.mean(np.concatenate(Q["free_market+rl_aux"]))
        h4a, h4b, txt = True, True, []
        for x in CIT_SLICE[1:]:
            gx = np.mean(np.concatenate(G[x])); ex_ = np.mean(np.concatenate(Q[x]))
            de, dg = ex_ - e0, gx - g0
            share = abs(dg * e0) / max(1e-12, abs(dg * e0) + abs(g0 * de))
            h4a &= abs(de) < 0.02
            h4b &= share > 0.5
            txt.append(f"{x}: d(1−Gini) {de:+.4f}, dGDP {dg:+.1f}, phan GDP {share*100:.0f}%")
        P(f"- H4 [{mode}] (tham do): {'; '.join(txt)} -> (a) {'DAT' if h4a else 'KHONG DAT'}, (b) {'DAT' if h4b else 'KHONG DAT'} -> H4 {'DUNG' if (h4a and h4b) else 'SAI'}")
    P("\nQuy tac dang ky truoc: bao cao moi nhanh, khong bo nhanh nao, khong chay lai tru khi loi chuong trinh; ket qua bao nguyen van.")
    txt = "\n".join(out)
    print(txt)
    with open(a.md_out, "w", encoding="utf-8") as f:
        f.write("# Ket qua quet thue phang + rl_mean_fixed (dang ky truoc: PREREG_flat_sweep.md, commit 4e0327d)\n" + txt + "\n")


if __name__ == "__main__":
    main()
