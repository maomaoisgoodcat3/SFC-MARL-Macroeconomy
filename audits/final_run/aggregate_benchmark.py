"""Tong hop benchmark qua NHIEU seed huan luyen (buoc 6 ke hoach cuoi, THESIS_DRAFT muc 9).

Dau vao: cac file JSON do `python -m be.benchmark ... --json-out <f>` sinh ra, MOT file cho moi (seed huan luyen, che do
policy). Mac dinh benchmark gieo seed tung episode (episode i cua MOI nhanh cung dieu kien ban dau) -> hieu giua nhanh
duoc tinh GHEP CAP theo episode.

Thong ke (chon theo khuyen nghi bao cao RL voi it lan chay):
  - IQM (trung binh lien tu phan vi, bo 25% hai dau) tren toan bo episode gop qua seed -- Agarwal, Schwarzer, Castro,
    Courville & Bellemare (2021), "Deep Reinforcement Learning at the Edge of the Statistical Precipice", NeurIPS 34.
  - Khoang tin cay 95% bang bootstrap PHAN TANG 2 cap (lay mau lai seed, roi lay mau lai episode trong seed) -- vi bien
    thien giua seed huan luyen va giua episode la hai nguon khac nhau; chi bootstrap episode se danh gia thap do bat
    dinh (Saravanan, Berman & Sober, 2020, "Application of the hierarchical bootstrap to multi-level data in
    neuroscience", Neurons, Behavior, Data Analysis, and Theory 3(5)). Voi 3 seed, CI rong la TRUNG THUC, khong phai loi.
  - So seed ma rl_learned > baseline (trung binh ghep cap trong seed) -- bao cao kem, khong thay kiem dinh.
So lieu Eq x Prod tinh TUNG EPISODE = GDP_ep x (1 - Gini_ep) (khac bang in cua benchmark.py: tich cua hai trung binh).

Dung: python audits/final_run/aggregate_benchmark.py <file1.json> <file2.json> ... [--boot 10000] [--md-out bang.md]
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

METRICS = ["eq_x_prod", "gdp", "equality", "deaths", "unemp"]


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def episode_metrics(arm):
    gdp, gini = np.asarray(arm["gdp"]), np.asarray(arm["gini"])
    return dict(eq_x_prod=gdp * (1.0 - gini), gdp=gdp, equality=1.0 - gini,
                deaths=np.asarray(arm["deaths"], dtype=float), unemp=np.asarray(arm["unemp"]))


def hier_boot(groups, stat, n_boot, rng):
    """groups: list (theo seed) cac mang episode. Tra CI 95% cua stat tren mau gop."""
    out = np.empty(n_boot)
    k = len(groups)
    for b in range(n_boot):
        pick = rng.integers(0, k, size=k)
        sample = np.concatenate([groups[i][rng.integers(0, len(groups[i]), size=len(groups[i]))] for i in pick])
        out[b] = stat(sample)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--md-out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    by_mode = defaultdict(list)
    for fp in a.files:
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        d["_file"] = os.path.basename(fp)
        by_mode["explore" if d["explore"] else "deterministic"].append(d)

    lines = []
    for mode, runs in by_mode.items():
        seeding = {r["episode_seeding"] for r in runs}
        inference = {r.get("inference", "per_agent") for r in runs}
        arms = [k for k in runs[0]["arms"]]
        lines.append(f"\n## Che do policy: {mode} -- {len(runs)} seed huan luyen ({', '.join(r['_file'] for r in runs)}); "
                     f"gieo episode: {seeding}; suy luan: {inference}")
        if len(inference) > 1:
            lines.append("CANH BAO: tron che do suy luan (batched/per_agent) giua cac file -- khong cung dieu kien do.")
        if any(set(r["arms"]) != set(arms) for r in runs):
            lines.append("CANH BAO: cac file khong cung tap nhanh -- chi tong hop nhanh chung.")
            arms = [k for k in arms if all(k in r["arms"] for r in runs)]
        per = {arm: [episode_metrics(r["arms"][arm]) for r in runs] for arm in arms}

        lines.append("\n| Nhanh | Eq x Prod IQM [CI95] | TB theo seed | GDP IQM | 1-Gini IQM | Tu vong/ep TB | That nghiep TB |")
        lines.append("|---|---|---|---|---|---|---|")
        for arm in arms:
            g = [m["eq_x_prod"] for m in per[arm]]
            ci = hier_boot(g, iqm, a.boot, rng)
            seed_means = ", ".join(f"{np.mean(x):.1f}" for x in g)
            lines.append(f"| {arm} | {iqm(np.concatenate(g)):.1f} [{ci[0]:.1f}, {ci[1]:.1f}] | {seed_means} | "
                         f"{iqm(np.concatenate([m['gdp'] for m in per[arm]])):.1f} | "
                         f"{iqm(np.concatenate([m['equality'] for m in per[arm]])):.3f} | "
                         f"{np.mean(np.concatenate([m['deaths'] for m in per[arm]])):.2f} | "
                         f"{np.mean(np.concatenate([m['unemp'] for m in per[arm]])):.3f} |")

        if "rl_learned" in arms and all(r["episode_seeding"] == "paired" for r in runs):
            lines.append("\nHieu GHEP CAP theo episode, Eq x Prod: rl_learned - nhanh khac (TB [CI95 bootstrap 2 cap]; so seed rl_learned > nhanh do)")
            lines.append("\n| So voi | Hieu TB [CI95] | Seed thang |")
            lines.append("|---|---|---|")
            for arm in arms:
                if arm == "rl_learned":
                    continue
                diffs = []
                for m_rl, m_b in zip(per["rl_learned"], per[arm]):
                    n = min(len(m_rl["eq_x_prod"]), len(m_b["eq_x_prod"]))
                    diffs.append(m_rl["eq_x_prod"][:n] - m_b["eq_x_prod"][:n])
                ci = hier_boot(diffs, np.mean, a.boot, rng)
                wins = sum(1 for x in diffs if np.mean(x) > 0)
                lines.append(f"| {arm} | {np.mean(np.concatenate(diffs)):+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}] | {wins}/{len(diffs)} |")
        elif "rl_learned" in arms:
            lines.append("\n(Khong tinh hieu ghep cap: co file gieo episode kieu legacy -- episode khong cung dieu kien ban dau.)")

    lines.append("\nGhi chu bat buoc khi dung bang nay: (a) bat doi xung cong cu -- doc cap rl_learned vs <baseline>+rl_aux; "
                 "(b) loi the san nha -- cac tac tu khac dong bang tu the gioi RL da cung hoc (THESIS_DRAFT, CLAUDE.md).")
    txt = "\n".join(lines)
    print(txt)
    if a.md_out:
        with open(a.md_out, "w", encoding="utf-8") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
