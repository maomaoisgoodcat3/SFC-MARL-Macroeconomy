"""YEU_CAU v4 P0.1 + P0.2: dung CSV tidy + bang (Markdown + doan LaTeX) tu JSON benchmark da co. CHI DOC — chay lai bat ky luc nao,
tu nhan seed nao da co file. Dung: python thesis_support/v4/make_tables.py [--boot 10000]

Nguon (bench-v1, tag bench-v1 -> 371799e, khong sua):
  - 7 nhanh: seed goc 42/202/303 -> audits/paper_extras/negwealth_7arms/bench7_seed{S}_{m}.json (84d3d8a; trung tuyet doi
    audits/final_run/logs/bench_seed{S}_{m}.json — da kiem 08/10); seed lap lai -> audits/paper_extras/v4_runs/v1/bench7_seed{S}_{m}.json
  - quet: seed goc -> audits/final_run/logs/flat_sweep/sweep_seed{S}_{m}.json; lap lai -> audits/paper_extras/v4_runs/v1/sweep_seed{S}_{m}.json
  - cot mo rong (births, min tai san, Kho bac/no cong cuoi, gia TB): lay tu bench-v2.1 CHE DO TAT DINH (v2.1 tat dinh trung v1 tung so —
    kiem lai tung episode truoc khi gan; lech -> de trong). Che do lay mau: v1 khong ghi -> de trong (KHONG CO).
Dinh dang o (yeu cau giang vien): mean^{±std}, mean = TB cac TB-theo-seed, std = do lech chuan (ddof=1) giua cac TB-theo-seed; kem IQM [CI95]
bootstrap 2 cap (10 000, rng(0), audits/final_run/aggregate_benchmark.py::hier_boot) — thuoc do chinh da dang ky.
Bo seed: goc (42, 202, 303), lap lai (404–909), gop 9 (PREREG_A8_replication.yaml)."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "audits", "final_run"))
from aggregate_benchmark import iqm, hier_boot  # noqa: E402

OUT = "thesis_support/v4"
ORIG, REPL = [42, 202, 303], [404, 505, 606, 707, 808, 909]
SETS = {"orig3": ORIG, "repl6": REPL, "all9": ORIG + REPL}
ARMS7 = ["rl_learned", "free_market", "free_market+rl_aux", "us_federal", "us_federal+rl_aux", "saez", "saez+rl_aux"]
MODES = {"det": "tất định", "exp": "lấy mẫu"}
CKSHA = {(int(r["seed"]), int(r["iter"])): r["sha256_dir"] for r in csv.DictReader(open(f"{OUT}/checkpoints_sha256.csv", encoding="utf-8"))}


def p7(s, m):
    return (f"audits/paper_extras/negwealth_7arms/bench7_seed{s}_{m}.json" if s in ORIG
            else f"audits/paper_extras/v4_runs/v1/bench7_seed{s}_{m}.json")


def psw(s, m):
    return (f"audits/final_run/logs/flat_sweep/sweep_seed{s}_{m}.json" if s in ORIG
            else f"audits/paper_extras/v4_runs/v1/sweep_seed{s}_{m}.json")


def pv2(s, m):
    return f"audits/paper_extras/v4_runs/v2/bench2_seed{s}_{m}.json"


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


COLS = ["seed", "mode", "arm", "episode", "episode_seed", "eqprod", "real_gdp_mean", "one_minus_gini_mean", "deaths", "births",
        "negwealth_step_share", "min_wealth_min", "taxes_pit", "taxes_cit", "taxes_total", "fines", "gov_purchases", "demand_injection",
        "relief_paid", "newborn_grants", "treasury_end", "public_debt_end", "unemployment_mean", "price_index_mean", "commit",
        "checkpoint_sha256", "source_json", "extended_cols_source"]


def rows_from(d, s, m, path, v2):
    out = []
    for arm, a in d["arms"].items():
        va = (v2 or {}).get("arms", {}).get(arm) if (m == "det" and v2) else None
        for e in range(len(a["gdp"])):
            ext_ok = bool(va) and abs(va["gdp"][e] - a["gdp"][e]) < 1e-12 and abs(va["gini"][e] - a["gini"][e]) < 1e-12
            g = lambda k: (va[k][e] if ext_ok else "")
            out.append(dict(seed=s, mode=m, arm=arm, episode=e, episode_seed=d["seed"] + e, eqprod=a["gdp"][e] * (1 - a["gini"][e]),
                            real_gdp_mean=a["gdp"][e], one_minus_gini_mean=1 - a["gini"][e], deaths=a["deaths"][e], births=g("births_episode"),
                            negwealth_step_share=a.get("neg_wealth_step_share", [""] * 99)[e], min_wealth_min=g("min_wealth_episode"),
                            taxes_pit="KHONG CO", taxes_cit="KHONG CO", taxes_total=a["tax"][e], fines=a["fines"][e], gov_purchases=a["purchases"][e],
                            demand_injection=a["injection"][e], relief_paid=a["relief_paid"][e], newborn_grants=a["newborn_outflow"][e],
                            treasury_end=g("treasury_end"), public_debt_end=g("public_debt_end"), unemployment_mean=a["unemp"][e],
                            price_index_mean=g("price_mean"), commit=d["provenance"]["git_head"][:7] or "(rong — xem NOTES_run_incidents.md muc 3: 0d964d8)",
                            checkpoint_sha256=CKSHA.get((s, 100), ""), source_json=path,
                            extended_cols_source=(pv2(s, m) if ext_ok else ("KHONG CO (che do lay mau v1)" if m == "exp" else "chua co v2.1"))))
    return out


def fmt(vals, nd=1):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return "—", "—"
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
    return f"{np.mean(v):.{nd}f}^{{±{sd:.{nd}f}}}", f"${np.mean(v):.{nd}f}^{{\\pm {sd:.{nd}f}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip().splitlines()

    # ---------------- P0.1: CSV tidy
    rows7, rowsw, have = [], [], defaultdict(list)
    for s in ORIG + REPL:
        for m in MODES:
            v2 = load(pv2(s, m))
            d = load(p7(s, m))
            if d:
                rows7 += [r for r in rows_from(d, s, m, p7(s, m), v2) if r["arm"] in ARMS7]
                have[m].append(s)
            w = load(psw(s, m))
            if w:
                rowsw += rows_from(w, s, m, psw(s, m), v2)
    for name, rows in (("bench_v1_episodes.csv", rows7), ("sweep_v1_episodes.csv", rowsw)):
        with open(f"{OUT}/{name}", "w", newline="", encoding="utf-8") as fo:
            w = csv.DictWriter(fo, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
    num = [c for c in COLS[5:24] if c not in ("taxes_pit", "taxes_cit")]
    with open(f"{OUT}/bench_v1_seed_summary.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["seed", "mode", "arm", "column", "mean", "sd_ddof1", "n_episodes"])
        grp = defaultdict(list)
        for r in rows7:
            grp[(r["seed"], r["mode"], r["arm"])].append(r)
        for (s, m, arm), rs in sorted(grp.items()):
            for c in num:
                v = [float(r[c]) for r in rs if r[c] != ""]
                if v:
                    w.writerow([s, m, arm, c, np.mean(v), np.std(v, ddof=1) if len(v) > 1 else "", len(v)])

    # ---------------- P0.2: bang
    E = defaultdict(dict)   # E[(mode, arm)][seed] = mang 10 episode cua 1 cot
    for r in rows7:
        for c in ("eqprod", "real_gdp_mean", "one_minus_gini_mean", "deaths", "negwealth_step_share", "taxes_total", "fines",
                  "gov_purchases", "demand_injection", "relief_paid", "newborn_grants"):
            E[(r["mode"], r["arm"], c)].setdefault(r["seed"], []).append(float(r[c]))
    md, tex = [f"# Bảng v4 (sinh tự động)\n\nHEAD `{head}`; file chưa commit: {len(dirty)}. Lệnh: `python thesis_support/v4/make_tables.py`. "
               f"Seed có dữ liệu bench-v1: det {have['det']}, exp {have['exp']}.\n"], []
    for setname, seeds in SETS.items():
        for m in MODES:
            ss = [s for s in seeds if s in have[m]]
            if len(ss) < 2:
                md.append(f"\n## [{setname}, {MODES[m]}] — chưa đủ seed ({ss})\n")
                continue
            rng = np.random.default_rng(0)
            n = len(ss)
            md.append(f"\n## [{setname}, {MODES[m]}] n = {n} seed: {ss}\n")
            # tab:main + tab:decomp
            md.append("| Nhánh | Eq×Prod mean^{±std} | IQM [CI95] | GDP thực | 1−Gini | Chết/ep |\n|---|---|---|---|---|---|")
            tex.append(f"% tab:main+decomp [{setname},{m}] n={n}")
            for arm in ARMS7:
                per = {c: [np.mean(E[(m, arm, c)][s]) for s in ss] for c in ("eqprod", "real_gdp_mean", "one_minus_gini_mean", "deaths")}
                groups = [np.asarray(E[(m, arm, "eqprod")][s]) for s in ss]
                ci = hier_boot(groups, iqm, a.boot, rng)
                cells = [fmt(per["eqprod"]), fmt(per["real_gdp_mean"]), fmt(per["one_minus_gini_mean"], 3), fmt(per["deaths"], 2)]
                md.append(f"| {arm} | {cells[0][0]} | {iqm(np.concatenate(groups)):.1f} [{ci[0]:.1f}, {ci[1]:.1f}] | {cells[1][0]} | {cells[2][0]} | {cells[3][0]} |")
                tex.append(f"{arm} & {cells[0][1]} & {iqm(np.concatenate(groups)):.1f} [{ci[0]:.1f}, {ci[1]:.1f}] & {cells[1][1]} & {cells[2][1]} & {cells[3][1]} \\\\")
            # tab:perseed
            md.append("\n| Nhánh | " + " | ".join(str(s) for s in ss) + " |\n|---|" + "---|" * n)
            for arm in ARMS7:
                md.append(f"| {arm} | " + " | ".join(f"{np.mean(E[(m, arm, 'eqprod')][s]):.1f}" for s in ss) + " |")
            # tab:paired
            md.append("\n| RL − nhánh | mean^{±std} theo seed | CI95 bootstrap 2 cấp | khoảng t (n−1) | số seed RL hơn |\n|---|---|---|---|---|")
            tex.append(f"% tab:paired [{setname},{m}] n={n}")
            for arm in ARMS7[1:]:
                diffs = [np.asarray(E[(m, "rl_learned", "eqprod")][s]) - np.asarray(E[(m, arm, "eqprod")][s]) for s in ss]
                ps = [float(np.mean(x)) for x in diffs]
                ci = hier_boot(diffs, np.mean, a.boot, rng)
                tq = stats.t.ppf(0.975, n - 1) * np.std(ps, ddof=1) / np.sqrt(n)
                c = fmt(ps)
                md.append(f"| {arm} | {c[0]} | [{ci[0]:+.1f}, {ci[1]:+.1f}] | [{np.mean(ps) - tq:+.1f}, {np.mean(ps) + tq:+.1f}] | {sum(p > 0 for p in ps)}/{n} |")
                tex.append(f"{arm} & {c[1]} & [{ci[0]:+.1f}, {ci[1]:+.1f}] & [{np.mean(ps) - tq:+.1f}, {np.mean(ps) + tq:+.1f}] & {sum(p > 0 for p in ps)}/{n} \\\\")
            # tab:flows + tab:negwealth
            md.append("\n| Nhánh | Thuế | Phạt | Chi G | Bơm cầu | Trợ cấp | Sơ sinh | % bước tài sản âm [min, max theo episode] |\n|---|---|---|---|---|---|---|---|")
            for arm in ARMS7:
                f = {c: [np.mean(E[(m, arm, c)][s]) for s in ss] for c in ("taxes_total", "fines", "gov_purchases", "demand_injection",
                                                                          "relief_paid", "newborn_grants", "negwealth_step_share")}
                allneg = np.concatenate([E[(m, arm, "negwealth_step_share")][s] for s in ss]) * 100
                md.append(f"| {arm} | " + " | ".join(fmt(f[c], 0)[0] for c in ("taxes_total", "fines", "gov_purchases", "demand_injection",
                                                                                 "relief_paid", "newborn_grants"))
                          + f" | {fmt(np.asarray(f['negwealth_step_share']) * 100)[0]} [{allneg.min():.1f}, {allneg.max():.1f}] |")
    # tab:actions (TB 5 cong cu Government RL moi seed, tu rl_full_mean cua JSON v1)
    md.append("\n## tab:actions — TB hành động Government RL (thô, 10 episode) theo seed\n\n| Chế độ | Seed | PIT | CIT | ρ | bơm cầu | trợ cấp |\n|---|---|---|---|---|---|---|")
    acts = defaultdict(list)
    for m in MODES:
        for s in have[m]:
            r = load(p7(s, m))["rl_full_mean"]
            acts[m].append(r)
            md.append(f"| {MODES[m]} | {s} | " + " | ".join(f"{x:.3f}" for x in r) + " |")
        if len(acts[m]) > 1:
            A = np.asarray(acts[m])
            md.append(f"| {MODES[m]} | mean^{{±std}} | " + " | ".join(fmt(A[:, j], 3)[0] for j in range(5)) + " |")
    # IQM [CI95] bang CHINH script da dang ky (audits/final_run/aggregate_benchmark.py) cho tung bo seed — trung tung chu so voi
    # benchmark_table_3seeds.md o bo goc (kiem 08/10). Quet thue phang: goi CHINH aggregate_flat_sweep.py (H1–H4 co dinh 0d964d8)
    # tren thu muc dung san (ten file giong ban goc), doi hang so SEEDS/LOG/SWEEP cua module — logic kiem khong doi.
    for setname, seeds in SETS.items():
        ss = [s for s in seeds if s in have["det"] and s in have["exp"]]
        if len(ss) < 2:
            continue
        files = [p7(s, m) for m in MODES for s in ss]
        subprocess.run([sys.executable, "audits/final_run/aggregate_benchmark.py", *files, "--boot", str(a.boot),
                        "--md-out", f"{OUT}/agg_benchmark_{setname}.md"], check=True, capture_output=True)
        sw = [s for s in ss if load(psw(s, "det")) and load(psw(s, "exp"))]
        if len(sw) >= 2:
            stage = f"{OUT}/_stage_{setname}"
            os.makedirs(f"{stage}/flat_sweep", exist_ok=True)
            for s in sw:
                for m in MODES:
                    shutil.copyfile(psw(s, m), f"{stage}/flat_sweep/sweep_seed{s}_{m}.json")
                    shutil.copyfile(p7(s, m), f"{stage}/bench_seed{s}_{m}.json")
            code = ("import sys; sys.path.insert(0, 'audits/final_run'); import aggregate_flat_sweep as A; "
                    f"A.SEEDS = tuple({sw!r}); A.LOG = r'{os.path.abspath(stage)}'; A.SWEEP = r'{os.path.abspath(stage)}/flat_sweep'; "
                    f"sys.argv = ['x', '--boot', '{a.boot}', '--md-out', r'{os.path.abspath(OUT)}/agg_flat_sweep_{setname}.md']; A.main()")
            subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
            shutil.rmtree(stage)
    open(f"{OUT}/tables_v1.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    open(f"{OUT}/tables_v1.tex", "w", encoding="utf-8").write("\n".join(tex) + "\n")
    print(f"ok: {len(rows7)} dong 7 nhanh, {len(rowsw)} dong quet; seed det {have['det']} exp {have['exp']}")


if __name__ == "__main__":
    main()
