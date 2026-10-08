"""YEU_CAU v4 P0.3: phan tich bench-v2.1 theo dung PREREG_A74_bench_v2.yaml (6a95505 + 452fb00 + 649087d). CHI DOC JSON.
Dau ra (thesis_support/v4/): bench_v2_episodes.csv, v2_comparisons.csv (quy tac bat dong tung so sanh x che do x thuoc do x bo seed),
v2_variance_reduction.csv (SD_v2/SD_v1), v2_report.md (gom: kiem tat dinh v2 == v1, Kendall tau, du doan co huong, ghep cap a/b/c).
Dung: python thesis_support/v4/analyze_v2.py [--boot 10000]"""
import argparse
import csv
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "audits", "final_run"))
sys.path.insert(0, os.path.join(REPO, "thesis_support", "v4"))
from aggregate_benchmark import hier_boot  # noqa: E402
import make_tables as MT  # noqa: E402  (duong dan file v1)

OUT = "thesis_support/v4"
V2 = "audits/paper_extras/v4_runs/v2"
CHK = "audits/paper_extras/v4_runs/checks"
MEASURES = {"primary": "gini", "gp_wealth": "gp_wealth_mean", "gp_net_wealth": "gp_net_wealth_mean", "gp_consumption": "gp_consumption_mean"}
CONTRASTS = ["free_market", "us_federal", "saez", "free_market+rl_aux", "rl_mean_fixed"]
BENCH7 = MT.ARMS7
PIT = ["free_market+rl_aux", "flat_pit0.10_cit0", "flat_pit0.20_cit0", "flat_pit0.30_cit0", "flat_pit0.50_cit0"]
CIT = ["free_market+rl_aux", "flat_pit0_cit0.10", "flat_pit0_cit0.20", "flat_pit0_cit0.30", "flat_pit0_cit0.50"]


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def eqp(arm, key):
    return np.asarray(arm["gdp"]) * (1.0 - np.asarray(arm[key]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    data = {(s, m): load(f"{V2}/bench2_seed{s}_{m}.json") for s in MT.ORIG + MT.REPL for m in MT.MODES}
    data = {k: v for k, v in data.items() if v}
    rep = ["# P0.3 — bench-v2.1 (sinh tự động bởi `python thesis_support/v4/analyze_v2.py`)\n",
           f"Seed × chế độ có dữ liệu: {sorted(data)}\n"]

    # bench_v2_episodes.csv
    with open(f"{OUT}/bench_v2_episodes.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["seed", "mode", "arm", "episode", "eqprod_gini_translated", "eqprod_GP_wealth", "eqprod_GP_netwealth", "eqprod_GP_consumption",
                    "real_gdp_mean", "gini_translated_mean", "gp_wealth_mean", "gp_net_wealth_mean", "gp_consumption_mean", "negwealth_step_share",
                    "k_share_mean", "bias_condition_share_among_neg_steps", "k_over_threshold_step_share", "tminus_over_tplus_mean_neg_steps",
                    "absmin_over_mean_mean_neg_steps", "min_wealth_episode", "deaths", "source_json", "sha256_match"])
        for (s, m), d in sorted(data.items()):
            for arm, x in d["arms"].items():
                for e in range(len(x["gdp"])):
                    w.writerow([s, m, arm, e] + [eqp(x, MEASURES[k])[e] for k in MEASURES] + [x["gdp"][e]]
                               + [x[MEASURES[k]][e] for k in MEASURES]
                               + [x["neg_step_share"][e], x["k_share_mean"][e], x["bias_condition_share_among_neg_steps"][e],
                                  x["k_over_threshold_step_share"][e], x["tminus_over_tplus_mean_neg_steps"][e],
                                  x["absmin_over_mean_mean_neg_steps"][e], x["min_wealth_episode"][e], x["deaths"][e],
                                  f"{V2}/bench2_seed{s}_{m}.json", d["sha256_match"]])

    # kiem tat dinh v2 == v1
    rep.append("\n## Tất định: v2.1 trùng v1?\n")
    for (s, m), d in sorted(data.items()):
        if m != "det":
            continue
        worst = 0.0
        for src in (MT.p7(s, m), MT.psw(s, m)):
            v1 = load(src)
            if not v1:
                continue
            for arm, x in v1["arms"].items():
                if arm in d["arms"]:
                    for k in ("gdp", "gini", "unemp", "deaths", "tax", "neg_wealth_step_share"):
                        worst = max(worst, float(np.max(np.abs(np.asarray(x[k], float) - np.asarray(d["arms"][arm][k], float)))))
        rep.append(f"- seed {s}: sai lệch lớn nhất mọi cột/nhánh/episode = {worst:.3e} ({'TRÙNG' if worst == 0 else 'KHÁC'})")

    # quy tac bat dong
    rows = []
    rep.append("\n## Quy tắc bất đồng (disagreement_rule) — dấu hiệu ghép cặp RL − nhánh\n")
    for setname, seeds in MT.SETS.items():
        for m in MT.MODES:
            ss = [s for s in seeds if (s, m) in data]
            if len(ss) < 2:
                continue
            rng = np.random.default_rng(0)
            n = len(ss)
            for arm in CONTRASTS:
                res = {}
                for meas, key in MEASURES.items():
                    diffs = [eqp(data[(s, m)]["arms"]["rl_learned"], key) - eqp(data[(s, m)]["arms"][arm], key) for s in ss]
                    ps = [float(np.mean(x)) for x in diffs]
                    ci = hier_boot(diffs, np.mean, a.boot, rng)
                    tq = stats.t.ppf(0.975, n - 1) * np.std(ps, ddof=1) / np.sqrt(n)
                    res[meas] = (float(np.mean(np.concatenate(diffs))), ci, np.mean(ps) - tq, np.mean(ps) + tq, sum(p > 0 for p in ps))
                s0 = np.sign(res["primary"][0])
                for meas, (mu, ci, tl, th, nwin) in res.items():
                    flip = bool(np.sign(mu) != s0)
                    sig_change = (ci[0] <= 0 <= ci[1]) != (res["primary"][1][0] <= 0 <= res["primary"][1][1])
                    rows.append(dict(set=setname, n_seeds=n, mode=m, contrast=f"rl_learned − {arm}", measure=meas, mean_diff=mu, ci_lo=ci[0],
                                     ci_hi=ci[1], t_lo=tl, t_hi=th, seeds_rl_better=f"{nwin}/{n}", sign_flip_vs_primary=flip,
                                     ci_contains0_changes_vs_primary=sig_change,
                                     conclusion=("ket qua phu thuoc vao thuoc do bat binh dang" if flip else "cung dau voi thuoc do chinh")))
    with open(f"{OUT}/v2_comparisons.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]) if rows else ["set"])
        w.writeheader()
        w.writerows(rows)
    flips = [r for r in rows if r["sign_flip_vs_primary"]]
    rep.append(f"Tổng {len(rows)} dòng; số dòng ĐẢO DẤU so với thước đo chính: {len(flips)}.")
    for r in flips:
        rep.append(f"- [{r['set']}, {r['mode']}] {r['contrast']} dưới {r['measure']}: {r['mean_diff']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] "
                   f"→ \"kết quả phụ thuộc vào thước đo bất bình đẳng\"")

    # Kendall tau thu tu 7 nhanh + du doan co huong + nhanh thue phang
    rep.append("\n## Thứ tự 7 nhánh (Kendall τ so với thước đo chính), dự đoán có hướng, lát cắt thuế\n")
    for setname, seeds in MT.SETS.items():
        for m in MT.MODES:
            ss = [s for s in seeds if (s, m) in data]
            if len(ss) < 2:
                continue
            means = {meas: [np.mean([np.mean(eqp(data[(s, m)]["arms"][arm], key)) for s in ss]) for arm in BENCH7] for meas, key in MEASURES.items()}
            taus = {meas: stats.kendalltau(means["primary"], v)[0] for meas, v in means.items() if meas != "primary"}
            rep.append(f"- [{setname}, {m}, n={len(ss)}] τ: " + ", ".join(f"{k} {v:.2f}" for k, v in taus.items()))
            for arm in ("us_federal", "saez"):
                d0 = np.mean([np.mean(eqp(data[(s, m)]["arms"]["rl_learned"], "gini") - eqp(data[(s, m)]["arms"][arm], "gini")) for s in ss])
                d1 = np.mean([np.mean(eqp(data[(s, m)]["arms"]["rl_learned"], "gp_wealth_mean") - eqp(data[(s, m)]["arms"][arm], "gp_wealth_mean")) for s in ss])
                bc = np.nanmean([np.nanmean(data[(s, m)]["arms"][arm]["bias_condition_share_among_neg_steps"]) for s in ss])
                kt = np.mean([np.mean(data[(s, m)]["arms"][arm]["k_over_threshold_step_share"]) for s in ss])
                rep.append(f"    - dự đoán (RL − {arm}) dưới G_P tài sản > dưới thước chính: {d1:+.1f} vs {d0:+.1f} → "
                           f"{'ĐÚNG' if d1 > d0 else 'SAI'}; % bước âm thoả điều kiện lệch = {bc * 100:.1f}%; % bước k > N²/(2N−1) = {kt * 100:.1f}%")
            for meas, key in MEASURES.items():
                pit = [np.mean([np.mean(eqp(data[(s, m)]["arms"][x], key)) for s in ss]) for x in PIT]
                cit = [np.mean([np.mean(eqp(data[(s, m)]["arms"][x], key)) for s in ss]) for x in CIT]
                rep.append(f"    - {meas}: lát cắt TNCN " + " > ".join(f"{v:.1f}" for v in pit)
                           + f" (giảm nghiêm ngặt: {all(pit[i] > pit[i + 1] for i in range(4))}); lát cắt DN " + " / ".join(f"{v:.1f}" for v in cit)
                           + f" (cao nhất: {CIT[int(np.argmax(cit))]})")

    # SD_v2/SD_v1 (che do lay mau)
    vr = []
    for s in MT.ORIG + MT.REPL:
        d2 = data.get((s, "exp"))
        v1a, v1s = load(MT.p7(s, "exp")), load(MT.psw(s, "exp"))
        if not d2 or not v1a:
            continue
        for arm in BENCH7[1:] + ["rl_mean_fixed", "flat_pit0_cit0.30", "flat_pit0_cit0.50"]:
            src = v1a if arm in v1a["arms"] else v1s
            if not src or arm not in src["arms"]:
                continue
            sd1 = np.std(eqp(src["arms"]["rl_learned"], "gini") - eqp(src["arms"][arm], "gini"), ddof=1)
            sd2 = np.std(eqp(d2["arms"]["rl_learned"], "gini") - eqp(d2["arms"][arm], "gini"), ddof=1)
            vr.append(dict(seed=s, contrast=f"rl_learned − {arm}", sd_v1=sd1, sd_v2=sd2, ratio=sd2 / sd1 if sd1 > 0 else float("nan")))
    if vr:
        with open(f"{OUT}/v2_variance_reduction.csv", "w", newline="", encoding="utf-8") as fo:
            w = csv.DictWriter(fo, fieldnames=list(vr[0]))
            w.writeheader()
            w.writerows(vr)
        rep.append(f"\n## SD_v2/SD_v1 (chế độ lấy mẫu, hiệu ghép cặp Eq×Prod chính, qua 10 episode)\n\nTrung vị tỷ số = "
                   f"{np.median([r['ratio'] for r in vr]):.3f} trên {len(vr)} cặp (seed × so sánh); tỷ số > 1 ở {sum(r['ratio'] > 1 for r in vr)} cặp.")
        by = defaultdict(list)
        for r in vr:
            by[r["contrast"]].append(r["ratio"])
        for k, v in by.items():
            rep.append(f"- {k}: trung vị {np.median(v):.3f} (n={len(v)})")

    # ghep cap (a)(b)(c)
    rep.append("\n## Ghép cặp nhiễu (P0.3 a/b/c)\n")
    pc = f"{CHK}/pairing_coverage_seed42.csv"
    if os.path.exists(pc):
        r = list(csv.DictReader(open(pc, encoding="utf-8")))
        by = defaultdict(list)
        for x in r:
            by[x["arm"]].append((int(x["step"]), float(x["share_same_noise"])))
        rep.append("(a) seed 42, lấy mẫu, 2 episode × 240 bước — tỷ lệ tác tử khác Government nhận CÙNG nhiễu với rl_learned:")
        for arm, v in by.items():
            arr = np.asarray(v)
            rep.append(f"- {arm}: TB {arr[:, 1].mean():.3f}; bước 0–19 {arr[arr[:, 0] < 20, 1].mean():.3f}; bước 100–119 "
                       f"{arr[(arr[:, 0] >= 100) & (arr[:, 0] < 120), 1].mean():.3f}; bước 220–239 {arr[arr[:, 0] >= 220, 1].mean():.3f}")
    else:
        rep.append("(a) CHƯA CÓ dữ liệu (chờ hàng đợi B).")
    rep.append("(b) Theo VỊ TRÍ trong lô: khoá gieo = sha256(seed|episode|bước|policy|thứ tự lần gọi trong bước), một lần gieo cho cả lô "
               "suy luận của policy; nhiễu từng hàng phụ thuộc vị trí hàng (thứ tự env.agents). Sinh/tử/firm vào-ra làm lệch vị trí các tác tử "
               "đứng sau → lệch nhiễu (audits/paper_extras/benchmark_v2.py, TorchPairing).")
    orv = load(f"{CHK}/order_reversed_seed42_exp.json")
    base = data.get((42, "exp"))
    if orv and base:
        worst = max(float(np.max(np.abs(np.asarray(orv["arms"][arm]["gdp"]) - np.asarray(base["arms"][arm]["gdp"]))))
                    for arm in orv["arms"] if arm in base["arms"])
        rep.append(f"(c) Đảo thứ tự nhánh (seed 42, lấy mẫu): sai lệch GDP lớn nhất theo episode so với thứ tự gốc = {worst:.3e} → "
                   f"{'ĐỘC LẬP thứ tự' if worst == 0 else 'PHỤ THUỘC thứ tự'}")
    else:
        rep.append("(c) CHƯA CÓ dữ liệu (chờ hàng đợi B).")
    open(f"{OUT}/v2_report.md", "w", encoding="utf-8").write("\n".join(rep) + "\n")
    print("ok", len(rows), "so sanh,", len(vr), "cap SD")


if __name__ == "__main__":
    main()
