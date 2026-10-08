"""YEU_CAU v4 P0.8: hien tuong cuoi chan troi — PHAN TICH KHAM PHA (khong phai kiem dinh dang ky truoc; sua doi dang ky 649087d).
Du lieu theo buoc cua bench-v2.1 (audits/paper_extras/v4_runs/v2/bench2_seed{S}_{m}_steps.csv.gz), nhanh rl_learned (va moi nhanh benchmark
de doi chung). So 20 buoc cuoi (t = 221..240) voi buoc 1..220 cho: Gini mo hinh, G_P tai san, GDP thuc, GDP danh nghia, gia, effort TB,
tieu dung TB, tong no ho, tu vong, hanh dong Government (thue thuc thi, rho, bom cau, tro cap tho). Dong gop 20 buoc cuoi vao Eq×Prod:
Eq×Prod(1..240) − Eq×Prod(1..220), Eq×Prod(T) = TB_{t<=T}(GDP thuc) × (1 − TB_{t<=T}(Gini)) — dung dinh nghia cua thuoc do chinh.
Dau ra: horizon_episodes.csv (seed, mode, arm, episode, bien, TB 1..220, TB 221..240, chenh), horizon_eqprod.csv, horizon_report.md.
Dung: python thesis_support/v4/analyze_horizon.py"""
import csv
import gzip
import os
from collections import defaultdict

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO)
OUT = "thesis_support/v4"
V2 = "audits/paper_extras/v4_runs/v2"
SEEDS = [42, 202, 303, 404, 505, 606, 707, 808, 909]
ARMS = ["rl_learned", "free_market", "us_federal", "saez", "free_market+rl_aux", "us_federal+rl_aux", "saez+rl_aux"]
VARS = ["gini_env", "gp_wealth", "real_gdp", "nominal_gdp", "price", "effort_mean", "consumption_mean", "household_debt_total",
        "deaths_step", "unemployment", "exec_pit", "exec_cit", "exec_rho", "exec_inj", "gov_a4", "treasury", "public_debt"]
CUT = 220


def main():
    rows, eq, rep = [], [], ["# P0.8 — Hiện tượng cuối chân trời (khám phá; sinh bởi `python thesis_support/v4/analyze_horizon.py`)\n"]
    for s in SEEDS:
        for m in ("det", "exp"):
            p = f"{V2}/bench2_seed{s}_{m}_steps.csv.gz"
            if not os.path.exists(p):
                continue
            by = defaultdict(list)
            for r in csv.DictReader(gzip.open(p, "rt", encoding="utf-8")):
                if r["arm"] in ARMS:
                    by[(r["arm"], int(r["episode"]))].append(r)
            for (arm, ep), rs in sorted(by.items()):
                rs.sort(key=lambda r: int(r["t"]))
                dc = np.asarray([float(r["deaths_cum"]) for r in rs])
                col = {v: np.asarray([float(r[v]) if r.get(v, "") not in ("", "None", "nan") else np.nan for r in rs]) for v in VARS if v != "deaths_step"}
                col["deaths_step"] = np.diff(np.concatenate([[0.0], dc]))
                t = np.asarray([int(r["t"]) for r in rs])
                early, late = t <= CUT, t > CUT
                for v in VARS:
                    x = col[v]
                    rows.append(dict(seed=s, mode=m, arm=arm, episode=ep, variable=v, mean_1_220=np.nanmean(x[early]),
                                     mean_221_240=np.nanmean(x[late]) if late.any() else np.nan,
                                     diff=(np.nanmean(x[late]) - np.nanmean(x[early])) if late.any() else np.nan))
                g, gi = col["real_gdp"], col["gini_env"]
                e240 = np.nanmean(g) * (1 - np.nanmean(gi))
                e220 = np.nanmean(g[early]) * (1 - np.nanmean(gi[early]))
                eq.append(dict(seed=s, mode=m, arm=arm, episode=ep, eqprod_1_240=e240, eqprod_1_220=e220, contribution=e240 - e220,
                               contribution_pct=100 * (e240 - e220) / e240 if e240 else np.nan, n_steps=len(rs)))
    if not rows:
        rep.append("CHƯA CÓ dữ liệu theo bước.")
    else:
        for name, data in (("horizon_episodes.csv", rows), ("horizon_eqprod.csv", eq)):
            with open(f"{OUT}/{name}", "w", newline="", encoding="utf-8") as fo:
                w = csv.DictWriter(fo, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
        rep.append("**Câu (1)** — quan sát KHÔNG chứa bước thời gian/thời gian còn lại (xem P0_6_P0_7_audit.md, mục P0.7).\n")
        rep.append("**Câu (2)** — đóng góp 20 bước cuối vào Eq×Prod (TB qua episode; Eq×Prod(1..240) − Eq×Prod(1..220)):\n")
        rep.append("| Chế độ | Seed | rl_learned: Eq×Prod 1..240 | 1..220 | chênh (%) | Gini 1..220 → 221..240 | GDP thực 1..220 → 221..240 | GDP d.nghĩa | giá |")
        rep.append("|---|---|---|---|---|---|---|---|---|")
        idx = defaultdict(list)
        for r in rows:
            idx[(r["seed"], r["mode"], r["arm"], r["variable"])].append(r)
        for m in ("det", "exp"):
            for s in SEEDS:
                e = [x for x in eq if x["seed"] == s and x["mode"] == m and x["arm"] == "rl_learned"]
                if not e:
                    continue
                f = lambda v, k: np.mean([x[k] for x in idx[(s, m, "rl_learned", v)]])
                rep.append(f"| {m} | {s} | {np.mean([x['eqprod_1_240'] for x in e]):.1f} | {np.mean([x['eqprod_1_220'] for x in e]):.1f} | "
                           f"{np.mean([x['contribution_pct'] for x in e]):+.2f}% | {f('gini_env', 'mean_1_220'):.3f} → {f('gini_env', 'mean_221_240'):.3f} | "
                           f"{f('real_gdp', 'mean_1_220'):.0f} → {f('real_gdp', 'mean_221_240'):.0f} | {f('nominal_gdp', 'mean_1_220'):.0f} → "
                           f"{f('nominal_gdp', 'mean_221_240'):.0f} | {f('price', 'mean_1_220'):.2f} → {f('price', 'mean_221_240'):.2f} |")
        rep.append("\nĐối chứng: cùng chênh lệch 20 bước cuối ở các nhánh baseline (Government cố định) — nếu baseline cũng có, hiện tượng đến từ "
                   "động lực tích luỹ của môi trường/tác tử khác, không riêng planner RL:\n")
        for m in ("det", "exp"):
            for arm in ARMS:
                c = [x["contribution_pct"] for x in eq if x["mode"] == m and x["arm"] == arm]
                dg = [x["diff"] for x in rows if x["mode"] == m and x["arm"] == arm and x["variable"] == "gini_env"]
                if c:
                    rep.append(f"- {m} {arm}: đóng góp TB {np.mean(c):+.2f}% [min {np.min(c):+.2f}, max {np.max(c):+.2f}]; ΔGini cuối TB {np.nanmean(dg):+.4f}")
    open(f"{OUT}/horizon_report.md", "w", encoding="utf-8").write("\n".join(rep) + "\n")
    print("ok", len(rows), "dong")


if __name__ == "__main__":
    main()
