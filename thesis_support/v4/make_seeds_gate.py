"""YEU_CAU v4 P0.4: seeds_gate.csv (moi seed x iter 40/100: gia tri do + nguong + DAT/TRUOT cua 7 dieu kien cong, ca 2 che do c3),
effort_histograms.csv (them seed 42 chay lai 08/10) va learning_curves.csv (9 seed). CHI DOC log/JSON san co.
Nguong lay tu audits/final_run/gate_eval.py (dong 177-189): c1 effort TB >= 0.5 va ty le <0.5 <= 15%; c2 khong buoc nao co wage=0 o moi
cua so; c3 original: TB chet tat dinh <= 0.5 va trung vi chet lay mau = 0; c3 split: nhu original nhung chi tinh ca KHONG thuoc #32 co che,
va ca #32 <= 6/episode, bo phan loai tu kiem hop le; c4 >= 2 cong cu Government co |d action| p5->p95 > 0.05; c5 SFC <= 1e-3;
c6 births/buoc max <= 0.35 va sinh khan cap TB <= 0.5; c7 tro cap khong bi chan khi con tien va khong vuot min(yeu cau, Kho bac con lai).
Dung: python thesis_support/v4/make_seeds_gate.py"""
import csv
import glob
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO)
OUT = "thesis_support/v4"
SHA = {(int(r["seed"]), int(r["iter"])): r["sha256_dir"] for r in csv.DictReader(open(f"{OUT}/checkpoints_sha256.csv", encoding="utf-8"))}

LOGS = {(42, 40): "audits/paper_extras/v4_runs/gate42/gate_seed42_iter40.log",
        (42, 100): "audits/paper_extras/v4_runs/gate42/gate_seed42_iter100.log",
        (202, 40): "audits/final_run/logs/final_v037_seed202_gate_iter40_20260929_163805.log",
        (202, 100): "audits/final_run/logs/final_v037_seed202_final_iter100.log",
        (303, 40): "audits/final_run/logs/final_v037_seed303_gate_iter40_20261001_185500.log",
        (303, 100): "audits/final_run/logs/final_v037_seed303_final_iter100.log"}
for s in (404, 505, 606, 707, 808, 909):
    for it in (40, 100):
        LOGS[(s, it)] = f"audits/paper_extras/extra_seeds/gate_seed{s}_iter{it}.log"

META = {  # script, retries, overrides, gio huan luyen, commit ma luc huan luyen (be/ khong doi tu ed57a19)
    42: ("thu cong (truoc khi co run_seed_with_gate.py) + gate_eval.py; cong iter 40 luc do che do original: TRUOT (#31 -> sua v0.37, chay lai "
         "tu dau; lan 2 TRUOT do #32) -> nguoi dung chon (A) chap nhan han che, chay tiep tu iter_50; cong o day CHAY LAI 08/10 tren checkpoint dong bang",
         "0", "1 (quyet dinh A, 2026-09-28)", "7.18 (iter 1-53: 3.87 h, dung 1-50; iter 51-100: 3.31 h)", "ed57a19"),
    202: ("run_seed_with_gate.py --c3-mode split (dung khi cong truot)", "0 (lan 1 DAT)", "0", "5.25", "ed57a19/9572b16 (be/ giong nhau)"),
    303: ("run_seed_with_gate.py --c3-mode split; goi lan 2 sau khi xong -> chay lai danh gia cuoi iter_100 (ghi de file final, checkpoint khong doi)",
          "0 (lan 1 DAT)", "0", "5.19", "9572b16"),
    404: ("run_day.sh (PREREG_extra_seeds: cong chi bao cao, KHONG dung; cong chay sau huan luyen)", "0", "0", "4.89", "84d3d8a"),
    505: ("run_day.sh", "0", "0", "5.58", "84d3d8a"),
    606: ("run_day.sh", "0", "0", "5.07", "d072f87"),
    707: ("run_day.sh", "0", "0", "5.25", "d072f87"),
    808: ("run_day.sh", "0", "0", "4.80", "e04cb35"),
    909: ("run_day.sh", "0", "0", "5.26", "e04cb35"),
}

R = {
    "c1": re.compile(r"^1 effort: TB=([\d.]+), ty le <0\.5=([\d.]+)%.*-> (DAT|KHONG DAT)", re.M),
    "c2": re.compile(r"^2 wage=0 .*max qua cac episode = (\[[^\]]*\]) % -> (DAT|KHONG DAT)", re.M),
    "c3": re.compile(r"^3 tu vong: tat dinh (\[[^\]]*\]), lay mau (\[[^\]]*\])", re.M),
    "c3s": re.compile(r"#32 CO CHE \(khong bao gio tuyen duoc\) tat dinh (\[[^\]]*\]) / lay mau (\[[^\]]*\])", re.M),
    "c3m": re.compile(r"che do original -> (DAT|KHONG DAT); che do split .*-> (DAT|KHONG DAT); DUNG CHO KET LUAN: (\w+)", re.M),
    "c3k": re.compile(r"tu kiem bo phan loai: (\d+) ca tuyen that, (\d+) ca vi pham", re.M),
    "c4": re.compile(r"^4 do nhay Government max\|d action\| p5->p95: (\{[^}]*\}); so cong cu > 0\.05: (\d+) -> (DAT|KHONG DAT)", re.M),
    "c5": re.compile(r"^5 SFC .*lech max=([\deE.+-]+) -> (DAT|KHONG DAT)", re.M),
    "c6": re.compile(r"^6 births/buoc max=([\d.]+), sinh khan cap TB=([\d.]+) -> (DAT|KHONG DAT)", re.M),
    "c7": re.compile(r"^7 tro cap.*?: (.*) -> (DAT|KHONG DAT)", re.M),
    "res": re.compile(r"GATE_RESULT: (PASS|FAIL)"),
}
COLS = ["seed", "iter", "source_log", "c1_effort_mean", "c1_low_share_pct", "c1_threshold", "c1", "c2_wage0_max_by_window_pct", "c2",
        "c3_deaths_det", "c3_deaths_exp", "c3_struct32_det", "c3_struct32_exp", "c3_classifier_hires_violations", "c3_original", "c3_split",
        "c3_mode_used", "c4_gov_sensitivity", "c4_n_tools_gt_0.05", "c4", "c5_sfc_max", "c5", "c6_births_max", "c6_emergency_mean", "c6",
        "c7_detail", "c7", "gate_result", "script", "retries", "overrides", "train_hours_wallclock", "train_code_commit", "checkpoint_sha256"]
rows = []
for (s, it), p in sorted(LOGS.items()):
    if not os.path.exists(p):
        rows.append(dict(seed=s, iter=it, source_log=p + " (CHUA CO)"))
        continue
    t = open(p, encoding="utf-8", errors="replace").read()
    g = {k: rx.search(t) for k, rx in R.items()}
    val = lambda k, i: (g[k].group(i) if g[k] else "KHONG TIM THAY")
    m = META[s]
    rows.append(dict(seed=s, iter=it, source_log=p, c1_effort_mean=val("c1", 1), c1_low_share_pct=val("c1", 2),
                     c1_threshold="TB>=0.5 & <0.5 <=15%", c1=val("c1", 3), c2_wage0_max_by_window_pct=val("c2", 1), c2=val("c2", 2),
                     c3_deaths_det=val("c3", 1), c3_deaths_exp=val("c3", 2), c3_struct32_det=val("c3s", 1), c3_struct32_exp=val("c3s", 2),
                     c3_classifier_hires_violations=(f"{val('c3k', 1)}/{val('c3k', 2)}" if g["c3k"] else "KHONG CO (ban cong cu)"),
                     c3_original=val("c3m", 1), c3_split=val("c3m", 2), c3_mode_used=val("c3m", 3),
                     c4_gov_sensitivity=val("c4", 1), **{"c4_n_tools_gt_0.05": val("c4", 2)}, c4=val("c4", 3), c5_sfc_max=val("c5", 1),
                     c5=val("c5", 2), c6_births_max=val("c6", 1), c6_emergency_mean=val("c6", 2), c6=val("c6", 3), c7_detail=val("c7", 1),
                     c7=val("c7", 2), gate_result=val("res", 1), script=m[0], retries=m[1], overrides=m[2], train_hours_wallclock=m[3],
                     train_code_commit=m[4], checkpoint_sha256=SHA.get((s, it), "")))
with open(f"{OUT}/seeds_gate.csv", "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)

# effort_histograms (bo sung seed 42) — cung dinh dang thesis_support/effort_histograms.csv
with open(f"{OUT}/effort_histograms.csv", "w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo)
    w.writerow(["seed", "iter", "bin_lo", "bin_hi", "count", "source_log"])
    for (s, it), p in sorted(LOGS.items()):
        if not os.path.exists(p):
            continue
        mm = re.search(r"histogram 10 bin=\[([0-9, ]+)\]", open(p, encoding="utf-8", errors="replace").read())
        if mm:
            for i, c in enumerate(int(x) for x in mm.group(1).split(",")):
                w.writerow([s, it, i / 10, (i + 1) / 10, c, p])
print("ok", len(rows), "dong;", sum(1 for r in rows if "CHUA CO" in r["source_log"]), "chua co log")
