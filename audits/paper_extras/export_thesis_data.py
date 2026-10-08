"""Xuat du lieu tho cho hinh/bang khoa luan (CHI DOC file san co; khong chay mo hinh). Dau ra: thesis_support/*.csv
  learning_curves.csv  : moi seed x iter: reward tung policy + vf_explained_var/kl/entropy tung policy (tu ~/ray_results/*/progress.csv)
                         seed 42 noi 2 run: iter 1-50 tu 2026-09-28_13-53-16, iter 51-100 tu 2026-09-28_18-13-35 (resume tu iter_50)
  effort_histograms.csv: histogram 10 bin effort (gom moi buoc x tac tu, 3 ep tat dinh + 10 ep lay mau) doc tu log cong gate_eval
  fiscal_flows.csv     : dong tien tai khoa moi episode theo nhanh (bench-v1, commit 84d3d8a, negwealth_7arms/bench7_*.json)
Dung: python audits/paper_extras/export_thesis_data.py"""
import csv, glob, json, os, re, sys
csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)  # progress.csv co cot hist_stats rat dai
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO)
RR = os.path.expanduser("~/ray_results")
RUNS = {42: [("PPO_RLlibMacroEnv_2026-09-28_13-53-16*", 1, 50), ("PPO_RLlibMacroEnv_2026-09-28_18-13-35*", 51, 100)],
        202: [("PPO_RLlibMacroEnv_2026-09-29_14-31-38*", 1, 100)], 303: [("PPO_RLlibMacroEnv_2026-10-01_16-34-53*", 1, 100)],
        404: [("PPO_RLlibMacroEnv_2026-10-06_11-57-23*", 1, 100)], 505: [("PPO_RLlibMacroEnv_2026-10-06_16-50-44*", 1, 100)],
        606: [("PPO_RLlibMacroEnv_2026-10-07_12-47-48*", 1, 100)], 707: [("PPO_RLlibMacroEnv_2026-10-07_17-51-44*", 1, 100)],
        808: [("PPO_RLlibMacroEnv_2026-10-08_09-41-29*", 1, 100)], 909: [("PPO_RLlibMacroEnv_2026-10-08_14-29-18*", 1, 100)]}
POL = ["government", "employee", "firm", "bank", "supervisor", "economy"]
os.makedirs("thesis_support", exist_ok=True)

cols = ["episode_reward_mean"] + [f"reward_{p}" for p in POL] + [f"{m}_{p}" for p in POL for m in ("vf_explained_var", "kl", "entropy")]
with open("thesis_support/learning_curves.csv", "w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo); w.writerow(["seed", "iter", "source_run"] + cols)
    for seed, parts in RUNS.items():
        for pat, lo, hi in parts:
            d = glob.glob(os.path.join(RR, pat)); assert len(d) == 1, (seed, pat, d)
            for r in csv.DictReader(open(os.path.join(d[0], "progress.csv"), encoding="utf-8")):
                it = int(r["training_iteration"])
                if not (lo <= it <= hi):
                    continue
                g = lambda k: r.get(k, "")
                row = [g("env_runners/episode_reward_mean")] + [g(f"env_runners/policy_reward_mean/policy_{p}") for p in POL] + \
                      [g(f"info/learner/policy_{p}/learner_stats/{m}") for p in POL for m in ("vf_explained_var", "kl", "entropy")]
                w.writerow([seed, it, os.path.basename(d[0])] + row)

logs = {(202, 40): "audits/final_run/logs/final_v037_seed202_gate_iter40_20260929_163805.log",
        (202, 100): "audits/final_run/logs/final_v037_seed202_final_iter100.log",
        (303, 40): "audits/final_run/logs/final_v037_seed303_gate_iter40_20261001_185500.log",
        (303, 100): "audits/final_run/logs/final_v037_seed303_final_iter100.log"}
for s in (404, 505, 606, 707, 808, 909):
    for it in (40, 100):
        p = f"audits/paper_extras/extra_seeds/gate_seed{s}_iter{it}.log"
        if os.path.exists(p):
            logs[(s, it)] = p
with open("thesis_support/effort_histograms.csv", "w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo); w.writerow(["seed", "iter", "bin_lo", "bin_hi", "count", "source_log"])
    for (s, it), p in sorted(logs.items()):
        m = re.search(r"histogram 10 bin=\[([0-9, ]+)\]", open(p, encoding="utf-8", errors="replace").read())
        for i, c in enumerate(int(x) for x in m.group(1).split(",")):
            w.writerow([s, it, i / 10, (i + 1) / 10, c, p])

KEYS = ["gdp", "gini", "unemp", "deaths", "tax", "fines", "purchases", "relief_paid", "injection", "newborn_outflow", "neg_wealth_step_share"]
with open("thesis_support/fiscal_flows.csv", "w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo); w.writerow(["seed", "mode", "arm", "episode"] + KEYS + ["source_json", "git_head"])
    for f in sorted(glob.glob("audits/paper_extras/negwealth_7arms/bench7_seed*_*.json")):
        s, mode = re.search(r"seed(\d+)_(det|exp)", f).groups()
        d = json.load(open(f, encoding="utf-8"))
        for arm, a in d["arms"].items():
            for e in range(len(a["gdp"])):
                w.writerow([s, mode, arm, e] + [a[k][e] for k in KEYS] + [f, d["provenance"]["git_head"][:7]])
print("ok")
