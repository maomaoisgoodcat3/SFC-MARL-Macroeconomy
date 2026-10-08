"""P0.3(a) — muc ghep cap nhieu cua bench-v2.1 du 240 buoc voi MOI nhanh (sua doi dang ky A7.4 2026-10-08).
Che do lay mau. Voi moi nhanh X != rl_learned: ty le tac tu KHAC Government nhan CUNG nhieu voi rl_learned o tung buoc
(cung khoa gieo (seed|episode|buoc|policy|stt) va cung VI TRI trong lo suy luan). Nhieu gan theo VI TRI trong lo, khong theo ID tac tu:
khi tap tac tu song khac nhau (chet/sinh/firm vao-ra), cac tac tu dung sau cho lech trong lo nhan nhieu khac.
Dung: python audits/paper_extras/pairing_coverage_v2.py --checkpoint <iter_100> --episodes 2 --csv-out <file.csv>"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_v2 as b2  # noqa: E402  (tu chdir ve goc repo)
from be import benchmark as v1  # noqa: E402
from be.env import MacroEnvironment  # noqa: E402
from be.scenario_config import ScenarioConfig  # noqa: E402

ARMS = ["free_market", "us_federal", "saez", "free_market+rl_aux", "us_federal+rl_aux", "saez+rl_aux", "rl_mean_fixed",
        "flat_pit0.10_cit0", "flat_pit0_cit0.30"]


def run(env_config, algo, pairing, arm, episodes, seed, aux, full):
    env = MacroEnvironment(**env_config)
    rec = b2.StepRecorder(env)
    pairing.rec = rec
    pairing.batch_log = {}
    np.random.seed(seed)
    acts = []
    for ep in range(episodes):
        pairing.ep = ep
        rec.new_episode()
        _, _, _, _, gov_acts, _ = v1.run_episode(env, algo, arm, 240, explore=True, aux_override=aux if v1.needs_rl_aux(arm) else None,
                                                 episode_seed=seed + ep, batched=True, full_override=full)
        if len(gov_acts):
            acts.append(gov_acts)
    log, pairing.batch_log = pairing.batch_log, None
    return log, acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="scenarios/em_baseline.yaml")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv-out", required=True)
    a = ap.parse_args()
    b2.self_test()
    cfg = ScenarioConfig.from_yaml(a.config)
    cfg.max_steps = 240
    env_config = cfg.to_env_kwargs()
    algo = v1.build_algo(env_config, a.checkpoint)
    pairing = b2.TorchPairing(algo, a.seed, mode="step_policy")
    ref, gov_acts = run(env_config, algo, pairing, v1.RL_LEARNED_GOV, a.episodes, a.seed, None, None)
    allacts = np.concatenate(gov_acts, axis=0)
    aux, full = allacts[:, v1.AUX_DIMS].mean(axis=0).astype(np.float32), allacts.mean(axis=0).astype(np.float32)
    out = []
    for arm in ARMS:
        log, _ = run(env_config, algo, pairing, arm, a.episodes, a.seed, aux, full)
        for ep in range(a.episodes):
            for t in range(240):
                tot = same = 0
                for key, ids in ref.items():
                    if key[0] != ep or key[1] != t or key[2] == "policy_government":
                        continue
                    pos = {aid: i for i, aid in enumerate(log.get(key, []))}
                    for i, aid in enumerate(ids):
                        tot += 1
                        same += int(pos.get(aid) == i)
                out.append(dict(arm=arm, episode=ep, step=t, n_agents=tot, share_same_noise=(same / tot if tot else float("nan"))))
        sh = [r["share_same_noise"] for r in out if r["arm"] == arm]
        print(f"{arm:<22} TB={np.nanmean(sh):.3f}  buoc 0-19={np.nanmean([r['share_same_noise'] for r in out if r['arm']==arm and r['step']<20]):.3f}"
              f"  buoc 220-239={np.nanmean([r['share_same_noise'] for r in out if r['arm']==arm and r['step']>=220]):.3f}")
    with open(a.csv_out, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("da ghi", a.csv_out)


if __name__ == "__main__":
    main()
