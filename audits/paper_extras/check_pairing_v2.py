"""Kiem tra ghep cap nhieu cua bench-v2 TRUOC khi chay that (PREREG_A74_bench_v2.yaml, amendment_2026-10-07, checks_before_real_run).
Chay tren checkpoint that, episode ngan (cong cu, khong phai danh gia):
  python audits/paper_extras/check_pairing_v2.py --checkpoint be/checkpoint/final_v037_seed42/iter_100 [--steps 8 --episodes 2]
(1) tat dinh: ket qua v2 (co TorchPairing + StepRecorder) trung v1 thuan (be.benchmark.run_episode tren env/algo khong boc) tung so.
(2) lay mau, HAI lan chay rl_learned giong het -> quy dao giong het tung so.
(3) lay mau, rl_learned vs us_federal -> hanh dong cua moi tac tu khong phai Government o buoc 0 giong het.
    Chan doan (khong phai tieu chi): ty le tac tu khac Government nhan CUNG nhieu (cung khoa gieo + cung vi tri trong lo) theo buoc.
    KHONG dung "ty le trung hanh dong" de do ghep cap: tu buoc 1 trang thai da khac (Government khac) nen hanh dong khac du nhieu
    giong het.
(4) so lan suy luan lam doi np.random = 0.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_v2 as b2  # noqa: E402  (b2 tu chdir ve goc repo)
from be import benchmark as v1  # noqa: E402
from be.env import MacroEnvironment  # noqa: E402
from be.scenario_config import ScenarioConfig  # noqa: E402


def run_arm(env_config, algo, arm, explore, steps, episodes, seed, pairing=None, record=True):
    env = MacroEnvironment(**env_config)
    acts_log = []  # [episode][buoc] -> {aid: action}
    orig = env.step

    def logging_step(actions):
        acts_log[-1].append({k: np.array(v, dtype=np.float64).copy() for k, v in actions.items()})
        return orig(actions)
    env.step = logging_step
    rec = b2.StepRecorder(env) if record else None
    if pairing is not None:
        pairing.rec = rec
    np.random.seed(seed)
    out = []
    for ep in range(episodes):
        if pairing is not None:
            pairing.ep = ep
        if rec is not None:
            rec.new_episode()
        acts_log.append([])
        gdp, gini, unemp, deaths, gov_acts, extra = v1.run_episode(env, algo, arm, steps, explore=explore,
                                                                   episode_seed=seed + ep, batched=True)
        out.append(dict(gdp=gdp.tolist(), gini=gini.tolist(), unemp=unemp.tolist(), deaths=int(deaths), **extra))
    return out, acts_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="scenarios/em_baseline.yaml")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    b2.self_test()
    cfg = ScenarioConfig.from_yaml(args.config)
    cfg.max_steps = args.steps
    env_config = cfg.to_env_kwargs()
    ok = True

    algo_plain = v1.build_algo(env_config, args.checkpoint)
    ref, _ = run_arm(env_config, algo_plain, v1.RL_LEARNED_GOV, False, args.steps, args.episodes, args.seed, record=False)
    algo = v1.build_algo(env_config, args.checkpoint)
    pairing = b2.TorchPairing(algo, args.seed, mode="step_policy")
    got, _ = run_arm(env_config, algo, v1.RL_LEARNED_GOV, False, args.steps, args.episodes, args.seed, pairing)
    c1 = ref == got
    print(f"(1) tat dinh v2 == v1 tung so: {'DAT' if c1 else 'KHONG DAT'}")
    ok &= c1

    r_a, acts_a = run_arm(env_config, algo, v1.RL_LEARNED_GOV, True, args.steps, args.episodes, args.seed, pairing)
    r_b, acts_b = run_arm(env_config, algo, v1.RL_LEARNED_GOV, True, args.steps, args.episodes, args.seed, pairing)
    same_acts = all(set(x) == set(y) and all(np.array_equal(x[k], y[k]) for k in x)
                    for ea, eb in zip(acts_a, acts_b) for x, y in zip(ea, eb))
    c2 = (r_a == r_b) and same_acts
    print(f"(2) lay mau, rl_learned x2: quy dao + hanh dong giong het: {'DAT' if c2 else 'KHONG DAT'}")
    ok &= c2

    pairing.batch_log = {}
    run_arm(env_config, algo, v1.RL_LEARNED_GOV, True, args.steps, args.episodes, args.seed, pairing)
    log_a, pairing.batch_log = pairing.batch_log, {}
    r_c, acts_c = run_arm(env_config, algo, "us_federal", True, args.steps, args.episodes, args.seed, pairing)
    log_c, pairing.batch_log = pairing.batch_log, None
    gov = "gov_1"
    c3 = True
    for e, (ea, ec) in enumerate(zip(acts_a, acts_c)):
        first = [k for k in ea[0] if k != gov]
        same0 = all(np.array_equal(ea[0][k], ec[0][k]) for k in first)
        c3 &= same0
        noise_share = []
        for t in range(len(ea)):
            tot = same = 0
            for key, ids_a in log_a.items():
                if key[0] != e or key[1] != t or key[2] == "policy_government":
                    continue
                pos_c = {aid: i for i, aid in enumerate(log_c.get(key, []))}
                for i, aid in enumerate(ids_a):
                    tot += 1
                    same += int(pos_c.get(aid) == i)
            noise_share.append(same / tot if tot else float("nan"))
        print(f"    ep {e}: ty le tac tu khac Gov nhan CUNG nhieu theo buoc = {np.round(noise_share, 3).tolist()}")
    print(f"(3) lay mau, rl_learned vs us_federal: buoc 0 moi tac tu khac Government trung hanh dong: {'DAT' if c3 else 'KHONG DAT'}")
    ok &= c3

    c4 = pairing.np_state_changed_calls == 0
    print(f"(4) suy luan lam doi np.random: {pairing.np_state_changed_calls} lan / {pairing.seeded_calls} lan goi co gieo -> "
          f"{'DAT' if c4 else 'KHONG DAT'}")
    ok &= c4
    print("KET LUAN KIEM GHEP CAP:", "DAT CA 4" if ok else "KHONG DAT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
