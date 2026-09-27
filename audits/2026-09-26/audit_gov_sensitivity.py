"""Government policy co phu thuoc trang thai khong? Do do nhay cua hanh dong tat dinh theo
quan sat THAT (thu thap tu rollout) + nhieu loan tung chieu quan sat, o 3 checkpoint."""
import os, sys
import numpy as np
REPO = r"C:\Users\piece\Downloads\TheAIEconomist\ai_economist_gpt"
sys.path.insert(0, REPO); os.chdir(REPO)
from be.env import MacroEnvironment
from be.scenario_config import ScenarioConfig
from be.rllib_wrapper import build_ppo_config

def pm(aid):
    if aid.startswith("emp_"): return "policy_employee"
    if aid.startswith("firm_"): return "policy_firm"
    if aid.startswith("bank_"): return "policy_bank"
    if aid == "gov_1": return "policy_government"
    if aid == "sup_1": return "policy_supervisor"
    return "policy_economy"

cfg = ScenarioConfig.from_yaml("scenarios/em_baseline.yaml"); kw = cfg.to_env_kwargs()
algo = build_ppo_config(kw, num_env_runners=0, seed=42).build()
for it in (180, 460, 500):
    algo.restore(f"be/checkpoint/official_baseline_v1/iter_{it}")
    env = MacroEnvironment(**kw); env.reset(seed=7)
    gov_obs, gov_act = [], []
    for t in range(240):
        raw = env.get_raw_environment_state(); acts = {}
        for aid in env.agents:
            o = env.observe_agent(aid, raw)
            a = np.asarray(algo.compute_single_action(o, policy_id=pm(aid), explore=False), dtype=np.float32)
            acts[aid] = a
            if aid == "gov_1": gov_obs.append(o); gov_act.append(a)
        env.step(acts)
    O, A = np.array(gov_obs), np.array(gov_act)
    # do nhay: day tung chieu quan sat tu p5 -> p95 cua phan phoi THAT, giu cac chieu khac o trung vi
    med = np.median(O, 0); sens = []
    for d in range(O.shape[1]):
        lo, hi = med.copy(), med.copy()
        lo[d], hi[d] = np.percentile(O[:, d], 5), np.percentile(O[:, d], 95)
        a_lo = np.asarray(algo.compute_single_action(lo, policy_id="policy_government", explore=False))
        a_hi = np.asarray(algo.compute_single_action(hi, policy_id="policy_government", explore=False))
        sens.append(np.abs(a_hi - a_lo).max())
    print(f"iter_{it}: gov act mean={np.round(A.mean(0),3)} std(trong episode)={np.round(A.std(0),4)}")
    print(f"          obs range (p5-p95) per dim = {np.round(np.percentile(O,95,0)-np.percentile(O,5,0),3)}")
    print(f"          max |d action| khi day tung chieu obs p5->p95 = {np.round(sens,4)}")
