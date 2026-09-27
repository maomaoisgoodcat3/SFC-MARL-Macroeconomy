"""So sanh explore=False (tat dinh, benchmark dung) vs explore=True (ngau nhien, giong luc train)
tren checkpoint, + phan ra nguyen nhan tu vong va hanh dong trung binh cua Employee/Firm."""
import os, sys
from collections import defaultdict, Counter
import numpy as np
REPO = r"C:\Users\piece\Downloads\TheAIEconomist\ai_economist_gpt"
sys.path.insert(0, REPO); os.chdir(REPO)
from be.env import MacroEnvironment
from be.scenario_config import ScenarioConfig
from be.core.enums import EventType, LifeCycleStatus
from be.agents.employee import Employee
from be.agents.firm import Firm
from be.rllib_wrapper import build_ppo_config

def pm(aid):
    if aid.startswith("emp_"): return "policy_employee"
    if aid.startswith("firm_"): return "policy_firm"
    if aid.startswith("bank_"): return "policy_bank"
    if aid == "gov_1": return "policy_government"
    if aid == "sup_1": return "policy_supervisor"
    return "policy_economy"

ckpt = sys.argv[1] if len(sys.argv) > 1 else "be/checkpoint/official_baseline_v1/iter_500"
cfg = ScenarioConfig.from_yaml("scenarios/em_baseline.yaml"); kw = cfg.to_env_kwargs()
algo = build_ppo_config(kw, num_env_runners=0, seed=42).build()
algo.restore(ckpt)
print("checkpoint:", ckpt)
for explore in (False, True):
    agg = defaultdict(list); reasons = Counter()
    emp_acts, firm_acts = [], []
    for ep in range(3):
        env = MacroEnvironment(**kw); np.random.seed(500 + ep); env.reset(seed=500 + ep)
        env.event_bus.subscribe(EventType.AGENT_DIED, lambda ev: reasons.update([ev.payload.get("reason")]))
        deaths = 0; unemp = []; pop = []; price = []
        for t in range(240):
            raw = env.get_raw_environment_state(); acts = {}
            for aid in env.agents:
                a = np.asarray(algo.compute_single_action(env.observe_agent(aid, raw), policy_id=pm(aid), explore=explore), dtype=np.float32)
                acts[aid] = a
                if aid.startswith("emp_"): emp_acts.append(a)
                elif aid.startswith("firm_"): firm_acts.append(a)
            env.step(acts)
            emps = [a for a in env.agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE]
            unemp.append(np.mean([e.employed_by is None for e in emps]) if emps else 1.0)
            pop.append(len(emps)); price.append(env.eco.base_living_cost)
        agg["deaths"].append(env.gov.dead_citizens_count); agg["unemp"].append(np.mean(unemp))
        agg["pop"].append(np.mean(pop)); agg["price"].append(np.mean(price)); agg["gini"].append(env.gov.current_gini)
    ea, fa = np.array(emp_acts), np.array(firm_acts)
    print(f"\nexplore={explore}: deaths/ep={agg['deaths']} unemp_mean={np.round(agg['unemp'],3)} pop_mean={np.round(agg['pop'],1)} price_mean={np.round(agg['price'],1)}")
    print("  ly do tu vong:", dict(reasons))
    print("  Employee act mean [effort, declare, consume, borrow] =", np.round(ea.mean(0), 3), " std =", np.round(ea.std(0), 3))
    print("  Firm act mean [hire, borrow, declare] =", np.round(fa.mean(0), 3), " std =", np.round(fa.std(0), 3))
