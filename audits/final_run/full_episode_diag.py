"""Chan doan TOAN BO 240 buoc (khong chi 30 buoc dau sau reset) tren checkpoint dang train.
Chi DOC checkpoint, khong sua code. Do theo thoi gian:
  - ty le nguoi co viec co thuoc tinh wage == 0 (bug #29g) va effort theo 2 nhom
  - that nghiep, so vu sa thai vi luoi, tu vong
  - Kho bac, tro cap da chi, ty le buoc Kho bac < 1000 (cong tro cap dong)
  - hanh dong Government
"""
import os, sys
import numpy as np
REPO = r"C:\Users\piece\Downloads\TheAIEconomist\ai_economist_gpt"
sys.path.insert(0, REPO); os.chdir(REPO)
from be.env import MacroEnvironment
from be.scenario_config import ScenarioConfig
from be.core.enums import EventType
from be.agents.employee import Employee
from be.rllib_wrapper import build_ppo_config

ckpt = sys.argv[1]
EXPLORE = (len(sys.argv) > 2 and sys.argv[2] == "explore")
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 999


def pm(aid):
    if aid.startswith("emp_"): return "policy_employee"
    if aid.startswith("firm_"): return "policy_firm"
    if aid.startswith("bank_"): return "policy_bank"
    if aid == "gov_1": return "policy_government"
    if aid == "sup_1": return "policy_supervisor"
    return "policy_economy"


kw = ScenarioConfig.from_yaml("scenarios/em_baseline.yaml").to_env_kwargs()
algo = build_ppo_config(kw, num_env_runners=0, seed=42).build()
algo.restore(ckpt)
env = MacroEnvironment(**kw)
env.reset(seed=SEED)
shirk_fires = [0]
env.event_bus.subscribe(EventType.FIRE, lambda ev: shirk_fires.__setitem__(0, shirk_fires[0] + ("Shirking" in str(ev.payload.get("reason", "")))))

rows = []
gov_acts = []
for t in range(240):
    raw = env.get_raw_environment_state()
    acts = {aid: np.asarray(algo.compute_single_action(env.observe_agent(aid, raw), policy_id=pm(aid), explore=EXPLORE), dtype=np.float32)
            for aid in env.agents}
    gov_acts.append(acts["gov_1"].copy())
    emps = [a for a in env.agents.values() if isinstance(a, Employee)]
    employed = [e for e in emps if e.employed_by is not None]
    w0 = [e for e in employed if e.wage == 0.0]
    wp = [e for e in employed if e.wage > 0.0]
    eff0 = np.mean([acts[e.agent_id][0] for e in w0]) if w0 else np.nan
    effp = np.mean([acts[e.agent_id][0] for e in wp]) if wp else np.nan
    rows.append(dict(t=t, n_emp=len(emps), n_employed=len(employed), share_w0=len(w0) / max(1, len(employed)),
                     eff_w0=eff0, eff_wpos=effp, unemp=1 - len(employed) / max(1, len(emps)),
                     treasury=env.gov.treasury, price=env.eco.base_living_cost, wage_mean=(np.mean([e.wage for e in wp]) if wp else np.nan), infl=env.eco.inflation_rate))
    env.step(acts)
    rows[-1]["subs"] = env.gov.last_subsidies_paid

R = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
print(f"checkpoint={ckpt} explore={EXPLORE} seed={SEED}")
for lo, hi in [(0, 30), (30, 60), (60, 120), (120, 180), (180, 240)]:
    s = slice(lo, hi)
    print(f"  buoc {lo:3d}-{hi:3d}: share_wage0={np.nanmean(R['share_w0'][s])*100:5.1f}%  eff(wage0)={np.nanmean(R['eff_w0'][s]):.3f}  "
          f"eff(wage>0)={np.nanmean(R['eff_wpos'][s]):.3f}  unemp={np.mean(R['unemp'][s])*100:5.1f}%  "
          f"treasury_mean={np.mean(R['treasury'][s]):8.0f}  subs/step={np.mean(R['subs'][s]):5.2f}  price={np.mean(R['price'][s]):6.2f} (min {np.min(R['price'][s]):.2f})  wage(>0)={np.nanmean(R['wage_mean'][s]):6.2f}  frac_infl0={np.mean(np.abs(R['infl'][s])<1e-9)*100:5.1f}%")
print(f"TOAN EPISODE: share_wage0 trung binh={np.nanmean(R['share_w0'])*100:.1f}%  unemp trung binh={np.mean(R['unemp'])*100:.1f}%  "
      f"(buoc cuoi {R['unemp'][-1]*100:.1f}%)")
print(f"  shirk-fires={shirk_fires[0]}  deaths={env.gov.dead_citizens_count}  births={env.births_episode} emergency={env.emergency_births_episode}")
print(f"  Kho bac < 1000 (cong tro cap DONG) o {np.mean(R['treasury'] < 1000)*100:.1f}% so buoc; buoc dau tien < 1000: "
      f"{int(np.argmax(R['treasury'] < 1000)) if np.any(R['treasury'] < 1000) else 'khong'}")
print(f"  off-book outflow: newborn={env.treasury_outflow_newborn_episode:.0f} firm_entry={env.treasury_outflow_firm_entry_episode:.0f} "
      f"bailout={env.treasury_outflow_bailout_episode:.0f}")
ga = np.array(gov_acts)
print(f"  gov act mean [wtax,ftax,rho,inj,relief]={np.round(ga.mean(0),3)}  std trong episode={np.round(ga.std(0),3)}")
