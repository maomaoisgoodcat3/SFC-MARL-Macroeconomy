"""
AUDIT (khong sua repo): do cac nghi van phat hien qua doc code tinh tren checkpoint iter_500
(tat ca tac tu dung policy RL da hoc) va doi chung policy ngau nhien.
Chay: python audit_diagnostics.py [--checkpoint PATH] [--episodes N]
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

REPO = r"C:\Users\piece\Downloads\TheAIEconomist\ai_economist_gpt"
sys.path.insert(0, REPO)
os.chdir(REPO)

from be.env import MacroEnvironment
from be.scenario_config import ScenarioConfig
from be.core.enums import EventType, LifeCycleStatus
from be.agents.employee import Employee
from be.agents.firm import Firm
from be.rllib_wrapper import (EMPLOYEE_ACT_SPACE, FIRM_ACT_SPACE, GOVERNMENT_ACT_SPACE,
                              BANK_ACT_SPACE, SUPERVISOR_ACT_SPACE, ECONOMY_ACT_SPACE)


def policy_mapping(aid):
    if aid.startswith("emp_"): return "policy_employee"
    if aid.startswith("firm_"): return "policy_firm"
    if aid.startswith("bank_"): return "policy_bank"
    if aid == "gov_1": return "policy_government"
    if aid == "sup_1": return "policy_supervisor"
    return "policy_economy"


def space_for(aid):
    return {"policy_employee": EMPLOYEE_ACT_SPACE, "policy_firm": FIRM_ACT_SPACE,
            "policy_bank": BANK_ACT_SPACE, "policy_government": GOVERNMENT_ACT_SPACE,
            "policy_supervisor": SUPERVISOR_ACT_SPACE, "policy_economy": ECONOMY_ACT_SPACE}[policy_mapping(aid)]


def run(env_kwargs, algo, episodes, seed0, label):
    rng = np.random.default_rng(seed0)
    D = defaultdict(list)
    for ep in range(episodes):
        env = MacroEnvironment(**env_kwargs)
        np.random.seed(seed0 + ep)
        obs, _ = env.reset(seed=seed0 + ep)
        bad = {"firm": 0.0, "death": 0.0, "firm_bankrupt_n": 0, "death_n": 0, "death_with_debt_n": 0}

        def on_bankrupt(ev):
            if ev.source_id.startswith("firm_"):
                bad["firm"] += float(ev.payload.get("bad_debt", 0.0)); bad["firm_bankrupt_n"] += 1

        def on_died(ev):
            b = float(ev.payload.get("bad_debt", 0.0))
            bad["death"] += b; bad["death_n"] += 1
            if b > 0: bad["death_with_debt_n"] += 1

        env.event_bus.subscribe(EventType.AGENT_BANKRUPT, on_bankrupt)
        env.event_bus.subscribe(EventType.AGENT_DIED, on_died)
        gov_actions, bank_actions = [], []
        g_rew_parts = defaultdict(float)
        for t in range(env.max_steps):
            raw = env.get_raw_environment_state()
            acts = {}
            for aid, ag in env.agents.items():
                if algo is not None:
                    a = algo.compute_single_action(env.observe_agent(aid, raw), policy_id=policy_mapping(aid), explore=False)
                else:
                    sp = space_for(aid)
                    a = rng.uniform(sp.low, sp.high)
                acts[aid] = np.asarray(a, dtype=np.float32)
            gov_actions.append(acts["gov_1"].copy())
            bank_actions.append(acts["bank_0"].copy())
            obs, rew, term, trunc, info = env.step(acts)

            firms = [a for a in env.agents.values() if isinstance(a, Firm) and a.status == LifeCycleStatus.ACTIVE]
            emps = [a for a in env.agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE]
            fc = [f.cash for f in firms]
            D["firm_cash_min"].append(min(fc) if fc else 0.0)
            D["firm_neg_frac"].append(np.mean([c < 0 for c in fc]) if fc else 0.0)
            D["firm_neg_total"].append(sum(min(0.0, c) for c in fc))
            D["firm_debt_total"].append(sum(f.debt for f in firms))
            D["firm_leverage_max"].append(max((f.debt / max(1.0, f.capital_stock)) for f in firms) if firms else 0.0)
            D["emp_debt_total"].append(sum(e.debt for e in emps))
            D["emp_neg_cash_total"].append(sum(min(0.0, e.cash) for e in emps))
            D["bank_loans"].append(sum(b.total_loans for b in env.banks))
            D["deposit_ledger_gap"].append(sum(b.total_deposits for b in env.banks) - sum(getattr(e, "bank_deposit", 0.0) for e in emps))
            md = env.get_raw_environment_state()["macro_indicators"]["market_demand_factor"]
            D["market_demand_factor"].append(md)
            D["treasury"].append(env.gov.treasury)
            D["gdp_nominal"].append(env.gov.current_gdp)
            D["real_gdp"].append(env.gov.current_real_gdp)
            D["public_debt"].append(env.gov.public_debt)
            D["bank_reserves"].append(sum(b.reserves for b in env.banks))
            D["gini"].append(env.gov.current_gini)
            D["price"].append(env.eco.base_living_cost)
            D["unemp"].append(np.mean([e.employed_by is None for e in emps]) if emps else 0.0)
            D["pop"].append(len(emps))
            D["nfirms"].append(len(firms))
            wages = [e.wage for e in emps if e.employed_by is not None]
            D["avg_wage"].append(np.mean(wages) if wages else 0.0)
            # Government reward decomposition (dung dung cong thuc government.py)
            g = env.gov
            g_rew_parts["gdp_level"] += g.current_real_gdp * 0.002
            g_rew_parts["gdp_growth"] += (g.current_real_gdp - g.last_real_gdp) * 0.005
            g_rew_parts["gini_pen"] += g.gini_penalty_coef * g.current_gini ** 2
            if trunc.get("__all__", False):
                break
        g_rew_parts["death_pen"] += bad["death_n"] * env.gov.death_penalty_coef
        D["bad_firm"].append(bad["firm"]); D["bad_death"].append(bad["death"])
        D["n_firm_bankrupt"].append(bad["firm_bankrupt_n"]); D["n_deaths"].append(bad["death_n"])
        D["n_death_with_debt"].append(bad["death_with_debt_n"])
        ga = np.array(gov_actions); ba = np.array(bank_actions)
        D["gov_act_mean"].append(ga.mean(0)); D["gov_act_std"].append(ga.std(0))
        D["bank_act_mean"].append(ba.mean(0))
        for k, v in g_rew_parts.items():
            D["gov_rew_" + k].append(v)

    print(f"\n================ {label} ({episodes} ep x 240 buoc) ================")
    def s(k, fmt="{:.3f}"):
        v = np.array(D[k], dtype=float)
        return f"mean={fmt.format(v.mean())} min={fmt.format(v.min())} max={fmt.format(v.max())}"
    for k in ["firm_cash_min", "firm_neg_frac", "firm_neg_total", "firm_debt_total", "firm_leverage_max",
              "emp_debt_total", "emp_neg_cash_total", "bank_loans", "deposit_ledger_gap",
              "market_demand_factor", "treasury", "gdp_nominal", "real_gdp", "public_debt", "bank_reserves",
              "gini", "price", "unemp", "pop", "nfirms", "avg_wage"]:
        print(f"{k:22s} {s(k)}")
    for k in ["bad_firm", "bad_death", "n_firm_bankrupt", "n_deaths", "n_death_with_debt"]:
        print(f"{k:22s} per-episode = {[round(x, 1) for x in D[k]]}")
    print("gov action mean [wtax, ftax, rho, inject, relief] =", np.round(np.mean(D["gov_act_mean"], 0), 3),
          " std within-ep =", np.round(np.mean(D["gov_act_std"], 0), 3))
    print("bank action mean [lend, dep, credit]     =", np.round(np.mean(D["bank_act_mean"], 0), 3))
    tot = {k: np.mean(D["gov_rew_" + k]) for k in ["gdp_level", "gdp_growth", "gini_pen", "death_pen"]}
    print("Government reward (TONG tho/episode): ", {k: round(v, 1) for k, v in tot.items()})
    tr = np.array(D["treasury"]); gd = np.array(D["gdp_nominal"])
    print(f"Treasury / GDP-moi-buoc (trung vi) = {np.median(tr / np.maximum(1, gd)):.0f} buoc GDP")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="be/checkpoint/official_baseline_v1/iter_500")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--random-only", action="store_true")
    args = ap.parse_args()
    cfg = ScenarioConfig.from_yaml("scenarios/em_baseline.yaml")
    env_kwargs = cfg.to_env_kwargs()

    run(env_kwargs, None, args.episodes, 1000, "POLICY NGAU NHIEN (doi chung)")
    if not args.random_only:
        from be.rllib_wrapper import build_ppo_config
        config = build_ppo_config(env_kwargs, num_env_runners=0, seed=42)
        algo = config.build_algo() if hasattr(config, "build_algo") else config.build()
        algo.restore(args.checkpoint)
        run(env_kwargs, algo, args.episodes, 1000, f"POLICY RL DA HOC ({args.checkpoint})")
