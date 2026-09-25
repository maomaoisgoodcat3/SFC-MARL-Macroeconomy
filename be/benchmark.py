"""
So sanh Government theo baseline rule-based (Free Market / US Federal / Saez, xem
be/baselines.py) doi chieu voi policy da hoc (RL) -- dung dung khung KPI cua Zheng
et al. (2020/2022): Productivity (GDP thuc trung binh), Equality (1-Gini trung
binh), va tich Equality x Productivity lam thuoc do phuc loi tong hop.

CHUA co checkpoint da train hop le tai thoi diem viet script nay (moi checkpoint-
freq dat sai trong cac run ablation/multi-seed truoc do deu > train-iters, xem
CLAUDE_HISTORY.md v0.26 "checkpoint-freq"). Khi CHUA co --checkpoint, script chay
CA 3 baseline VOI heuristic decide() cho moi tac tu KHAC Government (khong phai so
sanh "RL da hoc" vs "rule-based" that su -- chi kiem tra ha tang/plumbing va cho
so lieu tham khao ve chinh 3 bieu thue). Khi CO --checkpoint hop le, Employee/Firm/
Bank/Supervisor/Economy se dung policy da hoc that (qua algo.compute_single_action),
CHI RIENG Government bi ep theo baseline dang xet moi buoc -- dung "empirical best-
response" kieu Curry et al. 2022 (dong bang cac tac tu khac, chi doi 1 tac tu).

Dung: python -m be.benchmark --episodes 10 --max-steps 240 [--checkpoint <path>]
"""
import argparse
import os
import numpy as np

from be.env import MacroEnvironment
from be.scenario_config import ScenarioConfig
from be.core.enums import LifeCycleStatus
from be.agents.employee import Employee
from be.agents.government import Government
from be.baselines import BASELINE_NAMES, get_gov_baseline_action


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=240)
    p.add_argument("--num-employees", type=int, default=50)
    p.add_argument("--num-firms", type=int, default=5)
    p.add_argument("--hard-max-firms", type=int, default=7)
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Duong dan checkpoint RLlib da train hop le (khop dung obs/action space hien tai). "
                         "Neu bo trong, moi tac tu khac Government dung heuristic decide().")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_algo(env_config, checkpoint_path):
    """Nap policy da train neu co checkpoint hop le; tra None neu khong (dung heuristic)."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    from be.rllib_wrapper import build_ppo_config
    config = build_ppo_config(env_config, num_env_runners=0, seed=42)
    algo = config.build_algo() if hasattr(config, "build_algo") else config.build()
    try:
        algo.restore(checkpoint_path)
        print(f"[BENCHMARK] Da nap checkpoint: {checkpoint_path}")
        return algo
    except Exception as exc:
        print(f"[BENCHMARK] KHONG nap duoc checkpoint ({exc}) -- dung heuristic cho tat ca tac tu.")
        return None


def policy_mapping(agent_id: str) -> str:
    if agent_id.startswith("emp_"):
        return "policy_employee"
    if agent_id.startswith("firm_"):
        return "policy_firm"
    if agent_id.startswith("bank_"):
        return "policy_bank"
    if agent_id == "gov_1":
        return "policy_government"
    if agent_id == "sup_1":
        return "policy_supervisor"
    if agent_id == "eco_1":
        return "policy_economy"
    raise ValueError(f"Khong xac dinh duoc policy cho agent_id={agent_id}")


def run_episode(env, algo, gov_baseline_name, max_steps):
    obs, info = env.reset()
    gdp_hist = []
    gini_hist = []

    for t in range(max_steps):
        raw_state = env.get_raw_environment_state()
        actions = {}
        for aid, agent in env.agents.items():
            if aid == env.gov.agent_id:
                avg_wage = float(np.mean([e.wage for e in env.agents.values()
                                           if isinstance(e, Employee) and e.status == LifeCycleStatus.ACTIVE and e.employed_by is not None])) \
                    if any(isinstance(e, Employee) and e.employed_by is not None for e in env.agents.values()) else env.eco.base_living_cost
                actions[aid] = get_gov_baseline_action(gov_baseline_name, avg_wage, env.eco.base_living_cost)
                continue
            if algo is not None:
                obs_vec = agent.observe(raw_state).vector
                act = algo.compute_single_action(obs_vec, policy_id=policy_mapping(aid), explore=False)
                actions[aid] = np.asarray(act, dtype=np.float32)
            else:
                act = agent.decide(agent.observe(raw_state))
                actions[aid] = act.values

        obs, rew, term, trunc, info = env.step(actions)

        macro = env.get_raw_environment_state().get("macro_indicators", {})
        gdp_hist.append(float(env.gov.current_real_gdp))
        gini_hist.append(float(env.gov.current_gini))

        if trunc.get("__all__", False):
            break

    return np.array(gdp_hist), np.array(gini_hist)


def main():
    args = parse_args()
    cfg = ScenarioConfig(
        num_employees=args.num_employees, num_firms=args.num_firms,
        num_banks=1, max_steps=args.max_steps, hard_max_firms=args.hard_max_firms,
    )
    env_config = cfg.to_env_kwargs()
    algo = build_algo(env_config, args.checkpoint)

    print(f"\n{'Baseline':<15} {'Productivity(GDP)':>20} {'Equality(1-Gini)':>18} {'Eq x Prod':>12}")
    print("-" * 68)
    results = {}
    for name in BASELINE_NAMES:
        env = MacroEnvironment(**env_config)
        np.random.seed(args.seed)
        gdp_all, gini_all = [], []
        for ep in range(args.episodes):
            gdp_hist, gini_hist = run_episode(env, algo, name, args.max_steps)
            if len(gdp_hist) > 0:
                gdp_all.append(gdp_hist.mean())
                gini_all.append(gini_hist.mean())
        productivity = float(np.mean(gdp_all)) if gdp_all else float("nan")
        equality = float(1.0 - np.mean(gini_all)) if gini_all else float("nan")
        eq_x_prod = productivity * equality
        results[name] = (productivity, equality, eq_x_prod)
        print(f"{name:<15} {productivity:>20.2f} {equality:>18.3f} {eq_x_prod:>12.2f}")

    print("\n[BENCHMARK] Luu y: neu khong truyen --checkpoint hop le, day CHUA phai so sanh")
    print("'RL da hoc' vs 'rule-based' that su -- moi tac tu khac Government deu dung heuristic,")
    print("nen bang tren chi cho biet BAN THAN 3 bieu thue khac nhau the nao duoi CUNG mot moi")
    print("truong heuristic co dinh, chua danh gia duoc gia tri gia tang cua RL.")


if __name__ == "__main__":
    main()
