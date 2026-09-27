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

NHANH THU 4 "rl_learned" (v0.30, chi xuat hien khi co --checkpoint hop le): Government
CUNG dung chinh policy RL da hoc cua no (khong bi ep theo bat ky cong thuc nao) -- day
moi la phep so sanh "RL da hoc" vs "rule-based" THAT SU theo dung cau hoi trung tam cua
khoa luan, khac voi 3 nhanh baseline chi do "chinh sach thue nao hop voi the gioi da hoc".

QUAN TRONG (v0.30, phat hien qua phan bien Claude Web 2026-09-26): --initial-lending-rate/
--initial-deposit-rate PHAI khop DUNG voi gia tri da dung luc train checkpoint dang benchmark
(vd em_baseline.yaml dung 0.12/0.05, KHAC default class 0.06/0.02) -- phuong phap "empirical
best-response" (Curry et al., 2022) dua tren gia dinh cot loi la dong bang DUNG the gioi ma
agent da hoc cach thich nghi, chi doi RIENG chinh sach dang xet. Lech dieu kien khoi tao vi
pham dung gia dinh nay, khong phai nhieu nho co the bo qua.

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
    p.add_argument(
        "--config", type=str, default=None,
        help=(
            "Duong dan toi file YAML ScenarioConfig DA DUNG DE TRAIN checkpoint dang benchmark "
            "(vd. scenarios/em_baseline.yaml). LOI DA SUA (v0.35-fix3, phat hien qua phan bien "
            "Claude Web 2026-09-27 ve 'cau hinh train/benchmark phai khop'): truoc day benchmark.py "
            "tu dung mot ScenarioConfig() rieng CHI voi 5 field (--num-employees/--num-firms/ "
            "--hard-max-firms/--initial-lending-rate/--initial-deposit-rate), MOI field con lai "
            "(wage_renegotiation_prob, firm_entry_profitability_margin, shirking_monitor_prob, "
            "initial_treasury, government_reward_mode...) LUON dung default cua class ScenarioConfig "
            "chu KHONG doc tu scenario yaml da train -- vd hard_max_firms default=7 nhung "
            "em_baseline.yaml=25, firm_entry_profitability_margin KHONG THE ghi de qua CLI (luon "
            "dung 0.0 du checkpoint train voi 0.08) -- vi pham dung gia dinh 'dong bang the gioi da "
            "hoc' cua empirical best-response (Curry et al., 2022) ma docstring dau file da canh "
            "bao nhung chua sua het. Truyen --config = doc TOAN BO ScenarioConfig tu yaml (giong "
            "main.py), GHI DE moi --num-employees/--num-firms/--hard-max-firms/--initial-lending-rate/ "
            "--initial-deposit-rate ben duoi. Bo qua --config = hanh vi CU (tu chiu trach nhiem "
            "khop tay moi tham so qua CLI, de tai hien/ablation co chu dich)."
        )
    )
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=240)
    p.add_argument("--num-employees", type=int, default=50)
    p.add_argument("--num-firms", type=int, default=5)
    p.add_argument("--hard-max-firms", type=int, default=7)
    p.add_argument("--initial-lending-rate", type=float, default=0.06,
                    help="PHAI khop dung gia tri da dung luc train checkpoint dang benchmark "
                         "(vd em_baseline.yaml: 0.12) -- xem canh bao trong docstring dau file.")
    p.add_argument("--initial-deposit-rate", type=float, default=0.02,
                    help="PHAI khop dung gia tri da dung luc train checkpoint dang benchmark "
                         "(vd em_baseline.yaml: 0.05).")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Duong dan checkpoint RLlib da train hop le (khop dung obs/action space hien tai). "
                         "Neu bo trong, moi tac tu khac Government dung heuristic decide().")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--explore", action="store_true",
                    help="Lay mau hanh dong tu phan phoi policy (giong luc TRAIN) thay vi lay gia tri tat dinh. "
                         "Audit 2026-09-26: 2 che do cho ket qua khac han (170-270 vs 51-65 ca chet/episode o "
                         "iter_500) -- PHAI ghi ro che do nao dung khi bao cao.")
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


RL_LEARNED_GOV = "rl_learned"  # sentinel: Government dung CHINH policy RL da hoc, khong ep baseline nao
# NHANH "<baseline>+rl_aux" (v0.36, KNOWN_PATHOLOGIES.md #30/van de B): tach hieu ung "BIEU THUE" khoi hieu ung
# "NHIEU CONG CU HON". Government RL co 5 cong cu (thue TNCN, thue DN, ty le chi mua hang rho, bom cau, tro cap),
# con 3 baseline chi khac nhau o BIEU THUE (rho=1, bom cau=0, tro cap co dinh). Nhanh nay giu bieu thue cua
# baseline nhung dat 3 cong cu phu (rho, bom cau, tro cap) bang TRUNG BINH hanh dong ma chinh Government RL da
# chon trong nhanh rl_learned cung lan benchmark. So sanh:
#   rl_learned  vs  <baseline>+rl_aux  -> hieu ung bieu thue + tinh PHU THUOC TRANG THAI cua cong cu phu
#   <baseline>+rl_aux  vs  <baseline>  -> hieu ung muc cong cu phu (trung binh) duoi cung bieu thue
# Gioi han (ghi vao khoa luan): dat cong cu phu bang TRUNG BINH khong tai tao duoc tinh phan ung theo trang thai
# cua chinh sach RL; va cac tac tu khac van dong bang tu the gioi RL da cung hoc (loi the san nha).
RL_AUX_SUFFIX = "+rl_aux"
AUX_DIMS = slice(2, 5)  # [rho, demand_injection, relief] trong GOVERNMENT_ACT_SPACE (rllib_wrapper.py)


def run_episode(env, algo, gov_baseline_name, max_steps, explore=False, aux_override=None):
    obs, info = env.reset()
    gdp_hist = []
    gini_hist = []
    unemp_hist = []
    gov_actions = []
    base_name = gov_baseline_name[:-len(RL_AUX_SUFFIX)] if gov_baseline_name.endswith(RL_AUX_SUFFIX) else gov_baseline_name

    for t in range(max_steps):
        raw_state = env.get_raw_environment_state()
        actions = {}
        for aid, agent in env.agents.items():
            if aid == env.gov.agent_id and gov_baseline_name != RL_LEARNED_GOV:
                avg_wage = float(np.mean([e.wage for e in env.agents.values()
                                           if isinstance(e, Employee) and e.status == LifeCycleStatus.ACTIVE and e.employed_by is not None])) \
                    if any(isinstance(e, Employee) and e.employed_by is not None for e in env.agents.values()) else env.eco.base_living_cost
                gov_act = np.array(get_gov_baseline_action(base_name, avg_wage, env.eco.base_living_cost), dtype=np.float32)
                if aux_override is not None:
                    gov_act[AUX_DIMS] = aux_override
                actions[aid] = gov_act
                gov_actions.append(gov_act.copy())
                continue
            if algo is not None:
                # LOI DA SUA (v0.36): truoc day dung agent.observe(raw_state).vector -- BO QUA bo lam sach quan
                # sat (NaN/Inf) cua env.observe_agent() ma luc TRAIN dung -> vi pham quy tac "train va simulate
                # cung logic" (CLAUDE.md). Nay dung dung ham chung.
                obs_vec = env.observe_agent(aid, raw_state)
                act = algo.compute_single_action(obs_vec, policy_id=policy_mapping(aid), explore=explore)
                actions[aid] = np.asarray(act, dtype=np.float32)
                if aid == env.gov.agent_id:
                    gov_actions.append(actions[aid].copy())
            else:
                act = agent.decide(agent.observe(raw_state))
                actions[aid] = act.values

        obs, rew, term, trunc, info = env.step(actions)

        macro = env.get_raw_environment_state().get("macro_indicators", {})
        gdp_hist.append(float(env.gov.current_real_gdp))
        gini_hist.append(float(env.gov.current_gini))
        emps = [a for a in env.agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE]
        unemp_hist.append(float(np.mean([e.employed_by is None for e in emps])) if emps else 1.0)

        if trunc.get("__all__", False):
            break

    return np.array(gdp_hist), np.array(gini_hist), np.array(unemp_hist), int(env.gov.dead_citizens_count), np.array(gov_actions)


def main():
    args = parse_args()
    if args.config:
        # GHI DE TOAN BO --num-employees/--num-firms/--hard-max-firms/--initial-lending-rate/
        # --initial-deposit-rate (giong dung quy uoc cua main.py::resolve_scenario_config) --
        # KHONG merge tung phan, de tranh loai loi "tuong da khop nhung thuc ra con field cu
        # dung ngam default class" ma chinh --config nay duoc them de sua. Muon doi quy mo/lai
        # suat rieng cho mot lan benchmark -> sua truc tiep file yaml hoac bo --config.
        cfg = ScenarioConfig.from_yaml(args.config)
        cfg.max_steps = args.max_steps  # --max-steps van la tham so VAN HANH cua benchmark, khong phai calibration
        print(f"[BENCHMARK] Da doc ScenarioConfig tu: {args.config} (GHI DE toan bo --num-employees/--num-firms/"
              f"--hard-max-firms/--initial-lending-rate/--initial-deposit-rate)")
    else:
        cfg = ScenarioConfig(
            num_employees=args.num_employees, num_firms=args.num_firms,
            num_banks=1, max_steps=args.max_steps, hard_max_firms=args.hard_max_firms,
            initial_lending_rate=args.initial_lending_rate,
            initial_deposit_rate=args.initial_deposit_rate,
        )
    env_config = cfg.to_env_kwargs()
    algo = build_algo(env_config, args.checkpoint)

    # Nhanh rl_learned chay TRUOC (khi co checkpoint) de lay trung binh 3 cong cu phu cho cac nhanh "+rl_aux".
    names_to_run = ([RL_LEARNED_GOV] if algo is not None else []) + list(BASELINE_NAMES)
    aux_mean = None

    print(f"[BENCHMARK] Che do policy: {'LAY MAU (explore=True, giong luc train)' if args.explore else 'TAT DINH (explore=False)'}")
    print(f"\n{'Baseline':<18} {'Productivity(GDP)':>20} {'Equality(1-Gini)':>18} {'Eq x Prod':>12} {'Deaths/ep':>10} {'Unemp':>7}")
    print("-" * 89)
    results = {}
    idx = 0
    while idx < len(names_to_run):
        name = names_to_run[idx]
        idx += 1
        env = MacroEnvironment(**env_config)
        np.random.seed(args.seed)
        gdp_all, gini_all, unemp_all, deaths_all, gov_acts_all = [], [], [], [], []
        override = aux_mean if name.endswith(RL_AUX_SUFFIX) else None
        for ep in range(args.episodes):
            gdp_hist, gini_hist, unemp_hist, deaths, gov_acts = run_episode(env, algo, name, args.max_steps,
                                                                             explore=args.explore, aux_override=override)
            if len(gov_acts) > 0:
                gov_acts_all.append(gov_acts)
            if len(gdp_hist) > 0:
                gdp_all.append(gdp_hist.mean())
                gini_all.append(gini_hist.mean())
                unemp_all.append(unemp_hist.mean())
                deaths_all.append(deaths)
        productivity = float(np.mean(gdp_all)) if gdp_all else float("nan")
        productivity_std = float(np.std(gdp_all)) if gdp_all else float("nan")
        equality = float(1.0 - np.mean(gini_all)) if gini_all else float("nan")
        equality_std = float(np.std(gini_all)) if gini_all else float("nan")
        eq_x_prod = productivity * equality
        deaths_mean = float(np.mean(deaths_all)) if deaths_all else float("nan")
        unemp_mean = float(np.mean(unemp_all)) if unemp_all else float("nan")
        results[name] = (productivity, productivity_std, equality, equality_std, eq_x_prod, deaths_mean, unemp_mean)
        print(f"{name:<18} {productivity:>13.2f}+/-{productivity_std:<5.2f} {equality:>11.3f}+/-{equality_std:<5.3f} {eq_x_prod:>12.2f} {deaths_mean:>10.1f} {unemp_mean:>7.3f}")
        if name == RL_LEARNED_GOV and gov_acts_all:
            aux_mean = np.concatenate(gov_acts_all, axis=0)[:, AUX_DIMS].mean(axis=0).astype(np.float32)
            print(f"{'':<18} (trung binh cong cu phu RL [rho, bom cau, tro cap] = {np.round(aux_mean, 3)} -> dung cho cac nhanh '{RL_AUX_SUFFIX}')")
            names_to_run += [b + RL_AUX_SUFFIX for b in BASELINE_NAMES]

    print("\n[BENCHMARK] Luu y: neu khong truyen --checkpoint hop le, day CHUA phai so sanh")
    print("'RL da hoc' vs 'rule-based' that su -- moi tac tu khac Government deu dung heuristic,")
    print("nen bang tren chi cho biet BAN THAN 3 bieu thue khac nhau the nao duoi CUNG mot moi")
    print("truong heuristic co dinh, chua danh gia duoc gia tri gia tang cua RL.")
    if algo is not None:
        print(f"[BENCHMARK] Nhanh '{RL_LEARNED_GOV}': Government dung CHINH policy RL da hoc (khong ep baseline nao)")
        print("-- day moi la phep so sanh RL-vs-rule-based that su theo dung cau hoi trung tam cua khoa luan.")


if __name__ == "__main__":
    main()
