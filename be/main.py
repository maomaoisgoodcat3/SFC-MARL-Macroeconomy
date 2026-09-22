import os
import sys
import time
import argparse
import logging
from datetime import datetime
import ray

from be.rllib_wrapper import build_ppo_config
from be.env import MacroEnvironment
from be.logger import InstitutionalLogger
from be.training_monitor import TrainingMonitor
from be.scenario_config import ScenarioConfig

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")
logger = logging.getLogger("InstitutionalEconomist.Main")

def parse_args():
    parser = argparse.ArgumentParser(description="Institutional AI Economist System Controller")
    parser.add_argument("--mode", type=str, choices=["train", "simulate"], default="train", help="Run mode")
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Duong dan toi 1 file YAML ScenarioConfig (xem be/scenario_config.py, "
            "vd. scenarios/default.yaml, scenarios/em_baseline.yaml). Khi duoc chi "
            "dinh, GHI DE toan bo cac --flag calibration/quy mo dan so ben duoi "
            "(--num-employees, --inheritance-fraction, --gini-penalty-coef, v.v.) "
            "-- cac flag DIEU HANH (--train-iters, --num-workers, --checkpoint-*, "
            "--seed, --scenario-name) van hoat dong binh thuong. Bo qua --config "
            "= hanh vi giong het truoc day (dung gia tri --flag rieng le / mac dinh)."
        )
    )
    parser.add_argument("--num-employees", type=int, default=50, help="Total employee population")
    parser.add_argument("--num-firms", type=int, default=5, help="Total firm population")
    parser.add_argument("--num-banks", type=int, default=1, help="Total bank population (relationship-banking credit market)")
    parser.add_argument("--max-steps", type=int, default=240, help="Maximum timesteps (months) per episode")
    parser.add_argument(
        "--inheritance-fraction", type=float, default=0.15,
        help=(
            "Ty le tai san cha/me chuyen cho con khi sinh san noi sinh "
            "(Epstein & Axtell, 1996, Sugarscape; xem be/env.py Section A). "
            "0.0 = tat thua ke (newborn khong nhan gi tu parent, chi con lai "
            "phan an sinh toi thieu tu Kho bac)."
        )
    )
    parser.add_argument(
        "--min-newborn-cash", type=float, default=500.0,
        help="San an sinh toi thieu cho newborn (Kho bac bu them neu thua ke chua du)."
    )
    parser.add_argument("--min-reproduction-age", type=int, default=22, help="Tuoi toi thieu de sinh san (Sugarscape)")
    parser.add_argument("--max-reproduction-age", type=int, default=45, help="Tuoi toi da de sinh san (Sugarscape)")
    parser.add_argument("--min-reproduction-wealth-mult", type=float, default=3.0, help="So thang chi phi song can du tich luy de sinh san")
    parser.add_argument("--trait-mutation-sigma", type=float, default=0.05, help="Do lech chuan dot bien gen khi sinh san")
    parser.add_argument("--hard-min-emp", type=int, default=30, help="San dan so kich hoat luoi an sinh khan cap")
    parser.add_argument("--hard-max-emp", type=int, default=200, help="Tran dan so cho phep")
    parser.add_argument("--initial-lending-rate", type=float, default=0.06, help="Lai suat cho vay khoi tao (annual)")
    parser.add_argument("--initial-deposit-rate", type=float, default=0.02, help="Lai suat tien gui khoi tao (annual)")
    parser.add_argument("--gini-penalty-coef", type=float, default=25.0, help="He so phat Gini^2 trong reward Government")
    parser.add_argument("--death-penalty-coef", type=float, default=20.0, help="He so phat moi ca tu vong trong reward Government")
    parser.add_argument("--npl-flow-penalty-coef", type=float, default=0.06, help="He so phat no xau MOI phat sinh trong reward Bank")
    parser.add_argument("--npl-stock-penalty-coef", type=float, default=50.0, help="He so phat ty le ton kho NPL/tong du no trong reward Bank")
    parser.add_argument("--npl-writeoff-months", type=int, default=6, help="So thang no xau duoc mo truoc khi write-off (IFRS 9 / Basel NPL staging)")
    parser.add_argument("--emp-death-penalty-base", type=float, default=100.0, help="Muc phat tu vong goc cua Employee (ratio=0, tuc chet dung luc max_age)")
    parser.add_argument("--emp-death-penalty-horizon-multiplier", type=float, default=1.0, help="He so nhan them theo ty le quang doi con lai khi chet (Viscusi & Aldy VSL); can hieu chinh lai bang du lieu training thuc")
    parser.add_argument("--train-iters", type=int, default=500, help="Number of training iterations")
    parser.add_argument("--train-batch-size", type=int, default=4000, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel rollout workers")
    parser.add_argument("--minibatch-size", type=int, default=256, help="PPO SGD minibatch size")
    parser.add_argument("--checkpoint-freq", type=int, default=20, help="Save frequency (iterations)")
    parser.add_argument("--checkpoint-dir", type=str, default="be/checkpoint", help="Directory for checkpoints")
    parser.add_argument("--restore-checkpoint", type=str, default=None, help="Explicit checkpoint path to restore")
    parser.add_argument(
        "--scenario-name", type=str, default="training",
        help=(
            "Ten kich ban calibration/thi nghiem (vd. em_baseline, high_tax, "
            "low_reg). Checkpoint duoc luu/doc tai <checkpoint-dir>/<scenario-name>/. "
            "Mac dinh 'training' de tuong thich nguoc hoan toan voi cau truc thu "
            "muc be/checkpoint/training/ da co tu truoc (khong doi ten scenario "
            "mac dinh thi khong can di chuyen checkpoint cu). Dung ten khac de "
            "chay song song nhieu kich ban calibration ma khong ghi de len nhau."
        )
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def resolve_scenario_config(args) -> ScenarioConfig:
    """
    Xay dung ScenarioConfig tu 1 trong 2 nguon:
      - Neu --config duoc chi dinh: doc tu file YAML, GHI DE toan bo flag
        calibration/quy mo rieng le (xem help text cua --config).
      - Nguoc lai: xay dung truc tiep tu cac --flag rieng le (hanh vi giong
        het truoc khi co ScenarioConfig -- khong co thay doi ngam an).
    """
    if args.config:
        cfg = ScenarioConfig.from_yaml(args.config)
        logger.info(f"[SYSTEM] Loaded ScenarioConfig from: {args.config} (ghi de moi flag calibration/quy mo rieng le)")
        return cfg

    return ScenarioConfig(
        num_employees=args.num_employees,
        num_firms=args.num_firms,
        num_banks=args.num_banks,
        max_steps=args.max_steps,
        inheritance_fraction=args.inheritance_fraction,
        min_newborn_cash=args.min_newborn_cash,
        min_reproduction_age=args.min_reproduction_age,
        max_reproduction_age=args.max_reproduction_age,
        min_reproduction_wealth_mult=args.min_reproduction_wealth_mult,
        trait_mutation_sigma=args.trait_mutation_sigma,
        hard_min_emp=args.hard_min_emp,
        hard_max_emp=args.hard_max_emp,
        initial_lending_rate=args.initial_lending_rate,
        initial_deposit_rate=args.initial_deposit_rate,
        gini_penalty_coef=args.gini_penalty_coef,
        death_penalty_coef=args.death_penalty_coef,
        npl_flow_penalty_coef=args.npl_flow_penalty_coef,
        npl_stock_penalty_coef=args.npl_stock_penalty_coef,
        npl_writeoff_months=args.npl_writeoff_months,
        emp_death_penalty_base=args.emp_death_penalty_base,
        emp_death_penalty_horizon_multiplier=args.emp_death_penalty_horizon_multiplier,
    )

def find_latest_checkpoint(checkpoint_base: str, scenario_name: str = "training") -> str:
    """
    Quet chinh xac thu muc checkpoint co chi so iteration cao nhat trong
    KICH BAN (scenario) duoc chi dinh, tuong thich voi ca checkpoint dinh ky
    va checkpoint khan cap (_interrupt).
    """
    training_cp_dir = os.path.join(checkpoint_base, scenario_name)
    if not os.path.exists(training_cp_dir):
        return ""

    subdirs = [
        d for d in os.listdir(training_cp_dir) 
        if os.path.isdir(os.path.join(training_cp_dir, d)) and d.startswith("iter_")
    ]
    if not subdirs:
        return ""

    def parse_checkpoint_entry(dir_name: str):
        parts = dir_name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            iter_num = int(parts[1])
            mtime = os.path.getmtime(os.path.join(training_cp_dir, dir_name))
            return (iter_num, mtime, dir_name)
        return (-1, 0, dir_name)

    valid_entries = [parse_checkpoint_entry(d) for d in subdirs]
    valid_entries = [entry for entry in valid_entries if entry[0] >= 0]

    if not valid_entries:
        return ""

    # Sap xep uu tien theo: 1. So iteration cao nhat -> 2. Thoi gian ghi moi nhat
    valid_entries.sort(key=lambda x: (x[0], x[1]))
    target_dir_name = valid_entries[-1][2]
    
    return os.path.join(training_cp_dir, target_dir_name)

def run_training(args):
    logger.info("[SYSTEM] Initializing Institutional AI Economist Training Engine...")
    cfg = resolve_scenario_config(args)
    logger.info(f"[CONFIG] Scenario: {args.scenario_name} | Employees: {cfg.num_employees} | Firms: {cfg.num_firms} | Banks: {cfg.num_banks} | Max Steps: {cfg.max_steps}")
    logger.info(f"[CONFIG] Batch Size: {args.train_batch_size} | Total Iterations: {args.train_iters} | Checkpoint Freq: {args.checkpoint_freq}")

    ray.init(ignore_reinit_error=True)

    env_config = cfg.to_env_kwargs()

    # Cau hinh PPO/chinh sach DUNG CHUNG voi be/server.py (suy luan/simulate) qua
    # build_ppo_config -- chi khac nhau o tham so van hanh (so worker, batch size).
    config = build_ppo_config(
        env_config,
        num_env_runners=args.num_workers,   # truoc day bi hardcode = 8, bo qua --num-workers
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        sgd_minibatch_size=args.minibatch_size,   # Chia nho de tinh gradient nhanh tren CPU
        rollout_fragment_length=100,
        with_callbacks=True,
    )

    logger.info("[SYSTEM] Compiling PyTorch Neural Architectures...")
    algo = config.build_algo() if hasattr(config, "build_algo") else config.build()

    # Phuc hoi Checkpoint (Restore)
    target_checkpoint = args.restore_checkpoint
    if not target_checkpoint:
        target_checkpoint = find_latest_checkpoint(args.checkpoint_dir, args.scenario_name)

    current_iter = 0
    if target_checkpoint and os.path.exists(target_checkpoint):
        try:
            logger.info(f"[SYSTEM] Restoring policy weights from: {target_checkpoint}")
            algo.restore(target_checkpoint)
            current_iter = algo.iteration
            logger.info(f"[SYSTEM] Checkpoint successfully loaded. Resuming training from iteration {current_iter}.")
        except Exception as exc:
            # Bat buoc bat loi o day: checkpoint cu duoc train truoc khi khong
            # gian quan sat cua Employee tang tu 12 len 13 chieu (them
            # bank_deposit, xem be/rllib_wrapper.py) se luon lech shape va
            # khong the restore duoc nua. Khong bat exception se lam toan bo
            # tien trinh train sup do ngay khi khoi dong neu thu muc checkpoint
            # cu (be/checkpoint/training/) van con ton tai.
            #
            # QUAN TRONG: algo.restore() KHONG atomic -- khi no fail giua
            # chung (vd. o buoc nap trong so mo hinh), no co the da kip nap
            # MOT PHAN trang thai khac (vd. optimizer state/Adam exp_avg cua
            # checkpoint cu, kich thuoc 12) vao dung object `algo` hien tai
            # (da co tham so model moi, kich thuoc 13) truoc khi rai exception.
            # Neu chi bat loi roi tiep tuc dung LAI object `algo` do, buoc
            # optimizer.step() dau tien se crash vi exp_avg (12) khong khop
            # voi gradient/tham so (13) -- da xay ra thuc te. Cach an toan duy
            # nhat la HUY object algo cu (giai phong Ray actor) va BUILD LAI
            # hoan toan moi tu dau, dam bao khong con trang thai nhiem doc.
            logger.warning(
                f"[SYSTEM] Failed to restore checkpoint {target_checkpoint} (likely an "
                f"incompatible observation/action space from a previous model version): "
                f"{str(exc)}. Discarding this algorithm instance and building a fresh one "
                f"(a failed restore can leave partial/inconsistent internal state)."
            )
            try:
                algo.stop()
            except Exception:
                pass
            algo = config.build_algo() if hasattr(config, "build_algo") else config.build()
            current_iter = 0
    else:
        logger.info("[SYSTEM] No existing checkpoint identified. Training starting from iteration 0.")

    training_cp_dir = os.path.join(args.checkpoint_dir, args.scenario_name)
    os.makedirs(training_cp_dir, exist_ok=True)

    current_iter = algo.iteration
    monitor = TrainingMonitor(total_target_iters=args.train_iters)

    try:
        while current_iter < args.train_iters:
            train_results = algo.train()
            current_iter = train_results.get("training_iteration", current_iter + 1)

            env_stats = train_results.get("env_runners", {})
            mean_return = env_stats.get("episode_return_mean", float("nan"))

            custom_metrics = env_stats.get("custom_metrics", {})
            active_emp = custom_metrics.get("mean_active_employees_mean", 0.0)
            active_frm = custom_metrics.get("mean_active_firms_mean", 0.0)
            gdp = custom_metrics.get("gdp_mean", 0.0)
            gini = custom_metrics.get("gini_mean", 0.0)
            bank_res = custom_metrics.get("bank_reserves_mean", 0.0)
            npl = custom_metrics.get("bank_npl_mean", 0.0)
            unemp = custom_metrics.get("unemployment_rate_mean", float("nan"))
            npl_ratio = custom_metrics.get("npl_ratio_pct_mean", float("nan"))
            inflation = custom_metrics.get("inflation_pct_mean", float("nan"))
            treasury = custom_metrics.get("treasury_mean", float("nan"))
            avg_wage = custom_metrics.get("avg_wage_mean", float("nan"))
            births = custom_metrics.get("births_this_step_mean", float("nan"))

            print(
                f"[TRAIN] Iter: {current_iter:4d} | "
                f"Reward: {mean_return:9.2f} | "
                f"Emp: {active_emp:4.1f} | "
                f"Unemp: {unemp * 100:4.1f}% | "
                f"Firm: {active_frm:3.1f} | "
                f"GDP: {gdp:10.2f} | "
                f"Gini: {gini:4.2f} | "
                f"Infl: {inflation:5.2f}% | "
                f"Wage: {avg_wage:5.1f} | "
                f"Treasury: {treasury:9.0f} | "
                f"Reserves: {bank_res:9.1f} | "
                f"NPL: {npl:7.1f} ({npl_ratio:4.1f}%) | "
                f"Births: {births:3.2f}"
            )

            monitor.tick(mean_return)
            if current_iter % 5 == 0:
                monitor.render_summary(current_iter)

            if current_iter % args.checkpoint_freq == 0:
                save_path = os.path.join(training_cp_dir, f"iter_{current_iter}")
                os.makedirs(save_path, exist_ok=True)
                algo.save(checkpoint_dir=save_path)
                logger.info(f"[SYSTEM] Checkpoint persisted at: {save_path}")

    except KeyboardInterrupt:
        print("\n")
        logger.warning("[SYSTEM] Interrupt signal (Ctrl+C) caught. Preserving current model weights...")
        emergency_save_path = os.path.join(training_cp_dir, f"iter_{current_iter}_interrupt")
        os.makedirs(emergency_save_path, exist_ok=True)
        algo.save(checkpoint_dir=emergency_save_path)
        logger.info(f"[SYSTEM] Emergency checkpoint successfully preserved at: {emergency_save_path}")
        ray.shutdown()
        sys.exit(0)

    logger.info("[SYSTEM] Training process completed.")
    ray.shutdown()

def run_simulation(args):
    """Che do thuc thi mo phong thuan tuy (Inference / Headless Sim)."""
    logger.info("[SIMULATION] Initializing deterministic simulation run...")
    cfg = resolve_scenario_config(args)
    env = MacroEnvironment(**cfg.to_env_kwargs())
    # run_id gop ten scenario + timestamp de moi lan simulate ra dung 1 bo 5
    # file Parquet rieng biet, khong ghi de len lan chay truoc (xem parquet_io.py)
    run_id = f"{args.scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    inst_logger = InstitutionalLogger(flush_interval=50, run_id=run_id)
    logger.info(f"[SIMULATION] Parquet export run_id: {run_id} (be/exports/<category>/{run_id}.parquet)")

    for event_type in env.event_bus._subscribers.keys():
        env.event_bus.subscribe(event_type, inst_logger.log_event_for_ui)

    obs, _ = env.reset(seed=args.seed)

    for step in range(1, cfg.max_steps + 1):
        actions = {}
        for agent_id, agent in env.agents.items():
            agent_obs = agent.observe(env.get_raw_environment_state())
            agent_action = agent.decide(agent_obs)
            actions[agent_id] = agent_action.values

        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        active_w = sum(1 for a in env.agents.values() if a.agent_type.value == "employee" and a.status.name == "ACTIVE")
        active_f = sum(1 for a in env.agents.values() if a.agent_type.value == "firm" and a.status.name == "ACTIVE")
        employed_w = sum(1 for a in env.agents.values() if a.agent_type.value == "employee" and a.status.name == "ACTIVE" and getattr(a, "employed_by", None) is not None)
        unemployment_pct = (1.0 - employed_w / max(1, active_w)) * 100.0

        inst_logger.log_step(
            timestep=step,
            gov=env.gov,
            banks=env.banks,
            eco=env.eco,
            sup=env.sup,
            agents=env.agents,
            births_this_step=env.births_this_step
        )
        inst_logger.flush_ui_events()
        inst_logger.step_end(step)

        if step % 12 == 0:
            print(
                f"[SIM] Month: {step:3d} | "
                f"Active Emp: {active_w:2d} | "
                f"Unemp: {unemployment_pct:5.1f}% | "
                f"Active Firm: {active_f:2d} | "
                f"GDP: {env.gov.current_gdp:10.2f} | "
                f"Gini: {env.gov.current_gini:4.2f} | "
                f"Inflation: {env.eco.inflation_rate * 100:4.2f}%"
            )

        if terminateds.get("__all__", False):
            break

    inst_logger.close()
    logger.info(f"[SIMULATION] Simulation run finished. Parquet exported under be/exports/<category>/{run_id}.parquet")

def main():
    args = parse_args()
    if args.mode == "train":
        run_training(args)
    elif args.mode == "simulate":
        run_simulation(args)

if __name__ == "__main__":
    main()