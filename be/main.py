import os
import sys
import argparse
import logging
import ray
from ray.rllib.algorithms.ppo import PPOConfig

from be.rllib_wrapper import (
    RLlibMacroEnv,
    policy_mapping_fn,
    InstitutionalMetricsCallback,
    EMPLOYEE_OBS_SPACE,
    EMPLOYEE_ACT_SPACE,
    FIRM_OBS_SPACE,
    FIRM_ACT_SPACE,
    GOVERNMENT_OBS_SPACE,
    GOVERNMENT_ACT_SPACE,
    BANK_OBS_SPACE,
    BANK_ACT_SPACE,
    SUPERVISOR_OBS_SPACE,
    SUPERVISOR_ACT_SPACE,
    ECONOMY_OBS_SPACE,
    ECONOMY_ACT_SPACE
)
from be.env import MacroEnvironment
from be.logger import InstitutionalLogger

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")
logger = logging.getLogger("InstitutionalEconomist.Main")

def parse_args():
    parser = argparse.ArgumentParser(description="Institutional AI Economist System Controller")
    parser.add_argument("--mode", type=str, choices=["train", "simulate"], default="train", help="Run mode")
    parser.add_argument("--num-employees", type=int, default=50, help="Total employee population")
    parser.add_argument("--num-firms", type=int, default=5, help="Total firm population")
    parser.add_argument("--num-banks", type=int, default=1, help="Total bank population (relationship-banking credit market)")
    parser.add_argument("--max-steps", type=int, default=240, help="Maximum timesteps (months) per episode")
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
    logger.info(f"[CONFIG] Scenario: {args.scenario_name} | Employees: {args.num_employees} | Firms: {args.num_firms} | Banks: {args.num_banks} | Max Steps: {args.max_steps}")
    logger.info(f"[CONFIG] Batch Size: {args.train_batch_size} | Total Iterations: {args.train_iters} | Checkpoint Freq: {args.checkpoint_freq}")

    ray.init(ignore_reinit_error=True)

    policies = {
        "policy_employee": (None, EMPLOYEE_OBS_SPACE, EMPLOYEE_ACT_SPACE, {}),
        "policy_firm": (None, FIRM_OBS_SPACE, FIRM_ACT_SPACE, {}),
        "policy_government": (None, GOVERNMENT_OBS_SPACE, GOVERNMENT_ACT_SPACE, {}),
        "policy_bank": (None, BANK_OBS_SPACE, BANK_ACT_SPACE, {}),
        "policy_supervisor": (None, SUPERVISOR_OBS_SPACE, SUPERVISOR_ACT_SPACE, {}),
        "policy_economy": (None, ECONOMY_OBS_SPACE, ECONOMY_ACT_SPACE, {})
    }

    env_config = {
        "num_employees": args.num_employees,
        "num_firms": args.num_firms,
        "num_banks": args.num_banks,
        "max_steps": args.max_steps
    }

    config = (
        PPOConfig()
        .environment(env=RLlibMacroEnv, env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies.keys())
        )
        .training(
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.minibatch_size,      # Chia nhỏ để tính gradient nhanh trên CPU
            num_sgd_iter=10,
            model={"fcnet_hiddens": [64, 64]},
            vf_clip_param=500.0,
            grad_clip=0.5,
            lr=3e-4
        )
        .callbacks(InstitutionalMetricsCallback)
        .resources(
            num_gpus=0                   # Chạy thuần CPU
        )
        .env_runners(
            num_env_runners=args.num_workers,  # truoc day bi hardcode = 8, bo qua --num-workers
            rollout_fragment_length=100
        )
        .debugging(seed=args.seed)
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

            print(
                f"[TRAIN] Iter: {current_iter:4d} | "
                f"Reward: {mean_return:9.2f} | "
                f"Emp: {active_emp:4.1f} | "
                f"Firm: {active_frm:3.1f} | "
                f"GDP: {gdp:10.2f} | "
                f"Gini: {gini:4.2f} | "
                f"Reserves: {bank_res:9.1f} | "
                f"NPL: {npl:7.1f}"
            )

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
    env = MacroEnvironment(
        num_employees=args.num_employees,
        num_firms=args.num_firms,
        num_banks=args.num_banks,
        max_steps=args.max_steps
    )
    inst_logger = InstitutionalLogger(flush_interval=50)
    
    for event_type in env.event_bus._subscribers.keys():
        env.event_bus.subscribe(event_type, inst_logger.log_event_for_ui)

    obs, _ = env.reset(seed=args.seed)
    
    for step in range(1, args.max_steps + 1):
        actions = {}
        for agent_id, agent in env.agents.items():
            agent_obs = agent.observe(env.get_raw_environment_state())
            agent_action = agent.decide(agent_obs)
            actions[agent_id] = agent_action.values

        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        active_w = sum(1 for a in env.agents.values() if a.agent_type.value == "employee" and a.status.name == "ACTIVE")
        active_f = sum(1 for a in env.agents.values() if a.agent_type.value == "firm" and a.status.name == "ACTIVE")

        inst_logger.log_macro_step(
            timestep=step,
            gov=env.gov,
            banks=env.banks,
            eco=env.eco,
            sup=env.sup,
            active_workers=active_w,
            active_firms=active_f
        )
        inst_logger.log_micro_step(timestep=step, agents=env.agents)
        inst_logger.flush_ui_events()
        inst_logger.step_end(step)

        if step % 12 == 0:
            print(
                f"[SIM] Month: {step:3d} | "
                f"Active Emp: {active_w:2d} | "
                f"Active Firm: {active_f:2d} | "
                f"GDP: {env.gov.current_gdp:10.2f} | "
                f"Gini: {env.gov.current_gini:4.2f} | "
                f"Inflation: {env.eco.inflation_rate * 100:4.2f}%"
            )

        if terminateds.get("__all__", False):
            break

    inst_logger.flush_to_disk()
    logger.info("[SIMULATION] Simulation run finished and logs exported.")

def main():
    args = parse_args()
    if args.mode == "train":
        run_training(args)
    elif args.mode == "simulate":
        run_simulation(args)

if __name__ == "__main__":
    main()