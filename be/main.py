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
    parser.add_argument("--max-steps", type=int, default=240, help="Maximum timesteps (months) per episode")
    parser.add_argument("--train-iters", type=int, default=500, help="Number of training iterations")
    parser.add_argument("--train-batch-size", type=int, default=2000, help="Training batch size")
    parser.add_argument("--checkpoint-freq", type=int, default=20, help="Save frequency (iterations)")
    parser.add_argument("--checkpoint-dir", type=str, default="be/checkpoint", help="Directory for checkpoints")
    parser.add_argument("--restore-checkpoint", type=str, default=None, help="Explicit checkpoint path to restore")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def find_latest_checkpoint(checkpoint_base: str) -> str:
    training_cp_dir = os.path.join(checkpoint_base, "training")
    if not os.path.exists(training_cp_dir):
        return ""
    
    subdirs = [d for d in os.listdir(training_cp_dir) if d.startswith("iter_")]
    if not subdirs:
        return ""
    
    iter_nums = []
    for d in subdirs:
        parts = d.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            iter_nums.append(int(parts[1]))

    if not iter_nums:
        return ""

    latest_iter = max(iter_nums)
    return os.path.join(training_cp_dir, f"iter_{latest_iter}")

def run_training(args):
    logger.info("[SYSTEM] Initializing Institutional AI Economist Training Engine...")
    logger.info(f"[CONFIG] Employees: {args.num_employees} | Firms: {args.num_firms} | Max Steps: {args.max_steps}")
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
            model={"fcnet_hiddens": [64, 64]},
            vf_clip_param=500.0,
            grad_clip=0.5,
            lr=3e-4
        )
        .callbacks(InstitutionalMetricsCallback)
        .env_runners(
            num_env_runners=1,
            rollout_fragment_length="auto"
        )
        .debugging(seed=args.seed)
    )

    logger.info("[SYSTEM] Compiling PyTorch Neural Architectures...")
    algo = config.build_algo() if hasattr(config, "build_algo") else config.build()

    target_checkpoint = args.restore_checkpoint
    if not target_checkpoint:
        target_checkpoint = find_latest_checkpoint(args.checkpoint_dir)

    if target_checkpoint and os.path.exists(target_checkpoint):
        logger.info(f"[SYSTEM] Restoring policy weights from: {target_checkpoint}")
        algo.restore(target_checkpoint)
    else:
        logger.info("[SYSTEM] No existing checkpoint identified. Training starting from iteration 0.")

    training_cp_dir = os.path.join(args.checkpoint_dir, "training")
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
            bank=env.bank,
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