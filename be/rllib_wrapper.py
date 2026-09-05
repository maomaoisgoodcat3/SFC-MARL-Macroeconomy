from typing import Dict, Any, Tuple, Optional
import numpy as np
from gymnasium.spaces import Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from be.env import MacroEnvironment
from be.core.enums import LifeCycleStatus
from be.agents.employee import Employee
from be.agents.firm import Firm

# ==============================================================================
# KHONG GIAN QUAN SAT (OBSERVATION SPACES)
# ==============================================================================
EMPLOYEE_OBS_SPACE = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
FIRM_OBS_SPACE = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
GOVERNMENT_OBS_SPACE = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
BANK_OBS_SPACE = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
SUPERVISOR_OBS_SPACE = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
ECONOMY_OBS_SPACE = Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

# ==============================================================================
# KHONG GIAN HANH DONG (ACTION SPACES)
# ==============================================================================
EMPLOYEE_ACT_SPACE = Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
FIRM_ACT_SPACE = Box(
    low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    dtype=np.float32
)
GOVERNMENT_ACT_SPACE = Box(
    low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    high=np.array([0.5, 0.5, 0.4], dtype=np.float32),
    dtype=np.float32
)
BANK_ACT_SPACE = Box(
    low=np.array([0.01, 0.005, 0.0], dtype=np.float32),
    high=np.array([0.25, 0.15, 1.0], dtype=np.float32),
    dtype=np.float32
)
SUPERVISOR_ACT_SPACE = Box(
    low=np.array([0.01, 1.0, 0.0], dtype=np.float32),
    high=np.array([0.30, 3.0, 1.0], dtype=np.float32),
    dtype=np.float32
)
ECONOMY_ACT_SPACE = Box(
    low=np.array([0.90, 0.90, 0.0], dtype=np.float32),
    high=np.array([1.10, 1.10, 10.0], dtype=np.float32),
    dtype=np.float32
)

def policy_mapping_fn(agent_id: str, episode=None, worker=None, **kwargs) -> str:
    """
    Phan luong hanh vi tac tu vao dung chinh sach mang neural (Policy Mapping).
    Ho tro chia se tham so (Parameter Sharing) giua cac tac tu cung nhom.
    """
    if agent_id.startswith("emp_"):
        return "policy_employee"
    if agent_id.startswith("firm_"):
        return "policy_firm"
    if agent_id == "gov_1":
        return "policy_government"
    if agent_id == "bank_1":
        return "policy_bank"
    if agent_id == "sup_1":
        return "policy_supervisor"
    if agent_id == "eco_1":
        return "policy_economy"
    return "policy_employee"

class RLlibMacroEnv(MultiAgentEnv):
    """
    Lop boc MultiAgentEnv cho MacroEnvironment tuong thich Ray RLlib.
    Thuc thi tuong thich khong gian ma tran va co che chong loi so hoc (NaN Firewall).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = config or {}
        self.num_employees = cfg.get("num_employees", 50)
        self.num_firms = cfg.get("num_firms", 5)
        self.max_steps = cfg.get("max_steps", 240)

        self.env = MacroEnvironment(
            num_employees=self.num_employees,
            num_firms=self.num_firms,
            max_steps=self.max_steps
        )

        all_ids = ["gov_1", "bank_1", "eco_1", "sup_1"]
        all_ids.extend([f"firm_{i}" for i in range(self.num_firms)])
        all_ids.extend([f"emp_{j}" for j in range(self.num_employees)])

        self._agent_ids = set(all_ids)
        self.possible_agents = all_ids
        self.agents = list(all_ids)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self.agents = list(self.possible_agents)
        raw_obs, infos = self.env.reset(seed=seed)

        sanitized_obs: Dict[str, np.ndarray] = {}
        for agent_id, obs_vec in raw_obs.items():
            sanitized_obs[agent_id] = np.nan_to_num(obs_vec, nan=0.0, posinf=1000.0, neginf=-1000.0).astype(np.float32)

        return sanitized_obs, infos

    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray], 
        Dict[str, float], 
        Dict[str, bool], 
        Dict[str, bool], 
        Dict[str, Any]
    ]:
        sanitized_actions: Dict[str, np.ndarray] = {}
        for agent_id, act in action_dict.items():
            sanitized_actions[agent_id] = np.nan_to_num(act, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        obs, rewards, terminateds, truncateds, infos = self.env.step(sanitized_actions)

        # Loc so hoc toan dien cho dau ra truoc khi tra ve PyTorch
        clean_obs: Dict[str, np.ndarray] = {}
        for agent_id, obs_vec in obs.items():
            clean_obs[agent_id] = np.nan_to_num(obs_vec, nan=0.0, posinf=1000.0, neginf=-1000.0).astype(np.float32)

        clean_rewards: Dict[str, float] = {}
        for agent_id, r in rewards.items():
            val = float(r)
            if np.isnan(val) or np.isinf(val):
                clean_rewards[agent_id] = 0.0
            else:
                clean_rewards[agent_id] = float(np.clip(val, -100.0, 100.0))

        # Cap nhat danh sach tac tu dang hoat dong
        self.agents = [aid for aid, term in terminateds.items() if aid != "__all__" and not term]

        return clean_obs, clean_rewards, terminateds, truncateds, infos

class InstitutionalMetricsCallback(DefaultCallbacks):
    """
    Trinh ghi nhan chi so tuy chinh (Custom Metrics Callback).
    Trich xuat du lieu kinh te vi mo truc tiep tu the gioi vat ly ra TensorBoard/RLlib
    ma khong can thiep vao logic tinh reward hay transition.
    """
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        episode.user_data["active_employees_series"] = []
        episode.user_data["active_firms_series"] = []

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        sub_env = base_env.get_sub_environments()[env_index].env

        active_emp = sum(1 for a in sub_env.agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE)
        active_frm = sum(1 for a in sub_env.agents.values() if isinstance(a, Firm) and a.status == LifeCycleStatus.ACTIVE)

        episode.user_data["active_employees_series"].append(active_emp)
        episode.user_data["active_firms_series"].append(active_frm)

        episode.custom_metrics["gdp"] = float(sub_env.gov.current_gdp)
        episode.custom_metrics["gini"] = float(sub_env.gov.current_gini)
        episode.custom_metrics["treasury"] = float(sub_env.gov.treasury)
        episode.custom_metrics["bank_reserves"] = float(sub_env.bank.reserves)
        episode.custom_metrics["bank_npl"] = float(sub_env.bank.non_performing_loans)
        episode.custom_metrics["living_cost"] = float(sub_env.eco.base_living_cost)
        episode.custom_metrics["housing_price"] = float(sub_env.eco.housing_price)

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        emp_series = episode.user_data.get("active_employees_series", [])
        frm_series = episode.user_data.get("active_firms_series", [])

        if emp_series:
            episode.custom_metrics["mean_active_employees"] = float(np.mean(emp_series))
        if frm_series:
            episode.custom_metrics["mean_active_firms"] = float(np.mean(frm_series))