from typing import Dict, Any, Tuple, Optional
from dataclasses import fields
import numpy as np
from gymnasium.spaces import Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from be.env import MacroEnvironment
from be.scenario_config import ScenarioConfig
from be.core.enums import LifeCycleStatus
from be.agents.employee import Employee
from be.agents.firm import Firm
from be.agents.bank import compute_npl_ratio_pct

# ==============================================================================
# KHONG GIAN QUAN SAT (OBSERVATION SPACES)
# ==============================================================================
EMPLOYEE_OBS_SPACE = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)  # +1: bank_deposit (Section 8B)
FIRM_OBS_SPACE = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
GOVERNMENT_OBS_SPACE = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
BANK_OBS_SPACE = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
SUPERVISOR_OBS_SPACE = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
ECONOMY_OBS_SPACE = Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

# ==============================================================================
# KHONG GIAN HANH DONG (ACTION SPACES)
# ==============================================================================
# [3]: borrow_intensity [0,1] -- tin dung tieu dung khong the chap MOI (v0.23), xem
# rule_engine.py Section 3B + employee.py::decide(). Bien do [0,1] khop voi 3 chieu con lai.
EMPLOYEE_ACT_SPACE = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
FIRM_ACT_SPACE = Box(
    low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    dtype=np.float32
)
GOVERNMENT_ACT_SPACE = Box(
    # [2]: purchase_ratio rho [0, 1] -- ty le thu thue+phat ky truoc dung de CHI MUA HANG
    # (rule_engine.py Section 4C; Godley & Lavoie, 2007, mo hinh SIM). DA DOI tu [0, 0.4]
    # ("subsidy_budget_ratio" cu, hanh dong CHET). PHAI khop chinh xac bien trong
    # Government.validate_action(), neu khong policy bi RLlib gioi han sai khoang.
    # [3]: demand_injection_ratio [-0.20, 0.20] -- CHUYEN TU Economy sang day (v0.22, xem
    # KNOWN_PATHOLOGIES.md + government.py::decide()) sau audit tinh mach lac kinh te tong
    # the: 2 tac tu doc lap cung chi mot ngan sach vi pham nguyen tac hai tang (CLAUDE.md).
    low=np.array([0.0, 0.0, 0.0, -0.20], dtype=np.float32),
    high=np.array([0.5, 0.5, 1.0, 0.20], dtype=np.float32),
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
    # [0]: intervention_intensity [-1.0, 1.0] -- bình ổn thị trường bằng dự trữ đệm
    # (Buffer-Stock, Newbery & Stiglitz 1981; xem rule_engine.py Section 4D +
    # METHODOLOGY_NOTES.md mục 2). MỚI (v0.24) -- TRƯỚC là demand_injection_ratio (đã chuyển
    # sang Government action[3] ở v0.22), rồi "reserved" no-op (v0.22-v0.23, khi Economy hoàn
    # toàn mất đòn bẩy ảnh hưởng đến chính reward của nó -- xem CLAUDE.md).
    # [1]/[2]: housing price/supply factor -- vẫn placeholder no-op như từ đầu, chờ housing epic.
    low=np.array([-1.0, 0.90, 0.0], dtype=np.float32),
    high=np.array([1.0, 1.10, 10.0], dtype=np.float32),
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
    if agent_id.startswith("bank_"):
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
        # env_config den tu ScenarioConfig.to_env_kwargs() (be/main.py) va co
        # THE chua them cac key noi bo khac cua RLlib -- chi loc dung cac
        # truong ma ScenarioConfig biet, con lai dung mac dinh cua dataclass
        # (xem be/scenario_config.py) de an toan voi moi phien ban env_config cu/moi.
        scenario_field_names = {f.name for f in fields(ScenarioConfig)}
        scenario_kwargs = {k: v for k, v in cfg.items() if k in scenario_field_names}
        scenario = ScenarioConfig(**scenario_kwargs)

        self.num_employees = scenario.num_employees
        self.num_firms = scenario.num_firms
        self.num_banks = scenario.num_banks
        self.max_steps = scenario.max_steps

        self.env = MacroEnvironment(**scenario.to_env_kwargs())

        all_ids = ["gov_1", "eco_1", "sup_1"]
        all_ids.extend([f"bank_{i}" for i in range(self.num_banks)])
        all_ids.extend([f"firm_{i}" for i in range(self.num_firms)])
        all_ids.extend([f"emp_{j}" for j in range(self.num_employees)])

        self._agent_ids = set(all_ids)
        self.possible_agents = all_ids
        self.agents = list(all_ids)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        raw_obs, infos = self.env.reset(seed=seed)
        # Lay truc tiep danh sach cac tac tu dang ton tai thuc te trong env
        self.agents = list(self.env.agents.keys())

        # Quan sat DA duoc MacroEnvironment lam sach (sanitize_observation) -- wrapper
        # chi chuyen tiep, KHONG tu lam sach lai: train (RLlib), simulate va server phai
        # chia se DUNG MOT logic (be/env.py), khong nhan ban logic o nhieu noi.
        return raw_obs, infos

    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray], 
        Dict[str, float], 
        Dict[str, bool], 
        Dict[str, bool], 
        Dict[str, Any]
    ]:
        # Lam sach action/obs/reward (NaN, Inf, chuan hoa + kep reward) nam TRONG
        # MacroEnvironment.step() -- xem be/env.py. Truoc day lop nay tu kep reward
        # cung +-100 (loi that: cat mat hinh phat tu vong theo tuoi, -122,11 -> -100,00)
        # va CHI duong train di qua, con simulate/server thi khong -> hai logic khac nhau.
        obs, rewards, terminateds, truncateds, infos = self.env.step(action_dict)

        # Cap nhat self.agents chi chua cac tac tu hien con song (chua bi bao cao dead)
        self.agents = list(self.env.agents.keys())

        return obs, rewards, terminateds, truncateds, infos

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
        episode.custom_metrics["bank_reserves"] = float(sum(b.reserves for b in sub_env.banks))
        episode.custom_metrics["bank_npl"] = float(sum(b.non_performing_loans for b in sub_env.banks))
        episode.custom_metrics["living_cost"] = float(sub_env.eco.base_living_cost)
        episode.custom_metrics["housing_price"] = float(sub_env.eco.housing_price)

        # --- Thi truong lao dong: ty le that nghiep + luong trung binh ---
        # (unemployed = active_emp - employed; dung getattr cho an toan neu
        # agent chua qua initialize()).
        employed_emps = [
            a for a in sub_env.agents.values()
            if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE
            and getattr(a, "employed_by", None) is not None
        ]
        episode.custom_metrics["employed_count"] = float(len(employed_emps))
        episode.custom_metrics["unemployment_rate"] = float(
            1.0 - (len(employed_emps) / max(active_emp, 1))
        )
        episode.custom_metrics["avg_wage"] = float(
            np.mean([getattr(e, "wage", 0.0) for e in employed_emps])
        ) if employed_emps else 0.0

        # --- Ngan hang: NPL theo TY LE tren tong du no (khong phai NPL/Reserves
        # -- Reserves khong phai mau so dung, xem thao luan voi Claude Web) ---
        total_loans = sum(b.total_loans for b in sub_env.banks)
        total_npl = sum(b.non_performing_loans for b in sub_env.banks)
        episode.custom_metrics["bank_total_loans"] = float(total_loans)
        episode.custom_metrics["npl_ratio_pct"] = compute_npl_ratio_pct(total_npl, total_loans)

        # --- Suc khoe doanh nghiep ---
        active_firms_list = [
            a for a in sub_env.agents.values()
            if isinstance(a, Firm) and a.status == LifeCycleStatus.ACTIVE
        ]
        episode.custom_metrics["avg_firm_profit"] = float(
            np.mean([getattr(f, "last_profit", 0.0) for f in active_firms_list])
        ) if active_firms_list else 0.0

        # --- Lam phat + nhan khau hoc ---
        episode.custom_metrics["inflation_pct"] = float(sub_env.eco.inflation_rate * 100.0)
        episode.custom_metrics["births_this_step"] = float(getattr(sub_env, "births_this_step", 0))

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        emp_series = episode.user_data.get("active_employees_series", [])
        frm_series = episode.user_data.get("active_firms_series", [])

        if emp_series:
            episode.custom_metrics["mean_active_employees"] = float(np.mean(emp_series))
        if frm_series:
            episode.custom_metrics["mean_active_firms"] = float(np.mean(frm_series))

# ==============================================================================
# CAU HINH PPO DUNG CHUNG CHO TRAIN (be/main.py) VA SUY LUAN/SIMULATE (be/server.py)
# ==============================================================================
# Truoc day be/main.py va be/server.py MOI NOI TU DUNG chinh sach + PPOConfig rieng
# (server chi khai bao model). Nhan ban nhu vay de dan toi lech logic am tham khi mot
# ben doi (vd. kich thuoc mang) ma ben kia khong biet -- checkpoint se sai shape hoac
# tro nen vo nghia. Gom ve MOT nguon: doi o day thi ca hai cung doi.
POLICY_MODEL_CONFIG: Dict[str, Any] = {"fcnet_hiddens": [64, 64]}

# vf_clip_param: RLlib PPO cat BINH PHUONG sai so critic tai nguong nay (vuot => gradient
# = 0). 500 (gia tri cu) khien critic cua Government/Supervisor khong hoc trong ~20 lan
# train that (vf_explained_var ~ 0,0; vf_loss 450-499 sat tran). Voi reward da chuan hoa
# (MacroEnvironment.reward_scale, |return| p90 ~ 20) tran 2000 (|sai so| < ~44,7) chua
# ~2x du dia cho return tang len khi policy tot hon. HE SO HIEU CHINH, khong phai cong
# thuc. grad_clip=0.5 van gioi han do lon cap nhat.
PPO_VF_CLIP_PARAM: float = 2000.0
PPO_GRAD_CLIP: float = 0.5
PPO_LR: float = 3e-4
PPO_NUM_SGD_ITER: int = 10


def build_policy_specs() -> Dict[str, Any]:
    """Sáu policy dùng chung tham số theo nhóm (parameter sharing), cùng obs/action space."""
    return {
        "policy_employee": (None, EMPLOYEE_OBS_SPACE, EMPLOYEE_ACT_SPACE, {}),
        "policy_firm": (None, FIRM_OBS_SPACE, FIRM_ACT_SPACE, {}),
        "policy_government": (None, GOVERNMENT_OBS_SPACE, GOVERNMENT_ACT_SPACE, {}),
        "policy_bank": (None, BANK_OBS_SPACE, BANK_ACT_SPACE, {}),
        "policy_supervisor": (None, SUPERVISOR_OBS_SPACE, SUPERVISOR_ACT_SPACE, {}),
        "policy_economy": (None, ECONOMY_OBS_SPACE, ECONOMY_ACT_SPACE, {}),
    }


def build_ppo_config(env_config: Dict[str, Any], *, num_env_runners: int, seed: Optional[int] = None,
                     train_batch_size: Optional[int] = None, sgd_minibatch_size: Optional[int] = None,
                     rollout_fragment_length: Optional[int] = None, with_callbacks: bool = True):
    """PPOConfig DUY NHAT cho ca train va suy luan. Chi khac nhau o tham so van hanh
    (so worker, batch size...) -- moi thu anh huong logic hoc/quan sat/hanh dong la chung."""
    from ray.rllib.algorithms.ppo import PPOConfig  # import tre: tranh keo Ray khi chi can hang so

    policies = build_policy_specs()
    training_kwargs: Dict[str, Any] = dict(
        num_sgd_iter=PPO_NUM_SGD_ITER,
        model=POLICY_MODEL_CONFIG,
        vf_clip_param=PPO_VF_CLIP_PARAM,
        grad_clip=PPO_GRAD_CLIP,
        lr=PPO_LR,
    )
    if train_batch_size is not None:
        training_kwargs["train_batch_size"] = train_batch_size
    if sgd_minibatch_size is not None:
        training_kwargs["sgd_minibatch_size"] = sgd_minibatch_size

    runner_kwargs: Dict[str, Any] = dict(num_env_runners=num_env_runners)
    if rollout_fragment_length is not None:
        runner_kwargs["rollout_fragment_length"] = rollout_fragment_length

    config = (
        PPOConfig()
        .environment(env=RLlibMacroEnv, env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies.keys()),
        )
        .training(**training_kwargs)
        .resources(num_gpus=0)
        .env_runners(**runner_kwargs)
    )
    if with_callbacks:
        config = config.callbacks(InstitutionalMetricsCallback)
    if seed is not None:
        config = config.debugging(seed=seed)
    return config
