from typing import Dict, Any, Optional
import numpy as np
from be.core.enums import LifeCycleStatus, AgentType
from be.core.types import Observation, Action, ValidationResult, TransitionResult
from be.agents.base_agent import BaseAgent

class Employee(BaseAgent):
    """
    Tac tu Nguoi lao dong (Employee / Citizen).
    Tuan thu nghiem ngat contract cua BaseAgent theo SAS v1.0.
    """
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.agent_type = AgentType.EMPLOYEE
        
        # Dac tinh noi tai (Noi suy tam ly va the ly)
        self.skill_level: float = 1.0
        self.risk_aversion: float = 0.5
        self.tax_morale: float = 0.8
        self.age: int = 20
        self.max_age: int = 80
        
        # Chi so tai chinh va sinh hoc
        self.cash: float = 0.0
        self.energy: float = 1.0
        self.employed_by: Optional[str] = None
        self.wage: float = 0.0
        self.debt: float = 0.0
        
        # Luu tru hanh dong gan nhat de tinh toan Reward
        self.last_work_effort: float = 0.0
        self.last_declare_ratio: float = 1.0
        self.last_consumption: float = 0.0

    def initialize(self, 
                   skill_level: float = 1.0, 
                   risk_aversion: float = 0.5, 
                   tax_morale: float = 0.8,
                   initial_cash: float = 500.0,
                   initial_energy: float = 1.0,
                   age: int = 20) -> None:
        self.skill_level = float(skill_level)
        self.risk_aversion = float(risk_aversion) if risk_aversion != 1.0 else 0.99
        self.tax_morale = float(tax_morale)
        self.cash = float(initial_cash)
        self.energy = float(initial_energy)
        self.age = int(age)
        self.employed_by = None
        self.wage = 0.0
        self.debt = 0.0
        self.status = LifeCycleStatus.ACTIVE

    def observe(self, raw_environment_state: Dict[str, Any]) -> Observation:
        """
        Khong gian quan sat 12 chieu:
        [0-4] Vi mo: Lam phat, Chi phi sinh hoat co so, Thue TNCN, Lai suat, Ty le that nghiep
        [5-11] Vi mo noi tai: Tien mat, Nang luong, Ky nang, Luong hien tai, Trang thai viec lam, Tuoi, No
        """
        macro = raw_environment_state.get("macro_indicators", {})
        inflation = float(macro.get("inflation", 0.0))
        living_cost = float(macro.get("base_living_cost", 15.0))
        tax_rate = float(macro.get("worker_tax_rate", 0.1))
        interest_rate = float(macro.get("base_interest_rate", 0.05))
        unemployment_rate = float(macro.get("unemployment_rate", 0.0))
        
        obs_array = np.array([
            inflation,
            living_cost,
            tax_rate,
            interest_rate,
            unemployment_rate,
            self.cash,
            self.energy,
            self.skill_level,
            self.wage,
            1.0 if self.employed_by is not None else 0.0,
            float(self.age),
            self.debt
        ], dtype=np.float32)

        return Observation(
            agent_id=self.agent_id,
            timestep=raw_environment_state.get("timestep", 0),
            vector=obs_array,
            metadata={"employed_by": self.employed_by}
        )

    def decide(self, observation: Observation) -> Action:
        """
        Xuat khong gian hanh dong 3 chieu:
        [0]: Muc do no luc lam viec (Work Effort): [0.0, 1.0]
        [1]: Ty le khai bao thu nhap nop thue (Declare Ratio): [0.0, 1.0]
        [2]: Ty le tieu dung tren thu nhap kha dung (Consumption Ratio): [0.0, 1.0]
        """
        if "injected_action" in observation.metadata:
            raw_action = observation.metadata["injected_action"]
        else:
            # Heuristic mac dinh neu khong co Mang Neural truyen vao
            effort = 0.8 if self.employed_by is not None else 0.3
            raw_action = np.array([effort, 1.0, 0.5], dtype=np.float32)

        return Action(
            agent_id=self.agent_id,
            action_type="LABOR_AND_CONSUMPTION",
            values=np.asarray(raw_action, dtype=np.float32)
        )

    def validate_action(self, action: Action) -> ValidationResult:
        vals = action.values
        if len(vals) < 3:
            return ValidationResult(
                is_valid=False,
                sanitized_values=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                reason="Invalid action vector length"
            )

        # Gioi han vat ly noi tai
        work_effort = float(np.clip(vals[0], 0.0, 1.0))
        
        # Neu the luc qua thap, khong the lao dong cuong do cao
        if self.energy < 0.2:
            work_effort = min(work_effort, self.energy)

        declare_ratio = float(np.clip(vals[1], 0.0, 1.0))
        consumption_ratio = float(np.clip(vals[2], 0.0, 1.0))

        sanitized = np.array([work_effort, declare_ratio, consumption_ratio], dtype=np.float32)
        return ValidationResult(is_valid=True, sanitized_values=sanitized)

    def apply_result(self, transition_result: TransitionResult) -> None:
        delta = transition_result.state_delta
        self.cash += float(delta.get("cash_delta", 0.0))
        self.energy = float(np.clip(self.energy + delta.get("energy_delta", 0.0), 0.0, 2.0))
        self.debt = float(max(0.0, self.debt + delta.get("debt_delta", 0.0)))
        
        if "employed_by" in delta:
            self.employed_by = delta["employed_by"]
        if "wage" in delta:
            self.wage = float(delta["wage"])
        if "age_increment" in delta:
            self.age += int(delta["age_increment"])

        self.last_work_effort = float(delta.get("executed_work_effort", 0.0))
        self.last_declare_ratio = float(delta.get("executed_declare_ratio", 1.0))
        self.last_consumption = float(delta.get("executed_consumption", 0.0))

        # Kiem tra dieu kien sinh tu
        if self.cash < -300.0 or self.energy <= 0.0 or self.age >= self.max_age:
            self.terminate(reason="Depleted resources or reached maximum age")

    def calculate_reward(self, transition_result: TransitionResult) -> float:
        if self.status == LifeCycleStatus.TERMINATED:
            return -100.0

        # Ham thoa dung tieu dung CRRA
        consumption = max(0.01, self.last_consumption)
        u_consumption = (consumption ** (1.0 - self.risk_aversion) - 1.0) / (1.0 - self.risk_aversion)

        # Chi phi mat thoa dung do lao dong nang nhoc
        disutility_labor = (self.last_work_effort ** 3) / 3.0

        # Chi phi dao duc khi tron thue
        moral_cost = 2.0 * self.tax_morale * ((1.0 - self.last_declare_ratio) ** 2)

        # Tong hop Utility
        total_utility = u_consumption - disutility_labor - moral_cost
        return float(np.clip(total_utility, -50.0, 50.0))

    def export_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "status": self.status.name,
            "cash": self.cash,
            "energy": self.energy,
            "skill_level": self.skill_level,
            "employed_by": self.employed_by,
            "wage": self.wage,
            "debt": self.debt,
            "age": self.age
        }

    def reset(self) -> None:
        super().reset()
        self.skill_level = 1.0
        self.cash = 0.0
        self.energy = 1.0
        self.employed_by = None
        self.wage = 0.0
        self.debt = 0.0
        self.age = 20

    def terminate(self, reason: str = "") -> None:
        super().terminate(reason)
        self.employed_by = None
        self.wage = 0.0