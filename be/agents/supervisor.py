from typing import Dict, Any
import numpy as np
from be.core.enums import LifeCycleStatus, AgentType
from be.core.types import Observation, Action, ValidationResult, TransitionResult
from be.agents.base_agent import BaseAgent

class Supervisor(BaseAgent):
    """
    Tac tu Thanh tra Giam sat (Supervisor).
    Kiem toan thue, phat hien gian lan va cuong che thuc thi phap luat.
    Tuan thu nghiem ngat contract BaseAgent theo SAS v1.0.
    """
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.agent_type = AgentType.SUPERVISOR
        
        # Thong so quan ly va ngan sach dieu tra
        self.operational_budget: float = 0.0
        self.audit_cost_per_case: float = 20.0
        
        # Chinh sach thanh tra hien hanh
        self.audit_rate: float = 0.05
        self.fine_multiplier: float = 1.5
        self.target_firm_ratio: float = 0.5
        
        # Ket qua giam sat ky truoc
        self.audits_conducted: int = 0
        self.violations_detected: int = 0
        self.last_fines_collected: float = 0.0
        self.last_operational_expense: float = 0.0

    def initialize(self, 
                   initial_budget: float = 50000.0,
                   initial_audit_rate: float = 0.05,
                   initial_fine_multiplier: float = 1.5,
                   audit_cost_per_case: float = 20.0) -> None:
        self.operational_budget = float(initial_budget)
        self.audit_rate = float(initial_audit_rate)
        self.fine_multiplier = float(initial_fine_multiplier)
        self.audit_cost_per_case = float(audit_cost_per_case)
        self.target_firm_ratio = 0.5
        self.audits_conducted = 0
        self.violations_detected = 0
        self.last_fines_collected = 0.0
        self.last_operational_expense = 0.0
        self.status = LifeCycleStatus.ACTIVE

    def observe(self, raw_environment_state: Dict[str, Any]) -> Observation:
        """
        Khong gian quan sat 8 chieu:
        [0]: Ngan sach hoat dong con lai (scaled)
        [1]: Ty le kiem toan ky truoc
        [2]: He so phat hien hanh vi vi pham (Detection Rate)
        [3]: Muc do gian lan uoc tinh toan xa hoi
        [4]: He so phat ap dung hien tai
        [5]: Tong tien phat thu ve ky truoc (scaled)
        [6]: So vu vi pham da phat hien ky truoc
        [7]: Ty le phan bo kiem toan vao Doanh nghiep vs Ca nhan
        """
        macro = raw_environment_state.get("macro_indicators", {})
        estimated_evasion = float(macro.get("estimated_tax_evasion", 0.0)) * 0.001

        detection_rate = (float(self.violations_detected) / float(self.audits_conducted)) if self.audits_conducted > 0 else 0.0

        obs_array = np.array([
            self.operational_budget * 0.0001,
            self.audit_rate,
            float(np.clip(detection_rate, 0.0, 1.0)),
            estimated_evasion,
            self.fine_multiplier * 0.2,
            self.last_fines_collected * 0.001,
            float(self.violations_detected) * 0.1,
            self.target_firm_ratio
        ], dtype=np.float32)

        return Observation(
            agent_id=self.agent_id,
            timestep=raw_environment_state.get("timestep", 0),
            vector=obs_array,
            metadata={"budget": self.operational_budget, "violations": self.violations_detected}
        )

    def decide(self, observation: Observation) -> Action:
        """
        Khong gian hanh dong 3 chieu:
        [0]: Ty le lay mau kiem toan toan xa hoi (Audit Rate): [0.01, 0.30]
        [1]: He so phat tren so tien tron thue (Fine Multiplier): [1.0, 3.0]
        [2]: Ty le uu tien nguon luc cho Doanh nghiep vs Ca nhan: [0.0, 1.0] (1.0 = 100% Firm)
        """
        if "injected_action" in observation.metadata:
            raw_action = observation.metadata["injected_action"]
        else:
            raw_action = np.array([0.05, 1.5, 0.5], dtype=np.float32)

        return Action(
            agent_id=self.agent_id,
            action_type="AUDIT_POLICY",
            values=np.asarray(raw_action, dtype=np.float32)
        )

    def validate_action(self, action: Action) -> ValidationResult:
        vals = action.values
        if len(vals) < 3:
            return ValidationResult(
                is_valid=False,
                sanitized_values=np.array([0.05, 1.5, 0.5], dtype=np.float32),
                reason="Supervisor action vector must have 3 dimensions"
            )

        audit_rate = float(np.clip(vals[0], 0.01, 0.30))
        fine_multiplier = float(np.clip(vals[1], 1.0, 3.0))
        target_firm_ratio = float(np.clip(vals[2], 0.0, 1.0))

        # Kiem tra ngan sach hoat dong: neu thieu kinh phi, buoc phai ha ty le thanh tra
        if self.operational_budget < 500.0:
            audit_rate = min(audit_rate, 0.02)

        sanitized = np.array([audit_rate, fine_multiplier, target_firm_ratio], dtype=np.float32)
        return ValidationResult(is_valid=True, sanitized_values=sanitized)

    def apply_result(self, transition_result: TransitionResult) -> None:
        delta = transition_result.state_delta

        self.audits_conducted = int(delta.get("audits_conducted", 0))
        self.violations_detected = int(delta.get("violations_detected", 0))
        
        fines = float(delta.get("fines_collected", 0.0))
        expenses = float(delta.get("operational_cost", self.audits_conducted * self.audit_cost_per_case))
        
        self.last_fines_collected = fines
        self.last_operational_expense = expenses

        # Cap nhat ngan sach hoat dong
        self.operational_budget += float(delta.get("budget_allocation", 0.0)) - expenses

        # Cap nhat tham so thuc thi
        self.audit_rate = float(delta.get("executed_audit_rate", self.audit_rate))
        self.fine_multiplier = float(delta.get("executed_fine_multiplier", self.fine_multiplier))
        self.target_firm_ratio = float(delta.get("executed_target_firm_ratio", self.target_firm_ratio))

        if self.operational_budget < -5000.0:
            self.terminate(reason="Audit authority insolvency / operational collapse")

    def calculate_reward(self, transition_result: TransitionResult) -> float:
        """
        Mục tiêu kiểm toán (Allingham & Sandmo, 1972):
        Reward = Tiền phạt thu hồi + Thưởng bắt đúng - Phạt kiểm tra sai người - Chi phí vận hành

        false_positive = audits_conducted - violations_detected (audit người vô tội)
        Agent học: chọn đối tượng kiểm tra thông minh thay vì random bừa bãi
        """
        collection_reward = self.last_fines_collected * 0.01

        # Thưởng từng vụ phát hiện đúng
        true_positive_reward = float(self.violations_detected) * 2.0

        # Phạt kiểm tra người vô tội (tốn chi phí, không thu được gì)
        false_positives = max(0, self.audits_conducted - self.violations_detected)
        false_positive_penalty = float(false_positives) * 0.5

        expense_penalty = self.last_operational_expense * 0.005

        reward = collection_reward + true_positive_reward - false_positive_penalty - expense_penalty
        return float(np.clip(reward, -50.0, 50.0))

    def export_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "status": self.status.name,
            "operational_budget": self.operational_budget,
            "audit_rate": self.audit_rate,
            "fine_multiplier": self.fine_multiplier,
            "audits_conducted": self.audits_conducted,
            "violations_detected": self.violations_detected,
            "last_fines_collected": self.last_fines_collected
        }

    def reset(self) -> None:
        super().reset()
        self.operational_budget = 0.0
        self.audit_rate = 0.05
        self.fine_multiplier = 1.5
        self.target_firm_ratio = 0.5
        self.audits_conducted = 0
        self.violations_detected = 0
        self.last_fines_collected = 0.0
        self.last_operational_expense = 0.0

    def terminate(self, reason: str = "") -> None:
        super().terminate(reason)