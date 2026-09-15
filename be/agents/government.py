from typing import Dict, Any
import numpy as np
from be.core.enums import LifeCycleStatus, AgentType
from be.core.types import Observation, Action, ValidationResult, TransitionResult
from be.agents.base_agent import BaseAgent

class Government(BaseAgent):
    """
    Tac tu Chinh phu (Government).
    Dieu tiet tai khoa, an sinh xa hoi va toi uu hoa phuc loi cong dong.
    Tuan thu nghiem ngat contract BaseAgent theo SAS v1.0.
    """
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.agent_type = AgentType.GOVERNMENT
        
        # Thue suat va ngan sach
        self.tax_rate_worker: float = 0.15
        self.tax_rate_firm: float = 0.20
        self.subsidy_budget_ratio: float = 0.10
        
        # Kho bac va No cong
        self.treasury: float = 0.0
        self.public_debt: float = 0.0
        
        # Chi so giam sat vi mo de danh gia hieu qua
        self.current_gdp: float = 0.0
        self.last_gdp: float = 0.0
        self.current_gini: float = 0.0
        self.dead_citizens_count: int = 0
        self.last_tax_collected: float = 0.0
        self.last_subsidies_paid: float = 0.0

    def initialize(self, 
                   initial_treasury: float = 1000000.0,
                   initial_worker_tax: float = 0.15,
                   initial_firm_tax: float = 0.20) -> None:
        self.treasury = float(initial_treasury)
        self.tax_rate_worker = float(initial_worker_tax)
        self.tax_rate_firm = float(initial_firm_tax)
        self.subsidy_budget_ratio = 0.10
        self.public_debt = 0.0
        self.current_gdp = float(initial_treasury)
        self.last_gdp = float(initial_treasury)
        self.current_gini = 0.0
        self.dead_citizens_count = 0
        self.last_tax_collected = 0.0
        self.last_subsidies_paid = 0.0
        self.status = LifeCycleStatus.ACTIVE

    def observe(self, raw_environment_state: Dict[str, Any]) -> Observation:
        """
        Khong gian quan sat 10 chieu:
        [0]: GDP hien tai (scaled)
        [1]: Chenh lech GDP ky truoc
        [2]: He so bat binh dang Gini
        [3]: Ty le lam phat
        [4]: So luong cong dan tu vong luy ke
        [5]: Ty le that nghiep
        [6]: So du kho bac quoc gia (scaled)
        [7]: No cong (scaled)
        [8]: Thue suat TNCN hien tai
        [9]: Thue suat TNDN hien tai
        """
        macro = raw_environment_state.get("macro_indicators", {})
        inflation = float(macro.get("inflation", 0.0))
        unemployment_rate = float(macro.get("unemployment_rate", 0.0))
        delta_gdp = self.current_gdp - self.last_gdp

        obs_array = np.array([
            self.current_gdp * 0.0001,
            delta_gdp * 0.0001,
            self.current_gini,
            inflation,
            float(self.dead_citizens_count) * 0.01,
            unemployment_rate,
            self.treasury * 0.0001,
            self.public_debt * 0.0001,
            self.tax_rate_worker,
            self.tax_rate_firm
        ], dtype=np.float32)

        return Observation(
            agent_id=self.agent_id,
            timestep=raw_environment_state.get("timestep", 0),
            vector=obs_array,
            metadata={"treasury": self.treasury, "gini": self.current_gini}
        )

    def decide(self, observation: Observation) -> Action:
        """
        Khong gian hanh dong 3 chieu:
        [0]: Muc tieu Thue suat Thu nhap Ca nhan (Worker Tax Rate): [0.0, 0.5]
        [1]: Muc tieu Thue suat Doanh nghiep (Firm Tax Rate): [0.0, 0.5]
        [2]: Ty le ngan sach phan bo cho An sinh/Tro cap (Subsidy Ratio): [0.0, 0.4]
        """
        if "injected_action" in observation.metadata:
            raw_action = observation.metadata["injected_action"]
        else:
            # Chinh sach tai khoa can bang mac dinh
            raw_action = np.array([0.15, 0.20, 0.05], dtype=np.float32)

        return Action(
            agent_id=self.agent_id,
            action_type="FISCAL_POLICY",
            values=np.asarray(raw_action, dtype=np.float32)
        )

    def validate_action(self, action: Action) -> ValidationResult:
        vals = action.values
        if len(vals) < 3:
            return ValidationResult(
                is_valid=False,
                sanitized_values=np.array([0.15, 0.20, 0.05], dtype=np.float32),
                reason="Action vector must have 3 elements"
            )

        # Gioi han muc thue phu hop hien phap, khong cho phep ap dat thue qua cao triet tieu san xuat
        worker_tax = float(np.clip(vals[0], 0.0, 0.50))
        firm_tax = float(np.clip(vals[1], 0.0, 0.50))
        
        # Ty le tro cap khong the vuot qua quy dinh ngan sach
        subsidy_ratio = float(np.clip(vals[2], 0.0, 0.40))
        
        # Neu kho bac tham hut nghiem trong, bat buoc cat giam tro cap
        if self.treasury < 0.0:
            subsidy_ratio = min(subsidy_ratio, 0.02)

        sanitized = np.array([worker_tax, firm_tax, subsidy_ratio], dtype=np.float32)
        return ValidationResult(is_valid=True, sanitized_values=sanitized)

    def apply_result(self, transition_result: TransitionResult) -> None:
        delta = transition_result.state_delta
        
        self.last_gdp = self.current_gdp
        if "current_gdp" in delta:
            self.current_gdp = float(delta["current_gdp"])
            
        if "current_gini" in delta:
            self.current_gini = float(np.clip(delta["current_gini"], 0.0, 1.0))
            
        if "new_deaths" in delta:
            self.dead_citizens_count += int(delta["new_deaths"])

        # Cap nhat dong tien ngan sach thuc te
        tax_revenue = float(delta.get("tax_collected", 0.0))
        subsidies_spent = float(delta.get("subsidies_disbursed", 0.0))
        
        self.last_tax_collected = tax_revenue
        self.last_subsidies_paid = subsidies_spent
        
        net_budget = tax_revenue - subsidies_spent
        self.treasury += net_budget
        
        if self.treasury < 0.0:
            self.public_debt += abs(self.treasury)
            self.treasury = 0.0
        elif self.public_debt > 0.0 and self.treasury > 0.0:
            repayment = min(self.treasury * 0.5, self.public_debt)
            self.public_debt -= repayment
            self.treasury -= repayment

        # Cap nhat thue suat ap dung thuc te do RuleEngine chap thuan
        self.tax_rate_worker = float(delta.get("executed_worker_tax", self.tax_rate_worker))
        self.tax_rate_firm = float(delta.get("executed_firm_tax", self.tax_rate_firm))
        self.subsidy_budget_ratio = float(delta.get("executed_subsidy_ratio", self.subsidy_budget_ratio))

    def calculate_reward(self, transition_result: TransitionResult) -> float:
        """
        Social Welfare Objective:
        Reward = GDP_level (normalized) + GDP_growth - Gini_penalty - Death_penalty - Debt_penalty

        Thiết kế:
        - GDP level ~200-500 -> normalized ~0.2-0.5 (positive baseline)
        - Gini [0,1] -> penalty tối đa 25 (hệ số cấu trúc TỰ DO HIỆU CHỈNH,
          nâng từ 10 lên 25 sau khi quan sát thực nghiệm nhiều lần train dài
          hạn: hệ số 10 quá yếu so với gdp_level/gdp_growth nên Gini hội tụ
          bất động quanh 0.80-0.86 thay vì giảm -- xem thêm cơ chế thuế lãi
          tiền gửi mới thêm ở rule_engine.py Mục 8B giải quyết TẬN GỐC nguồn
          gây bất bình đẳng (lãi kép miễn thuế), còn hệ số này chỉ là tín
          hiệu reward bổ trợ, không thay thế cơ chế kinh tế)
        - Mỗi cái chết -> -5 (đủ để agent quan tâm nhưng không dominate)
        - Công nợ -> penalty nhỏ, dài hạn
        """
        # GDP level: positive signal để agent biết nền kinh tế đang hoạt động
        gdp_level = self.current_gdp * 0.002

        # GDP growth: thưởng tăng trưởng
        gdp_growth = (self.current_gdp - self.last_gdp) * 0.005

        # Gini penalty: [0,1]^2 * 25 -> tối đa -25
        gini_penalty = 25.0 * (self.current_gini ** 2)

        # Death penalty: mỗi cái chết = -5
        new_deaths = int(transition_result.state_delta.get("new_deaths", 0))
        death_penalty = float(new_deaths) * 5.0

        # Debt penalty: nhỏ, chỉ kích hoạt khi nợ lớn
        debt_penalty = (self.public_debt * 0.00005) if self.public_debt > 0 else 0.0

        social_welfare = gdp_level + gdp_growth - gini_penalty - death_penalty - debt_penalty
        return float(np.clip(social_welfare, -100.0, 100.0))

    def export_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "status": self.status.name,
            "treasury": self.treasury,
            "public_debt": self.public_debt,
            "tax_rate_worker": self.tax_rate_worker,
            "tax_rate_firm": self.tax_rate_firm,
            "current_gdp": self.current_gdp,
            "current_gini": self.current_gini,
            "dead_citizens_count": self.dead_citizens_count
        }

    def reset(self) -> None:
        super().reset()
        self.tax_rate_worker = 0.15
        self.tax_rate_firm = 0.20
        self.subsidy_budget_ratio = 0.10
        self.treasury = 0.0
        self.public_debt = 0.0
        self.current_gdp = 0.0
        self.last_gdp = 0.0
        self.current_gini = 0.0
        self.dead_citizens_count = 0

    def terminate(self, reason: str = "") -> None:
        super().terminate(reason)