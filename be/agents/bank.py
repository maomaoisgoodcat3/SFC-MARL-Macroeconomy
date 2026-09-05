from typing import Dict, Any
import numpy as np
from be.core.enums import LifeCycleStatus, AgentType
from be.core.types import Observation, Action, ValidationResult, TransitionResult
from be.agents.base_agent import BaseAgent

class Bank(BaseAgent):
    """
    Tac tu Ngan hang (Bank).
    Quan ly thanh khoan, lai suat, tin dung va no xau toan he thong.
    Tuan thu nghiem ngat contract BaseAgent theo SAS v1.0.
    """
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.agent_type = AgentType.BANK
        
        # Co cau tai san va nguon von
        self.reserves: float = 0.0
        self.total_deposits: float = 0.0
        self.total_loans: float = 0.0
        self.non_performing_loans: float = 0.0
        
        # Chinh sach lai suat va an toan von
        self.lending_rate: float = 0.06
        self.deposit_rate: float = 0.02
        self.reserve_requirement_ratio: float = 0.10
        self.credit_expansion_factor: float = 1.0
        
        # Ket qua hoat dong ky truoc
        self.last_interest_income: float = 0.0
        self.last_interest_expense: float = 0.0
        self.last_default_loss: float = 0.0

    def initialize(self, 
                   initial_reserves: float = 500000.0,
                   initial_lending_rate: float = 0.06,
                   initial_deposit_rate: float = 0.02,
                   reserve_requirement_ratio: float = 0.10) -> None:
        self.reserves = float(initial_reserves)
        self.total_deposits = 0.0
        self.total_loans = 0.0
        self.non_performing_loans = 0.0
        self.lending_rate = float(initial_lending_rate)
        self.deposit_rate = float(initial_deposit_rate)
        self.reserve_requirement_ratio = float(reserve_requirement_ratio)
        self.credit_expansion_factor = 1.0
        self.last_interest_income = 0.0
        self.last_interest_expense = 0.0
        self.last_default_loss = 0.0
        self.status = LifeCycleStatus.ACTIVE

    def observe(self, raw_environment_state: Dict[str, Any]) -> Observation:
        """
        Khong gian quan sat 10 chieu:
        [0]: Luong tien mat du tru tai ngan hang (Reserves scaled)
        [1]: Tong quy tien gui cua toan bo nen kinh te (Deposits scaled)
        [2]: Tong du no tin dung dang luu hanh (Loans scaled)
        [3]: Khoi luong no xau chua xu ly (NPL scaled)
        [4]: Ty le no xau tren tong du no (NPL Ratio)
        [5]: Ty le du tru thuc te tren tong tien gui (Current Reserve Ratio)
        [6]: Lai suat cho vay hien tai
        [7]: Lai suat tien gui hien tai
        [8]: Ty le lam phat thi truong
        [9]: Cau vay von cua Doanh nghiep toan thi truong
        """
        macro = raw_environment_state.get("macro_indicators", {})
        inflation = float(macro.get("inflation", 0.0))
        credit_demand = float(macro.get("total_credit_demand", 0.0)) * 0.001

        npl_ratio = (self.non_performing_loans / self.total_loans) if self.total_loans > 0.0 else 0.0
        reserve_ratio = (self.reserves / self.total_deposits) if self.total_deposits > 0.0 else 1.0

        obs_array = np.array([
            self.reserves * 0.0001,
            self.total_deposits * 0.0001,
            self.total_loans * 0.0001,
            self.non_performing_loans * 0.0001,
            float(np.clip(npl_ratio, 0.0, 1.0)),
            float(np.clip(reserve_ratio, 0.0, 2.0)),
            self.lending_rate,
            self.deposit_rate,
            inflation,
            credit_demand
        ], dtype=np.float32)

        return Observation(
            agent_id=self.agent_id,
            timestep=raw_environment_state.get("timestep", 0),
            vector=obs_array,
            metadata={"npl_ratio": npl_ratio, "reserves": self.reserves}
        )

    def decide(self, observation: Observation) -> Action:
        """
        Khong gian hanh dong 3 chieu:
        [0]: Muc tieu Lai suat cho vay (Lending Rate): [0.01, 0.25]
        [1]: Muc tieu Lai suat huy dong tien gui (Deposit Rate): [0.005, 0.15]
        [2]: He so mo rong / that chat tin dung (Credit Expansion Factor): [0.0, 1.0]
        """
        if "injected_action" in observation.metadata:
            raw_action = observation.metadata["injected_action"]
        else:
            # Chinh sach tien te on dinh mac dinh
            raw_action = np.array([0.06, 0.02, 0.8], dtype=np.float32)

        return Action(
            agent_id=self.agent_id,
            action_type="MONETARY_POLICY_AND_CREDIT",
            values=np.asarray(raw_action, dtype=np.float32)
        )

    def validate_action(self, action: Action) -> ValidationResult:
        vals = action.values
        if len(vals) < 3:
            return ValidationResult(
                is_valid=False,
                sanitized_values=np.array([0.06, 0.02, 0.5], dtype=np.float32),
                reason="Bank action requires 3 elements"
            )

        lending_rate = float(np.clip(vals[0], 0.01, 0.25))
        deposit_rate = float(np.clip(vals[1], 0.005, 0.15))
        
        # Nguyen tac song con cua Ngan hang: Lai suat cho vay luon phai cao hon Lai suat huy dong
        if lending_rate < deposit_rate + 0.01:
            lending_rate = deposit_rate + 0.01

        credit_factor = float(np.clip(vals[2], 0.0, 1.0))

        # Neu ty le du tru duoi nguong an toan, dong bang viec bom them tin dung
        required_cash = self.total_deposits * self.reserve_requirement_ratio
        if self.reserves < required_cash:
            credit_factor = 0.0

        sanitized = np.array([lending_rate, deposit_rate, credit_factor], dtype=np.float32)
        return ValidationResult(is_valid=True, sanitized_values=sanitized)

    def apply_result(self, transition_result: TransitionResult) -> None:
        delta = transition_result.state_delta
        
        # Bien dong luong tien gui va giai ngan tin dung thuc te do RuleEngine xac thuc
        self.total_deposits = float(max(0.0, self.total_deposits + delta.get("deposits_delta", 0.0)))
        self.total_loans = float(max(0.0, self.total_loans + delta.get("loans_delta", 0.0)))
        self.reserves += float(delta.get("reserves_delta", 0.0))
        
        # Xu ly no xau tu cac Doanh nghiep pha san trong ky
        new_defaults = float(delta.get("new_defaults", 0.0))
        self.non_performing_loans += new_defaults
        self.last_default_loss = new_defaults
        
        # Xoa bo no xau khoi tong du no (Write-off)
        if new_defaults > 0.0:
            self.total_loans = max(0.0, self.total_loans - new_defaults)

        self.last_interest_income = float(delta.get("interest_income", 0.0))
        self.last_interest_expense = float(delta.get("interest_expense", 0.0))

        # Cap nhat tham so thuc thi
        self.lending_rate = float(delta.get("executed_lending_rate", self.lending_rate))
        self.deposit_rate = float(delta.get("executed_deposit_rate", self.deposit_rate))
        self.credit_expansion_factor = float(delta.get("executed_credit_factor", self.credit_expansion_factor))

        # Ranh gioi pha san ngan hang (Mat kha nang thanh toan tram trong)
        if self.reserves < -100000.0:
            self.terminate(reason="Bank run / Total liquidity failure")

    def calculate_reward(self, transition_result: TransitionResult) -> float:
        """
        Hàm mục tiêu tài chính của Ngân hàng:
        Reward = Bien lai rong (NIM) - Phat No xau (NPL) - Phat Vi pham Du tru bat buoc
        """
        net_interest_margin = (self.last_interest_income - self.last_interest_expense) * 0.01
        npl_penalty = self.last_default_loss * 0.02
        
        # Phat neu du tru thuc te thap hon ty le bat buoc
        reserve_penalty = 0.0
        required_reserves = self.total_deposits * self.reserve_requirement_ratio
        if self.reserves < required_reserves:
            deficit = required_reserves - self.reserves
            reserve_penalty = deficit * 0.05

        reward = net_interest_margin - npl_penalty - reserve_penalty
        return float(np.clip(reward, -100.0, 100.0))

    def export_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "status": self.status.name,
            "reserves": self.reserves,
            "total_deposits": self.total_deposits,
            "total_loans": self.total_loans,
            "non_performing_loans": self.non_performing_loans,
            "lending_rate": self.lending_rate,
            "deposit_rate": self.deposit_rate,
            "last_nim": self.last_interest_income - self.last_interest_expense
        }

    def reset(self) -> None:
        super().reset()
        self.reserves = 0.0
        self.total_deposits = 0.0
        self.total_loans = 0.0
        self.non_performing_loans = 0.0
        self.lending_rate = 0.06
        self.deposit_rate = 0.02
        self.credit_expansion_factor = 1.0
        self.last_interest_income = 0.0
        self.last_interest_expense = 0.0
        self.last_default_loss = 0.0

    def terminate(self, reason: str = "") -> None:
        super().terminate(reason)