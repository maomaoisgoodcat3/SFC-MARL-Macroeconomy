from typing import Dict, Any
import numpy as np
from be.core.enums import LifeCycleStatus, AgentType
from be.core.types import Observation, Action, ValidationResult, TransitionResult
from be.agents.base_agent import BaseAgent

class Economy(BaseAgent):
    """
    Tac tu Tao lap Thi truong (Economy / Market Maker).
    Dinh gia ro hang hoa tieu dung, quan ly tai san nha dat va can bang vi mo.
    Tuan thu nghiem ngat contract BaseAgent theo SAS v1.0.
    """
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.agent_type = AgentType.ECONOMY
        
        # Ro hang hoa va chi phi sinh hoat
        self.base_living_cost: float = 15.0
        self.last_base_living_cost: float = 15.0
        
        # Thi truong bat dong san / nha o
        self.housing_inventory: int = 100
        self.housing_price: float = 1000.0
        self.last_housing_price: float = 1000.0
        
        # Chi so vi mo thi truong
        self.inflation_rate: float = 0.02
        self.cpi_index: float = 100.0
        self.step_trade_volume: float = 0.0
        self.market_liquidity_reserve: float = 0.0

    def initialize(self, 
                   initial_living_cost: float = 15.0,
                   initial_housing_inventory: int = 100,
                   initial_house_price: float = 1000.0,
                   target_inflation: float = 0.02) -> None:
        self.base_living_cost = float(initial_living_cost)
        self.last_base_living_cost = float(initial_living_cost)
        self.housing_inventory = int(initial_housing_inventory)
        self.housing_price = float(initial_house_price)
        self.last_housing_price = float(initial_house_price)
        self.inflation_rate = float(target_inflation)
        self.cpi_index = 100.0
        self.step_trade_volume = 0.0
        self.market_liquidity_reserve = 100000.0
        self.status = LifeCycleStatus.ACTIVE

    def observe(self, raw_environment_state: Dict[str, Any]) -> Observation:
        """
        Khong gian quan sat 9 chieu:
        [0]: Chi phi sinh hoat co so hien tai
        [1]: Gia nha trung binh thi truong (scaled)
        [2]: Ton kho quy nha o (Housing Inventory)
        [3]: Ty le lam phat hien hanh
        [4]: Chi so CPI (scaled)
        [5]: Khoi luong giao dich thi truong ky truoc (scaled)
        [6]: Tong cau tieu dung xa hoi (scaled)
        [7]: Tong cung lao dong / san pham (scaled)
        [8]: Do lech lam phat so voi muc tieu 2%
        """
        macro = raw_environment_state.get("macro_indicators", {})
        aggregate_demand = float(macro.get("aggregate_demand", 0.0)) * 0.001
        aggregate_supply = float(macro.get("aggregate_supply", 0.0)) * 0.001

        inflation_gap = self.inflation_rate - 0.02

        obs_array = np.array([
            self.base_living_cost * 0.05,
            self.housing_price * 0.0005,
            float(self.housing_inventory) * 0.01,
            self.inflation_rate,
            self.cpi_index * 0.01,
            self.step_trade_volume * 0.0005,
            aggregate_demand,
            aggregate_supply,
            inflation_gap
        ], dtype=np.float32)

        return Observation(
            agent_id=self.agent_id,
            timestep=raw_environment_state.get("timestep", 0),
            vector=obs_array,
            metadata={"housing_price": self.housing_price, "living_cost": self.base_living_cost}
        )

    def decide(self, observation: Observation) -> Action:
        """
        Khong gian hanh dong 3 chieu:
        [0]: He so dieu chinh chi phi sinh hoat (Living Cost Factor): [0.90, 1.10]
        [1]: He so dieu chinh gia bat dong san (Housing Price Factor): [0.90, 1.10]
        [2]: So luong quy nha/dat bo sung ra thi truong (Housing Supply Expansion): [0, 10]
        """
        if "injected_action" in observation.metadata:
            raw_action = observation.metadata["injected_action"]
        else:
            raw_action = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        return Action(
            agent_id=self.agent_id,
            action_type="MARKET_CLEARING_AND_PRICING",
            values=np.asarray(raw_action, dtype=np.float32)
        )

    def validate_action(self, action: Action) -> ValidationResult:
        vals = action.values
        if len(vals) < 3:
            return ValidationResult(
                is_valid=False,
                sanitized_values=np.array([1.0, 1.0, 0.0], dtype=np.float32),
                reason="Economy action vector must contain 3 elements"
            )

        # Tranh shock gia: moi chu ky chi cho phep bien dong toi da +- 10%
        living_cost_factor = float(np.clip(vals[0], 0.90, 1.10))
        housing_price_factor = float(np.clip(vals[1], 0.90, 1.10))
        housing_supply = float(np.clip(vals[2], 0.0, 10.0))

        sanitized = np.array([living_cost_factor, housing_price_factor, housing_supply], dtype=np.float32)
        return ValidationResult(is_valid=True, sanitized_values=sanitized)

    def apply_result(self, transition_result: TransitionResult) -> None:
        delta = transition_result.state_delta

        self.last_base_living_cost = self.base_living_cost
        self.last_housing_price = self.housing_price

        # Dieu chinh gia dua tren ket qua thuc thi tu RuleEngine
        cost_mult = float(delta.get("executed_cost_factor", 1.0))
        price_mult = float(delta.get("executed_housing_factor", 1.0))
        new_housing = int(delta.get("executed_housing_supply", 0))

        self.base_living_cost = float(np.clip(self.base_living_cost * cost_mult, 5.0, 100.0))
        self.housing_price = float(np.clip(self.housing_price * price_mult, 100.0, 50000.0))
        self.housing_inventory = max(0, self.housing_inventory + new_housing - int(delta.get("houses_sold", 0)))

        # Tinh toan Lam phat ky nay
        cost_growth = (self.base_living_cost - self.last_base_living_cost) / self.last_base_living_cost
        self.inflation_rate = cost_growth
        self.cpi_index *= (1.0 + self.inflation_rate)

        self.step_trade_volume = float(delta.get("total_market_turnover", 0.0))
        self.market_liquidity_reserve += float(delta.get("liquidity_delta", 0.0))

    def calculate_reward(self, transition_result: TransitionResult) -> float:
        """
        Muc tieu Tao lap Thi truong:
        Reward = On dinh gia ca (Phat sai lech lam phat khoi 2%) - Phat Thieu hut nha o + Quy mo thanh khoan
        """
        # Phat chenh lech muc tieu lam phat (2% hang nam ~ 0.0016 hang thang)
        inflation_penalty = 50.0 * ((self.inflation_rate - 0.0016) ** 2)

        # Phat tinh trang can kiet nha o tren thi truong
        housing_penalty = 0.0
        if self.housing_inventory < 5:
            housing_penalty = float(5 - self.housing_inventory) * 2.0

        # Thuong cho viec thi truong luu thong thanh khoan tot
        volume_reward = np.log1p(max(0.0, self.step_trade_volume)) * 0.5

        reward = volume_reward - inflation_penalty - housing_penalty
        return float(np.clip(reward, -50.0, 50.0))

    def export_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "status": self.status.name,
            "base_living_cost": self.base_living_cost,
            "housing_price": self.housing_price,
            "housing_inventory": self.housing_inventory,
            "inflation_rate": self.inflation_rate,
            "cpi_index": self.cpi_index,
            "step_trade_volume": self.step_trade_volume
        }

    def reset(self) -> None:
        super().reset()
        self.base_living_cost = 15.0
        self.last_base_living_cost = 15.0
        self.housing_inventory = 100
        self.housing_price = 1000.0
        self.last_housing_price = 1000.0
        self.inflation_rate = 0.02
        self.cpi_index = 100.0
        self.step_trade_volume = 0.0
        self.market_liquidity_reserve = 100000.0

    def terminate(self, reason: str = "") -> None:
        super().terminate(reason)