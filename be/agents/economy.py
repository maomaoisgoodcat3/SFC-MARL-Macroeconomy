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
        # Tổng chi phí khấu hao tư bản (overhead) toàn thị trường kỳ gần nhất --
        # kênh "rò rỉ tiền" duy nhất được cho phép ngoài DefaultedDebt trong đẳng
        # thức bảo toàn SFC (xem rule_engine.py, Section 3-4). Phơi bày tường
        # minh để kiểm toán tự động (be/tests/test_sfc_accounting.py).
        self.last_capital_depreciation: float = 0.0
        # Đòn bẩy chính sách tài khóa phản chu kỳ (rule_engine.py Section 4B) --
        # Blanchard & Perotti (2002). Lưu lại thuần để quan sát/audit, KHÔNG
        # nạp ngược vào observe() (tránh đổi obs shape, phá checkpoint cũ).
        self.last_demand_injection_ratio: float = 0.0
        self.last_demand_injection_value: float = 0.0
        # Mốc giá CỐ ĐỊNH tại thời điểm reset episode (KHÔNG đổi trong suốt
        # episode, khác self.base_living_cost là biến sống). Dùng làm mốc quy
        # đổi injection từ real quantity sang nominal (rule_engine.py Section
        # 4B) -- BẮT BUỘC dùng hằng số này thay vì expected_price/
        # base_living_cost đang biến động, nếu không sẽ tạo vòng lặp phản hồi
        # dương (price -> injection_base -> spending -> price) có thể phân kỳ
        # về mặt số học khi cung thực (total_real_supply) co lại nhanh hơn
        # dân số trong khủng hoảng thất nghiệp -- lỗi THẬT đã xảy ra và được
        # sửa ở đây, xem CLAUDE_HISTORY.md.
        self.initial_living_cost: float = 15.0

    def initialize(self, 
                   initial_living_cost: float = 15.0,
                   initial_housing_inventory: int = 100,
                   initial_house_price: float = 1000.0,
                   target_inflation: float = 0.02) -> None:
        self.base_living_cost = float(initial_living_cost)
        self.last_base_living_cost = float(initial_living_cost)
        self.initial_living_cost = float(initial_living_cost)
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
        [0]: Cuong do bom/rut CAU tai khoa phan chu ky (Fiscal Demand
             Injection Ratio), [-0.20, 0.20] -- xem rule_engine.py Section 4B
             va calculate_reward() o duoi. THAY THE cho "Living Cost Factor"
             cu (nhan truc tiep vao base_living_cost) vi thiet ke cu khong co
             hieu luc thuc te (key executed_cost_factor khong bao gio duoc
             RuleEngine ghi -- phat hien qua audit hardcode formula) VA ve
             ban chat kinh te khong co "khe ho" nao trong cong thuc Calvo
             (1983) de nhet mot hanh dong dieu tiet gia truc tiep vao ma
             khong bia them co che moi ngoai trich dan.
        [1]: He so dieu chinh gia bat dong san (Housing Price Factor): [0.90, 1.10]
             -- CHUA CO HIEU LUC: cho toi khi co market nha dat/dau gia that
             (xem ke hoach mo rong tuong lai), day van la placeholder no-op.
        [2]: So luong quy nha/dat bo sung ra thi truong (Housing Supply Expansion): [0, 10]
             -- CHUA CO HIEU LUC, ly do nhu tren.
        """
        if "injected_action" in observation.metadata:
            raw_action = observation.metadata["injected_action"]
        else:
            raw_action = np.array([0.0, 1.0, 1.0], dtype=np.float32)

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

        # He so cau truc TU DO HIEU CHINH: bien do bom/rut cau toi da +-20%
        # tong cau sinh ton co so cua dan so dang hoat dong (xem rule_engine.py
        # Section 4B). KHONG suy ra truc tiep tu Blanchard & Perotti (2002) --
        # paper do khong dua ra mot ty le % cu the nao, chi xac nhan huong tac
        # dong (chi tieu chinh phu -> tong cau); +-20% la calibration rieng
        # cua du an, tranh mot cu bom/rut lam thay doi qua dot ngot tong cau.
        demand_injection_ratio = float(np.clip(vals[0], -0.20, 0.20))
        housing_price_factor = float(np.clip(vals[1], 0.90, 1.10))
        housing_supply = float(np.clip(vals[2], 0.0, 10.0))

        sanitized = np.array([demand_injection_ratio, housing_price_factor, housing_supply], dtype=np.float32)
        return ValidationResult(is_valid=True, sanitized_values=sanitized)

    def apply_result(self, transition_result: TransitionResult) -> None:
        delta = transition_result.state_delta

        self.last_base_living_cost = self.base_living_cost
        self.last_housing_price = self.housing_price

        # base_living_cost/inflation_rate THUC SU duoc RuleEngine ghi de truc
        # tiep ngay sau khi goi apply_result nay (eco.base_living_cost =
        # actual_living_cost trong rule_engine.py Section 4) -- xem chi tiet
        # trong docstring class. Doan tinh toan o day chi la du phong/quan sat
        # noi bo, khong phai nguon su that cho 2 bien nay.
        self.last_demand_injection_ratio = float(delta.get("demand_injection_ratio", 0.0))
        self.last_demand_injection_value = float(delta.get("demand_injection_effect", 0.0))

        # [1]/[2]: Housing price/supply -- CHUA CO HIEU LUC (xem decide()),
        # cac key executed_housing_factor/executed_housing_supply khong duoc
        # RuleEngine ghi nen day van la no-op, giu nguyen cho toi khi co market
        # nha dat that.
        price_mult = float(delta.get("executed_housing_factor", 1.0))
        new_housing = int(delta.get("executed_housing_supply", 0))

        self.housing_price = float(np.clip(self.housing_price * price_mult, 100.0, 50000.0))
        self.housing_inventory = max(0, self.housing_inventory + new_housing - int(delta.get("houses_sold", 0)))

        # LOI DA PHAT HIEN VA SUA (ra soat lai khi wiring Section 4B): ban cu
        # tinh "cost_growth = (base_living_cost - last_base_living_cost) /
        # last_base_living_cost" roi gan vao self.inflation_rate. Nhung o
        # env.py::step(), RuleEngine.execute_cycle() (noi truc tiep mutate
        # eco.base_living_cost = actual_living_cost VA eco.inflation_rate =
        # calvo_inflation, xem rule_engine.py Section 4) LUON chay xong va
        # RETURN truoc khi apply_result nay duoc goi. Nghia la khi toi dong
        # "self.last_base_living_cost = self.base_living_cost" o tren, bien
        # base_living_cost DA BI RuleEngine cap nhat cho ky nay roi -- last_*
        # vo tinh bi gan bang chinh gia tri MOI, khien cost_growth LUON BANG 0
        # va tu tay xoa mat calvo_inflation dung RuleEngine vua tinh, RESET
        # inflation_rate VE 0.0 sau MOI buoc. Day la ly do inflation_penalty
        # trong calculate_reward() gan nhu luon la hang so vo nghia bat ke
        # lam phat/giam phat thuc te -- khong chi vi Economy thieu don bay
        # nhan qua (da sua o Section 4B, rule_engine.py) ma con vi chinh tin
        # hieu phan hoi bi hong. Sua: KHONG tinh lai/ghi de self.inflation_rate
        # o day nua (gia tri cua RuleEngine da dung, giu nguyen), chi doc lai
        # tu delta["inflation"] (nguon khong bi ghi de) de cap nhat cpi_index.
        calvo_inflation = float(delta.get("inflation", 0.0))
        self.cpi_index = float(np.clip(self.cpi_index * (1.0 + calvo_inflation), 1e-6, 1e6))

        self.step_trade_volume = float(delta.get("total_market_turnover", 0.0))
        self.market_liquidity_reserve += float(delta.get("liquidity_delta", 0.0))
        self.last_capital_depreciation = float(delta.get("capital_depreciation_cost", 0.0))

    def calculate_reward(self, transition_result: TransitionResult) -> float:
        """
        Mục tiêu Tạo lập Thị trường — phiên bản sửa lỗi exploit giá:

        Vấn đề cũ: volume_reward = log1p(nominal_spending) * 0.5
            -> Khi living_cost tăng, workers buộc phải chi nhiều tiền hơn để mua
               cùng lượng hàng -> nominal volume tăng -> Economy được thưởng dù
               thực chất không có thêm hàng hóa lưu thông. Đây là degenerate policy:
               Economy học tăng giá 10%/tháng liên tục để maximize reward.

        Fix: Đo REAL volume (số lượng hàng thực) = nominal_spending / living_cost.
             Thêm dead_worker_penalty để Economy chịu hậu quả trực tiếp khi workers
             chết vì không đủ tiền sống (mất demand vĩnh viễn).

        Nguồn:
        - Real vs nominal GDP distinction: Hicks (1946), "Value and Capital"
        - Market maker penalized for demand destruction: Tirole (1988),
          "The Theory of Industrial Organization", MIT Press, Ch.1
        - Inflation target penalty (quadratic loss): Svensson (1997),
          "Inflation Forecast Targeting", European Economic Review 41(6), 1111-1146
        """
        # 1. REAL volume reward: đo số lượng hàng thực tế lưu thông, không phải tiền
        #    real_volume = nominal_spending / price_level (deflate về giá gốc)
        #    log scale để tránh signal quá lớn khi volume tăng đột biến
        price_level = max(1.0, self.base_living_cost)
        real_volume = self.step_trade_volume / price_level
        real_volume_reward = np.log1p(max(0.0, real_volume)) * 2.0

        # 2. Inflation penalty quadratic (Svensson, 1997)
        #    Target: 0.0016/month (~2%/year). Penalty tăng theo bình phương độ lệch.
        #    Hệ số 50 giữ nguyên từ calibration cũ — đã ổn định trong training.
        inflation_penalty = 50.0 * ((self.inflation_rate - 0.0016) ** 2)

        # 3. Dead worker penalty: mỗi worker chết = mất 1 đơn vị demand vĩnh viễn
        #    Economy phải chịu hậu quả trực tiếp của việc đẩy giá quá cao.
        #    Hệ số 20: đủ lớn để outweigh volume gain từ việc tăng giá gây chết người.
        new_deaths = int(transition_result.state_delta.get("new_deaths", 0))
        dead_worker_penalty = float(new_deaths) * 20.0

        # 4. Housing inventory penalty (giữ nguyên)
        housing_penalty = 0.0
        if self.housing_inventory < 5:
            housing_penalty = float(5 - self.housing_inventory) * 2.0

        reward = real_volume_reward - inflation_penalty - dead_worker_penalty - housing_penalty
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
            "step_trade_volume": self.step_trade_volume,
            "last_capital_depreciation": round(self.last_capital_depreciation, 2),
            "last_demand_injection_ratio": round(self.last_demand_injection_ratio, 4),
            "last_demand_injection_value": round(self.last_demand_injection_value, 1)
        }

    def reset(self) -> None:
        super().reset()
        self.base_living_cost = 15.0
        self.last_base_living_cost = 15.0
        self.initial_living_cost = 15.0
        self.housing_inventory = 100
        self.housing_price = 1000.0
        self.last_housing_price = 1000.0
        self.inflation_rate = 0.02
        self.cpi_index = 100.0
        self.step_trade_volume = 0.0
        self.market_liquidity_reserve = 100000.0
        self.last_capital_depreciation = 0.0
        self.last_demand_injection_ratio = 0.0
        self.last_demand_injection_value = 0.0

    def terminate(self, reason: str = "") -> None:
        super().terminate(reason)