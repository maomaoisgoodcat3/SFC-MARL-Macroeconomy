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
    def __init__(self, agent_id: str, gini_penalty_coef: float = 25.0, death_penalty_coef: float = 20.0):
        super().__init__(agent_id)
        self.agent_type = AgentType.GOVERNMENT

        # Thue suat va ngan sach
        self.tax_rate_worker: float = 0.15
        self.tax_rate_firm: float = 0.20
        # ρ = ty le thu thue+phat ky truoc duoc Chinh phu CHI MUA HANG (G), [0, 1]; mac dinh 1 =
        # ngan sach can bang kieu SIM (Godley & Lavoie, 2007). Xem rule_engine.py Section 4C.
        # (Truoc day la subsidy_budget_ratio -- mot hanh dong CHET, chi duoc luu khong duoc dung.)
        self.purchase_ratio: float = 1.0
        self.last_purchase: float = 0.0

        # Kho bac va No cong
        self.treasury: float = 0.0
        self.public_debt: float = 0.0

        # Chi so giam sat vi mo de danh gia hieu qua
        self.current_gdp: float = 0.0
        self.last_gdp: float = 0.0
        # GDP THUC (khu gia): co so cua reward/quan sat, xem rule_engine.py Section 9
        self.current_real_gdp: float = 0.0
        self.last_real_gdp: float = 0.0
        self.current_gini: float = 0.0
        self.dead_citizens_count: int = 0
        self.last_tax_collected: float = 0.0
        self.last_fines_collected: float = 0.0
        self.last_subsidies_paid: float = 0.0

        # He so hieu chinh reward (calibration constants, xem calculate_reward)
        # -- co the cau hinh qua ScenarioConfig (be/scenario_config.py), KHONG
        # doi dang ham reward, chi doi gia tri hang so dau vao. Truyen o
        # constructor (khong phai initialize()) vi Government duoc tao lai moi
        # lan reset() (_create_world()) va he so calibration khong nen bi reset
        # theo tung episode, phai giu dung theo scenario dang chay.
        self.gini_penalty_coef: float = float(gini_penalty_coef)
        self.death_penalty_coef: float = float(death_penalty_coef)

    def initialize(self, 
                   initial_treasury: float = 1000000.0,
                   initial_worker_tax: float = 0.15,
                   initial_firm_tax: float = 0.20) -> None:
        self.treasury = float(initial_treasury)
        self.tax_rate_worker = float(initial_worker_tax)
        self.tax_rate_firm = float(initial_firm_tax)
        self.purchase_ratio = 1.0
        self.last_purchase = 0.0
        self.public_debt = 0.0
        self.current_gdp = float(initial_treasury)
        self.last_gdp = float(initial_treasury)
        self.current_real_gdp = 0.0
        self.last_real_gdp = 0.0
        self.current_gini = 0.0
        self.dead_citizens_count = 0
        self.last_tax_collected = 0.0
        self.last_fines_collected = 0.0
        self.last_subsidies_paid = 0.0
        self.status = LifeCycleStatus.ACTIVE

    def observe(self, raw_environment_state: Dict[str, Any]) -> Observation:
        """
        Khong gian quan sat 10 chieu:
        [0]: GDP THUC hien tai (scaled) -- cung dai luong duoc thuong
        [1]: Chenh lech GDP THUC ky truoc
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
        delta_gdp = self.current_real_gdp - self.last_real_gdp

        obs_array = np.array([
            self.current_real_gdp * 0.0001,
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
        [2]: Ty le thu thue+phat ky truoc dung de CHI MUA HANG (Purchase Ratio rho): [0.0, 1.0]
        """
        if "injected_action" in observation.metadata:
            raw_action = observation.metadata["injected_action"]
        else:
            # Chinh sach tai khoa can bang mac dinh (rho = 1: chi mua hang bang dung so thu ky truoc)
            raw_action = np.array([0.15, 0.20, 1.0], dtype=np.float32)

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
                sanitized_values=np.array([0.15, 0.20, 1.0], dtype=np.float32),
                reason="Action vector must have 3 elements"
            )

        # Gioi han muc thue phu hop hien phap, khong cho phep ap dat thue qua cao triet tieu san xuat
        worker_tax = float(np.clip(vals[0], 0.0, 0.50))
        firm_tax = float(np.clip(vals[1], 0.0, 0.50))
        
        # rho <= 1: chi mua hang khong vuot qua so thu ky truoc (ngan sach can bang la TRAN; muon
        # them thau chi phai la mot quyet dinh thiet ke rieng). Tran theo so du Kho bac duoc ap
        # tai rule_engine.py Section 4C nen khong can kep them o day.
        purchase_ratio = float(np.clip(vals[2], 0.0, 1.0))

        sanitized = np.array([worker_tax, firm_tax, purchase_ratio], dtype=np.float32)
        return ValidationResult(is_valid=True, sanitized_values=sanitized)

    def apply_result(self, transition_result: TransitionResult) -> None:
        delta = transition_result.state_delta
        
        self.last_gdp = self.current_gdp
        if "current_gdp" in delta:
            self.current_gdp = float(delta["current_gdp"])
        self.last_real_gdp = self.current_real_gdp
        if "current_real_gdp" in delta:
            self.current_real_gdp = float(delta["current_real_gdp"])
            
        if "current_gini" in delta:
            self.current_gini = float(np.clip(delta["current_gini"], 0.0, 1.0))
            
        if "new_deaths" in delta:
            self.dead_citizens_count += int(delta["new_deaths"])

        # Cap nhat dong tien ngan sach thuc te
        tax_revenue = float(delta.get("tax_collected", 0.0))
        subsidies_spent = float(delta.get("subsidies_disbursed", 0.0))
        # Chi phi bom/rut cau tai khoa phan chu ky do Economy quyet dinh cuong
        # do (rule_engine.py Section 4B; Blanchard & Perotti, 2002) -- duong
        # (bom them cau) tru vao Kho bac giong mot khoan chi tieu chinh phu G,
        # am (rut bot cau) lam Kho bac TANG (chinh phu thu ve suc mua da rut
        # khoi thi truong). Dam bao dang thuc SFC: phan cau Firm/khu vuc phi
        # chinh thuc nhan them/bot PHAI co nguon doi ung dung o day.
        demand_injection_cost = float(delta.get("demand_injection_cost", 0.0))
        # Tien phat trot thue do Supervisor thanh tra (rule_engine.py Section 7,
        # Allingham & Sandmo, 1972: kiem toan + phat la co che rang buoc hanh vi
        # khai bao) da bi TRU khoi vi nguoi vi pham; theo nguyen ly Stock-Flow
        # Consistent (Godley & Lavoie, 2007, "Monetary Economics", Palgrave
        # Macmillan) khoan tien do PHAI chuyen sang mot tac tu khac -- o day la
        # Kho bac (chu the thu phat). LOI DA SUA: truoc day rule_engine ghi
        # delta["fines_collected"] (tu commit de80d2f) nhung ham nay khong bao
        # gio doc, khien MOI dong tien phat bien mat khoi he thong (do duoc 685/
        # 720 buoc vi pham bao toan duoi policy ngau nhien; bo test SFC cu khong
        # thay vi heuristic luon khai bao 100%, khong bao gio bi phat).
        fines_collected = float(delta.get("fines_collected", 0.0))
        # Chi MUA HANG cua Chinh phu (rule_engine.py Section 4C; SIM, Godley & Lavoie 2007):
        # tien di tu Kho bac sang nguoi san xuat (doanh thu firm / thu nhap phi chinh thuc)
        # -- dang thuc SFC: Kho bac giam dung bang so tien khu vuc san xuat nhan them.
        government_purchase_cost = float(delta.get("government_purchase_cost", 0.0))
        self.last_purchase = government_purchase_cost

        self.last_tax_collected = tax_revenue
        self.last_fines_collected = fines_collected
        self.last_subsidies_paid = subsidies_spent

        net_budget = tax_revenue + fines_collected - subsidies_spent - demand_injection_cost - government_purchase_cost
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
        self.purchase_ratio = float(delta.get("executed_purchase_ratio", self.purchase_ratio))

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
        - Mỗi cái chết -> -20 (hệ số cấu trúc TỰ DO HIỆU CHỈNH, nâng từ 5 lên
          20 sau khi quan sát log train thực tế: dân số active sụt từ ~60
          xuống ~26 (mất >50%) trong lúc thất nghiệp leo tới 97% và lạm phát
          âm -20%, NHƯNG reward trung bình vẫn tăng liên tục suốt quá trình
          đó. Giả thuyết: đây là "survivorship bias" trong chính metric
          reward -- khi một Employee chết, agent đó biến mất khỏi tập hợp
          tính trung bình episode_return_mean của RLlib, nên việc loại bỏ
          dần những agent khổ sở nhất khỏi mẫu có thể khiến reward trung
          bình TĂNG dù tổng phúc lợi hệ thống đang sụp đổ thật. Hệ số 5 cũ
          quá yếu để tạo động lực đủ mạnh cho Government dùng đòn bẩy trợ
          cấp thất nghiệp ngăn chặn tử vong hàng loạt. Đây là
          điều chỉnh hệ số cấu trúc thuần tuý -- KHÔNG đổi dạng hàm reward,
          không đổi bất kỳ công thức kinh tế cốt lõi nào khác.
        - Công nợ -> penalty nhỏ, dài hạn
        """
        # GDP THỰC (khử giá), KHÔNG phải danh nghĩa: dùng GDP danh nghĩa thì siêu lạm phát tự
        # nó được thưởng (đo được reward 80,68/bước ở chế độ siêu lạm phát so với 8,54 ở chế
        # độ lành mạnh, trong khi GDP thực 101 so với 7.513) -- Chính phủ có động cơ phá nền
        # kinh tế. Khử giá theo khái niệm "volume measure" của hạch toán quốc gia (United
        # Nations et al. (2009), "System of National Accounts 2008", Ch.15; Hicks (1946),
        # "Value and Capital", phân biệt real/nominal). Hệ số 0,002/0,005 GIỮ NGUYÊN.
        # GDP level: positive signal để agent biết nền kinh tế đang hoạt động
        gdp_level = self.current_real_gdp * 0.002

        # GDP growth: thưởng tăng trưởng
        gdp_growth = (self.current_real_gdp - self.last_real_gdp) * 0.005

        # Gini penalty: [0,1]^2 * gini_penalty_coef (mac dinh 25 -> toi da -25)
        gini_penalty = self.gini_penalty_coef * (self.current_gini ** 2)

        # Death penalty: moi cai chet = -death_penalty_coef (mac dinh -20, xem giai thich hieu chinh o tren)
        new_deaths = int(transition_result.state_delta.get("new_deaths", 0))
        death_penalty = float(new_deaths) * self.death_penalty_coef

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
            "current_real_gdp": self.current_real_gdp,
            "purchase_ratio": self.purchase_ratio,
            "last_purchase": self.last_purchase,
            "current_gini": self.current_gini,
            "dead_citizens_count": self.dead_citizens_count
        }

    def reset(self) -> None:
        super().reset()
        self.tax_rate_worker = 0.15
        self.tax_rate_firm = 0.20
        self.purchase_ratio = 1.0
        self.last_purchase = 0.0
        self.current_real_gdp = 0.0
        self.last_real_gdp = 0.0
        self.treasury = 0.0
        self.public_debt = 0.0
        self.current_gdp = 0.0
        self.last_gdp = 0.0
        self.current_gini = 0.0
        self.dead_citizens_count = 0
        self.last_fines_collected = 0.0

    def terminate(self, reason: str = "") -> None:
        super().terminate(reason)