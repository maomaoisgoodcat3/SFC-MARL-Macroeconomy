from typing import Dict, Any, List, Optional
import numpy as np
from be.core.enums import LifeCycleStatus, AgentType
from be.core.types import Observation, Action, ValidationResult, TransitionResult
from be.agents.base_agent import BaseAgent

class Firm(BaseAgent):
    """
    Tac tu Doanh nghiep (Firm / Corporate).
    Tuan thu nghiem ngat contract cua BaseAgent theo SAS v1.0.
    """
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.agent_type = AgentType.FIRM
        
        # Nang luc san xuat va quan tri
        self.productivity_factor: float = 1.2
        self.risk_aversion: float = 0.3
        self.tax_morale: float = 0.6
        
        # Tai san va nhan su noi tai
        self.cash: float = 0.0
        self.capital_stock: float = 0.0
        self.debt: float = 0.0
        self.employee_ids: List[str] = []
        # Quan hệ tín dụng với MỘT ngân hàng cụ thể khi hệ thống có nhiều ngân
        # hàng (Petersen & Rajan, 1994, "The Benefits of Lending Relationships",
        # Journal of Finance 49(1)) -- xem rule_engine.py Section 5.
        self.creditor_bank_id: Optional[str] = None
        
        # Thong so tai chinh gan nhat de tinh Reward
        self.last_profit: float = 0.0
        self.last_revenue: float = 0.0
        self.last_declare_ratio: float = 1.0

    def initialize(self, 
                   productivity_factor: float = 1.2, 
                   risk_aversion: float = 0.3,
                   tax_morale: float = 0.6,
                   initial_capital: float = 2000.0) -> None:
        self.productivity_factor = float(productivity_factor)
        self.risk_aversion = float(risk_aversion) if risk_aversion != 1.0 else 0.99
        self.tax_morale = float(tax_morale)
        self.cash = float(initial_capital)
        self.capital_stock = float(initial_capital * 0.5)
        self.debt = 0.0
        self.employee_ids = []
        self.creditor_bank_id = None
        self.status = LifeCycleStatus.ACTIVE

    def observe(self, raw_environment_state: Dict[str, Any]) -> Observation:
        """
        Khong gian quan sat 13 chieu:
        [0-3] Vi mo: Thue TNDN, Lai suat cho vay cua Bank, Luong trung binh thi truong, He so cau thi truong
        [4-10] Vi mo noi tai: Tien mat, Von dau tu, No phai tra, So nhan vien, Doanh thu gan nhat, Loi nhuan gan nhat, Nang suat
        [11] risk_aversion, [12] tax_morale -- MOI (v0.28)

        LOI DA SUA cho [11]/[12] (phat hien qua audit chu dong theo yeu cau nguoi
        dung 2026-09-25, cung dot voi employee.py): risk_aversion anh huong that
        risk_discount khi vay von (Kimball 1990, rule_engine.py Section 5) va
        tax_morale anh huong truc tiep moral_cost trong calculate_reward() ben
        duoi -- CA HAI di bien theo tung Firm (khoi tao ngau nhien, xem
        env.py::_create_world/gia nhap nganh Section B) nhung CHUA BAO GIO duoc
        dua vao observe(). Vi Firm cung dung PARAMETER SHARING nhu Employee, mang
        chung khong the ca the hoa hanh vi vay no/khai bao thue theo dung dac diem
        rieng cua tung firm neu khong quan sat duoc 2 truong nay.
        """
        macro = raw_environment_state.get("macro_indicators", {})
        corp_tax = float(macro.get("firm_tax_rate", 0.2))
        lending_rate = float(macro.get("bank_lending_rate", 0.06))
        avg_market_wage = float(macro.get("average_wage", 30.0))
        demand_index = float(macro.get("market_demand_factor", 1.0))

        obs_array = np.array([
            corp_tax,
            lending_rate,
            avg_market_wage,
            demand_index,
            self.cash,
            self.capital_stock,
            self.debt,
            float(len(self.employee_ids)),
            self.last_revenue,
            self.last_profit,
            self.productivity_factor,
            self.risk_aversion,
            self.tax_morale,
        ], dtype=np.float32)

        return Observation(
            agent_id=self.agent_id,
            timestep=raw_environment_state.get("timestep", 0),
            vector=obs_array,
            metadata={"headcount": len(self.employee_ids)}
        )

    def decide(self, observation: Observation) -> Action:
        """
        Xuat khong gian hanh dong 3 chieu:
        [0]: Cuong do tuyen dung / sa thai (Hire/Fire Intensity): [-1.0, 1.0]
        [1]: Cuong do yeu cau vay von ngan hang (Borrow Intensity): [0.0, 1.0]
        [2]: Ty le khai bao loi nhuan dong thue (Declare Ratio): [0.0, 1.0]
        """
        if "injected_action" in observation.metadata:
            raw_action = observation.metadata["injected_action"]
        else:
            # Heuristic: neu con du tien thi co gang tuyen dung nhe, nguoc lai giam quy mo
            hire_signal = 0.2 if self.cash > 500.0 else -0.2
            borrow_signal = 0.1 if self.cash < 200.0 else 0.0
            raw_action = np.array([hire_signal, borrow_signal, 1.0], dtype=np.float32)

        return Action(
            agent_id=self.agent_id,
            action_type="PRODUCTION_AND_FINANCE",
            values=np.asarray(raw_action, dtype=np.float32)
        )

    def validate_action(self, action: Action) -> ValidationResult:
        vals = action.values
        if len(vals) < 3:
            return ValidationResult(
                is_valid=False,
                sanitized_values=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                reason="Action dimensions must be exactly 3"
            )

        hire_fire = float(np.clip(vals[0], -1.0, 1.0))
        borrow_intensity = float(np.clip(vals[1], 0.0, 1.0))
        declare_ratio = float(np.clip(vals[2], 0.0, 1.0))

        # Neu doanh nghiep khong con tien mat, khong the ra quyet dinh tuyen dung moi
        if self.cash <= 0.0 and hire_fire > 0.0:
            hire_fire = 0.0

        sanitized = np.array([hire_fire, borrow_intensity, declare_ratio], dtype=np.float32)
        return ValidationResult(is_valid=True, sanitized_values=sanitized)

    def apply_result(self, transition_result: TransitionResult) -> None:
        delta = transition_result.state_delta
        self.cash += float(delta.get("cash_delta", 0.0))
        self.capital_stock += float(delta.get("capital_delta", 0.0))
        self.debt = float(max(0.0, self.debt + delta.get("debt_delta", 0.0)))

        # Cap nhat danh sach nhan su thuc te do RuleEngine dieu phoi
        if "hired_employees" in delta:
            self.employee_ids.extend(delta["hired_employees"])
        if "fired_employees" in delta:
            for emp_id in delta["fired_employees"]:
                if emp_id in self.employee_ids:
                    self.employee_ids.remove(emp_id)

        self.last_revenue = float(delta.get("executed_revenue", 0.0))
        self.last_profit = float(delta.get("executed_profit", 0.0))
        self.last_declare_ratio = float(delta.get("executed_declare_ratio", 1.0))

        if "creditor_bank_id" in delta:
            self.creditor_bank_id = delta["creditor_bank_id"]

        # LOI DA SUA (v0.20, xem KNOWN_PATHOLOGIES.md muc moi + CLAUDE_HISTORY.md):
        # truoc day o day KHONG CO dong dong bo "status" tu delta (khac han
        # Employee.apply_result da co san dong nay) -- nghia la
        # deltas[firm.agent_id]["status"] = LifeCycleStatus.BANKRUPT ma
        # rule_engine.py Section 8 gan sau khi tinh dung mo hinh Merton (1974)
        # (thu hoi tai san, xoa no dung ngan hang chu no, sa thai nhan vien...)
        # CHUA BAO GIO DUOC DOC -- la dead code hoan toan. Trang thai BANKRUPT
        # cua chinh Firm truoc day CHI co the den tu dieu kien rieng, khong
        # trich dan, o ngay duoi day ("cash < 0 VA debt > 2*capital+500"), voi
        # mot bat loi nghiem trong: dieu kien nay doc self.debt SAU KHI da bi
        # rule_engine xoa ve 0 qua debt_delta (dong ben tren) trong dung buoc
        # rule_engine phat hien mat kha nang thanh toan -- nen hau nhu KHONG
        # BAO GIO tu no trung dung luc rule_engine da flag, khien Firm tro
        # thanh "zombie": no bi xoa, nhan vien bi sa thai dung, ngan hang duoc
        # hach toan NPL dung, nhung CHINH firm van "ACTIVE" mai mai, khong bao
        # gio duoc don khoi self.agents/khong bao gio nhan terminated=True cho
        # RLlib (cung lop loi da sua cho Employee/Firm o v0.11, xem
        # KNOWN_PATHOLOGIES.md muc 4). Sua: them dong dong bo "status" chuan
        # (giong Employee), xoa han dieu kien tu-pha-san rieng -- MOT nguon su
        # that duy nhat la rule_engine.py Section 8 (Merton, 1974), dung theo
        # dung hop dong kien truc BaseAgent da tuyen bo (be/agents/base_agent.py
        # dong 10: "khong chua luat kinh te").
        if "status" in delta:
            self.status = delta["status"]

    def calculate_reward(self, transition_result: TransitionResult) -> float:
        if self.status == LifeCycleStatus.BANKRUPT or self.status == LifeCycleStatus.TERMINATED:
            return -100.0

        # Profit reward: scale 0.1 để profit ~20-50/month -> reward ~2-5 (positive signal rõ ràng)
        profit_reward = self.last_profit * 0.1

        # Thưởng tạo việc làm — CHỈ khi lợi nhuận dương (Mortensen, D. T., &
        # Pissarides, C. A. (1994), "Job Creation and Job Destruction", Review
        # of Economic Studies 61(3), 397-415 -- job creation là ngoại tác xã
        # hội tích cực đáng được khuyến khích, cùng khung lý thuyết đã dùng
        # cho điều kiện gia nhập ngành ở env.py). Trước bản vá này, bonus vô
        # điều kiện (0.5 × headcount bất kể lãi/lỗ) tạo tín hiệu redundant và
        # có thể XUNG ĐỘT với chính cơ chế MRPL đã có ở rule_engine.py Section
        # 2: một firm thuê vượt mức MRPL biện minh (lợi nhuận âm do quỹ lương
        # vượt doanh thu biên) vẫn có thể được thưởng ròng dương nếu
        # headcount_bonus đủ lớn để bù profit_reward âm -- phát hiện qua audit
        # thực nghiệm, không có cơ sở lý thuyết nào biện minh cho việc thưởng
        # headcount KHÔNG ĐIỀU KIỆN. Gate theo last_profit > 0 giữ đúng tinh
        # thần khuyến khích tạo việc làm nhưng chỉ khi việc đó thực sự bền
        # vững về tài chính.
        headcount_bonus = (float(len(self.employee_ids)) * 0.5) if self.last_profit > 0.0 else 0.0

        # Phạt đạo đức nếu trốn thuế
        moral_cost = 2.0 * self.tax_morale * ((1.0 - self.last_declare_ratio) ** 2)

        # Phạt rủi ro đòn bẩy tài chính quá cao (Debt / Capital)
        leverage_penalty = 0.0
        if self.capital_stock > 0:
            leverage_ratio = self.debt / self.capital_stock
            if leverage_ratio > 1.5:
                leverage_penalty = (leverage_ratio - 1.5) * 3.0

        reward = profit_reward + headcount_bonus - moral_cost - leverage_penalty
        return float(np.clip(reward, -50.0, 50.0))

    def export_state(self) -> Dict[str, Any]:
        # Loai bo cac ID ma hoac trung lap
        valid_ids = list(set(self.employee_ids))
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "status": self.status.name,
            "cash": round(self.cash, 1),
            "capital_stock": round(self.capital_stock, 1),
            "debt": round(self.debt, 1),
            "headcount": len(valid_ids),
            "last_profit": round(self.last_profit, 1),
            "last_revenue": round(self.last_revenue, 1),
            "creditor_bank_id": self.creditor_bank_id
        }

    def reset(self) -> None:
        super().reset()
        self.cash = 0.0
        self.capital_stock = 0.0
        self.debt = 0.0
        self.employee_ids.clear()
        self.creditor_bank_id = None
        self.last_profit = 0.0
        self.last_revenue = 0.0

    def terminate(self, reason: str = "") -> None:
        super().terminate(reason)
        self.status = LifeCycleStatus.BANKRUPT
        self.employee_ids.clear()