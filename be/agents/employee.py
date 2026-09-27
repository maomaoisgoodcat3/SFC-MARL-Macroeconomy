from typing import Dict, Any, Optional
import numpy as np
from be.core.enums import LifeCycleStatus, AgentType
from be.core.types import Observation, Action, ValidationResult, TransitionResult
from be.agents.base_agent import BaseAgent

class Employee(BaseAgent):
    """
    Tac tu Nguoi lao dong (Employee / Citizen) v1.3.
    - Tiep nhan dong bo status va skill_delta tu RuleEngine.
    - Ham Reward phat that nghiep lien tuc de tao dong luc tim viec.
    """
    def __init__(self, agent_id: str, death_penalty_base: float = 100.0,
                 death_penalty_horizon_multiplier: float = 1.0):
        super().__init__(agent_id)
        self.agent_type = AgentType.EMPLOYEE

        # Dac tinh noi tai
        self.skill_level: float = 1.0
        self.risk_aversion: float = 0.5
        self.tax_morale: float = 0.8
        self.age: int = 20
        self.max_age: int = 75

        # He so hieu chinh reward tu vong (calibration constants, xem
        # calculate_reward) -- co the cau hinh qua ScenarioConfig, KHONG doi
        # dang ham reward, chi doi gia tri hang so dau vao. Truyen o
        # constructor (khong phai initialize()) vi Employee duoc tao lai moi
        # lan reset()/sinh san va he so calibration khong nen thay doi giua
        # cac episode, phai giu dung theo scenario dang chay -- cung pattern
        # da dung cho Government.gini_penalty_coef/death_penalty_coef.
        self.death_penalty_base: float = float(death_penalty_base)
        self.death_penalty_horizon_multiplier: float = float(death_penalty_horizon_multiplier)
        
        # Chi so tai chinh va the ly
        self.cash: float = 0.0
        self.bank_deposit: float = 0.0  # So du tien gui ngan hang (Section 8B, rule_engine.py)
        # Quan hệ tiền gửi với MỘT ngân hàng cụ thể khi hệ thống có nhiều ngân
        # hàng (cùng logic relationship banking như Firm.creditor_bank_id, xem
        # Petersen & Rajan, 1994).
        self.depository_bank_id: Optional[str] = None
        # Quan he TIN DUNG (vay, khac voi depository_bank_id la tien GUI) voi
        # MOT ngan hang cu the khi he thong co nhieu ngan hang -- cung logic
        # relationship banking nhu Firm.creditor_bank_id (Petersen & Rajan,
        # 1994). Xem rule_engine.py Section 3B (tin dung tieu dung khong the
        # chap, v0.23).
        self.creditor_bank_id: Optional[str] = None
        # Metadata thuan tuy phuc vu truy vet pha he cho nghien cuu (KHONG
        # thuoc observation/action space, khong anh huong bat ky cong thuc
        # kinh te nao) -- agent_id cua cha/me neu sinh ra qua co che tai san
        # xuat noi sinh Sugarscape (Epstein & Axtell, 1996, xem env.py Section
        # A); None neu sinh ra qua nhanh an sinh khan cap hoac la the he goc
        # luc khoi tao simulation.
        self.parent_id: Optional[str] = None
        self.energy: float = 1.0
        self.employed_by: Optional[str] = None
        self.wage: float = 0.0
        self.debt: float = 0.0
        self.unemployed_streak: int = 0
        # Moc timestep dau tien duoc phep tuyen lai sau khi bi sa thai vi lam luoi (dau an sa thai,
        # Gibbons & Katz 1991 -- xem rule_engine.py Section 2/3). -1 = khong co dau an. TRANG THAI AN:
        # khong nam trong observation (giong unemployed_streak).
        self.hire_lockout_until: int = -1

        # Luu tru hanh dong gan nhat
        self.last_work_effort: float = 0.0
        self.last_declare_ratio: float = 1.0
        self.last_consumption: float = 0.0

    def initialize(self, 
                   skill_level: float = 1.0, 
                   risk_aversion: float = 0.5, 
                   tax_morale: float = 0.8,
                   initial_cash: float = 300.0,
                   initial_energy: float = 1.0,
                   age: int = 20) -> None:
        self.skill_level = float(skill_level)
        self.risk_aversion = float(risk_aversion) if risk_aversion != 1.0 else 0.99
        self.tax_morale = float(tax_morale)
        self.cash = float(initial_cash)
        self.bank_deposit = 0.0
        self.depository_bank_id = None
        self.energy = float(initial_energy)
        self.age = int(age)
        self.employed_by = None
        self.wage = 0.0
        self.debt = 0.0
        self.creditor_bank_id = None
        self.unemployed_streak = 0
        self.hire_lockout_until = -1
        # Chuyen ngay sang ACTIVE khi da duoc nap thong so
        self.status = LifeCycleStatus.ACTIVE

    def observe(self, raw_environment_state: Dict[str, Any]) -> Observation:
        macro = raw_environment_state.get("macro_indicators", {})
        inflation = float(macro.get("inflation", 0.0))
        living_cost = float(macro.get("base_living_cost", 15.0))
        tax_rate = float(macro.get("worker_tax_rate", 0.15))
        interest_rate = float(macro.get("base_interest_rate", 0.05))
        unemployment_rate = float(macro.get("unemployment_rate", 0.0))
        
        # Khong gian quan sat 15 chieu: [0-4] vi mo, [5] tien mat, [6] tien gui
        # ngan hang (Section 8B, rule_engine.py), [7] the luc, [8] bac ky nang,
        # [9] luong, [10] dang co viec lam, [11] tuoi, [12] no, [13] risk_aversion,
        # [14] tax_morale.
        #
        # LOI DA SUA cho [13]/[14] (v0.28, phat hien qua audit chu dong theo yeu
        # cau nguoi dung 2026-09-25): risk_aversion anh huong that toi rule_engine.py
        # (risk_discount cho vay tieu dung, deposit_buffer_target -- Kimball 1990, xem
        # Section 3B/8B) va tax_morale anh huong truc tiep toi calculate_reward() ben
        # duoi (moral_cost) -- CA HAI deu di bien theo tung Employee (khoi tao ngau
        # nhien, di truyen + dot bien qua sinh san, env.py Section A) nhung CHUA BAO
        # GIO duoc dua vao observe(). Vi Employee dung PARAMETER SHARING (MOT mang
        # neural network duy nhat cho moi Employee), mang KHONG THE phan biet dang
        # dieu khien ca the co tax_morale/risk_aversion cao hay thap de toi uu rieng
        # hanh vi khai bao thue/vay no/tieu dung cho tung loai -- buoc phai hoc mot
        # chien luoc trung binh chung, mat kha nang ca the hoa dung ban chat thiet ke
        # "heterogeneous agents" cua mo hinh. Khong dua vao them scale rieng (ca hai
        # da nam san trong [0,1], cung bac do lon voi cac chieu quan sat khac).
        obs_array = np.array([
            inflation,
            living_cost,
            tax_rate,
            interest_rate,
            unemployment_rate,
            self.cash,
            self.bank_deposit,
            self.energy,
            self.skill_level,
            self.wage,
            1.0 if self.employed_by is not None else 0.0,
            float(self.age),
            self.debt,
            self.risk_aversion,
            self.tax_morale,
        ], dtype=np.float32)

        return Observation(
            agent_id=self.agent_id,
            timestep=raw_environment_state.get("timestep", 0),
            vector=obs_array,
            metadata={"employed_by": self.employed_by}
        )

    def decide(self, observation: Observation) -> Action:
        """
        Khong gian hanh dong 4 chieu:
        [0]: Cuong do lam viec (Work Effort): [0.0, 1.0]
        [1]: Ty le khai bao thu nhap dong thue (Declare Ratio): [0.0, 1.0]
        [2]: Ty le tieu dung tren phan thu nhap con lai (Consumption Ratio): [0.0, 1.0]
        [3]: Cuong do vay tin dung tieu dung (Borrow Intensity): [0.0, 1.0] -- MOI (v0.23,
             xem rule_engine.py Section 3B). CHI kich hoat khi tai san thanh khoan (cash +
             tien gui) khong du chi tieu sinh ton thang nay -- khong phai mot "nut bam vay
             tien tuy y" khong dieu kien.
        """
        if "injected_action" in observation.metadata:
            raw_action = observation.metadata["injected_action"]
        else:
            effort = 0.8 if self.employed_by is not None else 0.4
            raw_action = np.array([effort, 1.0, 0.6, 0.0], dtype=np.float32)

        return Action(
            agent_id=self.agent_id,
            action_type="LABOR_AND_CONSUMPTION",
            values=np.asarray(raw_action, dtype=np.float32)
        )

    def validate_action(self, action: Action) -> ValidationResult:
        vals = action.values
        if len(vals) < 4:
            return ValidationResult(
                is_valid=False,
                sanitized_values=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                reason="Invalid action vector length"
            )

        work_effort = float(np.clip(vals[0], 0.0, 1.0))
        if self.energy < 0.2:
            work_effort = min(work_effort, self.energy)

        declare_ratio = float(np.clip(vals[1], 0.0, 1.0))
        consumption_ratio = float(np.clip(vals[2], 0.0, 1.0))
        borrow_intensity = float(np.clip(vals[3], 0.0, 1.0))

        sanitized = np.array([work_effort, declare_ratio, consumption_ratio, borrow_intensity], dtype=np.float32)
        return ValidationResult(is_valid=True, sanitized_values=sanitized)

    def apply_result(self, transition_result: TransitionResult) -> None:
        delta = transition_result.state_delta
        self.cash += float(delta.get("cash_delta", 0.0))
        self.bank_deposit = float(max(0.0, self.bank_deposit + delta.get("deposit_delta", 0.0)))
        if "depository_bank_id" in delta:
            self.depository_bank_id = delta["depository_bank_id"]
        if "creditor_bank_id" in delta:
            self.creditor_bank_id = delta["creditor_bank_id"]
        self.energy = float(np.clip(self.energy + delta.get("energy_delta", 0.0), 0.0, 2.0))
        self.debt = float(max(0.0, self.debt + delta.get("debt_delta", 0.0)))
        
        # 1. DONG BO TRANG THAI TU RULE ENGINE
        if "status" in delta:
            self.status = delta["status"]

        # 2. CAP NHAT BIEN DONG KY NANG VA CHUOI THAT NGHIEP
        if "skill_delta" in delta:
            self.skill_level = float(np.clip(self.skill_level + delta["skill_delta"], 0.4, 4.0))
        if "unemployed_streak" in delta:
            self.unemployed_streak = int(delta["unemployed_streak"])
        if "hire_lockout_until" in delta:
            self.hire_lockout_until = int(delta["hire_lockout_until"])

        if "employed_by" in delta:
            self.employed_by = delta["employed_by"]
        if "wage" in delta:
            self.wage = float(delta["wage"])
        if "age_increment" in delta:
            self.age += int(delta["age_increment"])

        self.last_work_effort = float(delta.get("executed_work_effort", 0.0))
        self.last_declare_ratio = float(delta.get("executed_declare_ratio", 1.0))
        self.last_consumption = float(delta.get("executed_consumption", 0.0))

        # LOI DA SUA (v0.20, xem KNOWN_PATHOLOGIES.md muc moi + CLAUDE_HISTORY.md):
        # truoc day o day co MOT dieu kien tu vong RIENG cua chinh Employee
        # ("if self.cash < -300.0 or self.energy <= 0.0 or self.age >= self.max_age:
        # self.terminate(...)"), doc lap va KHAC CONG THUC voi dieu kien tu vong
        # THAT co trich dan (Shapiro & Stiglitz, 1984) o rule_engine.py Section 9
        # ("cash < -10*actual_living_cost VA energy < 0.15", co giai theo gia --
        # ngan cach voi -300.0 co dinh o day). Khi hai dieu kien lech nhau (de
        # xay ra nhat luc lam phat cao, vi nguong rule_engine gian theo living_cost
        # con -300.0 o day dung yen), Employee tu status=TERMINATED nhung
        # rule_engine KHONG he biet -- Government.new_deaths bo sot cai chet nay
        # (khong bi phat dung), va death_remaining_horizon_ratio mac dinh 0.0 khien
        # Employee chi nhan dung -100 phang thay vi phat theo tuoi nhu thiet ke.
        # Sua: XOA HAN kiem tra rieng nay, chi con MOT nguon su that duy nhat cho
        # tu vong (rule_engine.py Section 9, dong bo qua khoa "status" o tren) --
        # dung theo dung hop dong kien truc ma chinh BaseAgent da tuyen bo
        # ("Khong chua dinh danh tinh, khong chua luat kinh te, chi quan ly vong
        # doi", be/agents/base_agent.py dong 10): Employee KHONG duoc tu quyet
        # dinh luat kinh te (bao gom ca dieu kien tu vong) cho rieng minh.

    def calculate_reward(self, transition_result: TransitionResult) -> float:
        if self.status in [LifeCycleStatus.TERMINATED, LifeCycleStatus.DECEASED]:
            # HÌNH PHẠT TỬ VONG QUY MÔ THEO QUÃNG ĐỜI CÒN LẠI -- Viscusi &
            # Aldy (2003); Aldy & Viscusi (2008) (xem trích dẫn đầy đủ tại nơi
            # tính death_remaining_horizon_ratio, rule_engine.py Section 9).
            # Phát hiện qua audit (test_death_math.py, xét trên toàn bộ công
            # thức thật): một agent thất nghiệp có thu nhập phi chính thức đủ
            # sống (≈0.8-1.2x living_cost) có thể tích lũy TỔNG phạt âm hơn
            # NHIỀU LẦN so với mức chết cố định -100 (vì RLlib
            # MultiAgentEpisode.get_return() CỘNG DỒN toàn bộ phần thưởng
            # trong vòng đời, không loại trừ theo kiểu survivorship) -- khiến
            # "chết sớm" trông rẻ hơn "sống khổ kéo dài" một cách phi lý. Chết
            # ở đúng max_age (hết tuổi thọ tự nhiên, không còn quãng đời nào
            # để mất) giữ nguyên đúng -100 (ratio=0); chết càng trẻ (còn nhiều
            # quãng đời lẽ ra được sống) thì phạt càng nặng, tối đa gấp
            # (1 + multiplier) lần. death_penalty_horizon_multiplier mặc định
            # 1.0 là lựa chọn THẬN TRỌNG ban đầu (biên trên -200), KHÔNG phải
            # con số suy ra để triệt tiêu hoàn toàn kịch bản tệ nhất đã đo
            # (-854) -- cố ý để trống làm ScenarioConfig field, hiệu chỉnh lại
            # bằng dữ liệu training thật SAU KHI Mục 1/3/4 (NPL write-off,
            # headcount bonus gating, hire cap) đã chạy, thay vì suy luận từ
            # một kịch bản giả lập 400 tháng chưa chắc còn xảy ra trong
            # equilibrium đã cải thiện.
            horizon_ratio = float(transition_result.state_delta.get("death_remaining_horizon_ratio", 0.0))
            return float(-self.death_penalty_base * (1.0 + self.death_penalty_horizon_multiplier * horizon_ratio))

        # Utility tiêu dùng CRRA: scale bằng log để tránh số cực lớn
        # ln(consumption) đơn giản hơn và stable hơn CRRA khi consumption dao động mạnh
        consumption = max(0.01, self.last_consumption)
        u_consumption = float(np.log(consumption)) * 2.0

        # Chi phí mất thỏa dụng lao động
        disutility_labor = (self.last_work_effort ** 2.0) * 1.0
        moral_cost = 1.5 * self.tax_morale * ((1.0 - self.last_declare_ratio) ** 2)

        # Phạt thất nghiệp: tăng dần nhưng cap ở 8.0 để không dominate toàn bộ signal
        unemployment_penalty = 0.0
        if self.employed_by is None:
            unemployment_penalty = min(2.0 + (0.5 * self.unemployed_streak), 8.0)

        total_utility = u_consumption - disutility_labor - moral_cost - unemployment_penalty
        return float(np.clip(total_utility, -50.0, 50.0))

    def export_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "status": self.status.name,
            "cash": self.cash,
            "bank_deposit": round(self.bank_deposit, 1),
            "energy": self.energy,
            "skill_level": round(self.skill_level, 2),
            "employed_by": self.employed_by,
            "wage": round(self.wage, 1),
            "debt": self.debt,
            "creditor_bank_id": self.creditor_bank_id,
            "age": self.age,
            "unemployed_streak": self.unemployed_streak,
            "parent_id": self.parent_id
        }

    def reset(self) -> None:
        super().reset()
        self.cash = 0.0
        self.bank_deposit = 0.0
        self.depository_bank_id = None
        self.creditor_bank_id = None
        self.parent_id = None
        self.energy = 1.0
        self.employed_by = None
        self.wage = 0.0
        self.debt = 0.0
        self.age = 20
        self.unemployed_streak = 0
        self.hire_lockout_until = -1

    def terminate(self, reason: str = "") -> None:
        super().terminate(reason)
        self.status = LifeCycleStatus.DECEASED
        self.employed_by = None
        self.wage = 0.0