from typing import Dict, Any
import numpy as np
from be.core.enums import LifeCycleStatus, AgentType
from be.core.types import Observation, Action, ValidationResult, TransitionResult
from be.agents.base_agent import BaseAgent

class Economy(BaseAgent):
    """
    Tac tu Economy / Co quan Binh on Thi truong (Market Stabilization Authority).

    v0.24 -- khung "BA TANG TRACH NHIEM" (mo rong tu hai tang Inner-Outer Loop cua Zheng et al.,
    2022, xem METHODOLOGY_NOTES.md muc 1): Economy KHONG con la tac tu ich ky, ma la mot PLANNER
    o quy mo VI MO (an sinh ca the: ty le song sot, on dinh ty le sinh), song song voi Government
    la planner o quy mo VI MO (KPI tong hop: GDP thuc, Gini, no cong). Day dai dien thi truong/gia
    ca chung (chi phi sinh hoat, gia nha dat -- thu Gov/Bank khong truc tiep kiem soat), dung DUNG
    y dinh thiet ke goc, KHONG phai ngan hang trung uong (xem CLAUDE.md "Quyet dinh thiet ke da
    can nhac va tu choi").

    Cong cu chinh: co che binh on gia bang du tru dem (buffer-stock, Newbery & Stiglitz 1981; xem
    rule_engine.py Section 4D) va (tuong lai) quan ly kho nha o/dat.
    Tuan thu nghiem ngat contract BaseAgent theo SAS v1.0.
    """
    def __init__(self, agent_id: str, mortality_rate_floor: int = 30):
        super().__init__(agent_id)
        self.agent_type = AgentType.ECONOMY

        # He so hieu chinh reward (calibration constant, xem calculate_reward) -- truyen o
        # constructor (khong phai initialize()) vi Economy duoc tao lai moi lan reset()
        # (_create_world()), cung pattern da dung cho Government.gini_penalty_coef.
        # KHONG tai dung HARD_MIN_EMP (hang so rieng cho luoi an sinh dan so khan cap trong
        # env.py) -- xem METHODOLOGY_NOTES.md muc 3 (rui ro coupling an giua 2 co che khong
        # thiet ke de phoi hop, cung lop rui ro da gap voi capital_stock dung chung 6 cong thuc).
        self.mortality_rate_floor: int = int(mortality_rate_floor)

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

        # BANG CAN DOI KE TOAN RIENG cho co che binh on du tru dem (buffer-stock, v0.24) --
        # KHAC Treasury, dam bao SFC: xem rule_engine.py Section 4D + METHODOLOGY_NOTES.md muc 2.
        self.strategic_reserve_fund: float = 0.0   # tien, cap MOT LAN tu Treasury luc reset
        self.strategic_reserve_stock: float = 0.0  # hang thiet yeu (don vi vat ly), bat dau = 0

        # Ty le sinh KY TRUOC (v0.24, dung cho reward an sinh vi mo, xem calculate_reward).
        # LY DO DUNG DU LIEU KY TRUOC (khong phai ky nay): births_this_step duoc tinh trong
        # env.py Section A, chay SAU khi calculate_reward() cua TOAN BO agent (bao gom Economy)
        # da chay xong trong step() -- rule_engine.execute_cycle() khong he biet gi ve sinh san
        # (co che nay hoan toan thuoc env.py, khong phai rule_engine.py). env.py ghi truc tiep
        # vao day (khong qua delta) ngay sau khi tinh xong births_this_step moi buoc -- cung do
        # tre 1 buoc nhu purchase_base cua Government da dung cho ly do tuong tu (thue ky nay chi
        # tinh xong SAU khi chi tieu da chot).
        self.last_births_this_step: int = 0

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
        [0]: CUONG DO CAN THIEP BINH ON DU TRU DEM (Buffer-Stock Intervention Intensity):
             [-1.0, 1.0] -- MOI (v0.24, xem rule_engine.py Section 4D + METHODOLOGY_NOTES.md
             muc 2). Duong = MUA (ho tro gia khi giam phat, chi tu strategic_reserve_fund);
             am = BAN (ha gia khi lam phat, giai phong strategic_reserve_stock). Thay the
             cho action[0] "reserved" tu v0.22-v0.23 (khi Economy hoan toan mat don bay anh
             huong den chinh reward cua no sau khi demand_injection_ratio chuyen sang
             Government -- xem CLAUDE.md).
        [1]: He so dieu chinh gia bat dong san (Housing Price Factor): [0.90, 1.10]
             -- CHUA CO HIEU LUC: cho toi khi co market nha dat/dau gia that
             (xem ke hoach mo rong tuong lai housing epic), day van la
             placeholder no-op -- DUNG y dinh thiet ke goc cua Economy.
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
                sanitized_values=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                reason="Economy action vector must contain 3 elements"
            )

        intervention_intensity = float(np.clip(vals[0], -1.0, 1.0))
        housing_price_factor = float(np.clip(vals[1], 0.90, 1.10))
        housing_supply = float(np.clip(vals[2], 0.0, 10.0))

        sanitized = np.array([intervention_intensity, housing_price_factor, housing_supply], dtype=np.float32)
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
        # (v0.22: da bo doc "demand_injection_ratio"/"demand_injection_effect" o day --
        # don bay nay chuyen sang Government, xem decide() va KNOWN_PATHOLOGIES.md.)

        # [1]/[2]: Housing price/supply -- CHUA CO HIEU LUC (xem decide()),
        # cac key executed_housing_factor/executed_housing_supply khong duoc
        # RuleEngine ghi nen day van la no-op, giu nguyen cho toi khi co market
        # nha dat that.
        price_mult = float(delta.get("executed_housing_factor", 1.0))
        new_housing = int(delta.get("executed_housing_supply", 0))

        self.housing_price = float(np.clip(self.housing_price * price_mult, 100.0, 50000.0))
        self.housing_inventory = max(0, self.housing_inventory + new_housing - int(delta.get("houses_sold", 0)))

        # Buffer-stock (v0.24, rule_engine.py Section 4D) -- xem chu thich day du tai __init__.
        self.strategic_reserve_fund = max(0.0, self.strategic_reserve_fund + delta.get("strategic_reserve_fund_delta", 0.0))
        self.strategic_reserve_stock = max(0.0, self.strategic_reserve_stock + delta.get("strategic_reserve_stock_delta", 0.0))

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
        Mục tiêu An Sinh Vi Mô (Micro Social Welfare) — v0.24, xem METHODOLOGY_NOTES.md mục 1/3
        cho toàn bộ lập luận "khung ba tầng trách nhiệm" (mở rộng Zheng et al., 2022) + kiểm chứng
        số học cho từng thành phần TRƯỚC khi viết vào đây.

        Nguồn:
        - Real vs nominal GDP distinction: Hicks (1946), "Value and Capital"
        - Market maker penalized for demand destruction: Tirole (1988),
          "The Theory of Industrial Organization", MIT Press, Ch.1
        - Inflation target penalty (quadratic loss): Svensson (1997),
          "Inflation Forecast Targeting", European Economic Review 41(6), 1111-1146
        - Tỷ lệ sống sót: Sen, A. (1998), "Mortality as an Indicator of Economic Success and
          Failure", Economic Journal 108(446), 1-25.
        - Ổn định tỷ lệ sinh (dạng hàm quadratic-loss, nhất quán với inflation_penalty phía
          trên): không có trích dẫn cho TỶ LỆ MỤC TIÊU cụ thể (hệ số cấu trúc tự do hiệu chỉnh).
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

        # 3. TỶ LỆ SỐNG SÓT (Sen, 1998) — THAY THẾ HOÀN TOÀN dead_worker_penalty cũ (đếm tuyệt
        #    đối, méo theo quy mô dân số — không cộng thêm song song, tránh đếm trùng cùng hiện
        #    tượng tử vong hai lần trong reward, xem METHODOLOGY_NOTES.md mục 3).
        #    LỖI ĐÃ SỬA (v0.24, phát hiện khi nối reward mới): dead_worker_penalty cũ đọc
        #    "new_deaths" từ transition_result CỦA CHÍNH Economy, nhưng rule_engine.py TRƯỚC ĐÂY
        #    chỉ ghi khoá này vào deltas[gov.agent_id] — nghĩa là dead_worker_penalty đã LUÔN
        #    LUÔN = 0, chưa từng thực sự hoạt động (xem KNOWN_PATHOLOGIES.md). Đã sửa rule_engine
        #    ghi thêm vào deltas[eco.agent_id] (Section 9).
        #    SÀN mẫu số (mortality_rate_floor, mặc định 30, field RIÊNG — KHÔNG dùng chung
        #    HARD_MIN_EMP để tránh coupling ẩn, xem METHODOLOGY_NOTES.md mục 3): đây là
        #    VARIANCE-REDUCTION có chủ đích cho ổn định huấn luyện PPO, ĐÁNH ĐỔI lấy một phần độ
        #    chính xác kinh tế học ở vùng dân số dưới sàn — KHÔNG phải "vẫn đúng Sen ở mọi vùng".
        #    Hệ số K_SURVIVAL=1700.0 hiệu chỉnh để khớp thang đo dead_worker_penalty cũ (~20) tại
        #    quy mô dân số "bình thường" (~85-90, gần khớp log training gần nhất).
        new_deaths = int(transition_result.state_delta.get("new_deaths", 0))
        active_population = int(transition_result.state_delta.get("active_population_for_reward", 0))
        K_SURVIVAL = 1700.0
        death_rate = float(new_deaths) / max(active_population, self.mortality_rate_floor)
        survival_penalty = K_SURVIVAL * death_rate

        # 3b. ỔN ĐỊNH TỶ LỆ SINH (MỚI, chưa có tiền lệ) — quadratic-loss quanh mục tiêu thay thế
        #     dân số, nhất quán dạng hàm với inflation_penalty (Svensson, 1997), KHÔNG bịa dạng
        #     hàm mới. Dùng births_this_step KỲ TRƯỚC (self.last_births_this_step, xem __init__)
        #     vì sinh sản tính trong env.py Section A, chạy SAU calculate_reward() của bước này.
        #
        #     K_BIRTH ĐÃ HIỆU CHỈNH LẠI (v0.24, ngay sau smoke-test training thật đầu tiên --
        #     đúng tinh thần "để trống làm hằng số cấu trúc, hiệu chỉnh lại SAU KHI có dữ liệu
        #     training thật" đã ghi trước đó, và giờ ĐÃ CÓ dữ liệu đó). Bằng chứng: bản đầu
        #     K_BIRTH=1_000_000 khiến `vf_explained_var` của policy_economy ≈ 0 tuyệt đối suốt cả
        #     8 iteration -- nguyên nhân: births_this_step là số NGUYÊN rời rạc, chỉ CẦN 1 ca sinh
        #     trên dân số ~40 đã cho birth_rate=0.025 (gấp ~7,5 lần target 0.0033), birth_penalty
        #     = 1_000_000*(0.025-0.0033)^2 ≈ 471 -- vượt xa trần clip 50 của chính hàm reward này,
        #     khiến reward Economy gần như luôn bị KẸP CỨNG ở đáy mỗi khi có sinh sản (xảy ra
        #     ~1/4-1/5 số bước theo log thật) thay vì phản ánh đúng mức độ lệch. Hạ xuống
        #     K_BIRTH=4000 (birth_penalty tại rate=0.025 ≈ 1,9; tại rate=0.05 ≈ 8,7 -- cùng bậc độ
        #     lớn với inflation_penalty, không còn áp đảo toàn bộ ngân sách reward). ĐÃ XÁC NHẬN
        #     bằng smoke-test thứ 2 (cùng seed/scenario, chỉ đổi K_BIRTH): reward_mean policy_economy
        #     chuyển từ [-34,-29] (kẹp cứng, gần như hằng số) sang [+8.7,+14.0] (tăng dần, phản ánh
        #     đúng cải thiện chính sách) -- xác nhận hết clip-saturation.
        #     GIỚI HẠN CỦA FIX NÀY (quan trọng, đọc trước khi coi đây là "đã xong"): thoát
        #     clip-saturation KHÔNG kéo `vf_explained_var` lên -- đo lại ở smoke-test thứ 2 vẫn
        #     ~1e-5 suốt cả 8 iteration, KHÔNG khác biệt so với bản K_BIRTH=1_000_000 (cả hai đều
        #     dao động quanh 0, có iteration âm). Tức là "vf_explained_var≈0" và "reward bị kẹp
        #     cứng vì K_BIRTH quá lớn" là HAI triệu chứng riêng, KHÔNG PHẢI cùng nguyên nhân như
        #     suy đoán ban đầu (suy đoán này SAI, ghi lại để không lặp lại nhầm lẫn). Đối chiếu
        #     cùng smoke-test: policy_government và policy_supervisor (cũng đúng 1 instance/env,
        #     reward vĩ mô tổng hợp) có vf_explained_var≈0 GIỐNG HỆT, trong khi policy_employee
        #     (0,34→0,80), policy_firm (0,05→0,22), policy_bank (0,0001→0,14, dù cũng chỉ 1
        #     instance) học tốt trong đúng 8 iteration đó. Giả thuyết CHƯA kiểm chứng: 3 policy
        #     "tầng hoạch định" có reward phụ thuộc chủ yếu vào trạng thái tổng hợp/ngoại sinh
        #     (GDP, Gini, nợ công, sinh/tử) hơn là hành động riêng của chính nó mỗi bước -> bài
        #     toán gán tín dụng (credit assignment) khó hơn nhiều, có thể đơn giản là CẦN nhiều
        #     iteration hơn 8 mới hội tụ, không nhất thiết là lỗi thiết kế. CHƯA quyết định hướng
        #     xử lý (không đụng gì thêm nếu chưa có dữ liệu run dài thật) -- xem
        #     KNOWN_PATHOLOGIES.md mục 12 (bổ sung) để theo dõi tiếp khi có run dài.
        K_BIRTH = 4000.0
        target_birth_rate = 0.0033  # ~khớp Births~0.25/step tại dan so ~75 quan sat trong log gan day
        birth_rate = float(self.last_births_this_step) / max(active_population, self.mortality_rate_floor)
        birth_penalty = K_BIRTH * ((birth_rate - target_birth_rate) ** 2)

        # 4. Housing inventory penalty (giữ nguyên)
        housing_penalty = 0.0
        if self.housing_inventory < 5:
            housing_penalty = float(5 - self.housing_inventory) * 2.0

        reward = real_volume_reward - inflation_penalty - survival_penalty - birth_penalty - housing_penalty
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
            "strategic_reserve_fund": round(self.strategic_reserve_fund, 1),
            "strategic_reserve_stock": round(self.strategic_reserve_stock, 2)
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
        self.strategic_reserve_fund = 0.0
        self.strategic_reserve_stock = 0.0
        self.last_births_this_step = 0

    def terminate(self, reason: str = "") -> None:
        super().terminate(reason)