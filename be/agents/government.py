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
    def __init__(self, agent_id: str, gini_penalty_coef: float = 25.0, death_penalty_coef: float = 20.0,
                 reward_mode: str = "eq_x_prod", swf_reward_scale: float = 0.02):
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
        # Cuong do bom/rut cau tai khoa phan chu ky (rule_engine.py Section 4B) --
        # chuyen tu Economy sang day o v0.22, xem docstring day du tai decide().
        self.demand_injection_ratio: float = 0.0
        self.last_demand_injection_value: float = 0.0

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
        # DANG REWARD (v0.35, KNOWN_PATHOLOGIES.md #29b) -- LUA CHON MO HINH nen la field ScenarioConfig:
        #   "eq_x_prod" (MAC DINH, DA SUA): reward = swf_reward_scale * (1 - Gini) * GDP_thuc - phat tu vong -
        #       phat no cong -- CAN VOI thuoc do danh gia cua benchmark (be/benchmark.py: Eq x Prod), tuc la
        #       phuc loi xa hoi = cong bang x nang suat (Zheng et al., 2022, "The AI Economist: Taxation
        #       policy design via two-level deep multi-agent reinforcement learning", Science Advances
        #       8(18), eabk2607).
        #   "legacy": dang CU (0.002*GDP_thuc + 0.005*dGDP_thuc - gini_penalty_coef*Gini^2 - tu vong - no
        #       cong) -- KHONG khop thuoc do benchmark (audit 2026-09-26: chinh sach toi uu cho reward != chinh
        #       sach toi uu cho thuoc do danh gia), giu de ablation/tai hien.
        if reward_mode not in ("eq_x_prod", "legacy"):
            raise ValueError(f"reward_mode phai la 'eq_x_prod' hoac 'legacy', nhan '{reward_mode}'")
        self.reward_mode: str = reward_mode
        self.swf_reward_scale: float = float(swf_reward_scale)
        # Muc TRO CAP THAT NGHIEP hien hanh (action[4], xem decide()) -- luu de export/hien thi.
        self.unemployment_relief_level: float = 0.40

    def initialize(self, 
                   initial_treasury: float = 1000000.0,
                   initial_worker_tax: float = 0.15,
                   initial_firm_tax: float = 0.20) -> None:
        self.treasury = float(initial_treasury)
        self.tax_rate_worker = float(initial_worker_tax)
        self.tax_rate_firm = float(initial_firm_tax)
        self.purchase_ratio = 1.0
        self.last_purchase = 0.0
        self.demand_injection_ratio = 0.0
        self.last_demand_injection_value = 0.0
        self.unemployment_relief_level = 0.40
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
        Khong gian hanh dong 5 chieu:
        [0]: Muc tieu Thue suat Thu nhap Ca nhan (Worker Tax Rate): [0.0, 0.5]
        [1]: Muc tieu Thue suat Doanh nghiep (Firm Tax Rate): [0.0, 0.5]
        [2]: Ty le thu thue+phat ky truoc dung de CHI MUA HANG (Purchase Ratio rho): [0.0, 1.0]
        [3]: Cuong do bom/rut CAU tai khoa phan chu ky (Demand Injection Ratio): [-0.20, 0.20]
             -- CHUYEN TU Economy SANG day (v0.22, xem KNOWN_PATHOLOGIES.md muc moi + CLAUDE.md
             "Quyet dinh thiet ke da can nhac"). Ly do: day la MOT trong hai co che chi tieu tai
             khoa cua CUNG mot Kho bac (kenh con lai la [2] o tren) -- truoc day do MOT tac tu
             KHAC (Economy, tu nhan la "Market Maker") quyet dinh doc lap, vi pham truc tiep
             nguyen tac hai tang cua AI Economist (Zheng et al., 2022) da ghi trong CLAUDE.md
             ("chi Government moi mang trach nhiem phuc loi xa hoi qua chinh sach"). Xem
             rule_engine.py Section 4B.
        [4]: Muc TRO CAP THAT NGHIEP (Unemployment Relief Level): [0.0, 1.0] -- MOI (v0.35, KNOWN_PATHOLOGIES.md
             #29a). Truoc day muc tro cap la HANG SO hardcode (0.40 x chi phi sinh hoat trong 3 thang
             dau that nghiep, 0.15 x trong thang 4-6, 0 sau do; rule_engine.py Section 4), nen Government
             KHONG co cong cu an sinh nao du ban than la nguoi duy nhat mang trach nhiem phuc loi xa hoi
             (nguyen tac hai tang, CLAUDE.md) -- reward Government phat tu vong nhung Government khong the
             lam gi de giam tu vong do that nghiep keo dai ngoai chinh sach thue/chi tieu gian tiep.
             Action[4] = muc thay the thu nhap thang dau tien (ty le so voi chi phi sinh hoat); lich giam
             dan theo thoi gian that nghiep (0.375x o thang 4-6, 0 sau do -- ty le 0.15/0.40 cu) giu
             nguyen (Shavell & Weiss, 1979, "The Optimal Payment of Unemployment Insurance Benefits over
             Time", Journal of Political Economy 87(6), 1347-1362 -- loi ich toi uu giam dan theo thoi gian).
        """
        if "injected_action" in observation.metadata:
            raw_action = observation.metadata["injected_action"]
        else:
            # Chinh sach tai khoa can bang mac dinh (rho = 1: chi mua hang bang dung so thu ky
            # truoc; demand_injection_ratio = 0: khong bom/rut them; tro cap = 0.40 -- KHOP hanh vi cu)
            raw_action = np.array([0.15, 0.20, 1.0, 0.0, 0.40], dtype=np.float32)

        return Action(
            agent_id=self.agent_id,
            action_type="FISCAL_POLICY",
            values=np.asarray(raw_action, dtype=np.float32)
        )

    def validate_action(self, action: Action) -> ValidationResult:
        vals = action.values
        if len(vals) < 5:
            return ValidationResult(
                is_valid=False,
                sanitized_values=np.array([0.15, 0.20, 1.0, 0.0, 0.40], dtype=np.float32),
                reason="Action vector must have 5 elements"
            )

        # Gioi han muc thue phu hop hien phap, khong cho phep ap dat thue qua cao triet tieu san xuat
        worker_tax = float(np.clip(vals[0], 0.0, 0.50))
        firm_tax = float(np.clip(vals[1], 0.0, 0.50))

        # rho <= 1: chi mua hang khong vuot qua so thu ky truoc (ngan sach can bang la TRAN; muon
        # them thau chi phai la mot quyet dinh thiet ke rieng). Tran theo so du Kho bac duoc ap
        # tai rule_engine.py Section 4C nen khong can kep them o day.
        purchase_ratio = float(np.clip(vals[2], 0.0, 1.0))

        # He so cau truc TU DO HIEU CHINH: bien do bom/rut cau toi da +-20% tong cau sinh ton co
        # so cua dan so dang hoat dong (xem rule_engine.py Section 4B) -- KHONG suy ra truc tiep
        # tu Blanchard & Perotti (2002), chi xac lap huong tac dong; gia tri +-20% giu nguyen tu
        # ban thiet ke o Economy truoc khi chuyen sang day (v0.22).
        demand_injection_ratio = float(np.clip(vals[3], -0.20, 0.20))

        # Muc tro cap that nghiep [0, 1] (xem decide()). Bien tren 1.0 = thay the 100% chi phi sinh hoat
        # (tuong duong "khong con dong co tim viec", HE SO CAU TRUC TU DO HIEU CHINH). Rang buoc ngan
        # sach THAT (Kho bac > 1000 moi duoc chi, rule_engine.py Section 4) tu chan viec chi vo han.
        relief_level = float(np.clip(vals[4], 0.0, 1.0))

        sanitized = np.array([worker_tax, firm_tax, purchase_ratio, demand_injection_ratio, relief_level], dtype=np.float32)
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
        # Chi phi bom/rut cau tai khoa phan chu ky do CHINH Government quyet
        # dinh cuong do qua action[3] (rule_engine.py Section 4B; Blanchard &
        # Perotti, 2002) -- chuyen tu Economy sang day o v0.22 (xem
        # KNOWN_PATHOLOGIES.md, dung nguyen tac hai tang cua AI Economist:
        # chi Government moi mang trach nhiem phuc loi xa hoi qua chinh sach).
        # Duong (bom them cau) tru vao Kho bac giong mot khoan chi tieu chinh
        # phu G, am (rut bot cau) lam Kho bac TANG (chinh phu thu ve suc mua
        # da rut khoi thi truong). Dam bao dang thuc SFC: phan cau Firm/khu
        # vuc phi chinh thuc nhan them/bot PHAI co nguon doi ung dung o day.
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

        # No cuu tro Bank tra ve (goc + lai phat) -- Bagehot (1873), "at a high
        # rate" (xem rule_engine.py Section 5B, env.py::step() nhanh "NGUOI CHO
        # VAY CUOI CUNG"). Cung nguyen ly SFC nhu fines_collected o tren: tien
        # da chi ra luc bailout PHAI co duong quay lai tuong ming, khong duoc
        # coi la mat trang vinh vien.
        bailout_repayment = float(delta.get("bailout_repayment", 0.0))

        self.last_tax_collected = tax_revenue
        self.last_fines_collected = fines_collected
        self.last_subsidies_paid = subsidies_spent

        net_budget = tax_revenue + fines_collected + bailout_repayment - subsidies_spent - demand_injection_cost - government_purchase_cost
        self.treasury += net_budget
        
        if self.treasury < 0.0:
            self.public_debt += abs(self.treasury)
            self.treasury = 0.0
        elif self.public_debt > 0.0 and self.treasury > 0.0:
            # HE SO CAU TRUC TU DO HIEU CHINH (bo sung nhan con thieu, v0.20, phat hien
            # qua audit toan du an): 0.5 chi xac dinh TOC DO tra no cong moi buoc (tra
            # toi da 50% ngan sach thang du) -- khong suy ra truc tiep tu trich dan hoc
            # thuat nao, chi la lua chon can bang giua tra no nhanh (on dinh tai khoa)
            # va giu du du tru chi tieu (G, tro cap) cho buoc ke tiep.
            repayment = min(self.treasury * 0.5, self.public_debt)
            self.public_debt -= repayment
            self.treasury -= repayment

        # Cap nhat thue suat ap dung thuc te do RuleEngine chap thuan
        self.tax_rate_worker = float(delta.get("executed_worker_tax", self.tax_rate_worker))
        self.tax_rate_firm = float(delta.get("executed_firm_tax", self.tax_rate_firm))
        self.purchase_ratio = float(delta.get("executed_purchase_ratio", self.purchase_ratio))
        self.demand_injection_ratio = float(delta.get("demand_injection_ratio", self.demand_injection_ratio))
        self.last_demand_injection_value = float(delta.get("demand_injection_effect", 0.0))
        self.unemployment_relief_level = float(delta.get("executed_relief_level", self.unemployment_relief_level))

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
        # Death penalty: moi cai chet = -death_penalty_coef (mac dinh -20, xem giai thich hieu chinh o tren)
        new_deaths = int(transition_result.state_delta.get("new_deaths", 0))
        death_penalty = float(new_deaths) * self.death_penalty_coef

        # Debt penalty: nhỏ, chỉ kích hoạt khi nợ lớn
        debt_penalty = (self.public_debt * 0.00005) if self.public_debt > 0 else 0.0

        if self.reward_mode == "eq_x_prod":
            # CAN REWARD VOI THUOC DO DANH GIA (v0.35, KNOWN_PATHOLOGIES.md #29b): phuc loi xa hoi =
            # CONG BANG x NANG SUAT (Zheng et al., 2022, Science Advances 8(18), eabk2607 -- chinh la
            # thuoc do be/benchmark.py dung: Equality = 1 - Gini, Productivity = GDP thuc, Eq x Prod).
            # Truoc day reward la tong tuyen tinh khac (0.002*GDP + 0.005*dGDP - 25*Gini^2), trong do so
            # hang Gini^2 (toi da ~-11/buoc) ap dao GDP (~+1.6/buoc): chinh sach toi uu cho reward KHAC chinh
            # sach toi uu cho thuoc do benchmark -- khong the ket luan "RL vuot baseline tren Eq x Prod"
            # khi RL toi uu hoa mot muc tieu khac. swf_reward_scale=0.02 la HE SO CHUAN HOA TU DO HIEU
            # CHINH (dua Eq x Prod ~250 ve ~5/buoc, cung bac voi phat tu vong 20/ca), khong doi dang ham.
            equality = 1.0 - float(np.clip(self.current_gini, 0.0, 1.0))
            social_welfare = self.swf_reward_scale * equality * self.current_real_gdp - death_penalty - debt_penalty
            return float(np.clip(social_welfare, -100.0, 100.0))

        # --- DANG CU ("legacy", giu de tai hien) ---
        # GDP level: positive signal để agent biết nền kinh tế đang hoạt động
        gdp_level = self.current_real_gdp * 0.002

        # GDP growth: thưởng tăng trưởng
        gdp_growth = (self.current_real_gdp - self.last_real_gdp) * 0.005

        # Gini penalty: [0,1]^2 * gini_penalty_coef (mac dinh 25 -> toi da -25)
        gini_penalty = self.gini_penalty_coef * (self.current_gini ** 2)

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
            "demand_injection_ratio": round(self.demand_injection_ratio, 4),
            "unemployment_relief_level": round(self.unemployment_relief_level, 4),
            "last_demand_injection_value": round(self.last_demand_injection_value, 1),
            "current_gini": self.current_gini,
            "dead_citizens_count": self.dead_citizens_count
        }

    def reset(self) -> None:
        super().reset()
        self.tax_rate_worker = 0.15
        self.tax_rate_firm = 0.20
        self.purchase_ratio = 1.0
        self.last_purchase = 0.0
        self.demand_injection_ratio = 0.0
        self.last_demand_injection_value = 0.0
        self.unemployment_relief_level = 0.40
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