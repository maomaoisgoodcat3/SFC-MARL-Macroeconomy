from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from be.core.enums import LifeCycleStatus, EventType, AgentType
from be.core.event import Event, EventBus
from be.core.types import Action, TransitionResult, ValidationResult
from be.agents.base_agent import BaseAgent
from be.agents.employee import Employee
from be.agents.firm import Firm
from be.agents.government import Government
from be.agents.bank import Bank, compute_npl_ratio_pct
from be.agents.supervisor import Supervisor
from be.agents.economy import Economy
from be.rule_engine import RuleEngine


# ==============================================================================
# LOP LAM SACH DAU VAO/DAU RA -- MOT NGUON DUY NHAT cho moi duong chay
# ==============================================================================
# Truoc day cac buoc nay nam RIENG trong be/rllib_wrapper.py (chi duong TRAIN di
# qua), con be/main.py --mode simulate va be/server.py goi thang env.step() KHONG
# qua lop nay -- nghia la train va simulate xu ly NaN/Inf/reward KHAC NHAU. Dat
# vao day de moi tac nhan goi MacroEnvironment (RLlib wrapper, simulate, server,
# test) nhan DUNG CUNG mot logic.
def sanitize_observation(vec: np.ndarray) -> np.ndarray:
    """NaN -> 0, +-Inf -> +-1000 (chan lan truyen so hoc xau vao mang no-ron)."""
    return np.nan_to_num(np.asarray(vec, dtype=np.float32), nan=0.0, posinf=1000.0, neginf=-1000.0).astype(np.float32)


def sanitize_action(vals: np.ndarray) -> np.ndarray:
    """NaN -> 0, +-Inf -> +-1 (chan hanh dong hong tu policy truoc validate_action)."""
    return np.nan_to_num(np.asarray(vals, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


class MacroEnvironment:
    """
    Môi trường Kinh tế Vĩ mô Đa Tác tử Thể chế (Institutional MARL Environment).
    
    CƠ SỞ HỌC THUẬT & TRÍCH DẪN QUỐC TẾ:
    -------------------------------------------------------------------------
    1. Mincer, J. (1974). "Schooling, Experience, and Earnings". NBER.
       -> Phân phối kỹ năng liên tục Log-Normal (mu=0.0, sigma=0.35).
    2. Bain, J. S. (1956). "Barriers to New Competition". Harvard University Press.
       -> Quy mô hiệu dụng tối thiểu (Minimum Efficient Scale - MES) cho vốn mồi startup.
    3. Jorgenson, D. W. (1963). "Capital Theory and Investment Behavior". AER.
       -> Điều kiện gia nhập ngành dựa trên Suất sinh lời trên Vốn so với Lãi suất phi rủi ro.
    4. Mortensen, D. T., & Pissarides, C. A. (1994). "Job Creation and Job Destruction". RES.
       -> Dư địa lao động thất nghiệp kích thích khởi sự doanh nghiệp mới.
    5. Fisher, I. (1911). "The Purchasing Power of Money". Macmillan.
       -> Vận tốc lưu thông tiền tệ V = GDP / M2 trong tensor quan sát vĩ mô.
    -------------------------------------------------------------------------
    """
    def __init__(self, num_employees: int = 50, num_firms: int = 5, num_banks: int = 1, max_steps: int = 240,
                 inheritance_fraction: float = 0.15, min_newborn_cash: float = 500.0,
                 min_reproduction_age: int = 22, max_reproduction_age: int = 45,
                 min_reproduction_wealth_mult: float = 3.0, trait_mutation_sigma: float = 0.05,
                 hard_min_emp: int = 30, hard_max_emp: int = 200,
                 initial_lending_rate: float = 0.06, initial_deposit_rate: float = 0.02,
                 gini_penalty_coef: float = 25.0, death_penalty_coef: float = 20.0,
                 npl_flow_penalty_coef: float = 0.06, npl_stock_penalty_coef: float = 50.0,
                 npl_writeoff_months: int = 6,
                 emp_death_penalty_base: float = 100.0,
                 emp_death_penalty_horizon_multiplier: float = 1.0,
                 reward_scale_employee: float = 0.045, reward_scale_firm: float = 0.018,
                 reward_scale_government: float = 0.04, reward_scale_bank: float = 0.04,
                 reward_scale_supervisor: float = 0.012, reward_scale_economy: float = 0.02,
                 reward_clip: float = 100.0,
                 subsistence_indexation_ceiling_mult: float = 3.0,
                 initial_living_cost: float = 6.0,
                 initial_employment_rate: float = 0.90,
                 mrpl_scale_constant: float = 0.38,
                 hard_max_firms: int = 7,
                 firm_entry_probability: float = 0.15,
                 firm_entry_unemployment_threshold: float = 0.08,
                 firm_entry_profitability_margin: float = 0.0,
                 mortality_rate_floor: int = 30,
                 initial_economy_buffer_fund: float = 10000.0):
        self.num_employees: int = num_employees
        self.num_firms: int = num_firms
        self.num_banks: int = max(1, num_banks)
        self.max_steps: int = max_steps
        self.timestep: int = 0
        self.births_this_step: int = 0

        # Toan bo tham so ben duoi la HE SO CAU TRUC TU DO HIEU CHINH -- gia
        # tri mac dinh khop DUNG voi gia tri hardcode truoc khi co
        # ScenarioConfig (be/scenario_config.py), nen KHONG dung file cau
        # hinh nao van cho hanh vi giong het truoc day. Xem chi tiet trich
        # dan tung tham so tai noi su dung (Section A cua step() cho nhan
        # khau hoc, Government/Bank.calculate_reward cho he so reward).
        self.inheritance_fraction: float = float(inheritance_fraction)
        self.min_newborn_cash: float = float(min_newborn_cash)
        self.min_reproduction_age: int = int(min_reproduction_age)
        self.max_reproduction_age: int = int(max_reproduction_age)
        self.min_reproduction_wealth_mult: float = float(min_reproduction_wealth_mult)
        self.trait_mutation_sigma: float = float(trait_mutation_sigma)
        self.hard_min_emp: int = int(hard_min_emp)
        self.hard_max_emp: int = int(hard_max_emp)
        self.hard_max_firms: int = int(hard_max_firms)
        # He so cong gia nhap nganh (Section B duoi day) -- TRUOC DAY hardcode cuc bo, chuyen
        # sang ScenarioConfig (v0.26, phuc vu ablation tach bach anh huong so luong firm ban dau
        # / tran firm KHOI anh huong cua chinh dieu kien gia nhap, theo de xuat cua nguoi dung +
        # Claude Web: tang so firm khong tu dong lam pha phuc hoi nhanh hon neu cong gia nhap van
        # dong trong suy thoai). Jorgenson (1963)/Bain (1956)/Mortensen & Pissarides (1994) chi
        # xac lap DIEU KIEN dinh tinh gia nhap (loi nhuan vuot chi phi von, co du thua lao dong),
        # KHONG xac lap xac suat/nguong cu the -- ca hai la HE SO CAU TRUC TU DO HIEU CHINH.
        self.firm_entry_probability: float = float(firm_entry_probability)
        self.firm_entry_unemployment_threshold: float = float(firm_entry_unemployment_threshold)
        self.firm_entry_profitability_margin: float = float(firm_entry_profitability_margin)
        # Economy "an sinh vi mo" (v0.24, xem METHODOLOGY_NOTES.md muc 1/3) -- field RIENG cho
        # san mau so ty le tu vong cua Economy, KHONG dung chung hard_min_emp (rui ro coupling an
        # giua 2 co che khong thiet ke de phoi hop, xem METHODOLOGY_NOTES.md muc 3).
        self.mortality_rate_floor: int = int(mortality_rate_floor)
        # Von mo cap MOT LAN tu Treasury cho quy binh on du tru dem (buffer-stock) cua Economy
        # luc reset -- xem rule_engine.py Section 4D + METHODOLOGY_NOTES.md muc 2.
        self.initial_economy_buffer_fund: float = float(initial_economy_buffer_fund)
        self.initial_lending_rate: float = float(initial_lending_rate)
        self.initial_deposit_rate: float = float(initial_deposit_rate)
        self.gini_penalty_coef: float = float(gini_penalty_coef)
        self.death_penalty_coef: float = float(death_penalty_coef)
        self.npl_flow_penalty_coef: float = float(npl_flow_penalty_coef)
        self.npl_stock_penalty_coef: float = float(npl_stock_penalty_coef)
        self.npl_writeoff_months: int = int(npl_writeoff_months)
        self.emp_death_penalty_base: float = float(emp_death_penalty_base)
        self.emp_death_penalty_horizon_multiplier: float = float(emp_death_penalty_horizon_multiplier)

        # CHUAN HOA REWARD THEO TUNG LOAI TAC TU (he so hieu chinh, KHONG doi dang ham
        # reward). Ly do: PPO cua RLlib cat sai so gia tri o vf_clip_param (binh phuong
        # sai so > nguong => gradient bang 0). Do tren policy ngau nhien, |return| chiet
        # khau (gamma=0.99) cua Government/Bank/Supervisor ~ hang tram-nghin, nen 98-99%
        # mau bi cat va critic khong hoc: vf_explained_var ~ 0,0 o ~20 lan train that.
        # Nhan reward voi mot he so co dinh dua |return| p90 ve ~20 (< sqrt(vf_clip))
        # giu nguyen thu tu uu tien HANH VI (advantage duoc chuan hoa moi batch nen
        # policy gradient bat bien voi he so nhan) nhung cho critic tin hieu hoc duoc.
        # Trich dan: Engstrom, L. et al. (2020), "Implementation Matters in Deep Policy
        # Gradients: A Case Study on PPO and TRPO", ICLR; Andrychowicz, M. et al. (2021),
        # "What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical
        # Study", ICLR; van Hasselt, H. et al. (2016), "Learning values across many
        # orders of magnitude", NeurIPS. Gia tri cu the la CALIBRATION cua du an suy tu
        # phan phoi return do duoc (khong phai con so trong cac bai bao tren).
        # LUU Y HIEU CHINH LAI: he so cua Economy PHAI do lai moi khi doi phan phoi reward
        # cua no. Ban dau 0,20 (luc thuong khoi luong con chet, |r| trung vi ~0,19); sau
        # khi khoi phuc thuong khoi luong (rule_engine Section 4, total_market_turnover)
        # return cua Economy lon hon nhieu (p90 ~162 voi 0,20) nen ha xuong 0,025.
        # DO LAI LAN NUA (v0.16) sau khi dong vong chu chuyen (chi tieu tu tai san thanh
        # khoan + chi mua hang G + thuong Chinh phu tren GDP thuc): phan phoi reward doi vi
        # nen kinh te lanh manh hon (Employee/Firm p90 ~45-47 voi he so cu) -> Employee 0,045,
        # Firm 0,018, Government 0,04, Economy 0,02; Bank (return ~1) va Supervisor giu nguyen.
        self.reward_scale: Dict[AgentType, float] = {
            AgentType.EMPLOYEE: float(reward_scale_employee),
            AgentType.FIRM: float(reward_scale_firm),
            AgentType.GOVERNMENT: float(reward_scale_government),
            AgentType.BANK: float(reward_scale_bank),
            AgentType.SUPERVISOR: float(reward_scale_supervisor),
            AgentType.ECONOMY: float(reward_scale_economy),
        }
        self.reward_clip: float = float(reward_clip)
        # GUARD "fail loudly": reward_clip KHONG DUOC cat mat hinh phat tu vong theo tuoi.
        # (Loi that da xay ra: wrapper RLlib clip cung +-100 khien hinh phat -122,11 cua
        # agent 19 tuoi bi cat ve -100,00, vo hieu hoa hoan toan co che tuoi/horizon.)
        worst_death = (self.emp_death_penalty_base * (1.0 + max(0.0, self.emp_death_penalty_horizon_multiplier))
                       * self.reward_scale[AgentType.EMPLOYEE])
        if worst_death > self.reward_clip:
            raise ValueError(
                f"[MacroEnvironment] reward_clip={self.reward_clip} < hinh phat tu vong toi da sau chuan hoa "
                f"({worst_death:.2f}) -- clip se cat mat tin hieu tuoi/horizon. Tang reward_clip hoac giam he so."
            )

        # Tran chi so hoa chi tieu sinh ton theo gia (bo doi cua rule_engine Section 4),
        # tinh theo boi so cua initial_living_cost -- xem chu thich tai noi su dung.
        self.subsistence_indexation_ceiling_mult: float = float(subsistence_indexation_ceiling_mult)

        # GIA SINH HOAT KHOI TAO (v0.17, xem CLAUDE_HISTORY.md): truoc day hardcode = 20.0,
        # lech 3-4 lan so voi muc gia he thong thuc su hoi tu (~4-8, do thuc nghiem bang mo
        # phong policy da hoc -- xem CLAUDE_HISTORY.md v0.16.1) -- day la mot hang so KHOI TAO
        # SAI, khong phai lua chon can nhac: vi Calvo (1983) gan trong so theta=0.70 cho
        # expected_price (= gia ky truoc), mot gia khoi tao qua cao tao cu soc gia keo dai
        # nhieu thang dau moi episode. Gia tri 6.0 duoc DO thuc nghiem (chay policy ngau
        # nhien voi pre-employment o duoi, quan sat gia hoi tu ~5-7 trong 10 buoc dau) --
        # khong phai suy doan ly thuyet.
        self.initial_living_cost: float = float(initial_living_cost)

        # TY LE CO VIEC LAM LUC KHOI TAO (v0.17): truoc day 100% dan so bat dau THAT NGHIEP
        # (khong ai co employed_by), trong khi tran tuyen dung MAX_HIRES_PER_MONTH =
        # max(2, ceil(0.25*headcount)) chi cho phep toi da 2 nguoi/thang/firm khi headcount=0
        # -- tao ra mot "cua so khung hoang" co hoc keo dai 4-5 thang moi episode ma KHONG
        # policy nao tranh duoc bang cach hoc (gioi han co hoc cua he thong, khong phai hanh
        # vi). Do thuc nghiem (CLAUDE_HISTORY.md v0.16.1): duoi policy da hoc, ~30% dan so ban
        # dau chet trong ~30 thang dau vi Firm khai thac dung "cua so" nay (loi nhuan cao bat
        # thuong khi thue it luc dau). Sua: gan san viec lam cho 90% dan so luc reset() (xem
        # _create_world/reset), de lai 10% that nghiep lam ma sat tu nhien -- hop voi khoang
        # ty le that nghiep tu nhien/co ma sat thuong duoc dan chieu trong ly thuyet tim kiem
        # (Mortensen, D. T., & Pissarides, C. A. (1994), "Job Creation and Job Destruction",
        # RES 61(3) -- da trich dan cho chinh co che ma sat tim viec o rule_engine.py Section
        # 2). LUU Y: day la DIEU KIEN KHOI TAO, khong phai cong thuc hanh vi -- ty le 90%
        # cu the la HE SO HIEU CHINH tu do dieu chinh, KHONG suy truc tiep tu Mortensen &
        # Pissarides (paper do khong dua ra mot con so % cu the).
        self.initial_employment_rate: float = float(initial_employment_rate)

        # He so hieu chinh quy mo MRPL/san luong (v0.19) -- xem chu thich day du tai
        # RuleEngine.__init__ (be/rule_engine.py). Mac dinh = GIA TRI DA SUA (giai dai so tu
        # dieu kien can bang wage/price); dat = 1.0 de tai hien logic CU (lech chuan dinh co,
        # xem KNOWN_PATHOLOGIES.md muc #8).
        self.mrpl_scale_constant: float = float(mrpl_scale_constant)

        self.event_bus: EventBus = EventBus()
        self.rule_engine: RuleEngine = RuleEngine(
            event_bus=self.event_bus,
            subsistence_indexation_ceiling_mult=self.subsistence_indexation_ceiling_mult,
            mrpl_scale_constant=self.mrpl_scale_constant,
        )
        self.agents: Dict[str, BaseAgent] = {}
        self.banks: List[Bank] = []

        self.next_emp_id: int = self.num_employees
        self.next_firm_id: int = self.num_firms
        self.reported_dead_agents: set = set()
        self._create_world()

    def _create_world(self) -> None:
        self.agents.clear()
        self.gov = Government(agent_id="gov_1", gini_penalty_coef=self.gini_penalty_coef, death_penalty_coef=self.death_penalty_coef)
        self.eco = Economy(agent_id="eco_1", mortality_rate_floor=self.mortality_rate_floor)
        self.sup = Supervisor(agent_id="sup_1")

        # Hỗ trợ N ngân hàng đồng thời (mặc định 1). Toàn bộ ngân hàng chia sẻ
        # cùng một policy RL "policy_bank" qua parameter sharing, giống cách
        # firm_*/emp_* đã dùng (xem rllib_wrapper.policy_mapping_fn). self.bank
        # được giữ làm alias trỏ tới ngân hàng đầu tiên cho các chỗ chỉ cần một
        # đại diện hiển thị nhanh (vd. macro "headline" stats) -- mọi logic tài
        # chính thực sự (tín dụng, tiền gửi) dùng self.banks / RuleEngine.
        self.banks = [
            Bank(
                agent_id=f"bank_{i}",
                npl_flow_penalty_coef=self.npl_flow_penalty_coef,
                npl_stock_penalty_coef=self.npl_stock_penalty_coef,
                npl_writeoff_months=self.npl_writeoff_months
            )
            for i in range(self.num_banks)
        ]
        self.bank = self.banks[0]

        self.agents[self.gov.agent_id] = self.gov
        for b in self.banks:
            self.agents[b.agent_id] = b
        self.agents[self.eco.agent_id] = self.eco
        self.agents[self.sup.agent_id] = self.sup

        for i in range(self.num_firms):
            f_id = f"firm_{i}"
            self.agents[f_id] = Firm(agent_id=f_id)

        for j in range(self.num_employees):
            e_id = f"emp_{j}"
            self.agents[e_id] = Employee(
                agent_id=e_id,
                death_penalty_base=self.emp_death_penalty_base,
                death_penalty_horizon_multiplier=self.emp_death_penalty_horizon_multiplier
            )

    def _sample_skill(self) -> float:
        """Phân phối kỹ năng liên tục Log-Normal (Mincer, 1974; Saez, 2001)."""
        s_min = 0.50
        mu = 0.0
        sigma = 0.35
        raw_skill = float(np.random.lognormal(mean=mu, sigma=sigma))
        return float(np.clip(s_min + raw_skill * 0.45, 0.60, 4.0))

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        self.timestep = 0
        self.next_emp_id = self.num_employees
        self.next_firm_id = self.num_firms
        self.reported_dead_agents.clear()
        self.event_bus.clear()
        self._create_world()

        # KHỞI TẠO THỂ CHẾ VĨ MÔ
        self.gov.initialize(initial_treasury=1000000.0, initial_worker_tax=0.15, initial_firm_tax=0.20)
        self.gov.current_gdp = 0.0
        # Tổng dự trữ hệ thống ngân hàng được CHIA ĐỀU cho N ngân hàng để tổng
        # cung tín dụng ban đầu của toàn hệ thống không phụ thuộc vào num_banks
        # -- giữ các lần chạy với số lượng ngân hàng khác nhau có thể so sánh
        # được (comparable), tránh việc chỉ đơn thuần nhân đôi tổng tiền khi
        # tăng num_banks.
        per_bank_reserves = 500000.0 / len(self.banks)
        for b in self.banks:
            b.initialize(initial_reserves=per_bank_reserves, initial_lending_rate=self.initial_lending_rate, initial_deposit_rate=self.initial_deposit_rate)
        self.eco.initialize(initial_living_cost=self.initial_living_cost, initial_housing_inventory=100, initial_house_price=1000.0)
        # Von mo QUY BINH ON DU TRU DEM cua Economy (v0.24) -- chuyen MOT LAN tu Treasury, y het
        # mau hinh cap von cho Firm/Employee luc reset (seed_cash/start_cash) -- SFC-consistent
        # (chi chuyen giao noi bo, khong tao tien moi). Xem rule_engine.py Section 4D +
        # METHODOLOGY_NOTES.md muc 2.
        self.gov.treasury -= self.initial_economy_buffer_fund
        self.eco.strategic_reserve_fund = self.initial_economy_buffer_fund
        self.sup.initialize(initial_budget=50000.0, initial_audit_rate=0.05, initial_fine_multiplier=1.5)
        
        self.sup.violations_detected = 0
        self.sup.fines_collected = 0.0

        for agent in self.agents.values():
            if isinstance(agent, Firm):
                init_cap = float(np.random.uniform(8000.0, 14000.0))
                agent.initialize(
                    productivity_factor=float(np.random.uniform(1.3, 1.8)),
                    risk_aversion=float(np.random.uniform(0.2, 0.4)),
                    tax_morale=float(np.random.uniform(0.5, 0.85)),
                    initial_capital=init_cap
                )
                agent.age_months = 0
                agent.status = LifeCycleStatus.ACTIVE
                seed_cash = float(np.random.uniform(3000.0, 5000.0))
                self.gov.treasury -= seed_cash
                agent.cash = seed_cash

            elif isinstance(agent, Employee):
                start_cash = float(np.random.uniform(200.0, 400.0))
                self.gov.treasury -= start_cash
                agent.initialize(
                    skill_level=self._sample_skill(),
                    risk_aversion=float(np.random.uniform(0.3, 0.8)),
                    tax_morale=float(np.random.uniform(0.5, 0.95)),
                    initial_cash=start_cash,
                    initial_energy=float(np.random.uniform(0.8, 1.2)),
                    age=int(np.random.randint(18, 50))
                )
                agent.status = LifeCycleStatus.ACTIVE

        self._assign_initial_employment()

        raw_state = self.get_raw_environment_state()
        initial_obs = {aid: sanitize_observation(a.observe(raw_state).vector) for aid, a in self.agents.items()}
        infos = {aid: {"status": a.status.name} for aid, a in self.agents.items()}
        return initial_obs, infos

    def _assign_initial_employment(self) -> None:
        """Gán việc làm sẵn cho một phần dân số lúc reset() (v0.17, xem CLAUDE_HISTORY.md và
        chú thích tại self.initial_employment_rate ở __init__). Phân bổ số lượng nhân viên
        mỗi firm tỷ lệ thuận với capital_stock khởi tạo (vốn lớn hơn -> nhu cầu lao động bổ
        sung lớn hơn, nhất quán với tính bổ sung vốn-lao động ngầm định trong Cobb-Douglas ở
        rule_engine.py Section 3), dùng phương pháp số dư lớn nhất/Hamilton (Balinski, M. L.,
        & Young, H. P. (1982), "Fair Representation: Meeting the Ideal of One Man, One Vote",
        Yale University Press) để tổng số người được gán đúng bằng mục tiêu mà không thiên vị
        firm nào do làm tròn. Nhân viên được XÁO TRỘN NGẪU NHIÊN trước khi gán (không theo kỹ
        năng) để tránh vô tình tạo một phân bổ "tối ưu" nhân tạo ngay từ đầu."""
        active_firms = [a for a in self.agents.values() if isinstance(a, Firm)]
        active_employees = [a for a in self.agents.values() if isinstance(a, Employee)]
        if not active_firms or not active_employees:
            return

        n_target = int(round(len(active_employees) * self.initial_employment_rate))
        n_target = max(0, min(n_target, len(active_employees)))

        total_capital = sum(f.capital_stock for f in active_firms)
        if total_capital <= 0.0:
            shares = [n_target / len(active_firms)] * len(active_firms)
        else:
            shares = [f.capital_stock / total_capital * n_target for f in active_firms]
        quotas = [int(s) for s in shares]
        remainder = n_target - sum(quotas)
        # Hamilton/largest-remainder: phan du (do lam tron) di cho cac firm co phan thap phan
        # lon nhat truoc, dam bao tong dung bang n_target.
        order = sorted(range(len(active_firms)), key=lambda i: shares[i] - quotas[i], reverse=True)
        for i in order[:remainder]:
            quotas[i] += 1

        shuffled = list(active_employees)
        np.random.shuffle(shuffled)
        idx = 0
        for firm, quota in zip(active_firms, quotas):
            for _ in range(quota):
                if idx >= len(shuffled):
                    break
                emp = shuffled[idx]
                idx += 1
                emp.employed_by = firm.agent_id
                firm.employee_ids.append(emp.agent_id)

    def observe_agent(self, agent_id: str, raw_state: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Quan sat DA LAM SACH cua mot tac tu -- DUNG DUNG cach step()/reset() dung de
        tao quan sat cho RLlib. server.py/simulate PHAI goi ham nay (khong tu goi
        agent.observe()) de policy suy luan thay dung nhung gi no thay luc train."""
        if raw_state is None:
            raw_state = self.get_raw_environment_state()
        return sanitize_observation(self.agents[agent_id].observe(raw_state).vector)

    def _finalize_reward(self, agent: BaseAgent, raw_reward: float) -> float:
        """reward tho -> nhan he so chuan hoa theo loai tac tu -> NaN/Inf ve 0 -> kep [-clip, clip].
        Mot nguon duy nhat cho ca train lan simulate (xem chu thich reward_scale o __init__)."""
        r = float(raw_reward) * self.reward_scale[agent.agent_type]
        if not np.isfinite(r):
            return 0.0
        return float(np.clip(r, -self.reward_clip, self.reward_clip))

    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]
    ]:
        self.timestep += 1
        self.births_this_step = 0
        for a in self.agents.values():
            if isinstance(a, Firm):
                a.age_months = getattr(a, "age_months", 0) + 1

        # XỬ LÝ ĐẦY ĐỦ CẢ ACTIVE VÀ INITIALIZED
        validated_actions: Dict[str, Action] = {}
        for agent_id, raw_vals in action_dict.items():
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if agent.status in [LifeCycleStatus.ACTIVE, LifeCycleStatus.INITIALIZED]:
                    agent.status = LifeCycleStatus.ACTIVE  # Auto-transition sang ACTIVE
                    act = Action(agent_id=agent_id, action_type="STEP_ACTION", values=sanitize_action(raw_vals))
                    val_res: ValidationResult = agent.validate_action(act)
                    validated_actions[agent_id] = Action(
                        agent_id=agent_id,
                        action_type=act.action_type,
                        values=val_res.sanitized_values,
                        metadata={"is_valid": val_res.is_valid}
                    )

        transition_results: Dict[str, TransitionResult] = self.rule_engine.execute_cycle(
            agents=self.agents,
            validated_actions=validated_actions,
            timestep=self.timestep,
            max_steps=self.max_steps
        )

        rewards_all: Dict[str, float] = {}
        for agent_id, trans_res in transition_results.items():
            if agent_id in self.agents:
                self.agents[agent_id].apply_result(trans_res)
                if "status" in trans_res.state_delta:
                    self.agents[agent_id].status = trans_res.state_delta["status"]
                # Áp dụng kẹp thể lực chuẩn xác
                if "applied_energy" in trans_res.state_delta and isinstance(self.agents[agent_id], Employee):
                    self.agents[agent_id].energy = trans_res.state_delta["applied_energy"]
                rewards_all[agent_id] = self.agents[agent_id].calculate_reward(trans_res)

        sup_delta = transition_results.get(self.sup.agent_id)
        if sup_delta is not None:
            self.sup.fines_collected = sup_delta.state_delta.get("fines_collected", getattr(self.sup, "fines_collected", 0.0))
            self.sup.violations_detected = sup_delta.state_delta.get("violations_detected", getattr(self.sup, "violations_detected", 0))

        # QUẢN TRỊ TỬ VONG & BẢO TOÀN DI SẢN KHO BẠC (SFC CONSISTENCY)
        dead_emps = [aid for aid in self.agents.items() if isinstance(aid[1], Employee) and aid[1].status in [LifeCycleStatus.DECEASED, LifeCycleStatus.DEAD, LifeCycleStatus.TERMINATED]]
        bankrupt_firms = [aid for aid in self.agents.items() if isinstance(aid[1], Firm) and aid[1].status == LifeCycleStatus.BANKRUPT]
        # LOI DA SUA (v0.20, xem KNOWN_PATHOLOGIES.md muc moi): truoc day KHONG CO
        # nhanh tuong duong cho Bank o day -- Bank.apply_result co the tu goi
        # self.terminate() (status=TERMINATED) khi reserves < -100000 (mat kha
        # nang thanh toan tram trong), nhung khong bao gio duoc don khoi
        # self.agents (cung lop loi API MultiAgentEnv da sua cho Employee/Firm o
        # v0.11, xem chu thich ngay ben duoi) -- Bank "chet" tro thanh zombie vinh
        # vien: van nhan obs/reward/action moi buoc, khong bao gio terminated=True
        # cho RLlib. Nghiem trong hon: rule_engine.py::_get_all_agents(Bank) CHI
        # loc trang thai ACTIVE/INITIALIZED va RAISE RuntimeError neu KHONG con
        # bank ACTIVE nao -- voi cau hinh mac dinh num_banks=1, bank duy nhat pha
        # san se lam CRASH toan bo execute_cycle() (va do do ca vong training/
        # simulation) ngay buoc ke tiep. Da DO THUC NGHIEM (khong doan): qua hon
        # 4000 buoc heuristic/random-noise VA mot checkpoint PPO da train that
        # (iter_40, 3 seed x 480 buoc), reserves chua bao gio tien gan nguong nay
        # -- nhung day van la mot "qua bom no cham" kien truc can va truoc, dung
        # tinh than "phai tai hien duoc" cua du an (xem KNOWN_PATHOLOGIES.md).
        dead_banks = [aid for aid in self.agents.items() if isinstance(aid[1], Bank) and aid[1].status == LifeCycleStatus.TERMINATED]

        # LỖI API MultiAgentEnv (Gymnasium-style) ĐÃ SỬA -- xem CLAUDE.md để
        # biết bối cảnh phát hiện đầy đủ. Trước đây agent vừa chết/phá sản bị
        # pop() khỏi self.agents NGAY TẠI ĐÂY, TRƯỚC KHI observations/rewards/
        # terminateds cuối cùng được dựng (phía dưới, chỉ lặp qua
        # self.agents.keys() -- lúc đó agent đã biến mất khỏi dict). Hệ quả:
        # RLlib KHÔNG BAO GIỜ nhận được terminated=True, reward cuối cùng, hay
        # observation cuối cùng cho agent này -- toàn bộ tín hiệu hậu quả của
        # cái chết/phá sản (death_penalty, dead_worker_penalty, bankrupt
        # penalty...) chưa từng thực sự tới được policy gradient; agent chỉ
        # "biến mất" một cách vô hình, không phải kết thúc MDP có chủ đích.
        # Đây là lỗi API chuẩn: mọi agent rời khỏi tập hợp active PHẢI có đúng
        # 1 lần cuối terminated=True kèm reward/obs cuối cùng.
        # Sửa: giữ lại REFERENCE Python (không phải dict entry) tới agent vừa
        # chết/phá sản TRƯỚC KHI pop -- object vẫn còn hợp lệ (mang đúng
        # trạng thái lúc chết: cash âm, energy=0...) dù đã bị xoá khỏi dict,
        # để sau khi raw_state được tính, lấy observation CUỐI CÙNG từ chính
        # object này rồi gộp vào observations/rewards/terminateds/truncateds/
        # infos với terminated=True (KHÔNG PHẢI truncated -- đây là trạng thái
        # hấp thụ có chủ đích của MDP riêng agent đó, không phải bị cắt vì hết
        # giờ episode chung -- terminateds["__all__"] KHÔNG bị ảnh hưởng, vẫn
        # chỉ True khi toàn episode kết thúc).
        terminal_agents_this_step: Dict[str, BaseAgent] = {}

        for d_id, emp in dead_emps:
            self.reported_dead_agents.add(d_id)
            terminal_agents_this_step[d_id] = emp
            self.agents.pop(d_id, None)
            # Bảo toàn dòng tiền (Godley & Lavoie, 2007): TOÀN BỘ của cải còn lại
            # của người đã mất (tiền mặt + tiền gửi ngân hàng) được thu hồi về
            # Kho bạc -- kể cả khi cash âm (nợ cùng quẫn chưa trả), khoản nợ đó
            # được Kho bạc gánh chịu như một khoản mất mát xã hội TƯỜNG MINH
            # (cộng vào treasury dạng số âm), thay vì "bốc hơi" âm thầm khi agent
            # bị xoá khỏi self.agents -- trước đây chỉ cash > 0 mới được thu hồi
            # và bank_deposit không hề được xử lý, khiến cả hai chiều đều vi phạm
            # bảo toàn hệ thống.
            self.gov.treasury += emp.cash + getattr(emp, "bank_deposit", 0.0)
            if emp.employed_by and emp.employed_by in self.agents:
                employer = self.agents[emp.employed_by]
                if hasattr(employer, "employee_ids") and d_id in employer.employee_ids:
                    employer.employee_ids.remove(d_id)

        for b_id, firm in bankrupt_firms:
            self.reported_dead_agents.add(b_id)
            terminal_agents_this_step[b_id] = firm
            self.agents.pop(b_id, None)
            # Xem chú thích bảo toàn dòng tiền ở nhánh dead_emps phía trên -- áp
            # dụng cùng nguyên tắc: thu hồi toàn bộ cash còn lại (kể cả âm) về
            # Kho bạc thay vì chỉ thu phần dương.
            self.gov.treasury += firm.cash
            for a in self.agents.values():
                if isinstance(a, Employee) and a.employed_by == b_id:
                    a.employed_by = None
                    a.wage = 0.0

        # NGƯỜI CHO VAY CUỐI CÙNG (Lender of Last Resort) -- Bagehot, W. (1873),
        # "Lombard Street: A Description of the Money Market", Henry S. King & Co.
        # -- học thuyết kinh điển: "lend freely, at a HIGH RATE, against good
        # collateral". BẢN VÁ ĐẦU TIÊN (v0.20) chỉ tái cấp vốn MIỄN PHÍ, thiếu hẳn
        # vế "at a high rate" -- bị chỉ ra (Claude Web, 2026-09-23) là dùng tên
        # Bagehot cho một cơ chế KHÔNG đúng doctrine (loại bỏ hoàn toàn moral
        # hazard: với num_banks=1 mặc định, "toàn bộ bank chết" và "bank duy nhất
        # chết" là CÙNG một điều kiện -- Bank không bao giờ thực sự chịu hậu quả).
        # SỬA: khoản cứu trợ được ghi nhận là MỘT KHOẢN NỢ thật (`bailout_debt`,
        # xem Bank.__init__) mà Bank phải trả Kho bạc kèm lãi suất PHẠT cao hơn
        # hẳn trần lending_rate thị trường (`bailout_penalty_rate`, xem
        # rule_engine.py Section 5B) -- Bank vẫn được cứu (tránh crash/rỗng hệ
        # thống tín dụng), nhưng KHÔNG miễn phí, giữ được động cơ quản trị rủi ro
        # thay vì trung hoà hoàn toàn moral hazard.
        # Áp dụng CHỈ khi để TẤT CẢ bank chết cùng lúc sẽ làm rỗng hoàn toàn hệ
        # thống tín dụng (rule_engine.py::_get_all_agents(Bank) sẽ raise
        # RuntimeError, crash toàn bộ episode) -- không áp dụng bailout cho các
        # bank khác nếu còn ít nhất 1 bank khác vẫn ACTIVE (để nguyên tắc "chịu
        # trách nhiệm hậu quả tài chính" của RL vẫn có hiệu lực bình thường với
        # num_banks>=2). Ngân hàng được cứu là ngân hàng có reserves CAO NHẤT
        # trong số vừa phá sản (ít mất khả năng thanh toán nhất), tái cấp vốn về
        # đúng 0 (từ Kho bạc, một khoản chi ngân sách tường minh, bảo toàn SFC --
        # khoản này được ghi vào bailout_debt để đòi lại dần, không phải mất trắng).
        still_active_banks = any(
            isinstance(a, Bank) and a.status in (LifeCycleStatus.ACTIVE, LifeCycleStatus.INITIALIZED)
            for a in self.agents.values()
        )
        bailed_out_bank_id: Optional[str] = None
        if dead_banks and not still_active_banks:
            bailed_out_bank_id, bailed_bank = max(dead_banks, key=lambda item: item[1].reserves)
            bailout_cost = max(0.0, -bailed_bank.reserves)
            self.gov.treasury -= bailout_cost
            bailed_bank.reserves = 0.0
            # KHÔNG cho không (xem chú thích Bagehot ở trên): khoản cứu trợ là
            # một khoản NỢ thật, Bank phải trả dần kèm lãi phạt
            # (bailout_penalty_rate) qua rule_engine.py Section 5B từ bước sau.
            bailed_bank.bailout_debt += bailout_cost
            bailed_bank.status = LifeCycleStatus.ACTIVE
            self.event_bus.publish(Event(
                event_type=EventType.AGENT_BANKRUPT,
                source_id=self.gov.agent_id,
                target_id=bailed_out_bank_id,
                payload={"reason": "Lender of last resort bailout", "cost": round(bailout_cost, 1)},
                timestep=self.timestep
            ))

        for bk_id, bk in dead_banks:
            if bk_id == bailed_out_bank_id:
                continue
            self.reported_dead_agents.add(bk_id)
            terminal_agents_this_step[bk_id] = bk
            self.agents.pop(bk_id, None)
            # Cùng nguyên tắc bảo toàn dòng tiền ở 2 nhánh trên: reserves còn lại
            # (kể cả âm) được Kho bạc hấp thụ tường minh, KHÔNG "bốc hơi" âm thầm.
            self.gov.treasury += bk.reserves
            bk.reserves = 0.0  # tránh đếm trùng trong sum(b.reserves for b in self.banks) ở macro/M2
            # Dọn tham chiếu treo (stale) tới bank đã chết để rule_engine.py Section
            # 5/8B tự chọn lại ngân hàng còn sống ở bước kế tiếp (đã hỗ trợ sẵn qua
            # "existing_creditor/depository not in bank_lookup" -- xem rule_engine.py).
            for a in self.agents.values():
                if isinstance(a, Firm) and getattr(a, "creditor_bank_id", None) == bk_id:
                    a.creditor_bank_id = None
                if isinstance(a, Employee) and getattr(a, "depository_bank_id", None) == bk_id:
                    a.depository_bank_id = None

        # A. ĐIỀU TIẾT DÂN SỐ THEO SỨC TẢI KINH TẾ (Demographic Carrying Capacity)
        active_emps_list = [a for a in self.agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE]
        current_emp_count = len(active_emps_list)
        unemployed_count = sum(1 for e in active_emps_list if e.employed_by is None)
        unemployment_rate = (unemployed_count / current_emp_count) if current_emp_count > 0 else 0.0

        employed_emps = [e for e in active_emps_list if e.employed_by is not None]
        avg_wage = (sum(e.wage for e in employed_emps) / len(employed_emps)) if employed_emps else self.eco.base_living_cost
        living_standard_ratio = avg_wage / max(1.0, self.eco.base_living_cost)

        # HARD_MIN_EMP/HARD_MAX_EMP doc tu self.* (cau hinh qua constructor /
        # ScenarioConfig) -- mac dinh 30/200. Nang TRAN khong lam tang TOC DO
        # tang truong toi da moi step (van toi da 1-2 newborn/step nhu cu,
        # xem vong lap num_newborns ben duoi) -- chi mo rong bien tren cho
        # phep, khong tu dong lam dan so bung no nhanh hon.
        HARD_MIN_EMP = self.hard_min_emp
        HARD_MAX_EMP = self.hard_max_emp

        # --- CƠ CHẾ TĂNG TRƯỞNG DÂN SỐ: HAI NHÁNH TÁCH BẠCH RÕ RÀNG ---
        #
        # NHÁNH 1 (KHẨN CẤP, current_emp_count < HARD_MIN_EMP): lưới an sinh
        # bảo vệ đáy (population floor safety net). Đây là một BIỆN PHÁP KỸ
        # THUẬT thuần tuý để tránh trạng thái suy vong tuyệt đối (absorbing
        # extinction state) làm hỏng toàn bộ tiến trình RL training (môi
        # trường rỗng không còn agent nào thì không còn tín hiệu học) --
        # KHÔNG phải mô phỏng một quá trình nhân khẩu học/sinh học thực tế
        # nào (không nền kinh tế thật nào có "chính phủ tạo ra người lớn từ
        # hư không"). Newborn ở nhánh này được tài trợ trực tiếp từ Kho bạc,
        # KHÔNG gắn với cha/mẹ cụ thể nào -- giữ nguyên cơ chế gốc, chỉ làm
        # rõ bản chất "biên giới kỹ thuật" của nó trong comment.
        #
        # NHÁNH 2 (TỰ NGUYỆN, HARD_MIN_EMP <= count < HARD_MAX_EMP): sinh sản
        # nội sinh (endogenous), theo đúng mô hình Sugarscape kinh điển:
        #   Epstein, J. M., & Axtell, R. (1996). "Growing Artificial
        #   Societies: Social Science from the Bottom Up". Brookings
        #   Institution Press / MIT Press -- Chương 2 ("Sexual Reproduction,
        #   Inheritance, Culture"). Trong Sugarscape, một agent chỉ sinh sản
        #   được khi (a) nằm trong độ tuổi sinh sản hợp lệ, VÀ (b) tích luỹ đủ
        #   "sugar" (của cải) vượt ngưỡng tối thiểu; con cái KẾ THỪA một phần
        #   của cải VÀ một phần đặc điểm di truyền (ở đây: skill_level,
        #   risk_aversion, tax_morale) từ cha/mẹ, có pha trộn/đột biến ngẫu
        #   nhiên nhỏ. Phần chuyển giao của cải (inheritance) tham chiếu thêm
        #   Piketty, T. (2014), "Capital in the Twenty-First Century",
        #   Belknap/Harvard University Press, về vai trò của thừa kế trong
        #   bất bình đẳng liên thế hệ (r > g).
        #   (Đã xác minh lại: Neural MMO (Suárez et al., 2019, arXiv:1903.00784)
        #   CHỈ dùng permadeath + respawn slot NGẪU NHIÊN HOÀN TOÀN, KHÔNG có
        #   khái niệm cha/mẹ hay kế thừa gen -- không phù hợp làm nguồn trích
        #   dẫn cho cơ chế này.)
        # Toan bo hang so ben duoi doc tu self.* (cau hinh qua constructor /
        # ScenarioConfig, xem be/scenario_config.py) de phuc vu thu nghiem
        # nhieu kich ban calibration ma khong can sua code.
        MIN_REPRODUCTION_AGE = self.min_reproduction_age          # tuổi lao động đã ổn định (sau tuổi vào đời 18 ở cả 2 nhánh)
        MAX_REPRODUCTION_AGE = self.max_reproduction_age          # cận trên độ tuổi sinh sản còn năng động kinh tế
        MIN_REPRODUCTION_WEALTH_MULT = self.min_reproduction_wealth_mult  # phải dư >= N tháng chi phí sống mới đủ "sugar" để sinh sản (ngưỡng CAO HƠN K_LIQUIDITY_BUFFER=2.0 dùng cho đệm thanh khoản giao dịch ở rule_engine.py Section 8B -- có chủ đích, vì đây là thặng dư THẬT SỰ chứ không phải đệm giao dịch)
        INHERITANCE_FRACTION = self.inheritance_fraction          # tỷ lệ của cải cha/mẹ chuyển cho con
        TRAIT_MUTATION_SIGMA = self.trait_mutation_sigma          # độ lệch chuẩn đột biến Gaussian quanh đặc điểm cha/mẹ
        MIN_NEWBORN_CASH = self.min_newborn_cash                  # sàn an sinh tối thiểu, Kho bạc bù thêm nếu thừa kế chưa đủ

        num_newborns = 0
        newborn_parent: Optional[Employee] = None
        if current_emp_count < HARD_MIN_EMP:
            num_newborns = 2 if current_emp_count < 20 else 1
        elif current_emp_count < HARD_MAX_EMP:
            # HE SO CAU TRUC TU DO HIEU CHINH (v0.20, bo sung nhan con thieu -- phat hien
            # qua audit toan du an): cac he so 0.08/0.15/0.10 va bien [0.02, 0.35] CHI xac
            # dinh DO LON xac suat sinh san moi buoc de dan so tang truong hop ly trong
            # pham vi 1 episode (khong bung no/khong triet tieu) -- Epstein & Axtell (1996)
            # chi xac lap DIEU KIEN sinh san (tuoi + "sugar" toi thieu), KHONG cho cong
            # thuc xac suat cu the nao; day la lua chon hieu chinh so hoc rieng cua mo
            # phong nay, khong suy ra truc tiep tu trich dan.
            p_birth = 0.08 + 0.15 * max(0.0, 1.0 - unemployment_rate) + 0.10 * max(0.0, living_standard_ratio - 1.0)
            p_birth = float(np.clip(p_birth, 0.02, 0.35))
            eligible_parents = [
                e for e in active_emps_list
                if MIN_REPRODUCTION_AGE <= e.age <= MAX_REPRODUCTION_AGE
                and (e.cash + getattr(e, "bank_deposit", 0.0)) >= MIN_REPRODUCTION_WEALTH_MULT * self.eco.base_living_cost
            ]
            # "Sinh sản thật" đòi hỏi có ít nhất một cha/mẹ đủ điều kiện --
            # KHÔNG fallback về trợ cấp vô danh nếu không ai đủ điều kiện
            # (khác nhánh khẩn cấp phía trên).
            if (eligible_parents and np.random.rand() < p_birth
                    and self.gov.treasury >= (3.0 * self.eco.base_living_cost)):
                num_newborns = 1
                newborn_parent = eligible_parents[int(np.random.randint(len(eligible_parents)))]

        grant_per_newborn = float(3.0 * self.eco.base_living_cost)
        for _ in range(num_newborns):
            new_eid = f"emp_{self.next_emp_id}"
            self.next_emp_id += 1

            if newborn_parent is not None:
                # NHÁNH 2: sinh sản nội sinh có cha/mẹ thật (Epstein & Axtell, 1996)
                parent_wealth = newborn_parent.cash + getattr(newborn_parent, "bank_deposit", 0.0)
                inheritance = INHERITANCE_FRACTION * parent_wealth

                # Chuyển giao của cải cha/mẹ -> con: rút trước từ cash, nếu
                # không đủ rút tiếp từ bank_deposit (bảo toàn SFC tuyệt đối --
                # tổng hệ thống không đổi, chỉ chuyển sở hữu giữa 2 agent).
                cash_take = min(inheritance, newborn_parent.cash)
                newborn_parent.cash -= cash_take
                remaining = inheritance - cash_take
                if remaining > 0.0:
                    deposit_take = min(remaining, getattr(newborn_parent, "bank_deposit", 0.0))
                    newborn_parent.bank_deposit = getattr(newborn_parent, "bank_deposit", 0.0) - deposit_take
                    inheritance = cash_take + deposit_take

                # An sinh tối thiểu: nếu thừa kế chưa đạt sàn sống, Kho bạc bù
                # thêm phần thiếu (giữ đúng tinh thần lưới an sinh đã có, chỉ
                # áp dụng cho phần THIẾU thay vì toàn bộ như nhánh khẩn cấp).
                topup = max(0.0, MIN_NEWBORN_CASH - inheritance)
                if topup > 0.0 and self.gov.treasury >= topup:
                    self.gov.treasury -= topup
                else:
                    topup = 0.0
                newborn_cash = inheritance + topup

                # Kế thừa đặc điểm di truyền + đột biến Gaussian nhỏ quanh
                # giá trị cha/mẹ (Epstein & Axtell, 1996), thay vì random độc
                # lập hoàn toàn như nhánh khẩn cấp.
                child_skill = float(np.clip(
                    newborn_parent.skill_level + float(np.random.normal(0.0, TRAIT_MUTATION_SIGMA * newborn_parent.skill_level)),
                    0.60, 4.0
                ))
                child_risk_aversion = float(np.clip(
                    newborn_parent.risk_aversion + float(np.random.normal(0.0, TRAIT_MUTATION_SIGMA)),
                    0.10, 0.95
                ))
                child_tax_morale = float(np.clip(
                    newborn_parent.tax_morale + float(np.random.normal(0.0, TRAIT_MUTATION_SIGMA)),
                    0.30, 0.99
                ))
                birth_source_id = newborn_parent.agent_id
                parent_id_payload = newborn_parent.agent_id
            else:
                # NHÁNH 1: lưới an sinh bảo vệ đáy, không gắn parent (xem comment ở trên)
                grant = grant_per_newborn if self.gov.treasury >= (grant_per_newborn * 2.0) else (0.5 * grant_per_newborn)
                self.gov.treasury -= grant
                newborn_cash = grant
                child_skill = self._sample_skill()
                child_risk_aversion = float(np.random.uniform(0.3, 0.7))
                child_tax_morale = float(np.random.uniform(0.5, 0.95))
                birth_source_id = "SOCIETY"
                parent_id_payload = None

            new_emp = Employee(
                agent_id=new_eid,
                death_penalty_base=self.emp_death_penalty_base,
                death_penalty_horizon_multiplier=self.emp_death_penalty_horizon_multiplier
            )
            new_emp.initialize(
                skill_level=child_skill,
                risk_aversion=child_risk_aversion,
                tax_morale=child_tax_morale,
                initial_cash=newborn_cash,
                initial_energy=1.0,
                age=18
            )
            new_emp.status = LifeCycleStatus.ACTIVE
            new_emp.parent_id = parent_id_payload
            self.agents[new_eid] = new_emp
            self.births_this_step += 1
            self.event_bus.publish(Event(
                event_type=EventType.AGENT_BORN,
                source_id=birth_source_id,
                target_id=new_eid,
                payload={"cash": round(new_emp.cash, 1), "skill": round(new_emp.skill_level, 2), "parent_id": parent_id_payload},
                timestep=self.timestep
            ))

        # Economy doc births_this_step CUA BUOC NAY o buoc SAU (do tre 1 buoc co chu dich -- xem
        # Economy.__init__::last_births_this_step) -- rule_engine.execute_cycle() da chay VA
        # calculate_reward() cua MOI agent (bao gom Economy) da tinh xong TRUOC khi nhanh sinh
        # san nay chay, nen khong the phan anh dung buoc nay ma khong doi lai thu tu step().
        self.eco.last_births_this_step = self.births_this_step

        # B. GIA NHẬP THỊ TRƯỜNG THEO JORGENSON (1963) & QUY MÔ MES (Bain, 1956)
        active_firms_list = [a for a in self.agents.values() if isinstance(a, Firm) and a.status == LifeCycleStatus.ACTIVE]
        current_firm_count = len(active_firms_list)
        # HE SO CAU TRUC TU DO HIEU CHINH, doc tu self.* (v0.20, chuyen tu hardcode cuc
        # bo sang ScenarioConfig de nhat quan voi HARD_MIN_EMP/HARD_MAX_EMP o tren --
        # phat hien qua audit toan du an, xem KNOWN_PATHOLOGIES.md). Jorgenson (1963)/
        # Bain (1956) chi xac lap DIEU KIEN gia nhap nganh, KHONG cho gioi han so luong
        # firm toi da.
        HARD_MAX_FIRMS = self.hard_max_firms

        total_market_capital = sum(f.capital_stock for f in active_firms_list)
        total_market_profit = sum(getattr(f, "last_profit", 0.0) for f in active_firms_list)
        market_return_on_capital = (total_market_profit / max(100.0, total_market_capital)) if total_market_capital > 0 else 0.0

        # Điều kiện gia nhập: Lợi nhuận vốn vượt chi phí cơ hội vốn (lãi suất tiền gửi bình
        # quân toàn hệ thống ngân hàng) + có thặng dư lao động
        avg_deposit_rate = float(np.mean([b.deposit_rate for b in self.banks]))
        # LUU Y THIET KE (phat hien qua thao luan voi nguoi dung + Claude Web, 2026-09-24):
        # dieu kien is_profitable_industry NGUYEN BAN (margin=0.0, mac dinh) la MOT CONG
        # PROCYCLICAL tu than -- dung luc suy thoai (loi nhuan firm sup vi quy luong bi khoa
        # cao boi co che "cong lương" da ghi o KNOWN_PATHOLOGIES.md) can firm moi hap thu lao
        # dong du thua nhat thi dieu kien nay lai kho thoa nhat, keo dai pha xau thay vi rut
        # ngan. firm_entry_profitability_margin (HE SO CAU TRUC TU DO HIEU CHINH, mac dinh 0.0
        # = hanh vi CU khong doi) cho phep noi long: gia tri duong ha nguong loi nhuan can thiet
        # xuong duoi avg_deposit_rate, mo phong chinh sach khuyen khich gia nhap thi truong thoi
        # ky suy thoai (countercyclical entry subsidy) -- dung de ablation TACH BACH voi so
        # luong firm ban dau/tran firm (HARD_MAX_FIRMS): bien nay chi doi CHINH dieu kien gia
        # nhap co thoa hay khong, khac voi firm_entry_probability (toc do gia nhap KHI dieu
        # kien DA thoa) va num_firms/hard_max_firms (quy mo/tran so luong).
        is_profitable_industry = (market_return_on_capital > (avg_deposit_rate - self.firm_entry_profitability_margin))
        has_excess_labor = (unemployment_rate > self.firm_entry_unemployment_threshold and unemployed_count >= 2)
        emergency_repair = (current_firm_count < 2 and unemployed_count >= 2)

        # Xac suat gia nhap MOI BUOC khi du dieu kien loi nhuan/lao dong da thoa (HE SO CAU
        # TRUC TU DO HIEU CHINH, doc tu self.firm_entry_probability -- v0.20 bo sung hang so
        # nay, v0.26 chuyen tu hardcode cuc bo sang ScenarioConfig) -- Jorgenson (1963)/Bain
        # (1956) chi xac lap DIEU KIEN gia nhap, khong cho toc do/xac suat gia nhap cu the.
        should_incorporate = (
            emergency_repair or
            (current_firm_count < HARD_MAX_FIRMS and is_profitable_industry and has_excess_labor
             and np.random.rand() < self.firm_entry_probability)
        )

        if should_incorporate:
            new_fid = f"firm_{self.next_firm_id}"
            self.next_firm_id += 1

            # Vốn mồi quy mô tối thiểu MES (Bain, 1956)
            current_p = max(0.5, self.eco.base_living_cost)
            grant_firm = float(np.clip(80.0 * current_p, 1500.0, 4500.0))
            if self.gov.treasury >= (grant_firm * 2.0):
                self.gov.treasury -= grant_firm
            else:
                grant_firm = 1000.0
                self.gov.treasury -= grant_firm

            new_firm = Firm(agent_id=new_fid)
            new_firm.initialize(
                productivity_factor=float(np.random.uniform(1.3, 1.8)),
                risk_aversion=float(np.random.uniform(0.2, 0.4)),
                tax_morale=float(np.random.uniform(0.5, 0.85)),
                initial_capital=8000.0
            )
            new_firm.age_months = 0
            new_firm.cash = grant_firm
            new_firm.status = LifeCycleStatus.ACTIVE
            self.agents[new_fid] = new_firm
            self.event_bus.publish(Event(
                event_type=EventType.HIRE,
                source_id="MARKET",
                target_id=new_fid,
                payload={"capital": new_firm.capital_stock, "status": "INCORPORATED"},
                timestep=self.timestep
            ))

        raw_state = self.get_raw_environment_state()
        is_time_up = self.timestep >= self.max_steps

        observations = {aid: sanitize_observation(a.observe(raw_state).vector) for aid, a in self.agents.items()}
        rewards = {aid: self._finalize_reward(a, rewards_all.get(aid, 0.0)) for aid, a in self.agents.items()}
        terminateds = {"__all__": False}
        truncateds = {"__all__": is_time_up}
        infos = {aid: {"status": a.status.name} for aid, a in self.agents.items()}

        for aid in self.agents.keys():
            terminateds[aid] = False
            truncateds[aid] = is_time_up

        # Gộp lần cuối cho các agent vừa chết/phá sản bước NÀY (xem chú thích
        # đầy đủ tại nơi dựng terminal_agents_this_step phía trên). Observation
        # được tính từ chính object gốc (đã cập nhật đúng trạng thái lúc chết
        # qua apply_result trước đó trong step() này), KHÔNG dùng vector rỗng
        # -- đảm bảo policy network backup đúng giá trị tại state hấp thụ thật,
        # không phải nhiễu ngẫu nhiên. terminated=True, truncated=False (kết
        # thúc MDP có chủ đích, không phải bị cắt vì hết giờ episode chung).
        for t_id, t_agent in terminal_agents_this_step.items():
            observations[t_id] = sanitize_observation(t_agent.observe(raw_state).vector)
            rewards[t_id] = self._finalize_reward(t_agent, rewards_all.get(t_id, 0.0))
            terminateds[t_id] = True
            truncateds[t_id] = False
            infos[t_id] = {"status": t_agent.status.name}

        return observations, rewards, terminateds, truncateds, infos

    def get_raw_environment_state(self) -> Dict[str, Any]:
        """Chuẩn hóa tensor vĩ mô theo Fisher (1911) và Allingham & Sandmo (1972)."""
        active_employees = [a for a in self.agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE]
        unemployed_count = sum(1 for e in active_employees if e.employed_by is None)
        unemployment_rate = (unemployed_count / len(active_employees)) if active_employees else 0.0

        employed = [e for e in active_employees if e.employed_by is not None]
        avg_wage = (sum(e.wage for e in employed) / len(employed)) if employed else self.eco.base_living_cost

        active_firms = [a for a in self.agents.values() if isinstance(a, Firm) and a.status == LifeCycleStatus.ACTIVE]
        total_credit_demand = sum(f.debt for f in active_firms)

        total_emp_cash = sum(a.cash for a in active_employees)
        total_emp_deposits = sum(getattr(a, "bank_deposit", 0.0) for a in active_employees)
        total_firm_cash = sum(a.cash for a in active_firms)
        # M2 chuẩn = tiền mặt lưu hành + tiền gửi ngân hàng do khu vực TƯ (công
        # chúng) nắm giữ (Mishkin, F., 2019, "The Economics of Money, Banking
        # and Financial Markets", 12th ed., Pearson, Ch.3). Dự trữ ngân hàng
        # (bank.reserves) KHÔNG thuộc M2 theo định nghĩa chuẩn vì đó là tài sản
        # nội bộ của hệ thống ngân hàng, không phải tiền do công chúng nắm giữ
        # -- gộp cả hai sẽ tính trùng phần "hậu thuẫn" cho tiền gửi. Kho bạc
        # (Treasury) được cộng vào theo quy ước riêng của mô hình này để phản
        # ánh tổng sức mua danh nghĩa còn lưu hành trong nền kinh tế mô phỏng.
        m2_supply = max(1.0, self.gov.treasury + total_emp_cash + total_emp_deposits + total_firm_cash)

        # Vận tốc lưu thông tiền tệ Fisher: V = GDP / M2 (Fisher, 1911)
        velocity_of_money = float(self.gov.current_gdp / m2_supply)

        # Thất thu thuế ước tính từ số tiền phạt và số vụ phát hiện (Allingham & Sandmo, 1972)
        fines = getattr(self.sup, 'fines_collected', 0.0)
        multiplier = max(1.0, getattr(self.sup, 'fine_multiplier', 1.5))
        violations = getattr(self.sup, 'violations_detected', 0)
        real_evasion_estimate = float((fines / multiplier) + (violations * self.eco.base_living_cost * 0.5))

        # Lãi suất vĩ mô quan sát được (dùng cho Ω_i của Employee/Firm) là bình
        # quân gia quyền theo dự trữ (reserve-weighted average) trên toàn bộ hệ
        # thống ngân hàng -- một chỉ số "lãi suất thị trường" tổng hợp, tương tự
        # khái niệm prime rate thị trường; giao dịch tín dụng THỰC TẾ vẫn dùng
        # đúng lending_rate/deposit_rate của ngân hàng đối tác cụ thể (xem
        # rule_engine.py Section 5, 8B -- quan hệ tín dụng Petersen & Rajan, 1994).
        total_reserves = sum(b.reserves for b in self.banks)
        if total_reserves > 0.0:
            avg_lending_rate = sum(b.lending_rate * b.reserves for b in self.banks) / total_reserves
            avg_deposit_rate_w = sum(b.deposit_rate * b.reserves for b in self.banks) / total_reserves
        else:
            avg_lending_rate = float(np.mean([b.lending_rate for b in self.banks]))
            avg_deposit_rate_w = float(np.mean([b.deposit_rate for b in self.banks]))

        return {
            "timestep": self.timestep,
            "macro_indicators": {
                "inflation": self.eco.inflation_rate,
                "base_living_cost": self.eco.base_living_cost,
                "worker_tax_rate": self.gov.tax_rate_worker,
                "firm_tax_rate": self.gov.tax_rate_firm,
                "base_interest_rate": avg_deposit_rate_w,
                "bank_lending_rate": avg_lending_rate,
                "unemployment_rate": unemployment_rate,
                "average_wage": avg_wage,
                "market_demand_factor": float(np.clip(velocity_of_money * 10.0, 0.1, 5.0)),
                "total_credit_demand": total_credit_demand,
                "estimated_tax_evasion": real_evasion_estimate,
                "aggregate_demand": self.eco.step_trade_volume,
                "aggregate_supply": sum(f.capital_stock for f in active_firms)
            }
        }

    def export_full_world_state(self) -> Dict[str, Any]:
        total_emp_cash = sum(a.cash for a in self.agents.values() if isinstance(a, Employee))
        total_emp_deposits = sum(getattr(a, "bank_deposit", 0.0) for a in self.agents.values() if isinstance(a, Employee))
        total_firm_cash = sum(a.cash for a in self.agents.values() if isinstance(a, Firm))
        # Xem chú thích chi tiết định nghĩa M2 tại get_raw_environment_state().
        m2_supply = self.gov.treasury + total_emp_cash + total_emp_deposits + total_firm_cash

        active_employees_list = [a for a in self.agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE]
        employed_list = [e for e in active_employees_list if e.employed_by is not None]
        unemployment_rate = 1.0 - (len(employed_list) / max(1, len(active_employees_list)))
        avg_wage = (sum(e.wage for e in employed_list) / len(employed_list)) if employed_list else 0.0
        total_loans_all = sum(b.total_loans for b in self.banks)
        total_npl_all = sum(b.non_performing_loans for b in self.banks)

        return {
            "timestep": self.timestep,
            "agents": {agent_id: agent.export_state() for agent_id, agent in self.agents.items()},
            "macro": {
                "gdp": self.gov.current_gdp,
                "gini": self.gov.current_gini,
                "treasury": self.gov.treasury,
                "bank_reserves": sum(b.reserves for b in self.banks),
                "bank_deposits": sum(b.total_deposits for b in self.banks),
                "npl": total_npl_all,
                # npl_ratio_pct dung TOTAL LOANS lam mau so (khong phai reserves --
                # xem giai thich o rllib_wrapper.py InstitutionalMetricsCallback).
                "npl_ratio_pct": compute_npl_ratio_pct(total_npl_all, total_loans_all),
                "bank_loans": total_loans_all,
                "living_cost": self.eco.base_living_cost,
                "housing_price": self.eco.housing_price,
                "inflation": self.eco.inflation_rate,
                "m2_supply": m2_supply,
                "active_employees": len(active_employees_list),
                "employed_count": len(employed_list),
                "unemployment_rate_pct": float(unemployment_rate * 100.0),
                "avg_wage": float(avg_wage),
                "active_firms": sum(1 for a in self.agents.values() if isinstance(a, Firm) and a.status == LifeCycleStatus.ACTIVE)
            }
        }