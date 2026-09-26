from dataclasses import dataclass, asdict, fields
from typing import Any, Dict
import yaml


@dataclass
class ScenarioConfig:
    """
    Cau hinh MOT kich ban calibration duy nhat cho toan bo tham so kinh te
    tu do hieu chinh cua mo phong -- thay the viec them --flag CLI rieng le
    cho tung hang so moi lan can chinh tay (da xay ra lap di lap lai qua
    nhieu phien lam viec: inheritance_fraction, min_newborn_cash, gini_
    penalty_coef, death_penalty_coef, npl_*_penalty_coef...).

    PHAN LOAI RUI RO (Claude Web, ap dung dung nguyen tac "3 tang"): day
    la CAU HINH (tang rui ro THAP) -- KHONG doi bat ky dang ham (functional
    form) kinh te nao trong rule_engine.py/agents/*.py. Moi field duoi day
    chi thay doi GIA TRI hang so dau vao cho dung cong thuc da co san va da
    duoc trich dan; gia tri mac dinh cua MOI field = DUNG gia tri dang
    hardcode truoc khi co ScenarioConfig, nen bo qua --config (hoac dung
    scenarios/default.yaml) cho hanh vi giong het truoc day -- khong co
    thay doi ngam an nao.
    NGOAI LE CO CHU Y (v0.15, xem CLAUDE_HISTORY.md): cac field reward_scale_*,
    reward_clip va subsistence_indexation_ceiling_mult LA thay doi hanh vi co chu
    dich (chuan hoa reward cho critic; tran chi so hoa gia) -- khong con tuong duong
    hardcode cu. Muon tai tao tinh trang cu: reward_scale_* = 1.0, reward_clip >= 300,
    subsistence_indexation_ceiling_mult rat lon.

    Cach dung:
        cfg = ScenarioConfig.from_yaml("scenarios/em_baseline.yaml")
        env = MacroEnvironment(**cfg.to_env_kwargs())
    """
    # --- Quy mo dan so & thoi luong episode ---
    num_employees: int = 50
    num_firms: int = 5
    num_banks: int = 1
    max_steps: int = 240

    # --- Nhan khau hoc noi sinh (Epstein & Axtell, 1996, "Growing Artificial
    # Societies", Brookings/MIT Press -- co che sinh san Sugarscape; xem
    # be/env.py Section A cho chi tiet trich dan day du) ---
    inheritance_fraction: float = 0.15          # ty le tai san cha/me chuyen cho con khi sinh
    min_newborn_cash: float = 500.0             # san an sinh toi thieu cho newborn (Kho bac bu them neu thua ke chua du)
    min_reproduction_age: int = 22              # tuoi lao dong da on dinh, du tu cach sinh san
    max_reproduction_age: int = 45              # can tren do tuoi sinh san con nang dong kinh te
    min_reproduction_wealth_mult: float = 3.0   # so thang chi phi song can du tich luy ("sugar") de sinh san
    trait_mutation_sigma: float = 0.05          # do lech chuan dot bien Gaussian quanh dac diem di truyen cha/me
    hard_min_emp: int = 30                      # san dan so -- duoi muc nay kich hoat luoi an sinh khan cap
    hard_max_emp: int = 200                     # tran dan so cho phep (khong doi toc do tang truong toi da/step)
    # HE SO CAU TRUC TU DO HIEU CHINH (v0.20, chuyen tu hardcode cuc bo trong env.py sang
    # ScenarioConfig de nhat quan voi hard_min_emp/hard_max_emp o tren -- phat hien qua
    # audit toan du an). Jorgenson (1963)/Bain (1956) chi xac lap DIEU KIEN gia nhap nganh
    # (loi nhuan vuot chi phi von + du lao dong), KHONG cho gioi han so luong firm toi da.
    # MAC DINH DOI (v0.29, 2026-09-25): 7 -> 25, theo ket qua ablation "loose_entry_v2"
    # (KNOWN_PATHOLOGIES.md muc #21) -- tran 7 cu la rang buoc bo chat gay hieu ung phan chu ky
    # tu than (procyclical entry barrier), da do thuc nghiem 6 nhanh ablation khac nhau. Dat
    # hard_max_firms=7 de tai hien DUNG hanh vi CU (tran chat, chua sua).
    hard_max_firms: int = 25                     # tran so luong firm dong thoi cho phep

    # Co cau gia nhap nganh (v0.26, chuyen tu hardcode cuc bo trong env.py::step() Section B
    # sang ScenarioConfig -- phuc vu ablation tach bach "so luong firm" khoi "co che gia nhap",
    # phat hien qua thao luan voi nguoi dung + Claude Web 2026-09-24, xem KNOWN_PATHOLOGIES.md
    # va METHODOLOGY_NOTES.md).
    firm_entry_probability: float = 0.15         # xac suat gia nhap MOI BUOC khi du dieu kien
    firm_entry_unemployment_threshold: float = 0.08  # nguong that nghiep kich hoat "du lao dong"
    # MAC DINH DOI (v0.29, 2026-09-25): 0.0 -> 0.08, cung ket qua ablation "loose_entry_v2" o tren
    # (chi phat huy tac dung KHI hard_max_firms du cao -- 2 field nay phai doi CUNG NHAU, xem
    # KNOWN_PATHOLOGIES.md muc #21 "nhanh thu 2" cho bai hoc confound khi chi doi mot trong hai).
    # Dat =0.0 de tai hien DUNG hanh vi CU (procyclical, khong noi long dieu kien gia nhap).
    firm_entry_profitability_margin: float = 0.08    # ha nguong loi nhuan can de gia nhap

    # --- Economy "an sinh vi mo" + binh on thi truong bang du tru dem (v0.24, xem
    # METHODOLOGY_NOTES.md muc 1-3) ---
    # San mau so ty le tu vong/sinh cua Economy -- field RIENG, KHONG dung chung hard_min_emp
    # (rui ro coupling an giua 2 co che khong thiet ke de phoi hop).
    mortality_rate_floor: int = 30
    # Von mo cap MOT LAN tu Treasury cho quy binh on du tru dem cua Economy luc reset.
    initial_economy_buffer_fund: float = 10000.0

    # --- Ngan hang: lai suat khoi tao (QUY UOC: annual/nam, xem bank.py) ---
    initial_lending_rate: float = 0.06          # 6%/nam
    initial_deposit_rate: float = 0.02          # 2%/nam

    # --- He so hieu chinh reward (KHONG doi dang ham, chi doi HANG SO dau vao) ---
    gini_penalty_coef: float = 25.0             # Government.calculate_reward -- phat Gini^2
    death_penalty_coef: float = 20.0            # Government.calculate_reward -- phat moi ca tu vong
    npl_flow_penalty_coef: float = 0.06         # Bank.calculate_reward -- phat no xau MOI phat sinh
    npl_stock_penalty_coef: float = 50.0        # Bank.calculate_reward -- phat theo TY LE ton kho NPL/tong du no
    npl_writeoff_months: int = 6                # Bank.apply_result -- so thang no xau duoc "mo" truoc khi write-off (IFRS 9 / Basel NPL staging, xem bank.py)
    emp_death_penalty_base: float = 100.0                 # Employee.calculate_reward -- muc phat tu vong goc (ratio=0, tuc chet dung luc max_age)
    emp_death_penalty_horizon_multiplier: float = 1.0     # Employee.calculate_reward -- he so nhan them theo ty le quang doi con lai (Viscusi & Aldy VSL), xem rule_engine.py Section 9

    # --- Chuan hoa reward theo loai tac tu (MacroEnvironment._finalize_reward) ---
    # Dua |return| chiet khau p90 ve ~20 (< sqrt(vf_clip_param)) de critic PPO hoc duoc
    # (Engstrom et al., 2020; Andrychowicz et al., 2021). Suy tu phan phoi return do
    # duoc duoi policy ngau nhien -- HE SO HIEU CHINH, khong phai cong thuc.
    reward_scale_employee: float = 0.045
    reward_scale_firm: float = 0.018
    reward_scale_government: float = 0.04
    reward_scale_bank: float = 0.04
    reward_scale_supervisor: float = 0.012
    reward_scale_economy: float = 0.02
    reward_clip: float = 100.0                 # san/tran an toan SAU chuan hoa; env tu tu choi cau hinh clip < hinh phat tu vong toi da

    # --- Tran chi so hoa chi tieu sinh ton theo gia (rule_engine.py Section 4) ---
    subsistence_indexation_ceiling_mult: float = 3.0   # boi so cua initial_living_cost; xem rule_engine.py

    # --- Dieu kien khoi tao (env.py reset(), v0.17 -- xem CLAUDE_HISTORY.md) ---
    initial_living_cost: float = 6.0            # gia sinh hoat luc reset(); DO thuc nghiem (khong suy doan), truoc day hardcode=20 gay cu soc gia dau episode
    initial_employment_rate: float = 0.90       # ty le dan so co viec lam san luc reset(); con lai la that nghiep ma sat (Mortensen & Pissarides, 1994 -- ty le cu the la calibration)

    # --- He so hieu chinh quy mo MRPL/san luong (rule_engine.py Section 2/3, v0.19) ---
    mrpl_scale_constant: float = 0.38         # giai dai so tu dieu kien can bang wage/price; dat 1.0 de tai hien lech chuan dinh co CU (xem KNOWN_PATHOLOGIES.md muc #8)

    # --- Dam phan lai luong Calvo-style (rule_engine.py Section 3, v0.28 -- Taylor 1980;
    # Erceg, Henderson & Levin 2000) -- sua tan goc "coc luong" (KNOWN_PATHOLOGIES.md muc
    # wage-mrpl-ratchet): luong lao dong DA co viec gio co the duoc dam phan lai dinh ky thay vi
    # khoa vinh vien. =0.0 tai hien DUNG hanh vi CU (khoa vinh vien).
    wage_renegotiation_prob: float = 0.12

    @classmethod
    def from_yaml(cls, path: str) -> "ScenarioConfig":
        """Doc 1 file YAML kich ban. Truong khong duoc nhan dien se rai loi
        ro rang (thay vi am tham bi bo qua) de tranh go sai ten tham so ma
        khong hay biet."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        known_fields = {fld.name for fld in fields(cls)}
        unknown = set(raw.keys()) - known_fields
        if unknown:
            raise ValueError(
                f"[ScenarioConfig] File '{path}' co truong khong duoc nhan dien: "
                f"{sorted(unknown)}. Cac truong hop le: {sorted(known_fields)}"
            )
        return cls(**raw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_env_kwargs(self) -> Dict[str, Any]:
        """Tra ve dict khop CHINH XAC tham so constructor cua MacroEnvironment
        (be/env.py) -- dung truc tiep MacroEnvironment(**cfg.to_env_kwargs())."""
        return self.to_dict()
