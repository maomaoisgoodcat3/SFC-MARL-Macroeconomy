"""YEU_CAU v4 P1.1 (A5): free_constants.csv — danh muc hang so. CHI DOC. So dong tim TU DONG bang regex trong ma dong bang (khong ghi tay).
Cot 'loai': gan_moc_quy_mo (phu thuoc quy mo tien/gia/dan so — da/can quet khi doi quy mo, CLAUDE.md) / tu_do (he so tu do hieu chinh) /
lay_tu_nguon (gia tri lay truc tiep tu tai lieu) / chon_bang_thi_nghiem_co_lap (da do/quet co chu dich) / van_hanh (khong phai kinh te).
Phan loai + cot 'quet' do Claude Code gan tay tu comment ma va KNOWN_PATHOLOGIES (dong KP ghi kem); KHONG RO = chua kiem tay duoc.
Dung: python thesis_support/v4/make_free_constants.py"""
import csv
import dataclasses
import os
import re
import sys

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO)
sys.path.insert(0, REPO)
from be.scenario_config import ScenarioConfig  # noqa: E402

Y = yaml.safe_load(open("scenarios/em_baseline.yaml", encoding="utf-8"))


def find(path, pattern):
    for i, l in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
        m = re.search(pattern, l)
        if m and ("#" not in l[:m.start()]):  # bo qua dong comment
            return f"{path}:{i}", l.strip()[:160]
    return f"{path}:KHONG TIM THAY", ""


# (loai, don_vi, nguon_trong_comment / ghi chu, da_quet_do_nhay)
ANN = {
    "num_employees": ("van_hanh", "nguoi", "quy mo mo phong", "khong"),
    "num_firms": ("van_hanh", "firm", "quy mo mo phong", "co — ablation so firm/cong gia nhap KP#21 (dong 588-600)"),
    "num_banks": ("van_hanh", "bank", "", "khong (chi test SFC 2 bank)"),
    "max_steps": ("van_hanh", "buoc (thang)", "20 nam", "khong"),
    "inheritance_fraction": ("tu_do", "ty le", "", "khong"),
    "min_newborn_cash": ("gan_moc_quy_mo", "tien", "san an sinh tre so sinh; KHONG chi so hoa (KP#30 bang quet, #29c)", "khong (ghi nhan KHONG CHAN)"),
    "min_reproduction_age": ("tu_do", "nam", "", "khong"),
    "max_reproduction_age": ("tu_do", "nam", "", "khong"),
    "min_reproduction_wealth_mult": ("tu_do", "thang chi phi song", "Epstein & Axtell 1996 ('sugar')", "khong"),
    "trait_mutation_sigma": ("tu_do", "ty le", "", "khong"),
    "hard_min_emp": ("gan_moc_quy_mo", "nguoi", "san luoi an sinh khan cap", "khong"),
    "hard_max_emp": ("gan_moc_quy_mo", "nguoi", "", "khong"),
    "hard_max_firms": ("chon_bang_thi_nghiem_co_lap", "firm", "nhanh loose_entry_v2", "co — KP#21 ablation 3 nhanh x 40 iter, 1 seed (7 vs 25)"),
    "firm_entry_probability": ("tu_do", "xac suat/buoc", "", "khong"),
    "firm_entry_unemployment_threshold": ("tu_do", "ty le", "", "khong"),
    "firm_entry_profitability_margin": ("chon_bang_thi_nghiem_co_lap", "ty le", "bien gia nhap 8%", "co — KP#21 (0.0 vs 0.08)"),
    "mortality_rate_floor": ("tu_do", "nguoi", "san mau so reward Economy (METHODOLOGY_NOTES muc 3)", "khong"),
    "initial_economy_buffer_fund": ("gan_moc_quy_mo", "tien", "quy du tru 10 000", "co — KP#30 bang quet hang so quy mo"),
    "initial_treasury": ("gan_moc_quy_mo", "tien", "Kho bac 25 000 (sau cap von khoi tao)", "co — KP#29c: 25k vs 60k vs 1e6, rho 0.5/1 (KP dong 1189-1197; nguon so V6 KHONG TIM THAY)"),
    "treasury_funds_initial_endowments": ("van_hanh", "bool", "True = logic cu (KP#29c)", "khong"),
    "government_reward_mode": ("van_hanh", "-", "eq_x_prod (Zheng et al. 2022)", "khong"),
    "swf_reward_scale": ("tu_do", "he so", "chuan hoa Eq×Prod ~250 -> ~5/buoc", "khong (CLAUDE.md: KHONG CHAN)"),
    "initial_lending_rate": ("lay_tu_nguon", "/nam", "IMF Financial Access Survey 2023 (em_baseline.yaml)", "khong"),
    "initial_deposit_rate": ("lay_tu_nguon", "/nam", "World Bank GFDD 2023 (em_baseline.yaml)", "khong"),
    "gini_penalty_coef": ("tu_do", "he so", "chi dung o reward_mode legacy", "khong"),
    "death_penalty_coef": ("tu_do", "/ca chet", "nang 5 -> 20 sau quan sat log (government.py docstring)", "khong co quet he thong"),
    "npl_flow_penalty_coef": ("tu_do", "he so", "nang 0.02 -> 0.06 sau quan sat (bank.py docstring)", "khong"),
    "npl_stock_penalty_coef": ("tu_do", "he so", "", "khong"),
    "npl_writeoff_months": ("lay_tu_nguon", "thang", "IFRS 9 / Basel NPL (comment; chua doi chieu dieu khoan — A4)", "khong"),
    "bank_failure_criterion": ("van_hanh", "-", "equity (KP#28)", "khong"),
    "bank_failure_floor": ("gan_moc_quy_mo", "tien", "nguong pha san ngan hang 100 000", "khong"),
    "emp_death_penalty_base": ("tu_do", "reward", "", "khong"),
    "emp_death_penalty_horizon_multiplier": ("tu_do", "he so", "Viscusi & Aldy (comment)", "khong"),
    "reward_scale_employee": ("tu_do", "he so", "chuan hoa reward (KP#5, #12)", "khong"),
    "reward_scale_firm": ("tu_do", "he so", "", "khong"),
    "reward_scale_government": ("tu_do", "he so", "", "khong"),
    "reward_scale_bank": ("tu_do", "he so", "", "khong"),
    "reward_scale_supervisor": ("tu_do", "he so", "", "khong"),
    "reward_scale_economy": ("tu_do", "he so", "", "khong"),
    "reward_clip": ("van_hanh", "reward", "env tu choi neu cat mat phat tu vong", "khong"),
    "subsistence_indexation_ceiling_mult": ("gan_moc_quy_mo", "x initial_living_cost", "tran chi so hoa 3x", "co — KP#2/#8 (1e9 tai hien loi; do he so khuech dai)"),
    "initial_living_cost": ("chon_bang_thi_nghiem_co_lap", "tien/thang", "DO thuc nghiem (truoc 20 -> soc; KP#10)", "co — KP#10 (20 vs 6)"),
    "initial_employment_rate": ("tu_do", "ty le", "Mortensen & Pissarides 1994 (comment)", "co — KP#10 (0.0 vs 0.9)"),
    "mrpl_scale_constant": ("chon_bang_thi_nghiem_co_lap", "he so", "quet 3 chi so dong thoi (rule_engine.py:147-176)", "co — KP#8 (1.0 tai hien loi; 0.38 chon qua quet)"),
    "wage_renegotiation_prob": ("chon_bang_thi_nghiem_co_lap", "xac suat/thang", "Taylor 1980; Calvo 1983", "co — KP#24 dose-response 4 seed (0 vs 0.12; de nghi them 0.06)"),
    "shirking_monitor_prob": ("tu_do", "xac suat/thang", "Shapiro & Stiglitz 1984", "co — KP#27 (q=0 vs 0.05; test_effort_has_private_benefit_only_with_monitoring)"),
    "shirking_effort_threshold": ("tu_do", "effort", "", "khong"),
    "effort_signal_noise_sigma": ("tu_do", "do lech chuan", "Holmstrom 1979", "khong"),
    "subsidy_funding_rule": ("van_hanh", "-", "affordable (KP#30b)", "khong"),
    "buffer_stock_rule": ("van_hanh", "-", "band_scaled (KP#31)", "khong"),
    "buffer_max_market_share": ("tu_do", "ty le", "kappa", "co — KP#31 test hoi quy (logic cu vs moi), khong quet kappa"),
    "buffer_price_band": ("tu_do", "ty le", "+-5%", "khong"),
    "buffer_reference_halflife_months": ("tu_do", "thang", "EMA ban ra 12", "khong"),
    "shirker_rehire_lockout_months": ("chon_bang_thi_nghiem_co_lap", "thang", "Gibbons & Katz 1991", "co — KP#27: L=6 gap +0.34±0.36 (khong y nghia), L=9 +1.91±0.60 (16 seed)"),
}

# Hang so VIET CUNG trong ma (khong phai field ScenarioConfig)
HARD = [
    ("ALPHA_CAPITAL (so mu von)", "0.3", "be/rule_engine.py", r"ALPHA_CAPITAL = 0\.3", "tu_do", "-", "Cobb-Douglas", "khong"),
    ("BETA_LABOR (so mu lao dong)", "0.6", "be/rule_engine.py", r"BETA_LABOR = 0\.6", "lay_tu_nguon", "-", "ty trong thu nhap lao dong quan sat (comment)", "khong"),
    ("THETA_CALVO", "0.70", "be/rule_engine.py", r"THETA_CALVO = 0\.70", "lay_tu_nguon", "-", "Calvo 1983 (gia tri chuan DSGE)", "do he so khuech dai (test) — khong quet"),
    ("he so san luong phi chinh thuc", "0.35", "be/rule_engine.py", r"0\.35 \* e\.skill_level", "tu_do", "x skill", "'du song thoi thop'", "khong (de xuat B cho #32)"),
    ("HAIRCUT_LIQUIDATION", "0.30", "be/rule_engine.py", r"HAIRCUT_LIQUIDATION = 0\.30", "tu_do", "ty le", "Merton 1974 (dang ham); 0.30 tu do", "khong"),
    ("dieu kien thuê: thanh khoan > k x gia", "1.2", "be/rule_engine.py", r"expected_price \* 1\.2", "tu_do", "x gia", "", "khong"),
    ("tran tuyen/thang", "max(2, ceil(0.25 h))", "be/rule_engine.py", r"MAX_HIRES_PER_MONTH = ", "tu_do", "nguoi", "", "khong"),
    ("dieu kien nhan tro cap", "0.8 x actual_living_cost", "be/rule_engine.py", r"0\.8 \* actual_living_cost", "tu_do", "x chi phi song", "", "khong"),
    ("lich tro cap thang 4-6", "0.375", "be/rule_engine.py", r"0\.375 \* relief_level", "tu_do", "x relief", "ty le 0.15/0.40 cu", "khong"),
    ("bien do bom cau", "+-0.20", "be/agents/government.py", r"np\.clip\(vals\[3\], -0\.20, 0\.20\)", "tu_do", "x tong cau sinh ton", "giu tu thiet ke v0.22", "khong"),
    ("tran thue TNCN/DN", "0.50", "be/agents/government.py", r"np\.clip\(vals\[0\], 0\.0, 0\.50\)", "tu_do", "ty le", "'hien phap'", "khong"),
    ("tra no cong/buoc", "0.5 x Kho bac", "be/agents/government.py", r"self\.treasury \* 0\.5", "tu_do", "ty le", "", "khong"),
    ("phat no cong (reward Gov)", "0.00005", "be/agents/government.py", r"public_debt \* 0\.00005", "tu_do", "/don vi no", "", "khong"),
    ("K_SURVIVAL (reward Economy)", "1700", "be/agents/economy.py", r"K_SURVIVAL = 1700", "tu_do", "he so", "khop thang dead_worker_penalty cu (~20)", "khong"),
    ("K_BIRTH (reward Economy)", "4000", "be/agents/economy.py", r"K_BIRTH = 4000", "chon_bang_thi_nghiem_co_lap", "he so", "1e6 -> vf_ev ~0 (KP#19, dong 265-271)", "co — 2 lan huan luyen (1e6 vs 4000)"),
    ("muc tieu ty le sinh", "0.0033", "be/agents/economy.py", r"target_birth_rate = 0\.0033", "tu_do", "/nguoi/buoc", "khop log (~0.25/buoc o dan so ~75)", "khong"),
    ("muc tieu lam phat (reward Economy)", "0.0016", "be/agents/economy.py", r"0\.0016", "tu_do", "/thang", "Svensson 1997 (dang ham)", "khong"),
    ("he so phat lam phat", "50", "be/agents/economy.py", r"inflation_penalty = 50\.0", "tu_do", "he so", "", "khong"),
    ("reward Firm: profit x", "0.1", "be/agents/firm.py", r"last_profit \* 0\.1", "tu_do", "he so", "", "khong"),
    ("reward Firm: thuong quan so", "0.5/nguoi", "be/agents/firm.py", r"\* 0\.5\) if self\.last_profit > 0", "tu_do", "/nhan vien", "Mortensen & Pissarides 1994 (comment) — KP#35", "khong"),
    ("reward Worker: he so log tieu dung", "2.0", "be/agents/employee.py", r"np\.log\(consumption\)\) \* 2\.0", "tu_do", "he so", "", "khong"),
    ("reward Worker: tran phat that nghiep", "8.0 (2+0.5 streak)", "be/agents/employee.py", r"min\(2\.0 \+ \(0\.5", "tu_do", "reward", "", "khong"),
    ("PPO vf_clip_param", "2000", "be/rllib_wrapper.py", r"PPO_VF_CLIP_PARAM: float = 2000", "chon_bang_thi_nghiem_co_lap", "-", "500 -> critic khong hoc (KP#12)", "co — KP#12 (500 vs 2000)"),
    ("PPO lr / epoch / grad_clip", "3e-4 / 10 / 0.5", "be/rllib_wrapper.py", r"PPO_LR: float = 3e-4", "van_hanh", "-", "", "khong"),
]

rows = []
src = open("be/scenario_config.py", encoding="utf-8").read().splitlines()
for f in dataclasses.fields(ScenarioConfig):
    ln = next((i + 1 for i, l in enumerate(src) if re.match(rf"\s+{f.name}\s*:", l)), None)
    com = src[ln - 1].split("#", 1)[1].strip() if ln and "#" in src[ln - 1] else ""
    loai, unit, note, sweep = ANN.get(f.name, ("KHONG RO", "", "", "KHONG RO"))
    val = Y.get(f.name, f.default)
    rows.append(dict(name=f.name, value=val, default_class=f.default, unit=unit, file_line=f"be/scenario_config.py:{ln}",
                     yaml_override=("scenarios/em_baseline.yaml" if f.name in Y else ""), type=loai,
                     source_comment=(com or note)[:200], note=note, sensitivity_swept=sweep))
for name, val, path, pat, loai, unit, note, sweep in HARD:
    fl, line = find(path, pat)
    rows.append(dict(name=name, value=val, default_class="(viet cung)", unit=unit, file_line=fl, yaml_override="", type=loai,
                     source_comment=line, note=note, sensitivity_swept=sweep))
with open("thesis_support/v4/free_constants.csv", "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("ok", len(rows), "dong;", sum(1 for r in rows if "KHONG TIM THAY" in r["file_line"]), "khong tim thay dong")
