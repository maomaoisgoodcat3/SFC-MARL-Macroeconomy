"""
Baseline chinh sach RULE-BASED (khong hoc) de doi chieu voi policy RL da train --
dung khung so sanh kinh dien cua Zheng et al. (2020/2022, "The AI Economist"),
da xac minh truc tiep tu ban PDF goc (arXiv:2004.13332v1) trong phien lam viec
nay: Free Market / US Federal / Saez.

GIOI HAN KIEN TRUC QUAN TRONG (doc truoc khi dung): Government.decide() trong du
an nay chi co MOT scalar `worker_tax_rate` ap DONG NHAT cho moi nguoi
(rule_engine.py: `tax_due = taxable_income * worker_tax_rate`) -- KHONG phai bieu
thue LUY TIEN theo bac thu nhap nhu US Federal/Saez BAN GOC (ho cho cac muc thu
nhap khac nhau muc thue khac nhau). Doi action-space de ho tro thue luy tien thuc
su la mot thay doi RLlib action-space, can hoi truoc theo quy tac #3 CLAUDE.md --
CHUA lam o day. Thay vao do, moi baseline duoi day la MOT MUC THUE PHANG QUY DOI
TUONG DUONG (tinh dung cong thuc goc tai muc thu nhap DAI DIEN cua nen kinh te),
ghi ro la XAP XI, khong phai bieu thue day du. Xem FUTURE_WORK.md neu muon lam
dung ban goc (mo rong action-space).

Nguon xac minh (Agent doc truc tiep PDF trong phien lam viec 2026-09-24/25):
- US Federal (single-filer 2018): 7 bac [10,12,22,24,32,35,37]% theo nguong USD
  [0, 9700, 39475, 84200, 160725, 204100, 510300, inf] (scale 1000 USD = 1 Coin
  trong paper goc). O day quy doi nguong theo BOI SO cua `base_living_cost` (don
  vi tien te cua chinh du an nay) thay vi USD tuyet doi, giu dung TI LE giua cac
  bac (moi bac ~ gap 4.07/2.13/1.91/1.27/2.5/... lan bac truoc) -- khong suy doan
  ti le, tinh dung tu day so USD goc.
- Saez (Saez, E. (2001), "Using Elasticities to Derive Optimal Income Tax Rates",
  Review of Economic Studies 68(1)): tau(z) = (1-G(z)) / (1-G(z) + alpha(z)*e(z)).
  He so co gian e(z) trong ban goc UOC LUONG bang OLS tren du lieu training that
  cua chinh ho (khong the tai su dung truc tiep vi kinh te nay co phan phoi thu
  nhap khac han) -- o day dung HANG SO co gian tu dong thuan (Saez, Slemrod &
  Giertz (2012), "The Elasticity of Taxable Income with Respect to Marginal Tax
  Rates: A Critical Review", Journal of Economic Literature 50(1), e~0.25) --
  GHI RO day la HE SO TU DO HIEU CHINH (khong tu du lieu rieng cua du an nay),
  khac voi ban goc Zheng et al. uoc luong rieng.
"""
from typing import List
import numpy as np


# --- US Federal (2018 single-filer) -- HE SO CAU TRUC LAY THANG TU NGUON DA XAC MINH ---
# Ti le nguong bac (chia cho nguong bac dau tien >0, 9700 USD) -- BAT BIEN, khong doi theo
# don vi tien te cua tung nen kinh te, chi doi diem NEO (base_living_cost thay vi 1000 USD).
_US_FEDERAL_BRACKET_RATIOS = [0.0, 1.0, 4.070, 8.680, 16.570, 21.041, 52.608]  # 9700/9700, 39475/9700, ...
_US_FEDERAL_MARGINAL_RATES = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]


def us_federal_effective_rate(monthly_income: float, base_living_cost: float, anchor_income_mult: float = 3.0) -> float:
    """
    Tinh MUC THUE HIEU DUNG (effective average rate, khong phai marginal) cua bieu
    US Federal luy tien tai `monthly_income`, sau khi quy doi nguong bac theo boi so
    cua `base_living_cost` (thay the truc tiep cho 9700 USD trong ban goc).

    anchor_income_mult: HE SO CAU TRUC TU DO HIEU CHINH -- neo bac dau tien
    (~9700 USD/nam trong ban goc, tuong duong ~1 thang luong toi thieu) o
    `anchor_income_mult * base_living_cost`. Mac dinh 3.0 (mot ho gia dinh thu nhap
    bang 3x chi phi song toi thieu da vao bac thue thu 2) -- lua chon hop ly, KHONG
    suy ra truc tiep tu paper goc (ho dung USD tuyet doi, khong co khai niem
    base_living_cost).
    """
    if monthly_income <= 0.0:
        return 0.0
    bracket_edges = [anchor_income_mult * base_living_cost * r for r in _US_FEDERAL_BRACKET_RATIOS]
    tax_owed = 0.0
    remaining = monthly_income
    for i, rate in enumerate(_US_FEDERAL_MARGINAL_RATES):
        lo = bracket_edges[i]
        hi = bracket_edges[i + 1] if i + 1 < len(bracket_edges) else float("inf")
        width = max(0.0, min(monthly_income, hi) - lo)
        if width <= 0.0:
            continue
        tax_owed += width * rate
        remaining -= width
        if remaining <= 0.0:
            break
    return float(np.clip(tax_owed / monthly_income, 0.0, 0.99))


def saez_effective_rate(mean_income: float, base_living_cost: float, elasticity: float = 0.25,
                         pareto_alpha: float = 2.0) -> float:
    """
    Xap xi cong thuc Saez (2001) tau(z) = (1-G(z)) / (1-G(z) + alpha(z)*e(z)) tai
    muc thu nhap TRUNG BINH cua nen kinh te (khong phai tung ca nhan -- gioi han
    kien truc flat-rate da ghi o dau file).

    - alpha(z): he so Pareto nguoc cua duoi phan phoi thu nhap phia tren z (chuan
      Saez 2001 dung truc tiep alpha = z*f(z)/(1-F(z)); voi phan phoi Pareto thuan
      tuy, alpha(z) HANG SO = pareto_alpha). `pareto_alpha=2.0` la HE SO CAU TRUC
      TU DO HIEU CHINH (khong uoc luong tu du lieu rieng cua du an nay -- Saez
      (2001) bao cao alpha thuc nghiem My dao dong 1.5-3.0 tuy nam/nhom thu nhap,
      2.0 la diem giua thuong dung lam mac dinh trong literature ke thua).
    - e(z) = elasticity: HANG SO co gian tu dong thuan, xem trich dan o dau file.
    - G(z): trong so phuc loi xa hoi bien theo thu nhap (Saez dung g_i=1/z_i chuan
      hoa). Voi ho gia dinh o muc thu nhap TRUNG BINH cua chinh no, G(z) xap xi 0.5
      (nam giua phan phoi) -- xap xi hop ly cho MOT muc thue phang dai dien, khong
      phai tinh dung tich phan day du cho tung ca nhan (lai la gioi han flat-rate).
    """
    G_z = 0.5  # xap xi cho ho gia dinh tai muc thu nhap trung binh cua chinh no
    denom = (1.0 - G_z) + pareto_alpha * elasticity
    if denom <= 1e-9:
        return 0.0
    tau = (1.0 - G_z) / denom
    return float(np.clip(tau, 0.0, 0.99))


def free_market_gov_action() -> np.ndarray:
    """Khong thue, ngan sach can bang (rho=1 nhung khong co gi de chi), khong bom cau."""
    return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def us_federal_gov_action(avg_monthly_wage: float, base_living_cost: float) -> np.ndarray:
    rate = us_federal_effective_rate(avg_monthly_wage, base_living_cost)
    # Thue TNDN dung CUNG bieu (paper goc khong tach rieng thue DN, dung chung
    # 1 bieu luy tien cho ca ca nhan -- gia dinh don gian hoa nhat quan).
    return np.array([rate, rate, 1.0, 0.0], dtype=np.float32)


def saez_gov_action(avg_monthly_wage: float, base_living_cost: float) -> np.ndarray:
    rate = saez_effective_rate(avg_monthly_wage, base_living_cost)
    return np.array([rate, rate, 1.0, 0.0], dtype=np.float32)


def fixed_rate_bank_action(lending_rate: float = 0.06, deposit_rate: float = 0.02,
                            credit_factor: float = 0.8) -> np.ndarray:
    """Baseline lai suat co dinh cho Bank -- KHONG phai Taylor Rule thuc (khong co
    co che phan ung theo lam phat/GDP), chi la "chinh sach tien te on dinh mac
    dinh" da co san trong chinh Bank.decide() heuristic cua du an -- dung lam
    diem doi chieu don gian nhat, tach biet voi baseline thue cua Government.
    """
    return np.array([lending_rate, deposit_rate, credit_factor], dtype=np.float32)


BASELINE_NAMES = ["free_market", "us_federal", "saez"]


def get_gov_baseline_action(name: str, avg_monthly_wage: float, base_living_cost: float) -> np.ndarray:
    if name == "free_market":
        return free_market_gov_action()
    if name == "us_federal":
        return us_federal_gov_action(avg_monthly_wage, base_living_cost)
    if name == "saez":
        return saez_gov_action(avg_monthly_wage, base_living_cost)
    raise ValueError(f"Unknown baseline: {name}. Available: {BASELINE_NAMES}")
