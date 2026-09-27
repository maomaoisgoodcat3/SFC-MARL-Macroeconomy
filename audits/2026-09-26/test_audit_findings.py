"""
BO TEST AUDIT (2026-09-26) -- kiem chung bang so cac van de phat hien qua doc code tinh.
Moi test DUOC VIET DE PASS khi van de TON TAI (tuc la test "chung minh loi"), de sau nay khi sua
co the dao nguoc assert thanh test hoi quy. KHONG sua repo.
Chay: python -m pytest test_audit_findings.py -q -s
"""
import os, sys
import numpy as np
import pytest

REPO = r"C:\Users\piece\Downloads\TheAIEconomist\ai_economist_gpt"
sys.path.insert(0, REPO); os.chdir(REPO)
from be.env import MacroEnvironment
from be.scenario_config import ScenarioConfig
from be.core.enums import LifeCycleStatus, EventType
from be.agents.employee import Employee
from be.agents.firm import Firm
from be.rllib_wrapper import (EMPLOYEE_ACT_SPACE, FIRM_ACT_SPACE, GOVERNMENT_ACT_SPACE,
                              BANK_ACT_SPACE, SUPERVISOR_ACT_SPACE, ECONOMY_ACT_SPACE)

KW = ScenarioConfig.from_yaml("scenarios/em_baseline.yaml").to_env_kwargs()


def _space(aid):
    if aid.startswith("emp_"): return EMPLOYEE_ACT_SPACE
    if aid.startswith("firm_"): return FIRM_ACT_SPACE
    if aid.startswith("bank_"): return BANK_ACT_SPACE
    if aid == "gov_1": return GOVERNMENT_ACT_SPACE
    if aid == "sup_1": return SUPERVISOR_ACT_SPACE
    return ECONOMY_ACT_SPACE


def _rand_actions(env, rng, effort=None):
    acts = {}
    for aid in env.agents:
        sp = _space(aid)
        a = rng.uniform(sp.low, sp.high).astype(np.float32)
        if aid.startswith("emp_"):
            if effort is not None:
                a[0] = effort
            a[1] = 1.0  # khai bao day du -- co lap rieng kenh effort
        acts[aid] = a
    return acts


# ---------------------------------------------------------------------------------------------
# F1. LO HONG KHUYEN KHICH EFFORT (moral hazard, Shapiro & Stiglitz 1984 du bao dung)
# ---------------------------------------------------------------------------------------------
def test_F1a_effort_has_no_private_benefit_for_employee():
    """Cung trang thai, cung hanh dong khac: effort 0.1 vs 0.9 cho MOT employee co viec ->
    luong/viec lam/thu nhap cua chinh ho GIONG HET; reward cua ho CAO HON khi effort thap.
    [CAP NHAT v0.34] Test nay la phat bieu MOT BUOC nen VAN dung sau khi them giam sat Shapiro-Stiglitz:
    loi ich cua effort la ky vong DONG (de doa sa thai), khong phai trong 1 buoc. Kiem tra bang test
    khuyen khich nhieu buoc be/tests/test_env_contracts.py::test_effort_has_private_benefit_only_with_monitoring."""
    import copy
    env = MacroEnvironment(**KW); env.reset(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(5):
        env.step(_rand_actions(env, rng, effort=0.6))
    emp = next(a for a in env.agents.values() if isinstance(a, Employee) and a.employed_by)
    acts = _rand_actions(env, np.random.default_rng(99), effort=0.6)
    res = {}
    for eff in (0.1, 0.9):
        e2 = copy.deepcopy(env); a2 = {k: v.copy() for k, v in acts.items()}
        a2[emp.agent_id][0] = eff
        np.random.seed(123)
        _, rew, *_ = e2.step(a2)
        me = e2.agents.get(emp.agent_id)
        res[eff] = (me.wage, me.employed_by, rew[emp.agent_id], me.energy)
    print("\n[F1a] effort -> (wage, employer, reward, energy):", res)
    assert res[0.1][0] == pytest.approx(res[0.9][0]), "luong phu thuoc effort (loi da duoc sua?)"
    assert res[0.1][1] == res[0.9][1]
    assert res[0.1][2] > res[0.9][2], "reward khong cao hon khi effort thap"
    # energy co the bi kep o tran max_energy nen khong assert (khong phai loi co che)


def test_F1b_low_economy_wide_effort_causes_macro_collapse():
    """Co lap kenh effort o cap vi mo: MOI tac tu khac ngau nhien, chi ep effort toan dan
    0.1 (muc policy da hoc hoi tu) vs 0.8. Do tu vong, gia, GDP thuc, that nghiep."""
    out = {}
    for eff in (0.1, 0.8):
        deaths, prices, rgdp, unemp = [], [], [], []
        for s in (11, 12, 13):
            env = MacroEnvironment(**KW); env.reset(seed=s); rng = np.random.default_rng(s)
            for t in range(240):
                env.step(_rand_actions(env, rng, effort=eff))
                emps = [a for a in env.agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE]
                prices.append(env.eco.base_living_cost); rgdp.append(env.gov.current_real_gdp)
                unemp.append(np.mean([e.employed_by is None for e in emps]))
            deaths.append(env.gov.dead_citizens_count)
        out[eff] = dict(deaths=np.mean(deaths), price=np.mean(prices), real_gdp=np.mean(rgdp), unemp=np.mean(unemp))
    print("\n[F1b] effort=0.1:", {k: round(v, 3) for k, v in out[0.1].items()})
    print("[F1b] effort=0.8:", {k: round(v, 3) for k, v in out[0.8].items()})
    assert out[0.1]["real_gdp"] < out[0.8]["real_gdp"]
    assert out[0.1]["price"] > out[0.8]["price"]


# ---------------------------------------------------------------------------------------------
# F2. GHI NHAN NO XAU HAI LAN (reserves bi tru luc giai ngan VA luc vo no)
# ---------------------------------------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason="DA SUA v0.33 (KNOWN_PATHOLOGIES.md #28) -- ban hoi quy dao nguoc: be/tests/test_env_contracts.py::test_default_loss_is_recognized_once_in_bank_equity. xfail strict: neu loi quay lai thi test nay PASS => bao dong")
def test_F2_default_loss_counted_twice_in_bank_equity():
    """Employee no D chet vi tuoi (tat dinh). So sanh 2 ban sao CUNG buoc CUNG hanh dong: (A) nguoi
    do chet, (B) nguoi do song (max_age nang len). Chenh lech von ngan hang (reserves+loans-deposits)
    A-B chi do vo no gay ra. Ke toan dung: giam D (xoa khoan vay). Code hien tai: giam ~2D (xoa
    khoan vay VA tru reserves them lan nua -- reserves da bi tru luc giai ngan)."""
    import copy
    env = MacroEnvironment(**KW); env.reset(seed=5)
    bank = env.banks[0]
    emp = next(a for a in env.agents.values() if isinstance(a, Employee))
    D = 1000.0
    bank.reserves -= D; bank.total_loans += D; emp.cash += D; emp.debt += D   # giai ngan dung Section 3B
    emp.creditor_bank_id = bank.agent_id
    acts = _rand_actions(env, np.random.default_rng(5), effort=0.5)
    acts[emp.agent_id][3] = 0.0  # khong vay them
    eq = {}
    for dies in (True, False):
        e2 = copy.deepcopy(env); me = e2.agents[emp.agent_id]
        me.age = me.max_age if dies else 18
        np.random.seed(77); e2.step({k: v.copy() for k, v in acts.items()})
        b = e2.banks[0]; eq[dies] = b.reserves + b.total_loans - b.total_deposits
    diff = eq[False] - eq[True]
    print(f"\n[F2] chenh lech von ngan hang do rieng vo no = {diff:.2f} (du no ~{D}, ke toan dung ~{D})")
    assert diff > 1.8 * D, "von chi giam ~D -> khong con ghi nhan 2 lan"


# ---------------------------------------------------------------------------------------------
# F3. SO CAI TIEN GUI "MA": bank.total_deposits lech sum(emp.bank_deposit) sau chet/sinh
# ---------------------------------------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason="DA SUA v0.33 (KNOWN_PATHOLOGIES.md #28) -- ban hoi quy dao nguoc: be/tests/test_env_contracts.py::test_deposit_ledger_stays_consistent_after_death_and_inheritance")
def test_F3_deposit_ledger_drifts_after_deaths_and_births():
    env = MacroEnvironment(**KW); env.reset(seed=8); rng = np.random.default_rng(8)
    gaps = []
    for t in range(240):
        acts = _rand_actions(env, rng)
        if t % 10 == 0:  # ep vai nguoi CO TIEN GUI chet vi tuoi (tat dinh) de lo kenh ro ri
            for e in list(env.agents.values()):
                if isinstance(e, Employee) and getattr(e, "bank_deposit", 0.0) > 5.0:
                    e.age = e.max_age; break
        env.step(acts)
        emps = [a for a in env.agents.values() if isinstance(a, Employee)]
        gaps.append(sum(b.total_deposits for b in env.banks) - sum(getattr(e, "bank_deposit", 0.0) for e in emps))
    print(f"\n[F3] do lech so cai tien gui cuoi episode = {gaps[-1]:.2f}, max = {max(gaps):.2f}")
    assert max(gaps) > 1.0


# ---------------------------------------------------------------------------------------------
# F4. DAC TRUNG QUAN SAT "CHET": market_demand_factor luon = 0.1 (M2 gom Treasury ~1e6)
# ---------------------------------------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason="DA SUA v0.35 (KNOWN_PATHOLOGIES.md #29f): he so nhan 100 + Kho bac van hanh moi -> dac trung khong con hang so")
def test_F4_market_demand_factor_is_constant():
    env = MacroEnvironment(**KW); env.reset(seed=9); rng = np.random.default_rng(9)
    vals = set()
    for t in range(240):
        env.step(_rand_actions(env, rng))
        vals.add(round(env.get_raw_environment_state()["macro_indicators"]["market_demand_factor"], 6))
    print(f"\n[F4] cac gia tri market_demand_factor quan sat duoc: {sorted(vals)[:5]}")
    assert vals == {0.1}


# ---------------------------------------------------------------------------------------------
# F5. NGAN HANG LA "HO CHON TIEN": lai rong tich luy vao reserves, khong bao gio quay lai nen kinh te
# ---------------------------------------------------------------------------------------------
def test_F5_bank_net_interest_is_never_recycled():
    """Ep firm vay manh voi lai suat tran (giong hanh vi policy da hoc: lending 0.25/deposit 0.005).
    Tien tu khu vuc thuc chay vao reserves ma khong co kenh chi tieu/co tuc nao dua ve."""
    env = MacroEnvironment(**KW); env.reset(seed=10); rng = np.random.default_rng(10)
    r0 = sum(b.reserves for b in env.banks); cum_nim = 0.0
    circ0 = sum(a.cash for a in env.agents.values() if isinstance(a, (Employee, Firm)))
    for t in range(240):
        acts = _rand_actions(env, rng)
        for aid in acts:
            if aid.startswith("bank_"): acts[aid] = np.array([0.25, 0.005, 1.0], dtype=np.float32)
            if aid.startswith("firm_"): acts[aid][1] = 1.0
        env.step(acts)
        cum_nim += sum(b.last_interest_income - b.last_interest_expense for b in env.banks)
    r1 = sum(b.reserves for b in env.banks)
    print(f"\n[F5] lai rong tich luy = {cum_nim:.1f}; reserves {r0:.0f}->{r1:.0f}; tien luu thong ban dau (emp+firm cash) = {circ0:.0f}")
    assert cum_nim > 0.0
