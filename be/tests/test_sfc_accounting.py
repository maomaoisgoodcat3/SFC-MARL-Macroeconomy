"""
Bo Kiem toan Bao toan Tuyet doi (Automated SFC Audit Suite).

Xac nhan bang thuc nghiem rang dong tien trong MacroEnvironment tuan thu
nguyen ly Stock-Flow Consistent (SFC) da tuyen bo trong be/rule_engine.py:
Godley, W., & Lavoie, M. (2007), "Monetary Economics: An Integrated Approach
to Credit, Money, Income, Production and Wealth", Palgrave Macmillan -- moi
dong tien phat sinh tu tac tu nay deu phai la khoan chi tra/nhan vao cua tac
tu khac; khong co tien sinh ra tu hu khong hoac bien mat khoi he thong, NGOAI
TRU MOT kenh duy nhat duoc thiet ke tuong minh (xem rule_engine.py):

  Capital depreciation (overhead, Jorgenson 1963): chi phi khau hao tu ban
  ma Firm phai tra nhung khong chuyen cho tac tu nao khac trong mo phong
  (dai dien chi phi mua sam/bao tri tu khu vuc ben ngoai khong duoc mo
  hinh hoa tuong minh).

LOI DA SUA TRONG BO TEST NAY (v0.33, KNOWN_PATHOLOGIES.md #28): phien ban cu cho phep THEM kenh
"DefaultedDebt" (delta_total = -bad_debt - overhead). Do la SAI kinh te-ke toan: vo no chi xoa
so khoan vay (Bank.total_loans -- KHONG nam trong total_system_value) va chuyen tien mat con lai
cua con no ve chu no; khong co dong tien nao roi he thong. Vi Bank.apply_result tung tru them
`reserves -= new_defaults`, TIEN BI HUY ngoai y muon, va vế -bad_debt trong test cu "hop thuc
hoa" chinh loi do (test dung voi code sai, nen khong bat duoc). Nay dang thuc dung la
delta_total_system_value == -(capital_depreciation) va co them 2 bat bien so cai (xem
_assert_ledgers): sum(Bank.total_deposits) == sum(Employee.bank_deposit) va
sum(Bank.total_loans) == sum(Firm.debt) + sum(Employee.debt).

Dinh nghia "Tong gia tri he thong" (total_system_value) dung de kiem toan:

    total_system_value = Treasury + Sum(Bank.reserves)
                        + Sum(Firm.cash) + Sum(Employee.cash)
                        + Sum(Employee.bank_deposit)

Day la mot dai luong RONG HON dinh nghia M2 chuan (xem be/env.py) vi no CONG
CA du tru ngan hang (Bank.reserves) -- muc dich la bat duoc MOI ro ri tien
te o BAT KY dau trong he thong (ke ca von tu co cua ngan hang), khong chi
tien do cong chung nam giu.

Tieu chuan vuot qua moi buoc t:
    delta_total_system_value_t == -(capital_depreciation_t)
    (sai so tuyet doi <= SFC_TOLERANCE)

Neu chenh lech vuot qua dung sai, test lap tuc bao dong ro ri dong tien
(Capital Leakage) o buoc do.
"""
import os
import sys
from typing import List

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from be.env import MacroEnvironment
from be.core.enums import EventType
from be.core.event import Event
from be.agents.employee import Employee
from be.agents.firm import Firm

SFC_TOLERANCE = 1e-3  # dung sai tuyet doi (don vi tien te mo phong) do tich luy sai so float64
SEEDS = [42, 101, 2024]
STEPS_PER_SEED = 600


def _total_system_value(env: MacroEnvironment) -> float:
    emp_cash = sum(a.cash for a in env.agents.values() if isinstance(a, Employee))
    emp_deposits = sum(getattr(a, "bank_deposit", 0.0) for a in env.agents.values() if isinstance(a, Employee))
    firm_cash = sum(a.cash for a in env.agents.values() if isinstance(a, Firm))
    bank_reserves = sum(b.reserves for b in env.banks)
    # LOI DA SUA (v0.24, phat hien TRUOC khi code qua kiem chung so hoc -- xem
    # METHODOLOGY_NOTES.md muc 2): Economy.strategic_reserve_fund (quy binh on du tru dem) la
    # MOT PHAN tien that cua he thong (cap tu Treasury luc reset) -- neu KHONG cong vao day, moi
    # lan Economy chi tien tu quy (buffer-stock buy) se bao "ro ri" GIA vi tien chuyen noi bo
    # (fund giam, firm_cash tang) khong duoc doi chieu dung.
    economy_buffer_fund = getattr(env.eco, "strategic_reserve_fund", 0.0)
    # v0.36-fix1: TRU no cong (public_debt) -- khi Kho bac am, Government.apply_result chuyen phan am thanh
    # public_debt (no phai tra cua chinh phu, KHONG co chu no rieng: chi tieu tham hut duoc tai tro bang phat
    # hanh tien, dung mo hinh SIM cua Godley & Lavoie 2007 ch.3: dH = G - T). Dang thuc cu KHONG tru khoan no
    # nay nen se bao "ro ri" ~940-980/buoc moi khi co no cong -- bo test chua tung vao che do do (diem mu,
    # KNOWN_PATHOLOGIES.md #28 phu luc). Tai san rong cua chinh phu = Kho bac - no cong.
    return (env.gov.treasury - env.gov.public_debt) + bank_reserves + firm_cash + emp_cash + emp_deposits + economy_buffer_fund


def _assert_ledgers(env: MacroEnvironment, seed: int, step: int) -> None:
    """Bat bien SO CAI ngan hang (v0.33, KNOWN_PATHOLOGIES.md #28): so cai ngan hang phai khop
    dung so du thuc cua tac tu (audit 2026-09-26 test F3: tien gui tung lech "ma" sau chet/sinh)."""
    emps = [a for a in env.agents.values() if isinstance(a, Employee)]
    firms = [a for a in env.agents.values() if isinstance(a, Firm)]
    deposits_bank = sum(b.total_deposits for b in env.banks)
    deposits_emp = sum(getattr(e, "bank_deposit", 0.0) for e in emps)
    assert abs(deposits_bank - deposits_emp) <= SFC_TOLERANCE, (
        f"[seed={seed} step={step}] LEDGER tien gui lech: sum(Bank.total_deposits)={deposits_bank:.6f} "
        f"!= sum(Employee.bank_deposit)={deposits_emp:.6f}"
    )
    loans_bank = sum(b.total_loans for b in env.banks)
    debts = sum(f.debt for f in firms) + sum(e.debt for e in emps)
    assert abs(loans_bank - debts) <= SFC_TOLERANCE, (
        f"[seed={seed} step={step}] LEDGER cho vay lech: sum(Bank.total_loans)={loans_bank:.6f} "
        f"!= sum(Firm.debt)+sum(Employee.debt)={debts:.6f}"
    )


def _run_sfc_audit(seed: int, num_employees: int, num_firms: int, num_banks: int, steps: int) -> None:
    env = MacroEnvironment(num_employees=num_employees, num_firms=num_firms, num_banks=num_banks, max_steps=steps)

    env.reset(seed=seed)
    prev_total = _total_system_value(env)

    for step in range(1, steps + 1):
        actions = {}
        for agent_id, agent in env.agents.items():
            agent_obs = agent.observe(env.get_raw_environment_state())
            actions[agent_id] = agent.decide(agent_obs).values

        obs, rewards, terms, truncs, infos = env.step(actions)

        # Bat bien so hoc co ban: khong NaN/Inf o bat ky quan sat/reward nao.
        for aid, v in obs.items():
            assert np.all(np.isfinite(v)), f"[seed={seed} step={step}] Non-finite observation for {aid}: {v}"
        for aid, r in rewards.items():
            assert np.isfinite(r), f"[seed={seed} step={step}] Non-finite reward for {aid}: {r}"

        # Bat bien vat ly: du no tien gui khong the am (Bank.apply_result da
        # clamp bang max(0, ...), day la kiem tra hoi quy cho dieu do).
        for b in env.banks:
            assert b.total_deposits >= -1e-6, f"[seed={seed} step={step}] Negative bank deposits on {b.agent_id}: {b.total_deposits}"

        new_total = _total_system_value(env)
        delta_total = new_total - prev_total

        overhead_cost = float(env.eco.last_capital_depreciation)
        expected_delta = -overhead_cost
        unexplained = delta_total - expected_delta

        assert abs(unexplained) <= SFC_TOLERANCE, (
            f"[seed={seed} step={step}] SFC conservation VIOLATED: "
            f"delta_total_system_value={delta_total:.6f}, "
            f"expected(-overhead)={expected_delta:.6f} (overhead={overhead_cost:.6f}), "
            f"unexplained_leak={unexplained:.6f}"
        )
        _assert_ledgers(env, seed, step)

        prev_total = new_total


@pytest.mark.parametrize("seed", SEEDS)
def test_sfc_conservation_single_bank(seed: int) -> None:
    """He thong 1 Ngan hang (cau hinh mac dinh) phai bao toan dong tien qua
    nhieu seed doc lap."""
    _run_sfc_audit(seed=seed, num_employees=30, num_firms=4, num_banks=1, steps=STEPS_PER_SEED)


@pytest.mark.parametrize("seed", SEEDS)
def test_sfc_conservation_multi_bank(seed: int) -> None:
    """He thong nhieu Ngan hang (relationship banking, Petersen & Rajan, 1994)
    cung phai bao toan dong tien -- xac nhan viec dinh tuyen dong tien qua
    dung creditor_bank_id / depository_bank_id o rule_engine.py khong lam ro
    ri hoac nhan doi tien giua cac ngan hang."""
    _run_sfc_audit(seed=seed, num_employees=30, num_firms=4, num_banks=3, steps=STEPS_PER_SEED)


if __name__ == "__main__":
    for s in SEEDS:
        print(f"[SFC AUDIT] single-bank seed={s} ...")
        _run_sfc_audit(seed=s, num_employees=30, num_firms=4, num_banks=1, steps=STEPS_PER_SEED)
        print(f"[SFC AUDIT] single-bank seed={s}: PASS")
        print(f"[SFC AUDIT] multi-bank seed={s} ...")
        _run_sfc_audit(seed=s, num_employees=30, num_firms=4, num_banks=3, steps=STEPS_PER_SEED)
        print(f"[SFC AUDIT] multi-bank seed={s}: PASS")
    print("\nAll SFC conservation checks passed.")
