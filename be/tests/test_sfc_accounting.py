"""
Bo Kiem toan Bao toan Tuyet doi (Automated SFC Audit Suite).

Xac nhan bang thuc nghiem rang dong tien trong MacroEnvironment tuan thu
nguyen ly Stock-Flow Consistent (SFC) da tuyen bo trong be/rule_engine.py:
Godley, W., & Lavoie, M. (2007), "Monetary Economics: An Integrated Approach
to Credit, Money, Income, Production and Wealth", Palgrave Macmillan -- moi
dong tien phat sinh tu tac tu nay deu phai la khoan chi tra/nhan vao cua tac
tu khac; khong co tien sinh ra tu hu khong hoac bien mat khoi he thong, NGOAI
TRU hai kenh duy nhat duoc thiet ke tuong minh (xem rule_engine.py):

  1. DefaultedDebt: phan no khong the thu hoi duoc -- ton that tin dung thuc su cua ngan
     hang, tu HAI nguon: (a) Firm pha san (Merton, 1974, co thu hoi tai san the chap mot
     phan) va (b) Employee chet con no tin dung tieu dung (Section 3B, v0.23 -- khong co
     tai san the chap nen mat trang 100%).
  2. Capital depreciation (overhead, Jorgenson 1963): chi phi khau hao tu ban
     ma Firm phai tra nhung khong chuyen cho tac tu nao khac trong mo phong
     (dai dien chi phi mua sam/bao tri tu khu vuc ben ngoai khong duoc mo
     hinh hoa tuong minh).

Dinh nghia "Tong gia tri he thong" (total_system_value) dung de kiem toan:

    total_system_value = Treasury + Sum(Bank.reserves)
                        + Sum(Firm.cash) + Sum(Employee.cash)
                        + Sum(Employee.bank_deposit)

Day la mot dai luong RONG HON dinh nghia M2 chuan (xem be/env.py) vi no CONG
CA du tru ngan hang (Bank.reserves) -- muc dich la bat duoc MOI ro ri tien
te o BAT KY dau trong he thong (ke ca von tu co cua ngan hang), khong chi
tien do cong chung nam giu.

Tieu chuan vuot qua moi buoc t:
    delta_total_system_value_t == -(bad_debt_t) - (capital_depreciation_t)
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
    return env.gov.treasury + bank_reserves + firm_cash + emp_cash + emp_deposits + economy_buffer_fund


def _run_sfc_audit(seed: int, num_employees: int, num_firms: int, num_banks: int, steps: int) -> None:
    env = MacroEnvironment(num_employees=num_employees, num_firms=num_firms, num_banks=num_banks, max_steps=steps)

    bad_debt_this_step: List[float] = [0.0]

    def on_bad_debt(ev: Event) -> None:
        bad_debt_this_step[0] += float(ev.payload.get("bad_debt", 0.0))

    # AGENT_BANKRUPT: Firm mất khả năng thanh toán (Merton, Section 8, co thu hoi tai san).
    # AGENT_DIED: Employee chet con no tin dung tieu dung (Section 3B/9, v0.23) -- KHONG co
    # tai san the chap nen la mat trang 100%, nhung CUNG kenh "DefaultedDebt" duoc phep trong
    # dang thuc bao toan (xem docstring dau file) -- phai cong ca hai nguon vao cung bien dem.
    env.event_bus.subscribe(EventType.AGENT_BANKRUPT, on_bad_debt)
    env.event_bus.subscribe(EventType.AGENT_DIED, on_bad_debt)

    env.reset(seed=seed)
    prev_total = _total_system_value(env)

    for step in range(1, steps + 1):
        actions = {}
        for agent_id, agent in env.agents.items():
            agent_obs = agent.observe(env.get_raw_environment_state())
            actions[agent_id] = agent.decide(agent_obs).values

        bad_debt_this_step[0] = 0.0
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
        expected_delta = -bad_debt_this_step[0] - overhead_cost
        unexplained = delta_total - expected_delta

        assert abs(unexplained) <= SFC_TOLERANCE, (
            f"[seed={seed} step={step}] SFC conservation VIOLATED: "
            f"delta_total_system_value={delta_total:.6f}, "
            f"expected(-bad_debt-overhead)={expected_delta:.6f} "
            f"(bad_debt={bad_debt_this_step[0]:.6f}, overhead={overhead_cost:.6f}), "
            f"unexplained_leak={unexplained:.6f}"
        )

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
