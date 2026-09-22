"""
Kiem toan SFC DUOI POLICY NGAU NHIEN kieu RL (bo sung cho test_sfc_accounting.py).

Ly do ton tai: test_sfc_accounting.py chi dung agent.decide() heuristic -- heuristic
LUON khai bao thue 100% (declare_ratio=1.0) nen khong bao gio bi phat, va hanh dong
bom cau cua Economy luon = 0. Do do no MU truoc moi dong tien phat sinh tu hanh vi
"kieu RL" (tron thue -> tien phat; bom/rut cau tai khoa; vay/gui ngau nhien...).
Thuc te da xay ra: 685/720 buoc vi pham bao toan (lech toi da ~607) ma bo test cu van
PASS, vi tien phat cua Supervisor bi tru khoi vi nguoi vi pham nhung Government khong
bao gio cong vao Kho bac (xem CLAUDE_HISTORY.md v0.15).

Nguyen ly: Stock-Flow Consistent (Godley, W., & Lavoie, M. (2007), "Monetary
Economics: An Integrated Approach to Credit, Money, Income, Production and Wealth",
Palgrave Macmillan). Tieu chuan moi buoc (dung nhu test_sfc_accounting.py):

    delta_total_system_value == -(bad_debt) - (capital_depreciation)

TIEU CHUAN CO DINH tu nay: MOI thay doi cham vao dong tien phai chay bo test nay
(policy ngau nhien), khong chi bo test heuristic.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from be.env import MacroEnvironment
from be.core.enums import EventType
from be.agents.employee import Employee
from be.agents.firm import Firm
from be.rllib_wrapper import (
    EMPLOYEE_ACT_SPACE, FIRM_ACT_SPACE, GOVERNMENT_ACT_SPACE,
    BANK_ACT_SPACE, SUPERVISOR_ACT_SPACE, ECONOMY_ACT_SPACE,
)

SFC_TOLERANCE = 1e-3
STEPS = 150


def _space_for(agent_id: str):
    if agent_id.startswith("emp_"):
        return EMPLOYEE_ACT_SPACE
    if agent_id.startswith("firm_"):
        return FIRM_ACT_SPACE
    if agent_id == "gov_1":
        return GOVERNMENT_ACT_SPACE
    if agent_id.startswith("bank_"):
        return BANK_ACT_SPACE
    if agent_id == "sup_1":
        return SUPERVISOR_ACT_SPACE
    return ECONOMY_ACT_SPACE


def _random_actions(agent_ids, rng, std):
    """Xap xi policy PPO khoi dau: Gaussian quanh giua khoang, kep vao Box."""
    out = {}
    for aid in agent_ids:
        sp = _space_for(aid)
        mid, half = (sp.low + sp.high) / 2.0, (sp.high - sp.low) / 2.0
        out[aid] = np.clip(mid + half * rng.normal(0.0, std, size=sp.shape), sp.low, sp.high).astype(np.float32)
    return out


def _total_system_value(env: MacroEnvironment) -> float:
    emp = [a for a in env.agents.values() if isinstance(a, Employee)]
    firm = [a for a in env.agents.values() if isinstance(a, Firm)]
    return (env.gov.treasury + sum(b.reserves for b in env.banks)
            + sum(a.cash for a in firm) + sum(a.cash for a in emp)
            + sum(getattr(a, "bank_deposit", 0.0) for a in emp))


@pytest.mark.parametrize("num_banks", [1, 2])
@pytest.mark.parametrize("std", [0.5, 1.0])
@pytest.mark.parametrize("seed", [3, 17])
def test_sfc_conservation_random_policy(seed: int, std: float, num_banks: int) -> None:
    env = MacroEnvironment(num_employees=40, num_firms=5, num_banks=num_banks, max_steps=STEPS)
    bad_debt = [0.0]
    env.event_bus.subscribe(
        EventType.AGENT_BANKRUPT,
        lambda ev: bad_debt.__setitem__(0, bad_debt[0] + float(ev.payload.get("bad_debt", 0.0))),
    )
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    prev = _total_system_value(env)
    saw_injection = saw_fines = saw_purchase = False

    for step in range(1, STEPS + 1):
        bad_debt[0] = 0.0
        env.step(_random_actions(list(env.agents.keys()), rng, std))
        new = _total_system_value(env)
        unexplained = (new - prev) - (-bad_debt[0] - env.eco.last_capital_depreciation)
        assert abs(unexplained) <= SFC_TOLERANCE, (
            f"[seed={seed} std={std} banks={num_banks} step={step}] SFC VIOLATED: "
            f"unexplained_leak={unexplained:.6f} (fines={env.gov.last_fines_collected:.3f}, "
            f"injection={env.eco.last_demand_injection_value:.3f})"
        )
        saw_injection |= abs(env.eco.last_demand_injection_value) > 1e-9
        saw_fines |= env.gov.last_fines_collected > 1e-9
        saw_purchase |= env.gov.last_purchase > 1e-9
        prev = new

    # Bao dam test THUC SU kich hoat hai kenh da tung lam bo test cu mu (khong pass vi vo hieu).
    assert saw_injection, "Test khong bao gio kich hoat bom/rut cau (Economy action[0] != 0)"
    assert saw_fines, "Test khong bao gio sinh tien phat (kenh trot thue/thanh tra khong duoc kiem)"
    assert saw_purchase, "Test khong bao gio kich hoat chi mua hang cua Chinh phu (Section 4C)"
