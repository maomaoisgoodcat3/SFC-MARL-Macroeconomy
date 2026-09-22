"""
Hop dong (contracts) cua MacroEnvironment -- cac bat bien da tung bi vi pham THAT
va phai duoc khoa lai bang test tu dong (xem CLAUDE_HISTORY.md v0.15).

  1. Train == Simulate: RLlibMacroEnv (duong train) va MacroEnvironment (duong
     simulate/server) tra CUNG obs/reward cho cung seed + cung action (truoc day
     wrapper tu kep reward va lam sach rieng, con simulate/server thi khong).
  2. Hinh phat tu vong theo tuoi (Viscusi & Aldy, 2003; Aldy & Viscusi, 2008) KHONG
     bi cat mat boi reward_clip: agent tre chet phai bi phat NANG HON agent gia.
  3. reward_clip qua nho -> env TU CHOI khoi tao (fail loudly), khong am tham cat.
  4. Economy nhan duoc khoi luong giao dich > 0 (khoa total_market_turnover tung bi xoa).
  5. Cau hinh PPO/chinh sach cua train va server la MOT nguon.
  6. Tran chi so hoa chi tieu sinh ton theo gia: he so khuech dai mot buoc tren tran
     bang dung theta=0,70 (Calvo, 1983) o MOI muc cung (vong lap phan hoi bi cat).
"""
import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from be.env import MacroEnvironment, sanitize_action, sanitize_observation
from be.rllib_wrapper import (
    RLlibMacroEnv, build_ppo_config, build_policy_specs, POLICY_MODEL_CONFIG, PPO_VF_CLIP_PARAM,
    EMPLOYEE_ACT_SPACE, FIRM_ACT_SPACE, GOVERNMENT_ACT_SPACE, BANK_ACT_SPACE,
    SUPERVISOR_ACT_SPACE, ECONOMY_ACT_SPACE,
)
from be.scenario_config import ScenarioConfig

THETA_CALVO = 0.70


def _space_for(aid):
    if aid.startswith("emp_"):
        return EMPLOYEE_ACT_SPACE
    if aid.startswith("firm_"):
        return FIRM_ACT_SPACE
    if aid == "gov_1":
        return GOVERNMENT_ACT_SPACE
    if aid.startswith("bank_"):
        return BANK_ACT_SPACE
    if aid == "sup_1":
        return SUPERVISOR_ACT_SPACE
    return ECONOMY_ACT_SPACE


def _actions(ids, rng, std=0.5):
    out = {}
    for aid in ids:
        sp = _space_for(aid)
        mid, half = (sp.low + sp.high) / 2.0, (sp.high - sp.low) / 2.0
        out[aid] = np.clip(mid + half * rng.normal(0.0, std, size=sp.shape), sp.low, sp.high).astype(np.float32)
    return out


def test_train_and_simulate_paths_are_identical() -> None:
    cfg = ScenarioConfig(num_employees=30, num_firms=4, num_banks=1, max_steps=60)
    env_sim = MacroEnvironment(**cfg.to_env_kwargs())
    obs_sim, _ = env_sim.reset(seed=5)
    wrapper = RLlibMacroEnv(cfg.to_env_kwargs())
    obs_train, _ = wrapper.reset(seed=5)
    assert obs_sim.keys() == obs_train.keys()

    rng = np.random.default_rng(5)
    for step in range(40):
        acts = _actions(list(obs_sim.keys()), rng)
        # MacroEnvironment dung RNG TOAN CUC np.random (thanh tra ngau nhien, sinh san...)
        # -> gieo lai ngay truoc MOI buoc de hai env doc cung mot luong so ngau nhien.
        np.random.seed(9000 + step)
        o1, r1, t1, tr1, _ = env_sim.step(acts)
        np.random.seed(9000 + step)
        o2, r2, t2, tr2, _ = wrapper.step(acts)
        assert r1 == r2, f"reward train != simulate o buoc {step + 1}"
        assert t1 == t2 and tr1 == tr2
        assert o1.keys() == o2.keys()
        for k in o1:
            np.testing.assert_array_equal(o1[k], o2[k])
        obs_sim = o1


def test_death_penalty_scales_with_age_after_reward_pipeline() -> None:
    """Loi that: wrapper clip +-100 bien -122,11 thanh -100,00. Sau chuan hoa, agent tre
    van phai bi phat nang hon agent gia, va khong ai cham tran clip."""
    def scenario(make):
        env = make()
        e = env.env if hasattr(env, "env") else env
        for aid, age in (("emp_0", 19), ("emp_1", 74)):
            a = e.agents[aid]
            a.age, a.cash, a.energy = age, -5000.0, 0.0
        acts = _actions(list(e.agents.keys()), np.random.default_rng(5))
        _, rew, *_ = env.step(acts)
        return rew["emp_0"], rew["emp_1"]

    cfg = ScenarioConfig(num_employees=40, num_firms=5, num_banks=1, max_steps=200)

    def make_sim():
        env = MacroEnvironment(**cfg.to_env_kwargs()); env.reset(seed=21); return env

    def make_train():
        w = RLlibMacroEnv(cfg.to_env_kwargs()); w.reset(seed=21); return w

    young_s, old_s = scenario(make_sim)
    young_t, old_t = scenario(make_train)
    assert (young_s, old_s) == (young_t, old_t)
    assert young_s < old_s, "hinh phat tu vong theo tuoi bi vo hieu (agent tre khong bi phat nang hon)"
    assert abs(young_s) < cfg.reward_clip and abs(old_s) < cfg.reward_clip


def test_reward_clip_too_small_is_rejected() -> None:
    with pytest.raises(ValueError):
        MacroEnvironment(num_employees=10, num_firms=2, reward_clip=1.0)


def test_sanitizers_neutralize_non_finite_values() -> None:
    assert np.all(sanitize_action(np.array([np.nan, np.inf, -np.inf], dtype=np.float32)) == np.array([0.0, 1.0, -1.0]))
    o = sanitize_observation(np.array([np.nan, np.inf, -np.inf, 3.0], dtype=np.float32))
    assert np.all(np.isfinite(o)) and o[3] == 3.0


def test_economy_receives_positive_trade_volume() -> None:
    env = MacroEnvironment(num_employees=30, num_firms=4, num_banks=1, max_steps=30)
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    volumes = []
    for _ in range(25):
        env.step(_actions(list(env.agents.keys()), rng))
        volumes.append(env.eco.step_trade_volume)
    assert max(volumes) > 0.0, "step_trade_volume luon = 0: khoa total_market_turnover khong duoc ghi"


def test_train_and_server_share_one_ppo_config() -> None:
    env_cfg = ScenarioConfig().to_env_kwargs()
    train_cfg = build_ppo_config(env_cfg, num_env_runners=2, seed=1, train_batch_size=800, sgd_minibatch_size=128, rollout_fragment_length=100)
    serve_cfg = build_ppo_config(env_cfg, num_env_runners=0, with_callbacks=False)
    assert train_cfg.model["fcnet_hiddens"] == serve_cfg.model["fcnet_hiddens"] == POLICY_MODEL_CONFIG["fcnet_hiddens"]
    assert train_cfg.vf_clip_param == serve_cfg.vf_clip_param == PPO_VF_CLIP_PARAM
    assert set(train_cfg.policies.keys()) == set(serve_cfg.policies.keys()) == set(build_policy_specs().keys())


@pytest.mark.parametrize("supply_scale", [1.0, 0.25, 0.12, 0.03])
def test_price_indexation_loop_gain_equals_theta_above_ceiling(supply_scale: float) -> None:
    """Tren tran chi so hoa, d(P_t)/d(P_{t-1}) phai bang dung THETA_CALVO o MOI muc cung
    (cat vong lap khuech dai, khong chi lam cham). Do bang nhieu loan +2% tren 2 ban sao."""
    env = MacroEnvironment(num_employees=50, num_firms=5, num_banks=1, max_steps=100)
    env.reset(seed=11)
    rng = np.random.default_rng(11)
    for _ in range(8):
        env.step(_actions(list(env.agents.keys()), rng))
    for a in env.agents.values():
        if a.agent_id.startswith("firm_"):
            a.productivity_factor *= supply_scale
    ceiling = env.subsistence_indexation_ceiling_mult * env.eco.initial_living_cost
    env.eco.base_living_cost = 1.5 * ceiling
    acts = _actions(list(env.agents.keys()), np.random.default_rng(1011))
    acts["eco_1"] = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    a_env, b_env = copy.deepcopy(env), copy.deepcopy(env)
    p0 = a_env.eco.base_living_cost
    b_env.eco.base_living_cost = p0 * 1.02
    np.random.seed(77); a_env.step(dict(acts))
    np.random.seed(77); b_env.step(dict(acts))
    gain = (b_env.eco.base_living_cost - a_env.eco.base_living_cost) / (p0 * 1.02 - p0)
    assert gain < 1.0, f"vong lap van phan ky tren tran chi so hoa (gain={gain:.3f}, supply_scale={supply_scale})"
    assert abs(gain - THETA_CALVO) < 0.02, f"gain={gain:.3f} khac theta={THETA_CALVO} (con kenh phu thuoc gia khac?)"


# ==============================================================================
# TIEU CHI CHAP NHAN CHO VONG CHU CHUYEN (v0.16) -- do bang thuc nghiem, khong phai "trong co ve on"
# ==============================================================================
def _hire_fire_run(hire: float, seed: int, steps: int = 100):
    """Cac agent khac ngau nhien; chi ep dau hire/fire cua Firm va rho cua Chinh phu = 1."""
    env = MacroEnvironment(num_employees=50, num_firms=5, num_banks=1, max_steps=steps + 5)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    firm_rewards, employed = [], []
    for t in range(steps):
        acts = _actions(list(obs.keys()), rng)
        for aid in acts:
            if aid.startswith("firm_"):
                acts[aid][0] = hire
            if aid == "gov_1":
                acts[aid][2] = 1.0
        np.random.seed(4000 + t)
        obs, rew, *_ = env.step(acts)
        fr = [v for k, v in rew.items() if k.startswith("firm_")]
        firm_rewards.append(np.mean(fr) if fr else 0.0)
        emps = [a for a in env.agents.values() if a.agent_id.startswith("emp_")]
        employed.append(sum(1 for e in emps if e.employed_by is not None))
    alive = sum(1 for a in env.agents.values() if a.agent_id.startswith("firm_"))
    return np.mean(firm_rewards[30:]), np.mean(employed[-30:]), env.eco.base_living_cost, alive, env.gov.public_debt


def test_hiring_is_not_dominated_by_firing() -> None:
    """Tuong thich khuyen khich (incentive compatibility): mo hinh KHONG duoc thuong cho viec ngung
    san xuat. Loi that da xay ra: reward Firm khi SA THAI HET (-0,17) thang THUE (-0,59) vi doanh
    thu firm khong bu noi quy luong (thieu chi mua hang G + tien gui la tien chet), nen chinh sach
    hoc duoc la sa thai het => viec lam 0, gia 1e4 o moi episode (CLAUDE_HISTORY.md v0.15.1/v0.16)."""
    hire = [_hire_fire_run(+0.6, s) for s in (30, 31, 32, 33)]
    fire = [_hire_fire_run(-0.6, s) for s in (30, 31, 32, 33)]
    r_hire, e_hire, p_hire, alive_hire, debt_hire = (np.mean([x[i] for x in hire]) for i in range(5))
    r_fire, e_fire = np.mean([x[0] for x in fire]), np.mean([x[1] for x in fire])
    assert r_hire > r_fire, f"THUE khong thang SA THAI: {r_hire:.3f} <= {r_fire:.3f}"
    assert e_hire > e_fire + 10.0, f"viec lam khi thue ({e_hire:.1f}) khong vuot ro khi sa thai ({e_fire:.1f})"
    initial_price = 20.0  # env.reset(): eco.initialize(initial_living_cost=20)
    assert 0.25 * initial_price <= p_hire <= 3.0 * initial_price, f"gia khi thue ngoai [0,25x, 3x] gia goc: {p_hire:.2f}"
    assert alive_hire >= 4.0, f"qua nhieu firm pha san khi thue: song {alive_hire:.1f}/5"
    assert debt_hire == 0.0, "chi mua hang G khong duoc tao no cong"


def test_government_purchases_follow_taxes_and_treasury_cap() -> None:
    """G_t = min(rho * (thue + phat ky truoc), so du Kho bac dau ky) -- dung nhu thiet ke SIM."""
    env = MacroEnvironment(num_employees=40, num_firms=5, num_banks=1, max_steps=80)
    obs, _ = env.reset(seed=8)
    rng = np.random.default_rng(8)
    for rho in (1.0, 0.5, 0.0):
        for t in range(15):
            base_prev = env.gov.last_tax_collected + env.gov.last_fines_collected
            treasury_prev = env.gov.treasury
            acts = _actions(list(obs.keys()), rng)
            acts["gov_1"][2] = rho
            obs, *_ = env.step(acts)
            expected = min(rho * base_prev, max(0.0, treasury_prev))
            assert abs(env.gov.last_purchase - expected) < 1e-6, (
                f"rho={rho} t={t}: G={env.gov.last_purchase:.6f} != min(rho*thue_truoc, kho_bac)={expected:.6f}")
        if rho == 0.0:
            assert env.gov.last_purchase == 0.0


def test_household_consumption_draws_on_deposits() -> None:
    """Chi tieu tinh tren cash-on-hand (tien mat + tien gui): ho gia dinh 0 tien mat nhung co tien gui van
    mua duoc hang (truoc day tien gui la 'tien chet'), va tien gui giam tuong ung."""
    env = MacroEnvironment(num_employees=30, num_firms=4, num_banks=1, max_steps=20)
    env.reset(seed=3)
    emp = env.agents["emp_0"]
    emp.cash, emp.bank_deposit, emp.depository_bank_id = 0.0, 2000.0, "bank_0"
    acts = _actions(list(env.agents.keys()), np.random.default_rng(3))
    acts["emp_0"] = np.array([0.5, 1.0, 1.0], dtype=np.float32)
    env.step(acts)
    e = env.agents.get("emp_0")
    assert e is not None
    assert e.last_consumption > 1.0, "ho co tien gui nhung khong chi tieu duoc: tien gui van la tien chet"
    assert e.bank_deposit < 2000.0 - 1.0, "tien gui khong giam sau khi chi tieu tu tien gui"


def test_government_reward_is_based_on_real_gdp() -> None:
    """Thuong cua Chinh phu phai dua tren GDP THUC: nen kinh te lanh manh (GDP thuc lon, danh nghia
    nho) phai duoc thuong nhieu hon sieu lam phat (GDP thuc nho, danh nghia khong lo)."""
    from be.agents.government import Government
    from be.core.types import TransitionResult

    def reward(nominal, real):
        g = Government(agent_id="gov_t")
        g.initialize()
        g.current_gdp, g.last_gdp, g.current_real_gdp, g.last_real_gdp, g.current_gini = nominal, nominal, real, real, 0.3
        return g.calculate_reward(TransitionResult(agent_id="gov_t", state_delta={}, events_triggered=[], success=True))

    assert reward(nominal=6300.0, real=7500.0) > reward(nominal=200000.0, real=100.0)
