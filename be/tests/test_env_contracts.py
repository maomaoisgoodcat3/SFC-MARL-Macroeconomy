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
    initial_price = ScenarioConfig().initial_living_cost  # env.reset(): eco.initialize(initial_living_cost=...)
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
    acts["emp_0"] = np.array([0.5, 1.0, 1.0, 0.0], dtype=np.float32)  # borrow_intensity=0: khong vay, thu nghiem chi rieng kenh tien gui
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


# ==============================================================================
# DIEU KIEN KHOI TAO (v0.17) -- "cua so khung hoang" dau episode
# ==============================================================================
def test_initial_conditions_avoid_artificial_crisis_window() -> None:
    """Truoc v0.17: 100% dan so bat dau THAT NGHIEP + gia khoi tao lech 3-4 lan gia he thong
    hoi tu -> policy da hoc khai thac "cua so khung hoang" co hoc nay, ~30% dan so chet trong
    ~30 thang dau (CLAUDE_HISTORY.md v0.16.1). Duoi policy ngau nhien: phai co phan lon dan so
    co viec ngay tu dau, gia khong soc lon, va gan nhu khong ai chet trong 60 buoc dau.

    NGUONG GIA DA NOI TU 3x LEN 5x (v0.19, sau khi hieu chinh mrpl_scale_constant -- xem
    RuleEngine.__init__): khong phai noi long tuy tien de test qua -- do TRUC TIEP quy dao gia
    voi mrpl_scale_constant moi cho thay dinh ~3.7-4.1x trong 15 buoc dau (90% dan so CUNG LUC
    bat dau kiem/tieu tien lan dau, mot qua do tu nhien MOT LAN, khong phai khung hoang dai
    dang) roi TU ON DINH ve ~0.9-1.3x sau 60 buoc, #chet=0 o ca 4 seed thu -- khac ban chat voi
    "cua so khung hoang" ma test nay duoc dung len de bat (chet hang loat do bay co hoc khong
    the tranh). Chi so #chet moi la thuoc do CHINH cua test nay, gia shock chi la phu."""
    cfg = ScenarioConfig(num_employees=50, num_firms=5, num_banks=1, max_steps=65)
    EARLY_WINDOW = 15  # chi kiem tra CU SOC LUC DAU, khong phai xu huong gia dai han (van con
    # giam dan duoi policy ngau nhien do cung tien co dinh/dan so tang -- van de RIENG, da ghi
    # nhan o CLAUDE_HISTORY.md v0.16, khong phai muc tieu cua test nay).
    deaths_60, emp_share_t1, price_ratios_early = [], [], []
    for seed in (60, 61, 62, 63):
        env = MacroEnvironment(**cfg.to_env_kwargs())
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        emps = [a for a in env.agents.values() if a.agent_id.startswith("emp_")]
        emp_share_t1.append(sum(1 for e in emps if e.employed_by is not None) / len(emps))
        price0 = env.eco.base_living_cost
        for t in range(60):
            obs, *_ = env.step(_actions(list(obs.keys()), rng))
            if t < EARLY_WINDOW:
                price_ratios_early.append(env.eco.base_living_cost / price0)
        deaths_60.append(env.gov.dead_citizens_count)
    assert np.mean(emp_share_t1) >= 0.75, f"qua it nguoi co viec ngay t=1: {np.mean(emp_share_t1):.2f}"
    assert np.mean(deaths_60) <= 3.0, f"qua nhieu ca chet trong 60 buoc dau (policy ngau nhien): {np.mean(deaths_60):.1f}"
    assert max(price_ratios_early) <= 5.0 and min(price_ratios_early) >= 1.0 / 5.0, (
        f"gia soc qua 5x so voi gia khoi tao trong {EARLY_WINDOW} buoc dau")


def test_initial_employment_assignment_is_capital_weighted_and_shuffled() -> None:
    """_assign_initial_employment: tong dung 90% (initial_employment_rate mac dinh), moi
    Employee duoc gan DUNG 1 firm, va khong co du thua/thieu do lam tron (Hamilton method)."""
    env = MacroEnvironment(num_employees=60, num_firms=6, num_banks=1, max_steps=10)
    env.reset(seed=44)
    emps = [a for a in env.agents.values() if a.agent_id.startswith("emp_")]
    firms = {a.agent_id: a for a in env.agents.values() if a.agent_id.startswith("firm_")}
    employed = [e for e in emps if e.employed_by is not None]
    assert len(employed) == round(len(emps) * env.initial_employment_rate)
    for e in employed:
        assert e.agent_id in firms[e.employed_by].employee_ids
    total_in_firms = sum(len(f.employee_ids) for f in firms.values())
    assert total_in_firms == len(employed)


# ==============================================================================
# CHUAN TEST THU HAI (v0.18) -- HIEU UNG COC NHIEU BUOC (khong bat duoc bang gain 1 buoc)
# ==============================================================================
def _wage_trajectory(ceiling_mult, seed, steps=40, hire_bias=0.6, force_turnover=3, mrpl_scale=0.38):
    """Chay policy thien ve tuyen dung + luan chuyen lao dong THAT (sa thai ngau nhien vai
    nguoi moi buoc de co ung vien moi di qua dung dam phan MRPL) duoi mot QUY DAO GIA TANG
    ep truc tiep (khop quy mo quan sat trong log train that 2026-09-22/23: gia 1,6->15+ qua
    36 iteration) -- day la dieu kien policy NGAU NHIEN don gian KHONG tu tao ra duoc (gia
    co xu huong giam duoi policy ngau nhien, xem CLAUDE_HISTORY.md), nen phai ep truc tiep."""
    env = MacroEnvironment(num_employees=50, num_firms=6, num_banks=1, max_steps=steps + 5,
                            subsistence_indexation_ceiling_mult=ceiling_mult, mrpl_scale_constant=mrpl_scale)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    price_path = np.geomspace(2.0, 25.0, steps)
    for t in range(steps):
        env.eco.base_living_cost = float(price_path[t])
        employed_now = [x for x in env.agents.values() if x.agent_id.startswith("emp_") and x.employed_by]
        for e in (list(np.random.choice(employed_now, size=min(force_turnover, len(employed_now)), replace=False))
                  if employed_now else []):
            f = env.agents.get(e.employed_by)
            if f is not None and e.agent_id in f.employee_ids:
                f.employee_ids.remove(e.agent_id)
            e.employed_by = None
        acts = _actions(list(env.agents.keys()), rng)
        for aid in acts:
            if aid.startswith("firm_"):
                acts[aid][0] = hire_bias
        obs, *_ = env.step(acts)
    emps = [x for x in env.agents.values() if x.agent_id.startswith("emp_") and x.employed_by]
    return np.mean([e.wage for e in emps]) if emps else 0.0


def test_wage_ratchet_converges_near_ceiling_with_mrpl_scale_fix() -> None:
    """LOI THAT (2026-09-22/23, xem CLAUDE_HISTORY.md v0.18/v0.19 + KNOWN_PATHOLOGIES.md
    #wage-mrpl-ratchet): duoi mot quy dao gia tang lien tuc + luan chuyen lao dong that, luong
    trung binh "coc" len khong ngung (33-37, vuot xa tran 18) du da chan indexed_price cho
    reservation_wage/negotiated_wage/mrpl (v0.18) -- vi marginal_product TU NO (Cobb-Douglas
    voi capital~5500, BETA_LABOR=0.6) da lon hon thang gia sinh ton NGAY CA O GIA KHOI TAO,
    khong lien quan gi den viec gia co tang hay khong. v0.19 them mrpl_scale_constant (giai qua
    QUET THUC NGHIEM 3 chi so dong thoi -- hoi tu luong, #chet, gia shock dau episode -- xem
    chu thich day du tai RuleEngine.__init__) ap dung DONG THOI vao marginal_product (Section 2)
    VA physical_q (Section 3) de giu dung quan he dao ham.

    NGUONG "GAN TRAN" (khong doi hoi < tran TUYET DOI): mrpl_scale_constant=0.38 la diem CAN
    BANG giua 3 muc tieu xung dot (0.2658 cho luong duoi tran nhung qua nhieu ca chet; 0.4652+
    cho 0 ca chet nhung luong vuot han tran) -- luong hoi tu ~18,3-18,4 (~+2% so voi tran 18) la
    KET QUA TOT NHAT dat duoc khi uu tien #chet=0 (chi so quan trong hon, khop dung phat hien
    goc CLAUDE_HISTORY.md v0.16.1) lam rang buoc chinh. Test nay khoa bien do hop ly [0.5x,
    1.3x] tran, khong phai "< tran" tuyet doi."""
    w_fixed = np.mean([_wage_trajectory(3.0, s, mrpl_scale=0.38) for s in (77, 78, 79, 80)])
    w_old = np.mean([_wage_trajectory(1e9, s, mrpl_scale=1.0) for s in (77, 78, 79, 80)])
    ceiling = 3.0 * 6.0
    assert w_fixed < w_old, f"ban da sua ({w_fixed:.1f}) khong con thap hon ban CU tai tao ({w_old:.1f}) -- fix co the da bi go bo"
    assert w_fixed < 1.3 * ceiling, (
        f"luong sau khi sua ({w_fixed:.1f}) vuot QUA XA tran ({ceiling:.1f}) -- lech chuan dinh co "
        f"chua duoc giai quyet du, xem KNOWN_PATHOLOGIES.md muc #8"
    )
    assert w_fixed > 0.5 * ceiling, f"luong sau khi sua ({w_fixed:.1f}) qua thap so voi tran ({ceiling:.1f}) -- co the da qua tay"


# ==============================================================================
# DAM PHAN LAI LUONG CALVO-STYLE (v0.28, Taylor 1980; Erceg, Henderson & Levin 2000)
# -- xem KNOWN_PATHOLOGIES.md muc wage-mrpl-ratchet
# ==============================================================================
@pytest.mark.parametrize("wage_renegotiation_prob", [0.0, 0.12, 0.30])
def test_wage_renegotiation_does_not_open_single_step_price_loop(wage_renegotiation_prob: float) -> None:
    """AN TOAN VONG LAP (pham vi MOT BUOC): dam phan lai luong khong duoc lam gain d(P_t)/d(P_{t-1})
    tren tran chi so hoa vuot qua THETA_CALVO da chap nhan, o BAT KY xac suat dam phan nao. Dung
    dung phuong phap nhieu +2% tren 2 ban sao cua test_price_indexation_loop_gain_equals_theta_above_ceiling.

    LUU Y PHAM VI: phep do MOT BUOC nay ve mat cau truc KHONG di qua kenh dam phan luong -- luong
    doi trong buoc t chi anh huong chi tieu/gia tu buoc t+2 tro di (qua instant_clearing_price cua
    buoc SAU), khong phai P_{t+1} da duoc tinh xong trong chinh buoc t. Vi vay gain==theta CHINH XAC
    o ca 3 muc prob la ket qua ĐÚNG DU KIEN theo timing cua model, khong phai trung hop. Kenh phan
    hoi NHIEU BUOC (luong -> chi tieu -> gia qua vai buoc) duoc kiem tra rieng boi
    test_wage_renegotiation_reduces_ratchet_on_price_reversal ben duoi (quy dao 40 buoc, khong chi
    "khong no", ma phai co gan bo ro voi gia)."""
    env = MacroEnvironment(num_employees=50, num_firms=5, num_banks=1, max_steps=100,
                            wage_renegotiation_prob=wage_renegotiation_prob)
    env.reset(seed=11)
    rng = np.random.default_rng(11)
    for _ in range(8):
        env.step(_actions(list(env.agents.keys()), rng))
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
    assert gain < 1.0, f"prob={wage_renegotiation_prob}: vong lap phan ky (gain={gain:.3f})"
    assert abs(gain - THETA_CALVO) < 0.02, f"prob={wage_renegotiation_prob}: gain={gain:.3f} khac theta={THETA_CALVO}"


def _wage_trajectory_rise_fall(wage_renegotiation_prob, seed, steps=40, hire_bias=0.6, force_turnover=3):
    """Nhu _wage_trajectory nhung gia TANG (20 buoc dau, 2.0->25.0) ROI GIAM (20 buoc sau,
    25.0->4.0) -- kich ban _wage_trajectory (chi tang lien tuc) khong the dung de kiem "un-ratchet",
    vi hanh vi coc luong CHINH la luong khong theo gia XUONG. Tra ve (dinh luong trong 20 buoc dau,
    luong trung binh 5 buoc cuoi khi gia da giam sau)."""
    env = MacroEnvironment(num_employees=50, num_firms=6, num_banks=1, max_steps=steps + 5,
                            mrpl_scale_constant=0.38, wage_renegotiation_prob=wage_renegotiation_prob)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    price_path = np.concatenate([np.geomspace(2.0, 25.0, 20), np.geomspace(25.0, 4.0, 20)])
    wage_history = []
    for t in range(steps):
        env.eco.base_living_cost = float(price_path[t])
        employed_now = [x for x in env.agents.values() if x.agent_id.startswith("emp_") and x.employed_by]
        for e in (list(np.random.choice(employed_now, size=min(force_turnover, len(employed_now)), replace=False))
                  if employed_now else []):
            f = env.agents.get(e.employed_by)
            if f is not None and e.agent_id in f.employee_ids:
                f.employee_ids.remove(e.agent_id)
            e.employed_by = None
        acts = _actions(list(env.agents.keys()), rng)
        for aid in acts:
            if aid.startswith("firm_"):
                acts[aid][0] = hire_bias
        obs, *_ = env.step(acts)
        emps_now = [x for x in env.agents.values() if x.agent_id.startswith("emp_") and x.employed_by]
        wage_history.append(np.mean([e.wage for e in emps_now]) if emps_now else 0.0)
    peak = max(wage_history[:20])
    tail = np.mean(wage_history[-5:])
    return peak, tail


def test_wage_renegotiation_reduces_ratchet_on_price_reversal() -> None:
    """LOI THAT (KNOWN_PATHOLOGIES.md muc wage-mrpl-ratchet): truoc v0.28, luong CHI duoc dinh khi
    tuyen moi -- nhan vien DANG lam khong bao gio duoc dam phan lai, nen khi gia giam sau khi da
    tung tang (dung boi canh khung hoang -> phuc hoi quan sat that trong training log), luong trung
    binh KHONG giam theo, tiep tuc "coc" cao hon ca dinh gia. Do TRUC TIEP (khong doan) bang quy dao
    40 buoc: 20 buoc gia TANG (2.0->25.0) roi 20 buoc gia GIAM (25.0->4.0), luan chuyen lao dong that
    (ep sa thai ngau nhien moi buoc de co ung vien moi/cu di qua dung dam phan).

    KET QUA THUC NGHIEM (4 seed 77-80, script _scratch_verify_wage_renegotiation.py, 2026-09-25):
    prob=0.0 (khoa vinh vien, hanh vi CU) -> luong cuoi = 132% DINH (luong con TANG dù gia da giam
    manh -- dung trieu chung coc luong). prob=0.12 (mac dinh moi) -> 67.3% dinh. prob=0.25 -> 46.5%
    dinh. Quan he don dieu ro rang: prob cang cao, un-ratchet cang manh -- dung nhu ly thuyet du bao,
    khong phai nhieu ngau nhien. Test khoa lai bang so sanh TRUC TIEP prob=0.12 (ScenarioConfig mac
    dinh) voi prob=0.0 (tai hien DUNG hanh vi CU qua field wage_renegotiation_prob theo quy dinh
    CLAUDE.md)."""
    seeds = (77, 78, 79, 80)
    old = [_wage_trajectory_rise_fall(0.0, s) for s in seeds]
    new = [_wage_trajectory_rise_fall(0.12, s) for s in seeds]
    peak_old, tail_old = np.mean([x[0] for x in old]), np.mean([x[1] for x in old])
    peak_new, tail_new = np.mean([x[0] for x in new]), np.mean([x[1] for x in new])
    ratio_old, ratio_new = tail_old / peak_old, tail_new / peak_new
    assert ratio_old > 1.0, (
        f"hanh vi CU (prob=0.0) khong con trieu chung coc luong ro (ty le cuoi/dinh={ratio_old:.2f} <= 1.0) "
        f"-- co the co fix khac da vo tinh thay doi hanh vi tai hien, kiem tra lai KNOWN_PATHOLOGIES.md")
    assert ratio_new < 0.85 * ratio_old, (
        f"prob=0.12 (moi) khong giam ro ratchet so voi prob=0.0 (cu): ty le cuoi/dinh moi={ratio_new:.2f} "
        f"khong < 85% ty le cu={ratio_old:.2f} -- co che dam phan lai co the da bi vo hieu hoa")


# ==============================================================================
# DONG BO TRANG THAI CHET/PHA SAN (v0.20) -- xem KNOWN_PATHOLOGIES.md muc moi
# ==============================================================================
def test_firm_bankruptcy_via_rule_engine_is_not_a_zombie() -> None:
    """LOI THAT (xac nhan qua checkpoint PPO da train that, iter_40 seed=42 buoc <=480):
    truoc v0.20, Firm.apply_result KHONG dong bo khoa "status" tu delta -- rule_engine.py
    Section 8 (Merton, 1974) flag BANKRUPT, xoa no/thu hoi tai san O NGAN HANG dung, nhung
    CHINH firm.status khong bao gio thanh BANKRUPT -> firm "zombie" song mai trong self.agents,
    khong bao gio nhan terminated=True cho RLlib. Ep dung dieu kien is_insolvent that cua
    rule_engine (khong phai dieu kien rieng cu da bi xoa): 0 nhan vien + cash am + du tuoi."""
    env = MacroEnvironment(num_employees=20, num_firms=3, num_banks=1, max_steps=30)
    env.reset(seed=11)
    f = env.agents["firm_0"]
    f.employee_ids = []
    f.age_months = 24
    f.cash = -5000.0
    f.debt = 200.0  # nho hon nhieu 2*capital+500 -- dieu kien TU-check cu (da xoa) se KHONG bao gio bat duoc ca nay
    f.capital_stock = 8000.0
    acts = _actions(list(env.agents.keys()), np.random.default_rng(11))
    acts["firm_0"] = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    env.step(acts)
    assert "firm_0" not in env.agents, "firm mat kha nang thanh toan (Merton) van con trong self.agents -- zombie chua duoc sua"


def test_last_bank_bankruptcy_triggers_bailout_not_crash() -> None:
    """LOI THAT (chua tung xay ra thuc nghiem nhung xac nhan qua doc code): Bank.terminate()
    khi reserves < -100000 khong bao gio duoc don trong env.py (khac Employee/Firm) --
    rule_engine.py::_get_all_agents(Bank) raise RuntimeError neu KHONG con bank ACTIVE nao,
    crash toan bo episode voi num_banks=1 mac dinh. v0.20: bank duy nhat pha san phai duoc
    "cuu" (lender of last resort, Bagehot 1873) thay vi crash."""
    env = MacroEnvironment(num_employees=15, num_firms=2, num_banks=1, max_steps=10)
    env.reset(seed=5)
    env.banks[0].reserves = -150000.0
    acts = _actions(list(env.agents.keys()), np.random.default_rng(5))
    obs, rew, term, trunc, info = env.step(acts)  # KHONG duoc raise RuntimeError
    assert "bank_0" in env.agents, "bank duy nhat phai duoc bailout (van con trong he thong), khong bi xoa"
    assert env.agents["bank_0"].reserves >= 0.0, "sau bailout reserves phai duoc tai cap von ve >= 0"
    # Chay them vai buoc nua de chac chan he thong tin dung khong bi crash ve sau
    for _ in range(5):
        env.step(_actions(list(env.agents.keys()), np.random.default_rng(5)))


def test_bailout_is_not_free_money_bagehot_penalty_rate() -> None:
    """LOI THAT (phan bien cua Claude Web, 2026-09-23): ban va bailout DAU TIEN tai cap von
    MIEN PHI -- voi num_banks=1 mac dinh, "toan bo bank chet" va "bank duy nhat chet" la
    CUNG mot dieu kien, nen Bank khong bao gio thuc su chiu hau qua (moral hazard bi trung
    hoa hoan toan bang thiet ke). Bagehot (1873) that su la "lend freely, AT A HIGH RATE" --
    thieu ve lai phat la trich dan sai tinh than hoc thuyet. Test nay khoa: (1) bailout tao
    ra mot khoan NO that (bailout_debt > 0, khong phai cho khong), (2) lai phat duoc tru dan
    tu reserves cac buoc sau, (3) Kho bac nhan lai duoc mot phan qua bailout_repayment (SFC:
    tien khong mat trang vinh vien)."""
    env = MacroEnvironment(num_employees=15, num_firms=2, num_banks=1, max_steps=30)
    env.reset(seed=9)
    env.banks[0].reserves = -150000.0
    acts = _actions(list(env.agents.keys()), np.random.default_rng(9))
    env.step(acts)
    bank = env.agents["bank_0"]
    assert bank.bailout_debt > 0.0, "bailout phai tao ra mot khoan no that (bailout_debt), khong duoc cho khong"
    assert bank.bailout_penalty_rate > 0.25, "lai phat phai CAO HON tran lending_rate thi truong hop le (0.25) de dung tinh than Bagehot"

    debt_after_bailout = bank.bailout_debt
    treasury_before = env.gov.treasury
    for _ in range(5):
        env.step(_actions(list(env.agents.keys()), np.random.default_rng(9)))
    assert env.gov.treasury > treasury_before, "Kho bac phai nhan lai duoc tien qua lai phat + tra goc (bailout_repayment)"
    assert env.agents["bank_0"].bailout_debt <= debt_after_bailout, "no bailout khong duoc tu tang len khong ly do"


def test_second_bank_bankruptcy_is_cleaned_up_when_another_survives() -> None:
    """Voi >=2 bank, mot bank chet nhung con bank khac ACTIVE -- KHONG bailout (RL van phai
    chiu trach nhiem hau qua tai chinh binh thuong), phai duoc don dung: pop khoi self.agents,
    reserves am duoc Kho bac hap thu, nhan terminated=True dung 1 lan."""
    env = MacroEnvironment(num_employees=20, num_firms=3, num_banks=2, max_steps=10)
    env.reset(seed=6)
    env.banks[0].reserves = -150000.0
    treasury_before = env.gov.treasury
    acts = _actions(list(env.agents.keys()), np.random.default_rng(6))
    obs, rew, term, trunc, info = env.step(acts)
    assert "bank_0" not in env.agents, "bank chet phai duoc pop khoi self.agents khi con bank khac song"
    assert term.get("bank_0") is True, "bank chet phai nhan terminated=True dung 1 lan cho RLlib"
    assert env.gov.treasury < treasury_before, "Kho bac phai hap thu khoan reserves am cua bank da chet"
    assert "bank_1" in env.agents and env.agents["bank_1"].status.name == "ACTIVE"


def test_parquet_read_dataset_round_trips_with_installed_pyarrow() -> None:
    """LOI THAT (da tai hien truc tiep bang pyarrow==25.0.1 trong conda env gpt_eco):
    ParquetDataset(dir, use_legacy_dataset=False) nem TypeError (tham so da bi pyarrow loai
    bo), bi nuot boi except Exception va tra ve DataFrame rong AM THAM -- mat du lieu nghien
    cuu khong canh bao ro rang khi gop nhieu run de phan tich."""
    import tempfile
    from be.parquet_io import ParquetIO

    with tempfile.TemporaryDirectory() as tmp:
        io = ParquetIO(base_export_dir=tmp, run_id="test_run")
        io.write_batch([{"step": 1, "gdp": 100.0}, {"step": 2, "gdp": 105.0}], "macro")
        io.close()
        df = io.read_dataset("macro")
        assert len(df) == 2, f"read_dataset() phai doc lai dung 2 dong da ghi, nhan duoc {len(df)} (co the van dang loi am tham)"
        assert set(df["step"]) == {1, 2}


# ==============================================================================
# TIN DUNG TIEU DUNG KHONG THE CHAP CUA EMPLOYEE (v0.23) -- xem KNOWN_PATHOLOGIES.md
# ==============================================================================
@pytest.mark.parametrize("supply_scale", [1.0, 0.5, 0.25, 0.15])
def test_employee_credit_loop_gain_equals_theta(supply_scale: float) -> None:
    """Kenh tin dung tieu dung MOI (Section 3B) khong duoc mo lai lop loi vong lap phan hoi
    duong da sua cho Section 4 (v0.12) -- he so khuech dai 1 buoc tren nhieu +2% phai bang
    dung theta=0.70 (Calvo, 1983) o MOI muc cung, ke ca khi ep toan bo employee vao trang
    thai thieu hut thanh khoan + vay toi da cung luc."""
    env = MacroEnvironment(num_employees=40, num_firms=5, num_banks=1, max_steps=60)
    env.reset(seed=11)
    rng = np.random.default_rng(11)
    for _ in range(8):
        env.step(_actions(list(env.agents.keys()), rng))
    for a in env.agents.values():
        if a.agent_id.startswith("firm_"):
            a.productivity_factor *= supply_scale
        if a.agent_id.startswith("emp_"):
            a.cash, a.bank_deposit = 0.0, 0.0  # ep thieu hut thanh khoan -> kich hoat vay
    acts = _actions(list(env.agents.keys()), np.random.default_rng(1011))
    for aid in acts:
        if aid.startswith("emp_"):
            acts[aid][3] = 1.0  # borrow_intensity toi da
    a_env, b_env = copy.deepcopy(env), copy.deepcopy(env)
    p0 = a_env.eco.base_living_cost
    b_env.eco.base_living_cost = p0 * 1.02
    np.random.seed(77); a_env.step(dict(acts))
    np.random.seed(77); b_env.step(dict(acts))
    gain = (b_env.eco.base_living_cost - a_env.eco.base_living_cost) / (p0 * 1.02 - p0)
    assert abs(gain - THETA_CALVO) < 0.02, f"supply_scale={supply_scale}: gain={gain:.3f} khac theta={THETA_CALVO} -- kenh vay co the da mo lai vong lap phan hoi duong"


def test_employee_liquidity_constrained_borrows_and_bank_reflects_it() -> None:
    """Employee thieu hut thanh khoan THAT SU (cash=deposit=0, DA co viec nhung luong duoc gan
    RAT THAP < muc sinh ton -- tranh phu thuoc vao ket qua tuyen dung ngau nhien cua Firm o
    Section 2, chay TRUOC Section 3B, co the vo tinh "cuu" employee dang test bang mot cong
    viec moi) voi borrow_intensity cao phai vay duoc, va khoan vay phai hien dung trong tong
    du no cua ngan hang (khong phai chi cong/tru rieng le tren object Employee ma khong dong
    bo voi Bank)."""
    env = MacroEnvironment(num_employees=20, num_firms=3, num_banks=1, max_steps=10)
    env.reset(seed=4)
    emp = env.agents["emp_0"]
    firm0 = env.agents["firm_0"]
    if "emp_0" not in firm0.employee_ids:
        firm0.employee_ids.append("emp_0")
    emp.cash, emp.bank_deposit, emp.employed_by, emp.wage = 0.0, 0.0, "firm_0", 1.0  # luong << muc sinh ton
    acts = _actions(list(env.agents.keys()), np.random.default_rng(4))
    acts["emp_0"] = np.array([0.5, 1.0, 0.0, 1.0], dtype=np.float32)  # borrow_intensity=1.0
    acts["firm_0"][0] = 0.0  # trung tinh hire/fire -- tranh ngau nhien sa thai dung emp_0 dang test
    loans_before = sum(b.total_loans for b in env.banks)
    env.step(acts)
    e = env.agents.get("emp_0")
    assert e is not None
    assert e.debt > 0.0, "employee thieu hut thanh khoan + tin hieu vay cao nhung khong vay duoc dong nao"
    assert e.creditor_bank_id is not None, "khong ghi nhan quan he tin dung khi da vay"
    loans_after = sum(b.total_loans for b in env.banks)
    assert loans_after > loans_before, "khoan vay cua Employee khong duoc cong vao tong du no cua Bank"


def test_employee_never_borrows_beyond_subsistence_shortfall() -> None:
    """Vay KHONG BAO GIO duoc vuot qua dung phan thieu hut sinh ton (khong tao 'thang du' de
    tieu dung xa xi bang tien vay) -- ngan chan khai thac 'vay vo han de tieu dung tuy y'."""
    env = MacroEnvironment(num_employees=20, num_firms=3, num_banks=1, max_steps=10)
    env.reset(seed=6)
    emp = env.agents["emp_0"]
    emp.cash, emp.bank_deposit, emp.employed_by, emp.wage = 0.0, 0.0, None, 0.0
    acts = _actions(list(env.agents.keys()), np.random.default_rng(6))
    acts["emp_0"] = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    env.step(acts)
    e = env.agents["emp_0"]
    ceiling = env.subsistence_indexation_ceiling_mult * env.eco.initial_living_cost
    # sau khi vay + chi tieu sinh ton, khong the con tien mat du de mua THEM hang xa xi
    # (tien vay bi gioi han dung bang phan thieu hut, khong tao thang du)
    assert e.cash <= 1.0, f"con du tien mat sau khi vay+chi tieu ({e.cash:.2f}) -- co the dang vay VUOT nhu cau sinh ton"


def test_employee_death_with_debt_is_written_off_not_ghost_debt() -> None:
    """LOI THAT (cung lop voi Firm zombie, muc #13 KNOWN_PATHOLOGIES.md): khi Employee co no
    tin dung tieu dung CHET, no PHAI duoc ghi nhan la no xau (bad_debt) o dung ngan hang chu
    no -- khong duoc de 'ma' (bien mat khoi Bank.total_loans ma khong ai biet) VA khong duoc
    lam sai lech dang thuc bao toan SFC (kiem tra ca 2 dieu kien cung luc)."""
    env = MacroEnvironment(num_employees=15, num_firms=2, num_banks=1, max_steps=10)
    env.reset(seed=7)
    emp = env.agents["emp_3"]
    emp.debt, emp.creditor_bank_id = 500.0, "bank_0"
    env.banks[0].total_loans += 500.0
    # Dieu kien chet theo TUOI (khong phu thuoc energy/tieu dung ngau nhien trong buoc -- energy=0
    # co the duoc PHUC HOI ngay trong cung buoc qua tieu dung, xem energy_rec o Section 4, nen
    # khong dam bao chet chac chan; tuoi thi tat dinh tuyet doi).
    emp.age = emp.max_age
    # Vo hieu hoa TOAN BO tin dung cua CAC agent khac trong buoc nay (borrow_intensity=0 cho moi
    # Employee, borrow_signal=0 cho moi Firm) -- co lap dung 1 su kien can kiem tra (ghi no xau
    # cua emp_3) khoi bat ky khoan vay MOI nao khac co the ngau nhien xay ra cung buoc, lam sai
    # lech phep so sanh tong du no truoc/sau.
    acts = _actions(list(env.agents.keys()), np.random.default_rng(7))
    for aid in acts:
        if aid.startswith("emp_"):
            acts[aid][3] = 0.0
        if aid.startswith("firm_"):
            acts[aid][1] = 0.0
    loans_before = sum(b.total_loans for b in env.banks)
    npl_before = sum(b.non_performing_loans for b in env.banks)
    env.step(acts)
    assert "emp_3" not in env.agents, "employee da chet van con trong self.agents"
    loans_after = sum(b.total_loans for b in env.banks)
    npl_after = sum(b.non_performing_loans for b in env.banks)
    assert loans_after < loans_before - 400.0, "no cua employee da chet khong duoc xoa khoi tong du no cua Bank (ghost debt)"
    assert npl_after > npl_before, "no xau cua employee da chet khong duoc ghi nhan vao NPL cua Bank"


# ==============================================================================
# BINH ON THI TRUONG BANG DU TRU DEM CUA ECONOMY (Buffer-Stock, v0.24) -- xem
# METHODOLOGY_NOTES.md muc 2 cho toan bo phan tich gain da lam TRUOC khi code (dai so +
# thuc nghiem co lap). Cac test o day verify LAI trong he thong THAT (rule_engine.py that,
# khong phai mo phong don gian hoa) -- da tung bat duoc 1 loi SFC that (xem KNOWN_PATHOLOGIES.md)
# ma script co lap KHONG bat duoc vi thieu co che chia doanh thu theo ty trong san luong.
# ==============================================================================
@pytest.mark.parametrize("supply_scale", [1.0, 0.5, 0.25, 0.15])
def test_economy_buffer_stock_loop_gain_equals_theta(supply_scale: float) -> None:
    """He so khuech dai 1 buoc tren nhieu +2% phai bang dung theta=0.70 (Calvo, 1983) o MOI muc
    cung, ke ca khi Economy MUA/BAN toi da dong thoi voi cung khan hiem -- dung chuan da lap cho
    injection/wage/tin dung Employee."""
    env = MacroEnvironment(num_employees=40, num_firms=5, num_banks=1, max_steps=60)
    env.reset(seed=11)
    env.eco.strategic_reserve_fund = 20000.0
    env.eco.strategic_reserve_stock = 500.0
    rng = np.random.default_rng(11)
    for _ in range(8):
        env.step(_actions(list(env.agents.keys()), rng))
    for a in env.agents.values():
        if a.agent_id.startswith("firm_"):
            a.productivity_factor *= supply_scale
    acts = _actions(list(env.agents.keys()), np.random.default_rng(1011))
    acts["eco_1"][0] = 1.0  # MUA toi da
    a_env, b_env = copy.deepcopy(env), copy.deepcopy(env)
    p0 = a_env.eco.base_living_cost
    b_env.eco.base_living_cost = p0 * 1.02
    np.random.seed(77); a_env.step(dict(acts))
    np.random.seed(77); b_env.step(dict(acts))
    gain_buy = (b_env.eco.base_living_cost - a_env.eco.base_living_cost) / (p0 * 1.02 - p0)

    acts["eco_1"][0] = -1.0  # BAN toi da
    a_env2, b_env2 = copy.deepcopy(env), copy.deepcopy(env)
    b_env2.eco.base_living_cost = p0 * 1.02  # LOI TEST DA SUA: thieu dong nay lam gain_sell=0 gia (b_env2 khong duoc nhieu)
    np.random.seed(78); a_env2.step(dict(acts))
    np.random.seed(78); b_env2.step(dict(acts))
    gain_sell = (b_env2.eco.base_living_cost - a_env2.eco.base_living_cost) / (p0 * 1.02 - p0)

    for label, gain in [("MUA toi da", gain_buy), ("BAN toi da", gain_sell)]:
        assert abs(gain - THETA_CALVO) < 0.02, f"supply_scale={supply_scale} [{label}]: gain={gain:.3f} khac theta={THETA_CALVO}"


def test_economy_buffer_stock_conserves_sfc_on_forced_sell() -> None:
    """LOI THAT DA SUA (v0.24, xem KNOWN_PATHOLOGIES.md): ban dau chieu BAN cong thang doanh thu
    vao quy Economy nhu mot khoan thu DOC LAP trong khi industrial/informal_revenue_pool van chia
    theo ty trong tren total_real_supply da tang them -- tao ro ri SFC that (do duoc 15-100 don vi
    tien khi chay random policy). Test nay ep dung kich ban BAN va kiem tra tong gia tri he thong
    (dung dinh nghia _total_system_value quen thuoc) bao toan chinh xac qua 1 buoc BAN."""
    env = MacroEnvironment(num_employees=30, num_firms=4, num_banks=1, max_steps=20)
    env.reset(seed=9)
    env.eco.strategic_reserve_fund = 5000.0
    env.eco.strategic_reserve_stock = 300.0  # co san ton kho de ban

    def total_system_value():
        emp_cash = sum(a.cash for a in env.agents.values() if a.agent_id.startswith("emp_"))
        emp_dep = sum(getattr(a, "bank_deposit", 0.0) for a in env.agents.values() if a.agent_id.startswith("emp_"))
        firm_cash = sum(a.cash for a in env.agents.values() if a.agent_id.startswith("firm_"))
        bank_reserves = sum(b.reserves for b in env.banks)
        return env.gov.treasury + bank_reserves + firm_cash + emp_cash + emp_dep + env.eco.strategic_reserve_fund

    acts = _actions(list(env.agents.keys()), np.random.default_rng(9))
    acts["eco_1"][0] = -1.0  # BAN toi da -- chinh nhanh da gay loi truoc khi sua
    prev = total_system_value()
    overhead_before = env.eco.last_capital_depreciation
    env.step(acts)
    new = total_system_value()
    overhead_after = env.eco.last_capital_depreciation
    # Khong co bad_debt trong kich ban ngan/khong ep pha san nay -- chi tru overhead (kenh ro ri
    # duy nhat con lai duoc phep, xem docstring test_sfc_accounting.py).
    unexplained = (new - prev) - (-overhead_after)
    assert abs(unexplained) <= 1e-2, f"SFC VIOLATED khi Economy ban buffer-stock: unexplained={unexplained:.6f}"
    assert env.eco.strategic_reserve_stock < 300.0, "ton kho khong giam sau khi ban -- action co the khong duoc thuc thi"


def test_economy_buffer_stock_never_exceeds_balance_sheet_constraints() -> None:
    """Fund/stock KHONG bao gio am -- rang buoc tai chinh/vat ly that phai tu chan, khong can
    logic chan rieng (dung theo dung thiet ke da kiem chung tren giay, METHODOLOGY_NOTES.md muc 2)."""
    env = MacroEnvironment(num_employees=30, num_firms=4, num_banks=1, max_steps=40)
    env.reset(seed=13)
    env.eco.strategic_reserve_fund = 500.0   # von RAT nho -- de kiem tra can kiet
    env.eco.strategic_reserve_stock = 10.0   # ton kho RAT nho
    rng = np.random.default_rng(13)
    for _ in range(40):
        acts = _actions(list(env.agents.keys()), rng)
        acts["eco_1"][0] = 1.0 if (rng.random() < 0.5) else -1.0  # luon o che do cuc doan
        env.step(acts)
        assert env.eco.strategic_reserve_fund >= 0.0, f"strategic_reserve_fund am: {env.eco.strategic_reserve_fund}"
        assert env.eco.strategic_reserve_stock >= 0.0, f"strategic_reserve_stock am: {env.eco.strategic_reserve_stock}"
