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
from be.agents.employee import Employee
from be.agents.firm import Firm
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
    # shirking_monitor_prob=0.0 (v0.34): test nay CO LAP kenh coc luong (gia -> luong). Giam sat
    # Shapiro-Stiglitz voi effort NGAU NHIEN (~50% lao dong effort < 0.5) tao them ~54 vu sa thai/episode
    # = luan chuyen phu -> them hop dong moi khoa theo MRPL lam phong -> luong +1.0 (q=0.05) / +3.1
    # (q=0.10) so voi q=0 (do truc tiep, KNOWN_PATHOLOGIES.md #27). Voi effort tuan thu (>= nguong)
    # KHONG co vu sa thai nao va ket qua trung khit q=0 -- xem test_shirking_monitor_has_no_wage_effect_at_compliant_effort.
    env = MacroEnvironment(num_employees=50, num_firms=6, num_banks=1, max_steps=steps + 5,
                            subsistence_indexation_ceiling_mult=ceiling_mult, mrpl_scale_constant=mrpl_scale,
                            shirking_monitor_prob=0.0)
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
    1.5x] tran, khong phai "< tran" tuyet doi.

    DOI BIEN 1.3x -> 1.5x (v0.36, 2026-09-27) -- KHONG phai noi long de test pass, ma vi phep do CU bi
    NHIEM loi wage=0 (KNOWN_PATHOLOGIES.md #30a/#29g): nhom lao dong gan viec luc reset co thuoc tinh
    wage ket o 0 (van duoc TRA luong du phong nhung khong ghi lai, va bi loai khoi dam phan lai) -> keo
    TRUNG BINH luong xuong. Do tach nguyen nhan (4 seed 77-80): code v0.35 (co loi) ~23.1; chi sua ghi
    lai thuoc tinh o nhanh du phong rule_engine.py: 24.94; sua ca gan luong luc reset (env.py): 25.09;
    quy tac tai tro tro cap (sua A cung dot) KHONG anh huong (legacy_gate va affordable cho so giong het).
    Tuc muc "~18,3 gan tran" trong bang hieu chinh mrpl_scale_constant (RuleEngine.__init__) va muc ~23
    truoc day DEU do voi loi nay -- gia tri dung duoi kich ban ep lam phat nay la ~25.1 (1.39x tran).
    Ban sua van cat ~56% hien tuong coc luong so voi ban CU tai tao (25.1 vs 56.9) -- them assert dinh
    luong w_fixed < 0.6*w_old de test van bao ve dung muc dich goc (fix con hieu luc)."""
    w_fixed = np.mean([_wage_trajectory(3.0, s, mrpl_scale=0.38) for s in (77, 78, 79, 80)])
    w_old = np.mean([_wage_trajectory(1e9, s, mrpl_scale=1.0) for s in (77, 78, 79, 80)])
    ceiling = 3.0 * 6.0
    assert w_fixed < w_old, f"ban da sua ({w_fixed:.1f}) khong con thap hon ban CU tai tao ({w_old:.1f}) -- fix co the da bi go bo"
    assert w_fixed < 0.6 * w_old, (
        f"ban da sua ({w_fixed:.1f}) khong con cat duoc >=40% coc luong so voi ban CU ({w_old:.1f}) -- do: 0.44x (v0.36)"
    )
    assert w_fixed < 1.5 * ceiling, (
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
                            mrpl_scale_constant=0.38, wage_renegotiation_prob=wage_renegotiation_prob,
                            shirking_monitor_prob=0.0)  # co lap kenh coc luong -- xem chu thich o _wage_trajectory
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
    ceiling = env.subsistence_indexation_ceiling_mult * env.eco.initial_living_cost
    # Nhu cau sinh ton danh nghia = gio hang (1.0) x gia da chi so hoa (co tran) TAI THOI DIEM QUYET DINH VAY.
    subsistence_need = min(max(1.0, env.eco.base_living_cost), ceiling)
    env.step(acts)
    e = env.agents["emp_0"]
    # v0.35: DOI tu "cash cuoi buoc <= 1.0" sang HOP DONG TRUC TIEP "no <= nhu cau sinh ton". Ly do (do,
    # khong doan): cash cuoi buoc con LAN thu nhap phi chinh thuc (Section 4, tinh SAU quyet dinh vay ->
    # khong nam trong shortfall uoc tinh luc vay) nen phu thuoc vao so lan rut ngau nhien cua cac tac tu
    # khac; them 1 chieu hanh dong cua Government (v0.35) doi chuoi so ngau nhien cua test va lam
    # cash cuoi buoc = 4.7 du no chi 4.08 <= nhu cau. Hop dong that: khoan vay khong bao gio vuot phan thieu
    # hut sinh ton (khong tao 'thang du' de tieu dung xa xi bang tien vay).
    assert e.debt <= subsistence_need + 1e-6, f"vay {e.debt:.2f} VUOT nhu cau sinh ton {subsistence_need:.2f}"
    assert e.debt > 0.0, "employee thieu hut thanh khoan + borrow=1 phai vay duoc (test khong kiem tra suong)"


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
    # v0.37: dai gia (band) co the CHAN can thiep -> test se PASS RONG (gain = theta tam thuong). Ep gia tham chieu de
    # MO dai gia cho dung chieu dang do, va bat buoc can thiep THUC SU duoc thuc thi (xem assert fund/stock ben duoi).
    env.eco.buffer_reference_price = env.eco.base_living_cost * 2.0
    fund_before, stock_before = env.eco.strategic_reserve_fund, env.eco.strategic_reserve_stock  # so du NGAY truoc buoc do
    a_env, b_env = copy.deepcopy(env), copy.deepcopy(env)
    p0 = a_env.eco.base_living_cost
    b_env.eco.base_living_cost = p0 * 1.02
    np.random.seed(77); a_env.step(dict(acts))
    np.random.seed(77); b_env.step(dict(acts))
    gain_buy = (b_env.eco.base_living_cost - a_env.eco.base_living_cost) / (p0 * 1.02 - p0)

    assert a_env.eco.strategic_reserve_fund < fund_before, "MUA khong duoc thuc thi -- test gain chieu MUA se PASS RONG"
    acts["eco_1"][0] = -1.0  # BAN toi da
    env.eco.buffer_reference_price = env.eco.base_living_cost * 0.5  # mo dai gia chieu BAN (v0.37)
    a_env2, b_env2 = copy.deepcopy(env), copy.deepcopy(env)
    b_env2.eco.base_living_cost = p0 * 1.02  # LOI TEST DA SUA: thieu dong nay lam gain_sell=0 gia (b_env2 khong duoc nhieu)
    np.random.seed(78); a_env2.step(dict(acts))
    np.random.seed(78); b_env2.step(dict(acts))
    gain_sell = (b_env2.eco.base_living_cost - a_env2.eco.base_living_cost) / (p0 * 1.02 - p0)

    assert a_env2.eco.strategic_reserve_stock < stock_before, "BAN khong duoc thuc thi -- test gain chieu BAN se PASS RONG"
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
    env.eco.buffer_reference_price = env.eco.base_living_cost * 0.5  # v0.37: mo dai gia chieu BAN (neu khong, band chan lenh ban)

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


# ==============================================================================
# SO CAI NGAN HANG (v0.33, KNOWN_PATHOLOGIES.md #28) -- ghi nhan no xau DUNG MOT LAN, vo no
# ngan hang theo VON CHU SO HUU, so cai tien gui khop sau chet/thua ke. Ba test nay la ban
# HOI QUY cua audits/2026-09-26/test_audit_findings.py::F2/F3 (nguyen ban "PASS khi loi ton
# tai", nay dao nguoc assert sau khi sua).
# ==============================================================================
def _no_borrow_actions(env: MacroEnvironment, seed: int):
    """Hanh dong ngau nhien co dinh seed, TAT moi khoan vay MOI (Employee borrow=0, Firm borrow=0)
    de co lap mot su kien tin dung duy nhat khoi nhieu ngau nhien khac."""
    acts = _actions(list(env.agents.keys()), np.random.default_rng(seed))
    for aid in acts:
        if aid.startswith("emp_"):
            acts[aid][3] = 0.0
        if aid.startswith("firm_"):
            acts[aid][1] = 0.0
    return acts


def test_default_loss_is_recognized_once_in_bank_equity() -> None:
    """LOI THAT (audit 2026-09-26, F2): vo no bi ghi nhan 2 lan -- xoa khoan vay VA tru them
    reserves (reserves da giam luc giai ngan). Kiem chung: hai ban sao CUNG buoc CUNG hanh dong,
    chi khac viec nguoi vay co no D chet (A) hay song (B). Chenh lech von ngan hang
    (reserves + total_loans) B - A chi do vo no gay ra, PHAI ~ D (xoa khoan vay), khong phai ~2D."""
    env = MacroEnvironment(num_employees=15, num_firms=2, num_banks=1, max_steps=10)
    env.reset(seed=5)
    bank = env.banks[0]
    emp = next(a for a in env.agents.values() if isinstance(a, Employee))
    D = 1000.0
    bank.reserves -= D          # giai ngan dung Section 3B: reserves -D, loans +D, cash nguoi vay +D
    bank.total_loans += D
    emp.cash += D
    emp.debt += D
    emp.creditor_bank_id = bank.agent_id
    acts = _no_borrow_actions(env, 5)
    equity = {}
    for dies in (True, False):
        e2 = copy.deepcopy(env)
        me = e2.agents[emp.agent_id]
        me.age = me.max_age if dies else 18
        np.random.seed(77)
        e2.step({k: v.copy() for k, v in acts.items()})
        equity[dies] = e2.banks[0].equity
    diff = equity[False] - equity[True]
    assert 0.8 * D <= diff <= 1.3 * D, f"von ngan hang giam {diff:.1f} do 1 vu vo no D={D} (dung ~D; ~2D = ghi nhan 2 lan)"


def test_bank_failure_uses_equity_not_gross_reserves() -> None:
    """Tieu chi vo no theo VON CHU SO HUU (v0.33): reserves am nhung so du no lon (von duong) KHONG
    duoc coi la vo no; von am qua san moi vo no. Tieu chi CU (bank_failure_criterion='reserves')
    van tai hien duoc de lam ablation (KNOWN_PATHOLOGIES.md #28)."""
    def run(criterion: str, reserves: float, loans: float) -> bool:
        env = MacroEnvironment(num_employees=15, num_firms=2, num_banks=2, max_steps=10,
                               bank_failure_criterion=criterion)
        env.reset(seed=11)
        env.banks[0].reserves, env.banks[0].total_loans = reserves, loans
        env.step(_no_borrow_actions(env, 11))
        return "bank_0" not in env.agents or env.banks[0].status.name == "TERMINATED"

    assert not run("equity", -150000.0, 400000.0), "von duong (+250k) nhung bi tuyen vo no chi vi reserves am"
    assert run("equity", -150000.0, 10000.0), "von am -140k < -100k phai vo no"
    assert run("reserves", -150000.0, 400000.0), "tieu chi CU (reserves tho) phai con tai hien duoc"


def test_deposit_ledger_stays_consistent_after_death_and_inheritance() -> None:
    """LOI THAT (audit 2026-09-26, F3): tien gui roi ngan hang qua kenh chet (di san -> Kho bac) va
    thua ke cha/me->con khong tru Bank.total_deposits -> so cai phinh 'tien gui ma'."""
    env = MacroEnvironment(num_employees=40, num_firms=3, num_banks=1, max_steps=60)
    env.reset(seed=8)
    rng = np.random.default_rng(8)
    saw_deposit_death = False
    for t in range(60):
        acts = _actions(list(env.agents.keys()), rng)
        if t % 6 == 0:
            for e in list(env.agents.values()):
                if isinstance(e, Employee) and e.bank_deposit > 5.0:
                    e.age = e.max_age  # ep chet vi tuoi (tat dinh) NGUOI CO TIEN GUI
                    saw_deposit_death = True
                    break
        env.step(acts)
        emps = [a for a in env.agents.values() if isinstance(a, Employee)]
        gap = sum(b.total_deposits for b in env.banks) - sum(e.bank_deposit for e in emps)
        assert abs(gap) <= 1e-3, f"[t={t}] so cai tien gui lech {gap:.4f}"
    assert saw_deposit_death, "test khong bao gio ep duoc nguoi co tien gui chet (kenh khong duoc kiem)"


# ==============================================================================
# KY LUAT LAO DONG SHAPIRO-STIGLITZ (v0.34, KNOWN_PATHOLOGIES.md #27) -- effort PHAI co loi ich
# ca nhan (tranh bi sa thai), khong duoc chi la chi phi thuan tuy. Ban hoi quy cua
# audits/2026-09-26/test_audit_findings.py::F1a ("effort khong co loi ich ca nhan").
# ==============================================================================
def _forced_effort_actions(env: MacroEnvironment, seed: int, effort: float):
    acts = _actions(list(env.agents.keys()), np.random.default_rng(seed))
    for aid in acts:
        if aid.startswith("emp_"):
            acts[aid][0] = effort
    return acts


def test_shirking_is_detected_only_below_effort_threshold() -> None:
    """q=1 (kiem tra 100% de test tat dinh): effort duoi nguong -> nguoi co viec (khong vua tuyen,
    du nang luong) bi sa thai het; effort tren nguong -> khong ai bi sa thai vi 'lam luoi'."""
    from be.core.enums import EventType

    def shirk_fires(effort: float, q: float) -> int:
        env = MacroEnvironment(num_employees=40, num_firms=4, num_banks=1, max_steps=5,
                               shirking_monitor_prob=q, shirking_effort_threshold=0.5)
        env.reset(seed=3)
        n = [0]
        env.event_bus.subscribe(EventType.FIRE, lambda ev: n.__setitem__(0, n[0] + ("Shirking" in str(ev.payload.get("reason", "")))))
        employed = sum(1 for a in env.agents.values() if isinstance(a, Employee) and a.employed_by)
        env.step(_forced_effort_actions(env, 3, effort))
        assert employed > 10
        return n[0]

    assert shirk_fires(0.2, q=1.0) > 10, "effort duoi nguong + giam sat 100% phai sa thai nguoi lam luoi"
    assert shirk_fires(0.6, q=1.0) == 0, "effort tren nguong khong duoc bi coi la lam luoi"
    assert shirk_fires(0.2, q=0.0) == 0, "q=0 phai tai hien DUNG hanh vi CU (khong co giam sat)"


def test_shirking_monitor_has_no_wage_effect_at_compliant_effort() -> None:
    """Tuong tac voi coc luong (do o KNOWN_PATHOLOGIES.md #27): giam sat CHI anh huong luong khi co
    nguoi lam luoi; voi effort tuan thu (>= nguong) ket qua trung khit q=0 -- diem van hanh cua mot
    chinh sach da hoc dung ky luat khong bi tac dong phu."""
    def mean_wage(q: float) -> float:
        env = MacroEnvironment(num_employees=50, num_firms=6, num_banks=1, max_steps=45,
                               mrpl_scale_constant=0.38, shirking_monitor_prob=q)
        env.reset(seed=78)
        rng = np.random.default_rng(78)
        price_path = np.geomspace(2.0, 25.0, 40)
        for t in range(40):
            env.eco.base_living_cost = float(price_path[t])
            acts = _forced_effort_actions(env, 78 + t, 0.6)
            for aid in acts:
                if aid.startswith("firm_"):
                    acts[aid][0] = 0.6
            np.random.seed(1000 + t)
            env.step(acts)
        emps = [x for x in env.agents.values() if isinstance(x, Employee) and x.employed_by]
        return float(np.mean([e.wage for e in emps])) if emps else 0.0

    assert abs(mean_wage(0.0) - mean_wage(0.05)) < 1e-9, "giam sat khong duoc tac dong phu khi moi nguoi effort tuan thu"


def test_effort_has_private_benefit_only_with_monitoring() -> None:
    """TIEU CHI KHUYEN KHICH (incentive compatibility) cho kenh effort -- tuong tu vai tro cua
    test_hiring_is_not_dominated_by_firing cho kenh tuyen dung: mot lao dong duoc ep effort 0.1 (luoi)
    hay 0.6 (tuan thu), moi truong con lai co dinh (moi nguoi khac effort 0.6). Loi nhuan ca nhan
    (tong reward chiet khau gamma=0.99 sau 60 buoc, trung binh 8 seed; do 16 seed: khoang cach = +1.91 +- 0.60
    voi lockout 9 thang, chi +0.34 +- 0.36 voi lockout 6 thang -- KHONG co y nghia, nen mac dinh la 9):
      - q=0 (hanh vi CU): luoi > tuan thu (effort chi la chi phi) -- TAI HIEN loi khuyen khich cu lech;
      - q=0.05 (Shapiro-Stiglitz): tuan thu > luoi -- loi khuyen khich DUNG."""
    def discounted_return(q: float, effort: float, seed: int) -> float:
        env = MacroEnvironment(num_employees=40, num_firms=4, num_banks=1, max_steps=65,
                               shirking_monitor_prob=q, shirking_effort_threshold=0.5)
        env.reset(seed=seed)
        focal = next(a for a in env.agents.values() if isinstance(a, Employee) and a.employed_by)
        fid, total = focal.agent_id, 0.0
        for t in range(60):
            if fid not in env.agents:
                break
            acts = _forced_effort_actions(env, seed * 100 + t, 0.6)
            acts[fid][0] = effort
            np.random.seed(seed * 1000 + t)
            _, rew, *_ = env.step(acts)
            total += (0.99 ** t) * rew.get(fid, 0.0)
        return total

    seeds = range(8)
    gap_no_monitor = np.mean([discounted_return(0.0, 0.6, s) - discounted_return(0.0, 0.1, s) for s in seeds])
    gap_monitor = np.mean([discounted_return(0.05, 0.6, s) - discounted_return(0.05, 0.1, s) for s in seeds])
    assert gap_no_monitor < 0.0, f"q=0: effort khong co loi ich -> luoi phai thang (gap={gap_no_monitor:.3f})"
    assert gap_monitor > 0.0, f"q=0.05: tuan thu phai thang luoi (gap={gap_monitor:.3f}) -- ky luat Shapiro-Stiglitz khong du manh"


def test_dismissed_shirker_cannot_be_rehired_until_lockout_expires() -> None:
    """Dau an sa thai vi luoi (Gibbons & Katz 1991, v0.34): nguoi bi sa thai vi luoi bi loai khoi ung vien
    cua MOI firm dung `shirker_rehire_lockout_months` thang, roi moi duoc tuyen lai. Neu khong co dau an,
    thi truong lao dong 'khong ma sat' tuyen lai ngay nen sa thai gan nhu vo hai (do: q=0.05 chua du de
    tuan thu thang luoi, KNOWN_PATHOLOGIES.md #27)."""
    L = 4
    env = MacroEnvironment(num_employees=30, num_firms=4, num_banks=1, max_steps=30,
                           shirking_monitor_prob=1.0, shirking_effort_threshold=0.5,
                           shirker_rehire_lockout_months=L)
    env.reset(seed=4)
    focal = next(a for a in env.agents.values() if isinstance(a, Employee) and a.employed_by)
    fid = focal.agent_id
    acts = _forced_effort_actions(env, 4, 0.6)
    acts[fid][0] = 0.1            # chi nguoi nay luoi; q=1 -> bi phat hien chac chan
    for aid in acts:
        if aid.startswith("firm_"):
            acts[aid][0] = 0.9   # firm muon tuyen manh -- neu khong co dau an, se tuyen lai ngay
    env.step(acts)
    me = env.agents[fid]
    assert me.employed_by is None and me.hire_lockout_until == env.timestep + 1 + L, "nguoi lam luoi phai bi sa thai va dong dau an"
    for k in range(1, L + 1):     # thang t+1 ... t+L: khong duoc tuyen
        acts = _forced_effort_actions(env, 40 + k, 0.6)
        for aid in acts:
            if aid.startswith("firm_"):
                acts[aid][0] = 0.9
        env.step(acts)
        assert env.agents[fid].employed_by is None, f"thang +{k} van trong thoi gian dau an nhung da duoc tuyen lai"
    got_job = False
    for k in range(6):            # het dau an: co the duoc tuyen lai (firm dang tuyen manh, ky nang cao/thap tuy)
        acts = _forced_effort_actions(env, 90 + k, 0.6)
        for aid in acts:
            if aid.startswith("firm_"):
                acts[aid][0] = 0.9
        env.step(acts)
        got_job |= env.agents[fid].employed_by is not None
    assert got_job, "het thoi gian dau an ma van khong bao gio duoc tuyen lai -- dau an khong het han?"


# ==============================================================================
# CHINH PHU: TRO CAP THAT NGHIEP = ACTION[4], REWARD CAN VOI THUOC DO BENCHMARK, KHO BAC QUY MO THAT
# (v0.35, KNOWN_PATHOLOGIES.md #29a/#29b/#29c)
# ==============================================================================
def test_government_action_space_has_unemployment_relief_dimension() -> None:
    from be.agents.government import Government
    assert GOVERNMENT_ACT_SPACE.shape == (5,), "Government phai co 5 hanh dong (them muc tro cap that nghiep)"
    gov = Government(agent_id="gov_1")
    ok = gov.validate_action(type("A", (), {"values": np.array([0.1, 0.1, 0.5, 0.0, 7.0], dtype=np.float32)})())
    assert ok.is_valid and ok.sanitized_values[4] == 1.0, "muc tro cap phai duoc kep ve <= 1.0"
    bad = gov.validate_action(type("A", (), {"values": np.array([0.1, 0.1, 0.5, 0.0], dtype=np.float32)})())
    assert not bad.is_valid, "vector 4 chieu (cu) phai bi tu choi -- khong am tham chay voi hop dong hanh dong cu"


def test_government_relief_action_scales_subsidies_linearly_and_can_be_zero() -> None:
    """Muc tro cap la CONG CU that cua Government: 0 -> khong chi, 1.0 -> gap 2.5 lan 0.4 (tuyen tinh theo muc,
    cung lich giam dan theo thoi gian that nghiep)."""
    env = MacroEnvironment(num_employees=40, num_firms=4, num_banks=1, max_steps=10)
    env.reset(seed=12)
    for e in env.agents.values():
        if isinstance(e, Employee) and e.agent_id in ("emp_0", "emp_1", "emp_2", "emp_3", "emp_4", "emp_5"):
            e.cash, e.bank_deposit, e.employed_by, e.wage = 0.0, 0.0, None, 0.0
    paid = {}
    for level in (0.0, 0.4, 1.0):
        e2 = copy.deepcopy(env)
        acts = _actions(list(e2.agents.keys()), np.random.default_rng(12))
        acts["gov_1"][4] = level
        np.random.seed(5)
        e2.step(acts)
        paid[level] = e2.gov.last_subsidies_paid
    assert paid[0.0] == 0.0, f"muc 0 van chi tro cap ({paid[0.0]})"
    assert paid[0.4] > 0.0
    assert paid[1.0] == pytest.approx(2.5 * paid[0.4], rel=1e-6), f"khong tuyen tinh theo muc: {paid}"


def test_government_reward_is_equality_times_productivity_and_legacy_is_reproducible() -> None:
    """Reward Government khop thuoc do benchmark (Eq x Prod, Zheng et al. 2022) -- va dang cu tai hien duoc."""
    from be.agents.government import Government
    from be.core.types import TransitionResult

    def reward(mode, real_gdp, gini, deaths=0, debt=0.0, prev_gdp=0.0):
        g = Government(agent_id="gov_1", reward_mode=mode, swf_reward_scale=0.02)
        g.current_real_gdp, g.last_real_gdp, g.current_gini, g.public_debt = real_gdp, prev_gdp, gini, debt
        tr = TransitionResult(agent_id="gov_1", state_delta={"new_deaths": deaths}, events_triggered=[], success=True)
        return g.calculate_reward(tr)

    assert reward("eq_x_prod", 500.0, 0.4) == pytest.approx(0.02 * 0.6 * 500.0)
    assert reward("eq_x_prod", 500.0, 0.4, deaths=2) == pytest.approx(0.02 * 0.6 * 500.0 - 2 * 20.0)
    assert reward("eq_x_prod", 500.0, 0.6) < reward("eq_x_prod", 500.0, 0.4), "Gini cao hon phai giam reward"
    assert reward("eq_x_prod", 600.0, 0.4) > reward("eq_x_prod", 500.0, 0.4), "GDP cao hon phai tang reward"
    # dang CU: 0.002*GDP + 0.005*dGDP - 25*Gini^2
    assert reward("legacy", 500.0, 0.4, prev_gdp=400.0) == pytest.approx(0.002 * 500.0 + 0.005 * 100.0 - 25.0 * 0.16)


def test_initial_treasury_is_operating_balance_and_legacy_endowment_funding_reproducible() -> None:
    """initial_treasury = so du van hanh SAU khi cap von khoi tao (v0.35); treasury_funds_initial_endowments=True
    tai hien logic CU (Kho bac bi tru tien mat khoi tao cua Firm/Employee)."""
    new = MacroEnvironment(num_employees=40, num_firms=4, initial_treasury=25000.0, initial_economy_buffer_fund=10000.0)
    new.reset(seed=1)
    assert new.gov.treasury == pytest.approx(15000.0), f"Kho bac sau khi cap quy binh on phai = 25000 - 10000, nhan {new.gov.treasury}"
    old = MacroEnvironment(num_employees=40, num_firms=4, initial_treasury=1_000_000.0,
                           treasury_funds_initial_endowments=True)
    old.reset(seed=1)
    endow = sum(a.cash for a in old.agents.values() if isinstance(a, (Employee, Firm)))
    assert old.gov.treasury == pytest.approx(1_000_000.0 - 10000.0 - endow), "logic CU (Kho bac tai tro von khoi tao) khong tai hien duoc"


def test_market_demand_factor_is_an_informative_observation_not_a_constant() -> None:
    """LOI THAT (audit 2026-09-26, F4 -- v0.35, KNOWN_PATHOLOGIES.md #29f): Firm obs[3] 'market_demand_factor'
    = clip(V*10, 0.1, 5) luon bang san 0.1 -> dac trung CHET. Nay phai co phuong sai that (khong dinh san)."""
    env = MacroEnvironment(num_employees=40, num_firms=4, num_banks=1, max_steps=80)
    env.reset(seed=9)
    rng = np.random.default_rng(9)
    vals = []
    for _ in range(80):
        env.step(_actions(list(env.agents.keys()), rng))
        vals.append(env.get_raw_environment_state()["macro_indicators"]["market_demand_factor"])
    vals = np.array(vals)
    assert vals.max() - vals.min() > 0.3, f"market_demand_factor gan nhu hang so: [{vals.min():.3f}, {vals.max():.3f}]"
    assert np.mean(vals <= 0.1001) < 0.5, "market_demand_factor dinh san 0.1 o >50% buoc -- van la dac trung chet"


def test_treasury_outflow_counters_track_offbook_spending_channels() -> None:
    """v0.35-fix1 (KNOWN_PATHOLOGIES.md #29c/FUTURE_WORK.md #9): 3 bo dem chan doan phai khop DUNG
    tong tien Kho bac da chi qua tung kenh 'ngoai so' (khong di qua net_budget cua Government)."""
    # Nhanh 1 (luoi an sinh khan cap): ep dan so < hard_min_emp de chac chan kich hoat newborn grant.
    env = MacroEnvironment(num_employees=15, num_firms=2, num_banks=1, max_steps=5, hard_min_emp=30)
    env.reset(seed=1)
    treasury_before = env.gov.treasury
    env.step(_actions(list(env.agents.keys()), np.random.default_rng(1)))
    assert env.treasury_outflow_newborn_episode > 0.0, "luoi an sinh khan cap phai kich hoat newborn outflow"
    assert env.treasury_outflow_firm_entry_episode == 0.0
    assert env.treasury_outflow_bailout_episode == 0.0
    # Bo dem phai khop DUNG voi phan Kho bac giam do KENH NAY (co lap bang cach tat hoan toan cac
    # kenh chi khac trong dung buoc: khong thue/phat/tro cap/G/bom cau -> chi newborn outflow con lai).
    acts2 = _actions(list(env.agents.keys()), np.random.default_rng(2))
    acts2["gov_1"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    treasury_before2 = env.gov.treasury
    env.step(acts2)
    delta_outflow = env.treasury_outflow_newborn_episode  # tich luy tu buoc truoc + buoc nay, nhung buoc truoc da > 0
    # Kiem tra truc tiep hon: dung 1 moi truong MOI, 1 buoc, khong thue/phat/tro cap/G/bom cau.
    env2 = MacroEnvironment(num_employees=15, num_firms=2, num_banks=1, max_steps=5, hard_min_emp=30)
    env2.reset(seed=1)
    tre0 = env2.gov.treasury
    acts3 = _actions(list(env2.agents.keys()), np.random.default_rng(1))
    acts3["gov_1"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    env2.step(acts3)
    tre1 = env2.gov.treasury
    net_known = env2.gov.last_tax_collected + env2.gov.last_fines_collected - env2.gov.last_subsidies_paid \
        - env2.gov.last_purchase - env2.gov.last_demand_injection_value
    unexplained = (tre1 - tre0) - net_known
    assert unexplained == pytest.approx(-env2.treasury_outflow_newborn_episode, abs=1e-6), (
        f"bo dem newborn outflow ({env2.treasury_outflow_newborn_episode:.2f}) khong khop phan Kho bac "
        f"giam khong giai thich duoc ({-unexplained:.2f})"
    )

    # Gia nhap nganh: ep dieu kien loi nhuan + du lao dong de chac chan co firm moi.
    env3 = MacroEnvironment(num_employees=40, num_firms=1, num_banks=1, max_steps=5,
                            firm_entry_probability=1.0, firm_entry_unemployment_threshold=0.0,
                            firm_entry_profitability_margin=1.0, hard_max_firms=10)
    env3.reset(seed=3)
    n_firms_before = sum(1 for a in env3.agents.values() if isinstance(a, Firm))
    for _ in range(5):
        env3.step(_actions(list(env3.agents.keys()), np.random.default_rng(3)))
        n_firms_now = sum(1 for a in env3.agents.values() if isinstance(a, Firm))
        if n_firms_now > n_firms_before:
            break
    assert env3.treasury_outflow_firm_entry_episode > 0.0, "gia nhap nganh phai kich hoat firm-entry outflow"

    # reset() phai dua ca 3 bo dem ve 0.
    env3.reset(seed=3)
    assert env3.treasury_outflow_newborn_episode == 0.0
    assert env3.treasury_outflow_firm_entry_episode == 0.0
    assert env3.treasury_outflow_bailout_episode == 0.0


# ==============================================================================
# v0.36 (KNOWN_PATHOLOGIES.md #30a/#30b): wage=0 luc reset + cong tro cap tuyet doi
# ==============================================================================
def test_initial_employees_have_the_wage_they_are_actually_paid() -> None:
    """#30a: lao dong gan viec luc reset phai co thuoc tinh wage > 0 (dung muc luong mac dinh
    indexed_price*1.05 ma rule_engine thuc tra), va sau 1 buoc khong nguoi co viec nao con wage=0."""
    env = MacroEnvironment(num_employees=50, num_firms=5, num_banks=1, max_steps=10)
    env.reset(seed=4)
    employed = [a for a in env.agents.values() if isinstance(a, Employee) and a.employed_by]
    assert len(employed) > 30
    expected = min(max(1.0, env.eco.base_living_cost), env.subsistence_indexation_ceiling_mult * env.eco.initial_living_cost) * 1.05
    assert all(abs(e.wage - expected) < 1e-9 for e in employed), "lao dong gan viec luc reset khong co dung muc luong mac dinh"
    env.step(_actions(list(env.agents.keys()), np.random.default_rng(4)))
    still_zero = [e for e in env.agents.values() if isinstance(e, Employee) and e.employed_by and e.wage == 0.0]
    assert not still_zero, f"{len(still_zero)} nguoi co viec van co wage=0 sau 1 buoc"


def _relief_env(rule: str, treasury: float):
    env = MacroEnvironment(num_employees=40, num_firms=4, num_banks=1, max_steps=10, subsidy_funding_rule=rule)
    env.reset(seed=12)
    n = 0
    for e in env.agents.values():
        if isinstance(e, Employee) and n < 8:
            if e.employed_by and e.employed_by in env.agents:
                f = env.agents[e.employed_by]
                if e.agent_id in f.employee_ids:
                    f.employee_ids.remove(e.agent_id)
            e.cash, e.bank_deposit, e.employed_by, e.wage = 0.0, 0.0, None, 0.0
            n += 1
    env.gov.treasury = treasury
    acts = _actions(list(env.agents.keys()), np.random.default_rng(12))
    acts["gov_1"][2] = 0.0   # rho = 0: khong chi mua hang G
    acts["gov_1"][3] = 0.0   # khong bom cau
    acts["gov_1"][4] = 1.0   # muc tro cap toi da
    for aid in acts:
        if aid.startswith("firm_"):
            acts[aid][0] = -0.2  # khong tuyen lai nhom that nghiep trong buoc nay
    np.random.seed(3)
    env.step(acts)
    return env


def test_unemployment_relief_is_paid_within_treasury_budget() -> None:
    """#30b: tro cap KHONG con bi khoa boi nguong tuyet doi Kho bac > 1000 (lam action[4] vo hieu khi Kho bac
    nho), nhung cung KHONG duoc chi vuot so du con lai (rang buoc ngan sach, giong G). Logic CU tai hien duoc."""
    big = _relief_env("affordable", 50_000.0)
    requested = big.gov.last_subsidies_paid
    assert requested > 0.0, "Kho bac lon + nguoi du dieu kien + relief=1 nhung khong chi tro cap"
    small = _relief_env("affordable", 5.0)
    assert 0.0 < small.gov.last_subsidies_paid <= 5.0 + 1e-9, (
        f"Kho bac 5.0: tro cap phai > 0 va <= so du ({small.gov.last_subsidies_paid:.3f})")
    mid = _relief_env("affordable", 600.0)
    assert mid.gov.last_subsidies_paid == pytest.approx(big.gov.last_subsidies_paid, rel=1e-9), (
        "Kho bac 600 (du chi) phai chi DU muc yeu cau, khong bi nguong 1000 cu chan")
    legacy = _relief_env("legacy_gate", 600.0)
    assert legacy.gov.last_subsidies_paid == 0.0, "logic CU (Kho bac 600 < 1000 -> khong chi) khong con tai hien duoc"



# ==============================================================================
# v0.37 (KNOWN_PATHOLOGIES.md #31): du tru dem cua Economy -- quy mo theo thi truong hien tai + dai gia dong
# ==============================================================================
def _buffer_env(rule: str, ref_mult: float):
    env = MacroEnvironment(num_employees=40, num_firms=5, num_banks=1, max_steps=20, buffer_stock_rule=rule)
    env.reset(seed=11)
    rng = np.random.default_rng(11)
    for _ in range(5):
        env.step(_actions(list(env.agents.keys()), rng))
    env.eco.strategic_reserve_fund, env.eco.strategic_reserve_stock = 20000.0, 500.0
    env.eco.buffer_reference_price = env.eco.base_living_cost * ref_mult
    return env


def test_buffer_band_blocks_procyclical_intervention() -> None:
    """(b) Chi MUA khi gia < tham chieu x (1-band), chi BAN khi gia > tham chieu x (1+band) -- bat ke policy chon gi.
    Su co #31: Economy MUA ca khi gia da gap 3 -> phai bi chan."""
    buy_blocked = _buffer_env("band_scaled", 0.5)   # gia cao hon tham chieu nhieu -> KHONG duoc mua
    acts = _actions(list(buy_blocked.agents.keys()), np.random.default_rng(3)); acts["eco_1"][0] = 1.0
    buy_blocked.step(acts)
    assert buy_blocked.eco.strategic_reserve_fund == 20000.0, "dai gia khong chan lenh MUA khi gia tren tham chieu"
    sell_blocked = _buffer_env("band_scaled", 2.0)  # gia thap hon tham chieu nhieu -> KHONG duoc ban
    acts = _actions(list(sell_blocked.agents.keys()), np.random.default_rng(3)); acts["eco_1"][0] = -1.0
    sell_blocked.step(acts)
    assert sell_blocked.eco.strategic_reserve_stock == 500.0, "dai gia khong chan lenh BAN khi gia duoi tham chieu"
    legacy = _buffer_env("legacy_fixed_anchor", 0.5)  # logic CU: khong co dai gia -> van mua
    acts = _actions(list(legacy.agents.keys()), np.random.default_rng(3)); acts["eco_1"][0] = 1.0
    legacy.step(acts)
    assert legacy.eco.strategic_reserve_fund < 20000.0, "logic CU (khong dai gia) khong con tai hien duoc"


def test_buffer_buy_is_bounded_by_current_household_market() -> None:
    """(a) Luong mua <= kappa x chi tieu ho gia dinh cua chinh buoc do (khong neo hang so co dinh co the troi theo muc
    gia); logic CU mua theo initial_living_cost x dan so -> lon hon nhieu khi gia da giam."""
    new = _buffer_env("band_scaled", 2.0)
    new.eco.base_living_cost = 1.2  # tai hien boi canh giam phat cua su co
    acts = _actions(list(new.agents.keys()), np.random.default_rng(4)); acts["eco_1"][0] = 1.0
    new.step(acts)
    spent_new = 20000.0 - new.eco.strategic_reserve_fund
    household = new.eco.step_trade_volume  # chi tieu ho gia dinh (total_market_turnover) cua buoc vua chay
    assert 0.0 < spent_new <= 0.15 * household + 1e-6, f"mua {spent_new:.2f} vuot kappa x chi tieu ho gia dinh {household:.2f}"
    # Logic CU: luong mua = cuong do x initial_living_cost x dan so -- CO DINH, KHONG phu thuoc quy mo thi truong hien tai
    # (chinh la ly do no tro nen qua lon khi giam phat). Tai hien: thu voi 2 quy mo chi tieu ho gia dinh rat khac nhau.
    spends_old = []
    for cash_scale in (1.0, 0.05):
        old = _buffer_env("legacy_fixed_anchor", 2.0)
        old.eco.base_living_cost = 1.2
        for e in old.agents.values():
            if isinstance(e, Employee):
                e.cash *= cash_scale; e.bank_deposit *= cash_scale
        n_before = sum(1 for e in old.agents.values() if isinstance(e, Employee))
        old.step({k: v.copy() for k, v in acts.items()})
        spends_old.append((20000.0 - old.eco.strategic_reserve_fund, n_before))
    for spent_old, n_before in spends_old:
        assert spent_old == pytest.approx(1.0 * old.eco.initial_living_cost * n_before, rel=0.05), (
            f"logic CU phai mua dung initial_living_cost x dan so ({spent_old:.1f} vs {old.eco.initial_living_cost * n_before:.1f})")


def test_late_buffer_buy_after_deflation_no_longer_triples_prices() -> None:
    """Hoi quy #31 (tai hien dung dieu kien su co bang hanh dong ep, khong can checkpoint): nen kinh te giam phat
    (effort 0.72), Economy KHONG can thiep toi buoc 140 roi MUA toi da. Logic CU: gia tang ~2.9x trong 10 buoc (do
    2026-09-27); logic MOI phai giu dot bien 10 buoc <= 1.5x."""
    from be.scenario_config import ScenarioConfig
    kw0 = ScenarioConfig.from_yaml(os.path.join(os.path.dirname(__file__), "..", "..", "scenarios", "em_baseline.yaml")).to_env_kwargs()

    def max_jump(rule):
        kw = dict(kw0); kw.update(buffer_stock_rule=rule)
        env = MacroEnvironment(**kw); env.reset(seed=21); rng = np.random.default_rng(21)
        P = []
        for t in range(200):
            acts = _actions(list(env.agents.keys()), rng)
            for aid in acts:
                if aid.startswith("emp_"):
                    acts[aid][0] = 0.72
            acts["eco_1"][0] = 0.0 if t < 140 else 1.0
            env.step(acts)
            P.append(env.eco.base_living_cost)
        return max(P[t + 10] / P[t] for t in range(130, len(P) - 10))

    j_new = max_jump("band_scaled")
    assert j_new <= 1.5, f"logic MOI: gia van tang {j_new:.2f}x trong 10 buoc sau khi Economy mua muon"
    j_old = max_jump("legacy_fixed_anchor")
    assert j_old > 2.0, f"logic CU khong con tai hien su co (dot bien {j_old:.2f}x) -- kiem tra lai dieu kien test"
