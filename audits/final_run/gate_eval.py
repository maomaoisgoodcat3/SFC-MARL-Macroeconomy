"""CONG KIEM TRA DANG KY TRUOC (official_baseline_v1_report.md, muc "QUY TRINH DANG KY TRUOC cho seed 1").
Viet TRUOC khi co ket qua -- nguong co dinh o day, khong sua sau khi thay so.
Dung: python audits/final_run/gate_eval.py <checkpoint_path> [--c3-mode original|split] [--json-out f.json]

Ban goc (dung cho seed 1, 2026-09-27/28) nam o scratchpad phien; ban nay CHI THEM (2026-09-28, sau quyet dinh (A) cho #32):
  - phan loai tu vong: "chua tung co viec trong episode" (co che #32) vs "con lai"; LUON in ca hai, khong doi nguong nao;
  - --c3-mode split (DANG KY TRUOC cho seed 2/3, nguoi dung duyet 2026-09-28 theo ket luan Claude Web, xem
    official_baseline_v1_report.md): nhom #32 xac dinh theo CO CHE, khong theo so dem -- chet (khong phai tuoi gia) ma
    (i) chua tung co viec VA (ii) o MOI buoc trong doi, KHONG firm nao thoa dieu kien tuyen cua rule_engine Section 2:
        c * 0.6 * A_f * K_f^0.3 * max(0.5, L_f + s)^(-0.4) * s  >=  max(1.02, 0.8 + 0.3 s)
    (MRPL >= luong bao luu; gia indexed_price TRIET TIEU hai ve nen dieu kien doc lap voi muc gia). L_f lay LAC QUAN
    (chi nguoi CON O LAI firm sau buoc: tru nguoi roi di trong buoc, bo qua nguoi moi tuyen cung buoc; bo qua rang buoc
    thanh khoan/hire_signal/tran tuyen) -> vi tu "tuyen duoc" bi danh gia CAO, nhom #32 bi danh gia THAP (than trong).
    TU KIEM: moi ca tuyen THAT phai thoa vi tu; co vi pham -> bo phan loai sai -> che do split tu FAIL.
    Nhom "con lai" (ke ca nguoi chua co viec nhung TUNG tuyen duoc) ap nguong goc (TB tat dinh <= 0.5, trung vi lay mau 0).
    Nhom #32: moi episode (ca 13 episode cong) <= --c3-structural-cap (mac dinh 6). NGUONG 6 KHONG suy ra doc lap bang
    ly thuyet: = 2 x so ca lon nhat trong 1 episode quan sat o seed 1 (3) -- bien an toan theo phuong an du phong ma
    Claude Web chap nhan; phai ghi dung nhu vay trong khoa luan. Ly do khong suy ra doc lap duoc: ca so nguoi o vung ky
    nang khong tuyen duoc lan xac suat ho khong duoc tuyen deu noi sinh theo policy da hoc (so firm, so ca sinh, di
    truyen theo anh chi em); uoc tinh tu phan phoi ky nang ban dau (~1% duoi 0.7) bo qua cum anh chi em nen khong dung duoc.
  - dong may doc "GATE_RESULT: PASS|FAIL" + file JSON cho script dieu phoi run_seed_with_gate.py.
Mac dinh --c3-mode original = HANH VI Y HET ban goc.
"""
import os, sys, json, argparse, hashlib, subprocess
from collections import Counter
import numpy as np
REPO = r"C:\Users\piece\Downloads\TheAIEconomist\ai_economist_gpt"
sys.path.insert(0, REPO); os.chdir(REPO)
from be.env import MacroEnvironment
from be.scenario_config import ScenarioConfig
from be.core.enums import EventType
from be.agents.employee import Employee
from be.agents.firm import Firm
from be.rllib_wrapper import build_ppo_config

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--config", default="scenarios/em_baseline.yaml")
ap.add_argument("--c3-mode", choices=["original", "split"], default="original")
ap.add_argument("--c3-structural-cap", type=float, default=6.0)
ap.add_argument("--json-out", default=None)
ap.add_argument("--quick", action="store_true", help="CHI thu pipeline: 1+1 episode x 30 buoc -- KHONG dung cho ket luan")
ARGS = ap.parse_args()
CKPT = ARGS.checkpoint
WINDOWS = [(0, 30), (30, 60), (60, 120), (120, 180), (180, 240)]


def provenance():
    """Phien ban chinh xac cua cong cu + mo hinh da dung (yeu cau Claude Web 2026-09-28: truy nguoc duoc so lieu)."""
    def sh(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO).stdout.strip()
        except Exception as exc:
            return f"ERR {exc}"
    files = ["audits/final_run/gate_eval.py", "be/env.py", "be/rule_engine.py", "be/rllib_wrapper.py", "scenarios/em_baseline.yaml"]
    return dict(git_head=sh("rev-parse", "HEAD"),
                git_dirty=sh("status", "--porcelain", "--", "be", "audits", "scenarios").splitlines(),
                sha256={f: hashlib.sha256(open(os.path.join(REPO, f), "rb").read()).hexdigest()[:16] for f in files})


def pm(aid):
    if aid.startswith("emp_"): return "policy_employee"
    if aid.startswith("firm_"): return "policy_firm"
    if aid.startswith("bank_"): return "policy_bank"
    if aid == "gov_1": return "policy_government"
    if aid == "sup_1": return "policy_supervisor"
    return "policy_economy"


def tsv_net(env):  # dang thuc SFC CO tru no cong (v0.36-fix1)
    emp = [a for a in env.agents.values() if isinstance(a, Employee)]
    firm = [a for a in env.agents.values() if isinstance(a, Firm)]
    return (env.gov.treasury - env.gov.public_debt + sum(b.reserves for b in env.banks) + sum(a.cash for a in firm)
            + sum(a.cash for a in emp) + sum(a.bank_deposit for a in emp) + env.eco.strategic_reserve_fund)


# --- Dieu kien 7 (bo sung 2026-09-27 theo Claude Web, TRUOC khi co ket qua iter_40): tro cap KHONG bi chan khi co
# nguoi du dieu kien. Boc RuleEngine.execute_cycle (CHI trong script do, khong sua code mo hinh) de ghi moi buoc:
# Kho bac dau buoc, G, bom cau, tro cap YEU CAU, tro cap THUC CHI. Dung: paid == min(requested, max(0, Kho bac - G - max(0, bom cau))).
from be.rule_engine import RuleEngine
RELIEF_LOG = []
_orig_cycle = RuleEngine.execute_cycle
def _logged_cycle(self, agents, validated_actions, timestep, max_steps=999999):
    tre0 = agents["gov_1"].treasury if "gov_1" in agents else float("nan")
    res = _orig_cycle(self, agents, validated_actions, timestep, max_steps)
    d = res["gov_1"].state_delta if "gov_1" in res else {}
    RELIEF_LOG.append(dict(treasury=tre0, G=float(d.get("government_purchase_cost", 0.0)),
                           inj=float(d.get("demand_injection_cost", 0.0)),
                           req=float(d.get("subsidies_requested", 0.0)), paid=float(d.get("subsidies_disbursed", 0.0))))
    return res
RuleEngine.execute_cycle = _logged_cycle

kw = ScenarioConfig.from_yaml(ARGS.config).to_env_kwargs()
algo = build_ppo_config(kw, num_env_runners=0, seed=42).build()
algo.restore(CKPT)
print("checkpoint:", CKPT)

runs = [(False, s) for s in (999, 1001, 1002)] + [(True, s) for s in range(2001, 2011)]
T_STEPS = 240
if ARGS.quick:
    runs, T_STEPS = [(False, 999), (True, 2001)], 30
    print("!!! CHE DO --quick: chi thu pipeline, KET QUA KHONG DUNG CHO KET LUAN !!!")
R = []
gov_obs = []
HIRE_CHECK = [0, 0]  # [so ca tuyen that da kiem, so ca vi pham vi tu "tuyen duoc"] -- phai = 0 vi pham
for explore, seed in runs:
    env = MacroEnvironment(**kw); env.reset(seed=seed); np.random.seed(seed)
    reasons = Counter()
    env.event_bus.subscribe(EventType.AGENT_DIED, lambda ev: reasons.update([ev.payload.get("reason")]))
    # Phan loai #32: ai CHUA TUNG co viec (quan sat dau moi buoc, tu luc xuat hien) den luc chet; tach rieng tu vong tuoi gia.
    ever_employed, ever_hireable, died = set(), set(), []
    C_MRPL = env.rule_engine.mrpl_scale_constant
    env.event_bus.subscribe(EventType.AGENT_DIED, lambda ev: died.append((ev.source_id, str(ev.payload.get("reason")))))
    efforts, w0_share = [], np.zeros(240)  # giu 240 de cac cua so WINDOWS khong rong o che do --quick
    max_sfc, prev = 0.0, tsv_net(env)
    for t in range(T_STEPS):
        raw = env.get_raw_environment_state(); acts = {}
        for aid in env.agents:
            o = env.observe_agent(aid, raw)
            acts[aid] = np.asarray(algo.compute_single_action(o, policy_id=pm(aid), explore=explore), dtype=np.float32)
            if aid == "gov_1" and not explore:
                gov_obs.append(o)
        employed = [e for e in env.agents.values() if isinstance(e, Employee) and e.employed_by is not None]
        ever_employed.update(e.agent_id for e in employed)
        efforts += [float(acts[e.agent_id][0]) for e in employed]
        w0_share[t] = np.mean([e.wage == 0.0 for e in employed]) if employed else 0.0
        firms_pre = {f.agent_id: (f.productivity_factor, f.capital_stock) for f in env.agents.values() if isinstance(f, Firm)}
        emp_pre = {e.agent_id: (e.employed_by, e.skill_level) for e in env.agents.values() if isinstance(e, Employee)}
        env.step(acts)
        emp_post = {e.agent_id: e.employed_by for e in env.agents.values() if isinstance(e, Employee)}
        L_opt = {f: 0.0 for f in firms_pre}
        for aid_, (fb, sk) in emp_pre.items():
            if fb in L_opt and emp_post.get(aid_) == fb:
                L_opt[fb] += sk  # chi tinh nguoi CON O LAI firm sau buoc (nguoi roi di trong buoc bi tru -> lac quan)

        def pred(sk, f):
            A, K = firms_pre[f]
            return C_MRPL * 0.6 * A * (K ** 0.3) * (max(0.5, L_opt[f] + sk) ** (-0.4)) * sk >= max(1.02, 0.8 + 0.3 * sk)
        for aid_, (fb, sk) in emp_pre.items():
            if fb is not None:
                continue
            post = emp_post.get(aid_)
            if post is not None and post in firms_pre:
                HIRE_CHECK[0] += 1
                if not pred(sk, post):
                    HIRE_CHECK[1] += 1
            if aid_ not in ever_hireable and any(pred(sk, f) for f in firms_pre):
                ever_hireable.add(aid_)
        new = tsv_net(env); max_sfc = max(max_sfc, abs((new - prev) + env.eco.last_capital_depreciation)); prev = new
    non_old = [d for d in died if d[1] != "Tuổi già"]  # tu vong tu nhien khong thuoc ca 2 nhom
    d_never = sum(1 for d in non_old if d[0] not in ever_employed)
    d_struct = sum(1 for d in non_old if d[0] not in ever_employed and d[0] not in ever_hireable)
    d_other = len(non_old) - d_struct
    R.append(dict(explore=explore, seed=seed, effort=np.array(efforts), w0=w0_share, deaths=env.gov.dead_citizens_count,
                  d_never=d_never, d_struct=d_struct, d_other=d_other, d_old=len(died) - len(non_old),
                  reasons=dict(reasons), births=env.births_episode / float(T_STEPS), emerg=env.emergency_births_episode, sfc=max_sfc))

det = [r for r in R if not r["explore"]]; exp = [r for r in R if r["explore"]]
eff_all = np.concatenate([r["effort"] for r in R])
hist = np.histogram(eff_all, bins=10, range=(0, 1))[0]
frac_low = float(np.mean(eff_all < 0.5))
w0_by_win = [max(float(r["w0"][lo:hi].mean()) for r in R) for lo, hi in WINDOWS]
d_det = [r["deaths"] for r in det]; d_exp = [r["deaths"] for r in exp]
all_reasons = Counter(); [all_reasons.update(r["reasons"]) for r in R]
births = [r["births"] for r in R]; emerg = [r["emerg"] for r in R]
sfc = max(r["sfc"] for r in R)

O = np.array(gov_obs); med = np.median(O, 0); S = np.zeros((O.shape[1], 5))
for d in range(O.shape[1]):
    lo, hi = med.copy(), med.copy()
    lo[d], hi[d] = np.percentile(O[:, d], 5), np.percentile(O[:, d], 95)
    S[d] = np.abs(np.asarray(algo.compute_single_action(hi, policy_id="policy_government", explore=False))
                  - np.asarray(algo.compute_single_action(lo, policy_id="policy_government", explore=False)))
sens = S.max(axis=0); names = ["wtax", "ftax", "rho", "inj", "relief"]
n_sens = int(np.sum(sens > 0.05))

c1 = eff_all.mean() >= 0.5 and frac_low <= 0.15
c2 = all(v == 0.0 for v in w0_by_win)
c3_original = (np.mean(d_det) <= 0.5) and (np.median(d_exp) == 0)
dn_det = [r["d_never"] for r in det]; dn_exp = [r["d_never"] for r in exp]
ds_det = [r["d_struct"] for r in det]; ds_exp = [r["d_struct"] for r in exp]
do_det = [r["d_other"] for r in det]; do_exp = [r["d_other"] for r in exp]
classifier_ok = HIRE_CHECK[0] > 0 and HIRE_CHECK[1] == 0
c3_split = (classifier_ok and (np.mean(do_det) <= 0.5) and (np.median(do_exp) == 0)
            and max(ds_det + ds_exp) <= ARGS.c3_structural_cap)
c3 = c3_original if ARGS.c3_mode == "original" else c3_split
c4 = n_sens >= 2
c5 = sfc <= 1e-3
c6 = max(births) <= 0.35 and np.mean(emerg) <= 0.5
print("\n=== CONG KIEM TRA (6 dieu kien dang ky truoc) ===")
print(f"1 effort: TB={eff_all.mean():.3f}, ty le <0.5={frac_low*100:.1f}% (nguong <=15%), histogram 10 bin={hist.tolist()} -> {'DAT' if c1 else 'KHONG DAT'}")
print(f"2 wage=0 theo cua so {WINDOWS}: max qua cac episode = {[round(v*100,1) for v in w0_by_win]} % -> {'DAT' if c2 else 'KHONG DAT'}")
print(f"3 tu vong: tat dinh {d_det}, lay mau {d_exp} (trung vi {np.median(d_exp)}), ly do {dict(all_reasons)}")
print(f"  tu kiem bo phan loai: {HIRE_CHECK[0]} ca tuyen that, {HIRE_CHECK[1]} ca vi pham vi tu tuyen duoc -> "
      f"{'HOP LE' if classifier_ok else 'KHONG HOP LE (split tu FAIL)'}")
print(f"  phan loai (tach tuoi gia): chua tung co viec tat dinh {dn_det} / lay mau {dn_exp}; trong do #32 CO CHE (khong bao gio "
      f"tuyen duoc) tat dinh {ds_det} / lay mau {ds_exp}; CON LAI tat dinh {do_det} / lay mau {do_exp}; tuoi gia {[r['d_old'] for r in R]}")
print(f"  che do original -> {'DAT' if c3_original else 'KHONG DAT'}; che do split (#32 <= {ARGS.c3_structural_cap}/episode) -> "
      f"{'DAT' if c3_split else 'KHONG DAT'}; DUNG CHO KET LUAN: {ARGS.c3_mode} -> {'DAT' if c3 else 'KHONG DAT'}")
print(f"4 do nhay Government max|d action| p5->p95: {dict(zip(names, np.round(sens,3)))}; so cong cu > 0.05: {n_sens} -> {'DAT' if c4 else 'KHONG DAT'}")
print(f"5 SFC (co tru no cong): lech max={sfc:.2e} -> {'DAT' if c5 else 'KHONG DAT'}")
print(f"6 births/buoc max={max(births):.3f}, sinh khan cap TB={np.mean(emerg):.2f} -> {'DAT' if c6 else 'KHONG DAT'}")
elig = [r for r in RELIEF_LOG if r["req"] > 1e-12]
viol = [r for r in elig if abs(r["paid"] - min(r["req"], max(0.0, r["treasury"] - r["G"] - max(0.0, r["inj"])))) > 1e-6 * max(1.0, r["req"])]
blocked_with_money = [r for r in elig if r["paid"] <= 1e-12 and (r["treasury"] - r["G"] - max(0.0, r["inj"])) > 1e-9]
if not elig:
    c7_txt, c7 = "KHONG KIEM DUOC TREN QUY DAO (0 buoc co nguoi du dieu kien -- dua vao test don vi)", True
else:
    c7 = (not viol) and (not blocked_with_money)
    c7_txt = "DAT" if c7 else "KHONG DAT"
print(f"7 tro cap khong bi chan: {len(elig)} buoc co nguoi du dieu kien / {len(RELIEF_LOG)} buoc; vi pham min(yeu cau, Kho bac con lai)={len(viol)}; "
      f"bi chan du con tien={len(blocked_with_money)}; tong yeu cau={sum(r['req'] for r in elig):.1f}, tong thuc chi={sum(r['paid'] for r in elig):.1f} -> {c7_txt}")
ok = all([c1, c2, c3, c4, c5, c6, c7])
print(f"\nKET LUAN CONG: {'DAT CA 7' if ok else 'KHONG DAT -- DUNG seed theo quy trinh'}")
print(f"GATE_RESULT: {'PASS' if ok else 'FAIL'}", flush=True)
if ARGS.json_out:
    with open(ARGS.json_out, "w", encoding="utf-8") as f:
        json.dump(dict(checkpoint=CKPT, c3_mode=ARGS.c3_mode, passed=bool(ok),
                       conditions=dict(effort=bool(c1), wage0=bool(c2), deaths=bool(c3), deaths_original=bool(c3_original),
                                       deaths_split=bool(c3_split), gov_sensitivity=bool(c4), sfc=bool(c5), births=bool(c6),
                                       relief=bool(c7)),
                       effort_mean=float(eff_all.mean()), effort_low_share=frac_low, deaths_det=d_det, deaths_exp=d_exp,
                       deaths_never_employed_det=dn_det, deaths_structural_det=ds_det, deaths_structural_exp=ds_exp,
                       deaths_other_det=do_det, deaths_other_exp=do_exp, classifier_hire_check=HIRE_CHECK,
                       c3_structural_cap=ARGS.c3_structural_cap, provenance=provenance(), gov_sens=dict(zip(names, map(float, sens))),
                       sfc_max=float(sfc), births_max=float(max(births))), f, ensure_ascii=False, indent=1)
