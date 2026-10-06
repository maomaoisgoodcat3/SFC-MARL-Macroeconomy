"""
bench-v2 (A7.4) — dang ky truoc: audits/paper_extras/PREREG_A74_bench_v2.yaml (commit 6a95505, nguoi dung duyet 2026-10-07).

Thiet ke: KHONG sua be/benchmark.py (bench-v1, tag -> 371799e). File nay goi lai NGUYEN VAN be.benchmark.run_episode (cung logic
moi nhanh, cung thu tu nhanh, cung np.random.seed(seed) dau moi nhanh, cung reset(seed=seed+i)) va chi them:
  (a) ghep cap nhieu lay mau hanh dong giua cac nhanh (KNOWN_PATHOLOGIES #36) — SUA DOI 2026-10-07 (PREREG amendment, commit
      452fb00): truoc MOI lan suy luan cua MOI policy o MOI buoc (che do lay mau), torch.manual_seed(K(seed, episode, buoc,
      policy, stt)) — xem TorchPairing. Che do tat dinh: khong gieo gi. (--torch-seeding episode = cach cu, chi de doi chung);
  (b) bo ghi so do sau MOI env.step (boc env.step, khong doi hanh vi): G_P cua Raffinetti, Siletti & Vernizzi (2015) tren
      tai san / tai san rong / tieu dung + chan doan tai san am. Thuoc do CHINH van la env.gov.current_gini (Gini cu cua mo hinh).
=> o che do tat dinh, cac cot cua v1 (gdp, gini, unemp, deaths, ...) phai TRUNG v1 tung so (kiem khi chay that).

G_P (Muc 3.2, Eq. 5 va dinh nghia ngay sau; khop Bang 3 cua bai: 0.5556 / 1 / 0.8346):
    G_P = sum_i sum_j |Y_i - Y_j| / [2 (N-1) (T+ + T-)],  T+ = sum max(0, Y_i),  T- = |sum min(0, Y_i)|.
Dang N HUU HAN cua bai, KHONG phai GiniWegNeg::Gini_RSV (goi R chia cho (T+ + T-)/N = dang tiem can).

Dung (KHONG chay song song voi huan luyen — dieu kien 8):
  python audits/paper_extras/benchmark_v2.py --config scenarios/em_baseline.yaml \
      --checkpoint be/checkpoint/final_v037_seed42/iter_100 --episodes 10 --max-steps 240 --batched-inference [--explore] \
      [--arms ...] --json-out <file>
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)  # giong audits/final_run/gate_eval.py: duong dan tuong doi tinh tu goc repo

from be import benchmark as v1  # noqa: E402
from be.agents.employee import Employee  # noqa: E402
from be.core.enums import LifeCycleStatus  # noqa: E402
from be.env import MacroEnvironment  # noqa: E402
from be.rule_engine import RuleEngine  # noqa: E402
from be.scenario_config import ScenarioConfig  # noqa: E402

TOOL_VERSION = "bench-v2"
PREREG = "audits/paper_extras/PREREG_A74_bench_v2.yaml"
GINI_KEYS = ("gini_model_recomputed", "gp_wealth", "gp_net_wealth", "gp_consumption")
# Ham Gini GOC cua mo hinh (giu tham chieu truoc khi StepRecorder cai bo bat mang tai san, xem StepRecorder.__init__).
_ORIG_GINI = RuleEngine._compute_gini_with_negatives


# --------------------------------------------------------------------------------------------- thuoc do
def gini_p(values) -> float:
    """G_P cua Raffinetti, Siletti & Vernizzi (2015), Muc 3.2 Eq. (5), dang N huu han. ∈ [0, 1].
    sum_i sum_j |Y_i - Y_j| = 2 * sum_i (2i - N - 1) * Y_(i) (Y sap tang dan, i = 1..N) -> O(N log N)."""
    y = np.sort(np.asarray(values, dtype=np.float64))
    n = y.size
    if n < 2:
        return 0.0
    t_plus = float(y[y > 0].sum())
    t_minus = float(-y[y < 0].sum())
    if t_plus + t_minus <= 0.0:
        return 0.0
    i = np.arange(1, n + 1, dtype=np.float64)
    ss = 2.0 * float(np.sum((2.0 * i - n - 1.0) * y))
    return ss / (2.0 * (n - 1) * (t_plus + t_minus))


def gini_model(values) -> float:
    """Gini cu cua mo hinh (chuan hoa theo N, tinh tien khi co am) — goi THANG ham dong bang, khong chep lai."""
    return _ORIG_GINI(list(values))


def self_test():
    """3 test dang ky truoc (PREREG_A74_bench_v2.yaml: tests). Chay truoc moi lan dung v2; sai -> dung chuong trinh."""
    table3 = {"a": ([-5.0] * 9 + [45.01], 0.5556), "b": ([-45.0] + [0.0] * 8 + [45.01], 1.0),
              "c": ([-15.0, -10.0, -8.0, -7.0, -5.0, 0.0, 0.0, 0.0, 0.0, 45.01], 0.8346)}
    for name, (y, want) in table3.items():
        got = gini_p(y)
        assert abs(got - want) < 1e-4, f"Bang 3 kich ban ({name}): G_P = {got:.6f}, bai = {want}"
    rng = np.random.default_rng(12345)
    for n in (2, 10, 50, 97):
        x = rng.lognormal(5.0, 0.8, n)
        assert abs(gini_p(x) - gini_model(x) * n / (n - 1)) < 1e-9, f"khong am, N={n}: G_P != Gini_mo_hinh x N/(N-1)"
    assert gini_p([0.0] * 7) == 0.0 and gini_p([3.0]) == 0.0 and gini_p([]) == 0.0
    for _ in range(200):
        y = rng.normal(0.0, 100.0, int(rng.integers(2, 80)))
        g = gini_p(y)
        assert -1e-12 <= g <= 1.0 + 1e-12, f"G_P ngoai [0,1]: {g}"
    return True


# --------------------------------------------------------------------------------------------- ghep cap nhieu torch
class TorchPairing:
    """Boc algo.compute_actions / algo.compute_single_action (v1 goi ca hai bang tu khoa policy_id=, explore=).
    Khi explore=True: torch.manual_seed(K) voi K = 15 hex dau sha256("seed|episode|buoc|policy_id|stt"), stt = so lan policy
    do da duoc goi trong buoc (0 voi suy luan theo lo) -> nhieu cua moi policy o moi buoc giong nhau giua cac nhanh, bat ke nhanh
    co lay mau Government hay khong. Khi explore=False: khong lam gi (che do tat dinh trung v1 tung bit).
    Kiem them: dem so lan goi lam doi trang thai np.random (RNG cua moi truong) — phai = 0, neu khac thi gieo/lay mau cua
    RLlib dang cham vao dong luc moi truong."""

    def __init__(self, algo, seed, mode="step_policy"):
        self.seed, self.mode = int(seed), mode
        self.ep, self.rec = 0, None
        self._counts, self._count_key = {}, None
        self.seeded_calls = 0
        self.np_state_changed_calls = 0
        self.batch_log = None  # dat = {} de ghi (episode, buoc, policy, stt) -> thu tu id tac tu trong lo (chan doan ghep cap)
        algo.compute_actions = self._wrap(algo.compute_actions)
        algo.compute_single_action = self._wrap(algo.compute_single_action)

    def key(self, policy_id):
        t = self.rec.t if self.rec is not None else -1
        if self._count_key != (self.ep, t):
            self._counts, self._count_key = {}, (self.ep, t)
        n = self._counts.get(policy_id, 0)
        self._counts[policy_id] = n + 1
        return int(hashlib.sha256(f"{self.seed}|{self.ep}|{t}|{policy_id}|{n}".encode()).hexdigest()[:15], 16)

    def _wrap(self, fn):
        def wrapped(*a, **kw):
            if not kw.get("explore"):
                return fn(*a, **kw)
            if self.mode == "step_policy":
                pid = kw.get("policy_id")
                if pid is None:
                    raise RuntimeError("TorchPairing: thieu policy_id (v1 luon truyen bang tu khoa)")
                k = self.key(pid)
                torch.manual_seed(k)
                self.seeded_calls += 1
                if self.batch_log is not None and a and isinstance(a[0], dict):
                    self.batch_log[(self.ep, self.rec.t if self.rec is not None else -1, pid,
                                    self._counts[pid] - 1)] = list(a[0].keys())
            st = np.random.get_state()
            out = fn(*a, **kw)
            st2 = np.random.get_state()
            if st[2] != st2[2] or not np.array_equal(st[1], st2[1]):
                self.np_state_changed_calls += 1
            return out
        return wrapped


# --------------------------------------------------------------------------------------------- bo ghi theo buoc
class StepRecorder:
    """Boc env.step (khong doi hanh vi env).
    - TAI SAN (thuoc do chinh + gp_wealth + chan doan am): lay DUNG mang ma rule_engine dua vao Gini cua mo hinh
      (rule_engine.py:1710-1716), bat bang cach boc RuleEngine._compute_gini_with_negatives — ham boc tra NGUYEN gia tri goc,
      chi chep lai mang dau vao -> gp_wealth va Gini chinh tinh tren CUNG mot anh chup (smoke 2026-10-07: tinh lai sau buoc
      lech ~0.002 vi trang thai sau buoc khac thoi diem mo hinh tinh Gini).
    - TAI SAN RONG va TIEU DUNG: doc tu Employee sau buoc (ACTIVE/INITIALIZED, co mat ca truoc va sau buoc — giong tap
      active_employees cua rule_engine.py:254 tru nguoi chet; tre sinh trong buoc KHONG tinh)."""

    LIVE = (LifeCycleStatus.ACTIVE, LifeCycleStatus.INITIALIZED)

    def __init__(self, env):
        self.env = env
        self._orig_step = env.step
        env.step = self._step
        self.t = 0
        self.rows = []
        self._captured = []
        rec = self

        def _capture(wealth_array):
            rec._captured.append(np.asarray(wealth_array, dtype=np.float64).copy())
            return _ORIG_GINI(wealth_array)
        RuleEngine._compute_gini_with_negatives = staticmethod(_capture)

    def new_episode(self):
        self.t = 0
        self.rows = []

    def _live(self):
        return {aid: a for aid, a in self.env.agents.items() if isinstance(a, Employee) and a.status in self.LIVE}

    def _step(self, actions):
        before = set(self._live())
        self._captured.clear()
        out = self._orig_step(actions)
        self.t += 1
        emps = [a for aid, a in self._live().items() if aid in before]
        post_wealth = np.array([e.cash + getattr(e, "bank_deposit", 0.0) for e in emps], dtype=np.float64)
        net = post_wealth - np.array([getattr(e, "debt", 0.0) for e in emps], dtype=np.float64)
        engine_ok = len(self._captured) == 1
        wealth = self._captured[-1] if engine_ok else post_wealth
        cons = np.array([getattr(e, "last_consumption", 0.0) for e in emps], dtype=np.float64)
        n = int(wealth.size)
        row = dict(n=n, gini_env=float(self.env.gov.current_gini), wealth_from_engine=engine_ok,
                   gini_model_poststep=(gini_model(post_wealth) if post_wealth.size else float("nan")))
        if n == 0:
            row.update({k: float("nan") for k in GINI_KEYS})
            row.update(k=0, t_plus=0.0, t_minus=0.0, min_w=float("nan"), mean_w=float("nan"), bias_condition=None)
        else:
            t_plus = float(wealth[wealth > 0].sum())
            t_minus = float(-wealth[wealth < 0].sum())
            min_w = float(wealth.min())
            c = (-min_w + 1e-5) if min_w < 0 else 0.0
            row.update(
                gini_model_recomputed=gini_model(wealth), gp_wealth=gini_p(wealth), gp_net_wealth=gini_p(net),
                gp_consumption=gini_p(cons), k=int((wealth < 0).sum()), t_plus=t_plus, t_minus=t_minus,
                min_w=min_w, mean_w=float(wealth.mean()), min_net=float(net.min()), k_net=int((net < 0).sum()),
                # dieu kien CHINH XAC de Gini tinh tien < G_P (PREREG directional_prediction); chi co nghia khi co nguoi am
                bias_condition=(bool(t_plus + n * n * c > (2 * n - 1) * t_minus) if min_w < 0 else None),
            )
        self.rows.append(row)
        return out

    def summary(self):
        """Tom tat 1 episode: TB theo buoc cua moi Gini + chan doan tai san am."""
        r = self.rows
        if not r:
            return {}
        def m(key):
            v = np.array([x.get(key, np.nan) for x in r], dtype=np.float64)
            return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
        neg = [x for x in r if x.get("k", 0) > 0]
        n_steps = len(r)
        thr = [x["k"] > x["n"] ** 2 / (2 * x["n"] - 1) for x in r if x["n"] > 0]
        out = {f"{k}_mean": m(k) for k in GINI_KEYS}
        out.update(
            gini_env_mean=m("gini_env"),
            # sai lech giua Gini mo hinh tinh lai (tap Employee cua bo ghi) va env.gov.current_gini — kiem tap nguoi trung
            gini_recompute_max_abs_err=float(np.nanmax([abs(x["gini_model_recomputed"] - x["gini_env"]) for x in r
                                                        if x["n"] > 0] or [np.nan])),   # phai = 0 (cung mang)
            gini_poststep_max_abs_err=float(np.nanmax([abs(x["gini_model_poststep"] - x["gini_env"]) for x in r
                                                       if x["n"] > 0] or [np.nan])),   # do lech anh chup sau buoc
            wealth_from_engine_share=float(np.mean([x["wealth_from_engine"] for x in r])),
            n_mean=m("n"),
            neg_step_share=len(neg) / n_steps,
            k_share_mean=float(np.mean([x["k"] / x["n"] for x in r if x["n"] > 0])) if any(x["n"] > 0 for x in r) else float("nan"),
            k_max=int(max(x["k"] for x in r)),
            k_over_threshold_step_share=float(np.mean(thr)) if thr else float("nan"),
            bias_condition_share_among_neg_steps=(float(np.mean([x["bias_condition"] for x in neg])) if neg else float("nan")),
            tminus_over_tplus_mean_neg_steps=(float(np.mean([x["t_minus"] / x["t_plus"] if x["t_plus"] > 0 else np.inf
                                                              for x in neg])) if neg else 0.0),
            absmin_over_mean_mean_neg_steps=(float(np.mean([-x["min_w"] / x["mean_w"] if x["mean_w"] > 0 else np.inf
                                                             for x in neg])) if neg else 0.0),
            min_wealth_episode=float(min(x["min_w"] for x in r if x["n"] > 0)),
            neg_net_step_share=float(np.mean([x.get("k_net", 0) > 0 for x in r])),
        )
        return out


# --------------------------------------------------------------------------------------------- chay
def sha256_files(paths):
    out = {}
    for p in paths:
        if os.path.isdir(p):
            h = hashlib.sha256()
            for root, _, files in sorted(os.walk(p)):
                for f in sorted(files):
                    fp = os.path.join(root, f)
                    h.update(os.path.relpath(fp, p).replace("\\", "/").encode())
                    h.update(open(fp, "rb").read())
            out[p] = h.hexdigest()
        elif os.path.exists(p):
            out[p] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        else:
            out[p] = "MISSING"
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="yaml ScenarioConfig DA DUNG DE TRAIN (vd scenarios/em_baseline.yaml)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=240)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--explore", action="store_true")
    p.add_argument("--batched-inference", action="store_true")
    p.add_argument("--arms", type=str, default=None, help="giong be/benchmark.py --arms (bo trong = 3 baseline + 3 +rl_aux)")
    p.add_argument("--json-out", type=str, required=True)
    p.add_argument("--torch-seeding", choices=["step_policy", "episode"], default="step_policy",
                   help="step_policy = theo sua doi 2026-10-07 (mac dinh); episode = cach cu (seed+i moi episode), chi de doi chung")
    p.add_argument("--self-test-only", action="store_true", help="chi chay 3 test dang ky truoc roi thoat")
    return p.parse_args()


def main():
    args = parse_args()
    self_test()
    print(f"[{TOOL_VERSION}] self-test G_P: DAT (Bang 3 Raffinetti 2015; G_P = Gini x N/(N-1) khi khong am; bien)")
    if args.self_test_only:
        return

    integrity_paths = ["be/env.py", "be/rule_engine.py", "be/benchmark.py", "be/baselines.py", args.checkpoint, args.config]
    sha_before = sha256_files(integrity_paths)

    cfg = ScenarioConfig.from_yaml(args.config)
    cfg.max_steps = args.max_steps
    env_config = cfg.to_env_kwargs()
    algo = v1.build_algo(env_config, args.checkpoint)
    if algo is None:
        raise SystemExit(f"[{TOOL_VERSION}] KHONG nap duoc checkpoint -> dung (v2 chi dung cho checkpoint that).")
    pairing = TorchPairing(algo, args.seed, mode=args.torch_seeding)

    # Thu tu nhanh + cach lay TB hanh dong RL: CHEP DUNG be/benchmark.py::main (v1) de tai lap.
    custom_arms = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else None
    names_to_run = [v1.RL_LEARNED_GOV] + (list(v1.BASELINE_NAMES) if custom_arms is None else [])
    aux_mean = None
    rl_full_mean = None
    per_episode = {}

    print(f"[{TOOL_VERSION}] Che do policy: {'LAY MAU (explore)' if args.explore else 'TAT DINH'}; "
          f"gieo torch: {args.torch_seeding}")
    print(f"\n{'Nhanh':<22} {'GDP':>8} {'1-Gini':>7} {'1-GPw':>7} {'1-GPnet':>8} {'1-GPc':>7} {'%buoc am':>9} {'k/N':>6}")
    idx = 0
    while idx < len(names_to_run):
        name = names_to_run[idx]
        idx += 1
        env = MacroEnvironment(**env_config)
        rec = StepRecorder(env)
        pairing.rec = rec
        np.random.seed(args.seed)  # giong v1
        override = aux_mean if v1.needs_rl_aux(name) else None
        cols = {}
        gov_acts_all = []
        for ep in range(args.episodes):
            pairing.ep = ep
            if args.torch_seeding == "episode":
                torch.manual_seed(args.seed + ep)  # cach cu (truoc sua doi 2026-10-07) — chi de doi chung
            rec.new_episode()
            gdp_hist, gini_hist, unemp_hist, deaths, gov_acts, extra = v1.run_episode(
                env, algo, name, args.max_steps, explore=args.explore, aux_override=override, episode_seed=args.seed + ep,
                batched=args.batched_inference, full_override=rl_full_mean)
            if len(gov_acts) > 0:
                gov_acts_all.append(gov_acts)
            if len(gdp_hist) == 0:
                continue
            row = dict(gdp=float(gdp_hist.mean()), gini=float(gini_hist.mean()), unemp=float(unemp_hist.mean()),
                       deaths=int(deaths), **{k: float(v) for k, v in extra.items()}, **rec.summary())
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
        per_episode[name] = cols
        g = np.mean(cols["gdp"])
        print(f"{name:<22} {g:>8.2f} {1 - np.mean(cols['gini']):>7.3f} {1 - np.mean(cols['gp_wealth_mean']):>7.3f} "
              f"{1 - np.mean(cols['gp_net_wealth_mean']):>8.3f} {1 - np.mean(cols['gp_consumption_mean']):>7.3f} "
              f"{100 * np.mean(cols['neg_step_share']):>8.1f}% {np.mean(cols['k_share_mean']):>6.3f}")
        if name == v1.RL_LEARNED_GOV and gov_acts_all:
            acts = np.concatenate(gov_acts_all, axis=0)
            aux_mean = acts[:, v1.AUX_DIMS].mean(axis=0).astype(np.float32)
            rl_full_mean = acts.mean(axis=0).astype(np.float32)
            names_to_run += ([b + v1.RL_AUX_SUFFIX for b in v1.BASELINE_NAMES] if custom_arms is None else custom_arms)

    sha_after = sha256_files(integrity_paths)
    if sha_after != sha_before:
        print(f"[{TOOL_VERSION}] CANH BAO: sha256 TRUOC != SAU — file mo hinh/checkpoint bi doi trong luc chay!")
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(dict(tool=TOOL_VERSION, prereg=PREREG, checkpoint=args.checkpoint, config=args.config,
                       explore=bool(args.explore), seed=args.seed, episodes=args.episodes, max_steps=args.max_steps,
                       episode_seeding="paired", torch_seeding=args.torch_seeding, prereg_amendment="452fb00",
                       torch_seeded_calls=pairing.seeded_calls, np_state_changed_by_inference=pairing.np_state_changed_calls,
                       inference="batched" if args.batched_inference else "per_agent",
                       provenance=v1.provenance(), argv=sys.argv,
                       sha256_before=sha_before, sha256_after=sha_after, sha256_match=(sha_before == sha_after),
                       aux_mean=None if aux_mean is None else [float(x) for x in aux_mean],
                       rl_full_mean=None if rl_full_mean is None else [float(x) for x in rl_full_mean],
                       arms_requested=custom_arms, arms=per_episode), f, ensure_ascii=False, indent=1)
    if pairing.np_state_changed_calls:
        print(f"[{TOOL_VERSION}] CANH BAO: {pairing.np_state_changed_calls} lan suy luan lam doi np.random (RNG moi truong)!")
    print(f"[{TOOL_VERSION}] Da ghi: {args.json_out} (sha256 truoc/sau trung: {sha_before == sha_after})")


if __name__ == "__main__":
    main()
