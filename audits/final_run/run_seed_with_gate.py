"""Dieu phoi MOT seed huan luyen voi cong kiem tra dang ky truoc (official_baseline_v1_report.md) chay TU DONG.

Ly do ton tai (2026-09-28): o seed 1, cong iter 40 phu thuoc vao mot job hen gio trong phien Claude -- job khong kich hoat,
cong chi duoc chay tay khi training da toi iter 50. Script nay la tien trinh DOC LAP, khong phu thuoc phien chat:
  1. khoi chay training (cung lenh voi seed 1, chi doi --seed/--scenario-name), ghi log ra file;
  2. theo doi log, khi checkpoint iter_<gate_iter> duoc luu xong -> chay audits/final_run/gate_eval.py tren checkpoint do
     (training van chay song song, 20 loi CPU du cho ca hai);
  3. cong KHONG DAT hoac cong LOI -> dung training ngay (taskkill ca cay tien trinh), theo quy tac dung da dang ky truoc;
     checkpoint da luu van con, co the resume neu nguoi dung quyet dinh khac;
  4. cong DAT -> de training chay het, roi (mac dinh) chay lai cung bo danh gia tren checkpoint cuoi de ghi nhan
     (danh gia cuoi CHI de bao cao, khong phai tieu chi dung).

KHONG tu noi seed: moi lan goi = dung 1 seed. Moi lan chay dai phai duoc nguoi dung phe duyet rieng (CLAUDE.md).
Chay lai cung --scenario-name sau khi bi dung: tu choi neu cong da ghi FAIL, tru khi co --override-failed-gate.

Vi du (seed 2, sau khi duoc duyet):
  python audits/final_run/run_seed_with_gate.py --seed 202 --scenario-name final_v037_seed202 --c3-mode split
Thu pipeline (vai phut, khong dung cho ket luan):
  python audits/final_run/run_seed_with_gate.py --seed 7 --scenario-name smoke_gate --checkpoint-dir <tmp> \
      --train-iters 3 --checkpoint-freq 1 --gate-iter 1 --gate-quick --no-final-eval
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GATE = os.path.join(REPO, "audits", "final_run", "gate_eval.py")


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def other_training_pids():
    """PID cac tien trinh `be.main --mode train` DANG chay (tru chinh minh) -- de khong bao gio chay chong 2 training."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -match 'be\\.main' -and $_.CommandLine -match '--mode train' } | "
          "ForEach-Object { $_.ProcessId }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None  # khong kiem duoc -> nguoi goi xu ly than trong
    return [int(x) for x in out.split() if x.strip().isdigit() and int(x) != os.getpid()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scenario-name", required=True)
    ap.add_argument("--config", default="scenarios/em_baseline.yaml")
    ap.add_argument("--train-iters", type=int, default=100)
    ap.add_argument("--checkpoint-freq", type=int, default=10)
    ap.add_argument("--checkpoint-dir", default="be/checkpoint")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--train-batch-size", type=int, default=4000)
    ap.add_argument("--minibatch-size", type=int, default=256)
    ap.add_argument("--gate-iter", type=int, default=40)
    ap.add_argument("--c3-mode", choices=["original", "split"], required=True,
                    help="bat buoc chon tuong minh -- day la quyet dinh dang ky truoc cua nguoi dung")
    ap.add_argument("--c3-structural-cap", type=float, default=6.0)
    ap.add_argument("--gate-quick", action="store_true", help="CHI de thu pipeline: gate_eval --quick")
    ap.add_argument("--no-final-eval", action="store_true")
    ap.add_argument("--override-failed-gate", action="store_true")
    ap.add_argument("--log-dir", default="audits/final_run/logs")
    ap.add_argument("--gate-script", default=GATE, help="CHI de thu orchestrator: thay cong that bang cong gia")
    ap.add_argument("--gate-retries", type=int, default=2,
                    help="so lan chay LAI cong khi cong LOI (khong phai TRUOT) truoc khi dung training. Quy tac dang ky "
                         "truoc chi noi ve TRUOT; LOI la su co cong cu (thu nghiem 28/9: 1 lan loi thoang qua sau 13 s).")
    ap.add_argument("--rest-minutes", type=float, default=0.0,
                    help="cho moi training khac (be.main --mode train) ket thuc, roi NGHI them N phut truoc khi bat dau "
                         "(quy tac nghi may giua cac lan chay, CLAUDE.md). Luon tu choi chay chong training khac.")
    a = ap.parse_args()

    os.chdir(REPO)
    os.makedirs(a.log_dir, exist_ok=True)
    tag = a.scenario_name
    train_log = os.path.join(a.log_dir, f"{tag}_train.log")
    status_path = os.path.join(a.log_dir, f"{tag}_status.json")
    gate_json = os.path.join(a.log_dir, f"{tag}_gate_iter{a.gate_iter}.json")
    run_dir = os.path.join(a.checkpoint_dir, tag)
    gate_ckpt = os.path.join(run_dir, f"iter_{a.gate_iter}")
    status = dict(seed=a.seed, scenario_name=tag, c3_mode=a.c3_mode, gate_iter=a.gate_iter, events=[])
    if os.path.exists(status_path):
        with open(status_path, encoding="utf-8") as f:
            status = json.load(f)
        status.setdefault("events", [])

    def note(msg, **kw):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        status["events"].append(dict(t=now(), msg=msg, **kw))
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=1)

    if status.get("gate_result") == "FAIL" and not a.override_failed_gate:
        note("TU CHOI chay: cong da ghi FAIL cho scenario nay (dung --override-failed-gate neu nguoi dung quyet dinh khac).")
        return 2

    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")

    others = other_training_pids()
    if others is None:
        note("KHONG kiem tra duoc tien trinh training khac (powershell loi) -> dung de an toan")
        return 5
    if others or a.rest_minutes > 0:
        if others:
            note(f"dang co training khac chay (PID {others}) -> cho ket thuc truoc", other_pids=others)
        while others:
            time.sleep(60)
            others = other_training_pids() or []
        if a.rest_minutes > 0:
            note(f"nghi may {a.rest_minutes:.0f} phut truoc khi bat dau (bat dau luc ~"
                 f"{(dt.datetime.now() + dt.timedelta(minutes=a.rest_minutes)).strftime('%H:%M')})")
            time.sleep(60.0 * a.rest_minutes)
        others = other_training_pids()
        if others is None or others:
            note(f"sau khi cho van co training khac/khong kiem duoc ({others}) -> dung de an toan")
            return 5

    def run_gate(ckpt, out_json, out_log):
        cmd = [sys.executable, a.gate_script, ckpt, "--config", a.config, "--c3-mode", a.c3_mode,
               "--c3-structural-cap", str(a.c3_structural_cap), "--json-out", out_json]
        if a.gate_quick:
            cmd.append("--quick")
        with open(out_log, "w", encoding="utf-8") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=REPO)
        txt = open(out_log, encoding="utf-8", errors="replace").read()
        m = re.search(r"GATE_RESULT: (PASS|FAIL)", txt)
        return (m.group(1) if (m and rc == 0) else "ERROR"), rc

    def stop_training(proc, why):
        note(f"DUNG training ({why}): taskkill cay tien trinh PID {proc.pid}")
        subprocess.call(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            note("CANH BAO: tien trinh training chua thoat sau 120 s -- kiem tra tay (tasklist)")

    train_cmd = [sys.executable, "-m", "be.main", "--mode", "train", "--config", a.config,
                 "--train-iters", str(a.train_iters), "--num-workers", str(a.num_workers),
                 "--train-batch-size", str(a.train_batch_size), "--minibatch-size", str(a.minibatch_size),
                 "--checkpoint-freq", str(a.checkpoint_freq), "--checkpoint-dir", a.checkpoint_dir,
                 "--scenario-name", tag, "--seed", str(a.seed)]
    gate_done = status.get("gate_result") == "PASS" or (status.get("gate_result") == "FAIL" and a.override_failed_gate)
    note("BAT DAU training: " + " ".join(train_cmd[1:]), gate_already_done=gate_done)
    lf = open(train_log, "a", encoding="utf-8")
    lf.write(f"\n===== {now()} run_seed_with_gate: bat dau/tiep tuc =====\n")
    lf.flush()
    proc = subprocess.Popen(train_cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=REPO,
                            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    status["train_pid"] = proc.pid

    saved_re = re.compile(r"Checkpoint persisted at: .*[\\/]iter_(\d+)\s*$")
    pos = 0
    while not gate_done:
        time.sleep(20)
        with open(train_log, "rb") as f:  # nhi phan: vi tri byte on dinh giua cac lan mo file
            f.seek(pos)
            data = f.read()
        cut = data.rfind(b"\n") + 1  # chi xu ly toi dong HOAN CHINH cuoi cung; phan duoi doc lai lan sau
        pos += cut
        chunk = data[:cut].decode("utf-8", errors="replace")
        hit = any(int(m.group(1)) >= a.gate_iter for m in map(saved_re.search, chunk.splitlines()) if m)
        if not hit and os.path.isdir(gate_ckpt) and os.path.isdir(os.path.join(run_dir, f"iter_{a.gate_iter + a.checkpoint_freq}")):
            hit = True  # da co checkpoint sau cong (vd. resume) -> checkpoint cong chac chan da luu xong
        if hit:
            note(f"checkpoint cong da luu: {gate_ckpt} -> chay gate_eval (c3-mode={a.c3_mode})")
            for attempt in range(1 + max(0, a.gate_retries)):
                stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")  # moi lan 1 log rieng -- khong ghi de bang chung
                res, rc = run_gate(gate_ckpt, gate_json, os.path.join(a.log_dir, f"{tag}_gate_iter{a.gate_iter}_{stamp}.log"))
                note(f"KET QUA CONG iter {a.gate_iter} (lan {attempt + 1}): {res} (rc={rc})")
                if res != "ERROR":
                    break
                if attempt < a.gate_retries:
                    time.sleep(60)
            status["gate_result"] = res
            if res != "PASS":
                if proc.poll() is None:
                    stop_training(proc, f"cong {res}")
                lf.close()
                return 1
            gate_done = True
        elif proc.poll() is not None:
            note(f"training THOAT truoc khi toi cong (rc={proc.returncode}) -- xem {train_log}")
            lf.close()
            return 3

    rc = proc.wait()
    lf.close()
    note(f"training ket thuc (rc={rc})")
    if rc != 0:
        return 4
    if not a.no_final_eval:
        saved = sorted((int(d.split("_")[1]) for d in os.listdir(run_dir) if re.fullmatch(r"iter_\d+", d)), reverse=True)
        final_ckpt = os.path.join(run_dir, f"iter_{saved[0]}")
        note(f"danh gia checkpoint cuoi (CHI de bao cao): {final_ckpt}")
        res, rc2 = run_gate(final_ckpt, os.path.join(a.log_dir, f"{tag}_final_iter{saved[0]}.json"),
                            os.path.join(a.log_dir, f"{tag}_final_iter{saved[0]}.log"))
        status["final_eval_result"] = res
        note(f"danh gia cuoi iter {saved[0]}: {res} (rc={rc2}) -- KHONG phai tieu chi dung")
    note("XONG seed. KHONG tu chay seed tiep theo (can nguoi dung phe duyet rieng).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
