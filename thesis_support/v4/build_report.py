"""Gom MOI ket qua dot v4 thanh MOT file cho Claude Web: BAO_CAO_v4_cho_Claude_Web.md (goc repo, KHONG commit — sao luu Documents).
Chi doc cac file da sinh trong thesis_support/v4 + log hang doi. Sinh lai bat ky luc nao: python thesis_support/v4/build_report.py"""
import csv
import datetime as dt
import os
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO)
V4 = "thesis_support/v4"
OUTF = "BAO_CAO_v4_cho_Claude_Web.md"


def rd(p):
    return open(p, encoding="utf-8").read() if os.path.exists(p) else f"_(chưa có: `{p}`)_\n"


def csv_md(p, cols=None, maxrows=200):
    if not os.path.exists(p):
        return f"_(chưa có: `{p}`)_\n"
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    cols = cols or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows[:maxrows]:
        out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |")
    return "\n".join(out) + "\n"


def tail(p, n=40):
    return "".join(open(p, encoding="utf-8").readlines()[-n:]) if os.path.exists(p) else "(chưa có)\n"


head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
S = []
S.append(f"""# BÁO CÁO ĐỢT v4 — Claude Code gửi Claude Web (qua người dùng)

Tạo: {now}. Repo code HEAD `{head}`; file chưa commit: {len(dirty.splitlines())} (chủ yếu log đang chạy). Lệnh sinh lại: `python thesis_support/v4/build_report.py`.
Mã mô hình đóng băng (`be/` không đổi từ `ed57a19`; mốc đóng băng `14406fd`). Mọi CSV/.py/.tex nằm trong `thesis_support/v4/` (đã commit);
file .md (báo cáo này và các báo cáo con) KHÔNG commit vào repo code — sao lưu ở repo Documents `TheAIEconomist/project_docs/v4/`.

## 0. Trạng thái tổng quát (theo thứ tự yêu cầu)

| Mục | Trạng thái |
|---|---|
| P0.0 | XONG — 808/909 commit `fea7552`, sao lưu Documents; sha256 checkpoint 9 seed; `PREREG_A8_replication.yaml` commit `649087d` (21:00 08/10) TRƯỚC mọi kết quả 404–909; ước tính giờ máy ở mục 2 |
| P0.5 | XONG — 22 mục, trích nguyên văn mã (mục 3) |
| P0.7 | XONG — `metric_semantics.csv` + kiểm câu #33 (mục 4) |
| P0.6 | XONG — V1, V2, V5, V7, phần cứng; **V6 KHÔNG TÌM THẤY nguồn** (mục 4) |
| P0.1/P0.2 | 3 seed gốc XONG, tái lập **trùng từng chữ số**; 6 seed lặp lại: bench-v1 ĐANG CHẠY đêm 08/10 (mục 6) |
| P0.4 | XONG — `seeds_gate.csv` 9 seed (cổng seed 42 chạy lại 08/10) (mục 5) |
| P0.3 (bench-v2.1) | DỜI SANG 09/10 theo người dùng (máy nghỉ); đã xong seed 42 tất định (chưa phân tích — chờ đủ seed theo bản đăng ký) |
| P0.8 (cuối chân trời) | câu (1) đã trả lời từ mã (mục 4); câu (2) chờ dữ liệu theo bước của v2.1 (09/10) |
| P1, P2 | chưa làm (P1 trước 12/10, P2 trước 25/10) |

## 1. Mâu thuẫn / thất bại / thiếu (nêu trước)

1. **P0.5 — hai dòng KHÔNG KHỚP mã**: (2) thu nhập phi chính thức: `0.35·s_i` là SẢN LƯỢNG; thu nhập = chi tiêu hộ × (0.35·s_i / tổng cung thực)
   (`rule_engine.py:616-618, 1047, 1066`); `0.35·s_i·P_idx` chỉ là thu nhập kỳ vọng cho hạn mức vay (`:668`). (9) firm vỡ nợ khi
   (0 nhân viên ∧ tiền dự kiến ≤ 0 ∧ tuổi > 6) ∨ (tài sản ròng < 0 ∧ tiền dự kiến < −chi phí cố định) (`rule_engine.py:1432-1435`).
   Khoá luận thiếu: số mũ vốn 0.3; L_eff = Σ effort·skill; điều kiện nhận trợ cấp (tài sản thanh khoản < 0.8 × chi phí sinh hoạt thị trường);
   reward mọi lớp bị env nhân `reward_scale_*` (0.045/0.018/0.04/0.04/0.012/0.02) rồi clip ±100 (`env.py:492-495`); Firm "P" trong điều kiện
   thuê là giá kỳ vọng thô (không chặn trần).
2. **Dòng [TRAIN] = bước cuối episode, GDP danh nghĩa** (`rllib_wrapper.py:162-217`). Khoá luận (bảng "3 seed cuối", THESIS_MASTER:470):
   "Thất nghiệp train iter 1→100" là thất nghiệp BƯỚC CUỐI → ghi rõ hoặc thay bằng `unemp` benchmark (TB mọi bước). "thất nghiệp ~0.1–0.4%"
   (THESIS_MASTER:559): KHÔNG xác định được nguồn. Các con số "GDP thực" trong báo cáo tiến độ 404–909 phải đọc là GDP danh nghĩa bước cuối.
3. **#33 "Eq×Prod RL khớp log"**: [TRAIN] không có Eq×Prod; log in màn hình benchmark cũ không được lưu. Thay bằng: lần chạy lại 06/10 (`84d3d8a`)
   có "Da nap checkpoint" ở cả 6 log và trùng JSON từng episode (sai lệch 0.0).
4. **V6 KHÔNG TÌM THẤY nguồn** (#29c "834 vs 924 …"): suy luận số học → (834; 0.55; 379) = Kho bạc 25 000 (sau sửa), (924; 0.68; 289) = nhánh so
   sánh (nhiều khả năng Kho bạc 1e6, không chắc); thiết lập ép hành động (effort 0.6, ρ=1), không phải policy RL. Khuyến nghị không trích.
5. **V5**: "mục 15" trong #20 là tham chiếu sai — bản sửa trước là **#9** (thước đo NPL, v0.15).
6. **Seed 303**: điều phối gọi lần 2 sau khi xong → chạy lại đánh giá cuối iter_100, ghi đè file final (checkpoint không đổi). Ghi chú "nghỉ
   máy 60 phút" của 202/303 thuộc các lần gọi bị dừng/khởi lại (xem `final_v037_seed{{202,303}}_status.json`).
7. **KHÔNG CÓ**: thuế tách TNCN/DN (mô hình chỉ lưu tổng — cột `taxes_pit/taxes_cit` = KHONG CO, có `taxes_total`); cột mở rộng (births,
   min tài sản, Kho bạc/nợ cuối, giá TB) của v1 ở chế độ LẤY MẪU (v1 không ghi; v2.1 lấy mẫu là quỹ đạo khác). Ở chế độ tất định các cột này
   lấy từ v2.1 (trùng v1 từng số — kiểm từng episode trước khi gán); hiện mới có seed 42.
8. **Tôi (Claude Code) không báo trước thời lượng benchmark** khi khởi chạy tối 08/10 (ước ~5 h + ~7 h song song) — người dùng đã dừng v2.1.

## 2. P0.0 — sha256, đăng ký, ước tính giờ máy

Checkpoint (sha256 thư mục: đường dẫn tương đối + nội dung mọi file, duyệt có sắp xếp):

{csv_md(f"{V4}/checkpoints_sha256.csv")}
Ước tính (đo thật 06–08/10): bench-v1 7 nhánh ≈ 8 phút / seed×chế độ; quét 11 nhánh ≈ 16 phút; bench-v2.1 16 nhánh ≈ 25 phút (chạy một mình
≈ 18–20 phút). 6 seed mới × 2 chế độ bench-v1 + quét ≈ 4.8 giờ (một tiến trình); v2.1 9 seed × 2 chế độ ≈ 6–7.5 giờ + kiểm ghép cặp ≈ 0.5 giờ.
Tổng ≈ 11.5–12.5 giờ máy.

`PREREG_A8_replication.yaml` (`649087d`): 3 seed gốc giữ nguyên là phép kiểm gốc; 404–909 là mẫu lặp lại độc lập (cùng nhánh, chỉ số, H1–H4
nguyên văn 4e0327d/0d964d8, quy tắc bất đồng v2); quy tắc diễn giải "khác thì viết đúng như vậy"; khai báo mù: chưa chạy/xem benchmark hay
quét nào của 404–909; ĐÃ xem log [TRAIN], kết quả cổng, learning_curves/effort_histograms (404/505) và hiện tượng cuối episode của 404/909.

## 3. P0.5 — Kiểm toán công thức (nguyên văn)

{rd(f"{V4}/P0_5_formula_audit.md")}
## 4. P0.6 + P0.7

{rd(f"{V4}/P0_6_P0_7_audit.md")}
### metric_semantics.csv

{csv_md(f"{V4}/metric_semantics.csv")}
### V2 — đầu ra `gini_negative_check.py`

```
{rd(f"{V4}/gini_negative_check_output.txt")}```

## 5. P0.4 — Seed và cổng (9 seed × iter 40/100)

{csv_md(f"{V4}/seeds_gate.csv", ["seed", "iter", "c1_effort_mean", "c1_low_share_pct", "c1", "c2", "c3_deaths_det", "c3_deaths_exp",
                                  "c3_struct32_det", "c3_struct32_exp", "c3_original", "c3_split", "c4_n_tools_gt_0.05", "c4_gov_sensitivity",
                                  "c5_sfc_max", "c6_births_max", "c7", "gate_result", "train_hours_wallclock", "train_code_commit"])}
Ngưỡng (`gate_eval.py:177-189`): c1 effort TB ≥ 0.5 và tỷ lệ < 0.5 ≤ 15%; c2 không có wage = 0; c3 original TB chết tất định ≤ 0.5 và trung vị
lấy mẫu = 0 — split tương tự nhưng chỉ tính ca KHÔNG thuộc #32 và #32 ≤ 6/episode; c4 ≥ 2 công cụ có |Δaction| p5→p95 > 0.05; c5 SFC ≤ 1e-3;
c6 births/bước ≤ 0.35 và sinh khẩn cấp TB ≤ 0.5; c7 trợ cấp không bị chặn khi còn tiền. Episode cổng: 3 tất định (seed 999/1001/1002) + 10 lấy
mẫu (2001–2010), 240 bước, suy luận từng tác tử. Script/retry/override/giờ huấn luyện từng seed: cột `script`, `retries`, `overrides` trong
`seeds_gate.csv`. Seed 42: cổng gốc 28/09 ở chế độ original TRƯỢT ở iter 40 (lần 1 #31 → sửa v0.37 chạy lại từ đầu; lần 2 #32) → người dùng
chọn (A); số ở bảng trên là CHẠY LẠI 08/10 trên checkpoint đóng băng (iter 40: original TRƯỢT [0,0,3], split ĐẠT; khớp lịch sử).
Thời điểm quy tắc `split` so với kết quả seed 42 (theo CLAUDE_HISTORY): cổng iter 40 seed 42 trượt lúc 17:50 28/09 (v0.37-gate); `--c3-mode split`
được thêm sau đó cùng ngày (v0.37-tool, trần #32 = 3); định nghĩa #32 theo cơ chế + trần 6/episode đặt 20:05 28/09 (v0.37-tool3) — tức là
ĐỀU SAU khi đã thấy kết quả seed 42; trước seed 202/303/404–909. Phải khai báo nguyên văn như vậy trong khoá luận.

Phụ: `effort_histograms.csv` (9 seed × iter 40/100), `learning_curves.csv` (9 seed × 100 iter).

## 6. P0.1/P0.2 — Bảng (hiện mới đủ 3 seed gốc; 6 seed lặp lại đang chạy)

Kiểm tái lập: `aggregate_benchmark.py` trên JSON mới (84d3d8a) → bảng số TRÙNG `benchmark_table_3seeds.md`; `aggregate_flat_sweep.py` →
bảng + H1–H4 TRÙNG `flat_sweep_table.md`. Bảng mean^{{±std}} (đơn vị lặp = seed, ddof = 1):

{rd(f"{V4}/tables_v1.md")}
### Bảng gộp bằng script đã đăng ký — bộ 3 seed gốc

{rd(f"{V4}/agg_benchmark_orig3.md")}
{rd(f"{V4}/agg_flat_sweep_orig3.md")}
## 7. Tiến độ hàng đợi bench-v1 (đuôi log)

```
{tail("audits/paper_extras/v4_runs/queue_v1.log")}```

## 8. Lệnh tái lập (bản hiện tại — P2.5 hoàn chỉnh sau)

```
# huấn luyện: 202/303 audits/final_run/run_seed_with_gate.py --seed S --scenario-name final_v037_seedS --c3-mode split
#            404–909 bash audits/paper_extras/extra_seeds/run_day.sh S1 S2
bash audits/paper_extras/v4_runs/queue_v1.sh          # cổng seed 42 + bench-v1 (7 nhánh + quét) cho 404–909
bash audits/paper_extras/v4_runs/queue_v2.sh          # bench-v2.1 9 seed + kiểm ghép cặp (09/10)
python thesis_support/v4/make_tables.py               # CSV tidy + bảng mean^±std + IQM/CI (gọi aggregate_benchmark.py / aggregate_flat_sweep.py)
python thesis_support/v4/make_seeds_gate.py           # seeds_gate.csv, effort_histograms.csv
python audits/paper_extras/export_thesis_data.py      # learning_curves.csv
python thesis_support/v4/analyze_v2.py                # P0.3 (sau khi có v2.1)
python thesis_support/v4/analyze_horizon.py           # P0.8 (sau khi có v2.1)
python thesis_support/v4/make_p05_audit.py; python thesis_support/v4/gini_negative_check.py
pytest be/tests -q                                    # 73 ca
```
""")
open(OUTF, "w", encoding="utf-8").write("\n".join(S))
print("ok", OUTF, os.path.getsize(OUTF), "bytes")
