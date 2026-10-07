"""B4 (danh muc benh ly: trang thai, nhom, phien ban/ngay/commit sua) + B5.1 (danh sach 73 test) cho khoa luan.
CHI DOC: KNOWN_PATHOLOGIES.md, CLAUDE_HISTORY.md (khong nam trong git), be/tests/*.py, git log. Dau ra thesis_support/B4_*.csv, B5_tests.csv.
Phan loai trang thai va nhom do Claude Code GAN TAY tu dong "Trang thai" cua tung muc (cot status_src_line de doi chieu); nhom la DE XUAT.
Phien ban -> commit: git chi co ~35 commit, nhieu phien ban gop 1 commit; cot commit_upper_bound = commit dau tien co ngay > ngay
cua phien ban (CAN TREN, khong phai commit chinh xac). Dung: python audits/paper_extras/export_b4_b5.py"""
import ast
import collections
import csv
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO)
OUT = "thesis_support"
os.makedirs(OUT, exist_ok=True)

# ---------------- git log
log = subprocess.run(["git", "log", "--reverse", "--date=iso-strict", "--pretty=%h|%ad|%s"], capture_output=True, text=True,
                     encoding="utf-8").stdout.strip().splitlines()
commits = [l.split("|", 2) for l in log]
PAT_RE = re.compile(r"#(\d{1,2})\b")
with open(f"{OUT}/B4_git_log.csv", "w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo)
    w.writerow(["commit", "datetime", "subject", "pathology_ids_in_subject"])
    for h, d, s in commits:
        w.writerow([h, d, s, ";".join(sorted(set(PAT_RE.findall(s)), key=int))])

# ---------------- phien ban -> ngay (CLAUDE_HISTORY) -> commit can tren
ver_date = {}
for l in open("CLAUDE_HISTORY.md", encoding="utf-8"):
    m = re.match(r"^## (v[0-9][^ ]*) — (\d{4}-\d{2}-\d{2})", l)
    if m and m.group(1) not in ver_date:
        ver_date[m.group(1)] = m.group(2)


def commit_upper(date):
    for h, d, s in commits:
        if d[:10] > date:
            return h
    return ""


# ---------------- benh ly: trang thai/nhom/phien ban sua (gan tay, doi chieu cot status_src_line)
CAT = {  # id: (status_category, group_de_xuat, fixed_version, ghi_chu)
    1: ("đã sửa", "vòng lặp & mốc neo quy mô", "v0.12", ""),
    2: ("đã sửa", "vòng lặp & mốc neo quy mô", "v0.18", "sửa vòng lặp 1 bước ở 3 kênh (v0.16/v0.17/v0.18); kênh trợ cấp actual_living_cost cùng loại KHÔNG CHẶN (CLAUDE.md)"),
    3: ("đã sửa", "kế toán", "v0.15", ""),
    4: ("đã sửa", "tích hợp RL", "v0.11", ""),
    5: ("đã sửa", "tích hợp RL", "v0.15", ""),
    6: ("đã sửa", "tích hợp RL", "v0.16", "quy hồi: khoá total_market_turnover bị xoá → reward Economy mất tín hiệu"),
    7: ("đã sửa", "khuyến khích", "v0.16", "nguyên nhân gốc siêu lạm phát v0.15.1"),
    8: ("đã sửa", "vòng lặp & mốc neo quy mô", "v0.19", ""),
    9: ("đã sửa", "kế toán", "v0.15", "thước đo NPL"),
    10: ("đã sửa", "thể chế & khởi tạo", "v0.17", "hiệu quả với policy đã học mới kiểm gián tiếp"),
    11: ("đã sửa", "khuyến khích", "v0.16", ""),
    12: ("đã sửa", "tích hợp RL", "v0.16", "v0.15 sửa, v0.16 hiệu chỉnh lại"),
    13: ("đã sửa", "tích hợp RL", "v0.20", ""),
    14: ("đã sửa", "thể chế & khởi tạo", "v0.21", "v0.20 sửa, v0.21 vá (lãi phạt Bagehot)"),
    15: ("đã sửa", "công cụ & thiết kế benchmark", "v0.20", "lỗi công cụ đọc dữ liệu"),
    16: ("đã sửa", "thể chế & khởi tạo", "v0.21", "v0.20 sửa, v0.21 hiệu chỉnh trích dẫn"),
    17: ("đã sửa", "thể chế & khởi tạo", "v0.22", ""),
    18: ("đã sửa", "kế toán", "v0.23", ""),
    19: ("đã sửa", "thể chế & khởi tạo", "v0.24", "2 lỗi khi triển khai buffer-stock"),
    20: ("đã sửa", "kế toán", "v0.26", ""),
    21: ("chấp nhận làm hạn chế", "không chặn", "", "KHÔNG CHẶN (cập nhật 2026-10-02)"),
    22: ("chấp nhận làm hạn chế", "không chặn", "", "KHÔNG CHẶN (cập nhật 2026-10-02)"),
    23: ("mở", "không chặn", "", "KHÔNG CHẶN; chưa đo lại trên checkpoint multi-seed cuối"),
    24: ("đã sửa", "vòng lặp & mốc neo quy mô", "v0.28", ""),
    25: ("chấp nhận làm hạn chế", "không chặn", "", "rủi ro thuộc run 500 iteration cũ"),
    26: ("đã sửa", "công cụ & thiết kế benchmark", "v0.37", "xử lý bằng phương pháp (3 seed × 2 chế độ), không phải sửa mã"),
    27: ("đã sửa", "khuyến khích", "v0.34", "kiểm chứng sau bằng validation_v035 và 3 seed"),
    28: ("đã sửa", "kế toán", "v0.33", ""),
    29: ("đã sửa một phần", "thể chế & khởi tạo", "v0.35", "(a)(b)(c)(f) v0.35, (h)(i) v0.32, (g) v0.36 (#30a); (d) lưới an sinh khẩn cấp, (e) cổ tức ngân hàng: KHÔNG CHẶN"),
    30: ("đã sửa một phần", "vòng lặp & mốc neo quy mô", "v0.36", "(a)(b) v0.36; (c) giảm phát tới sàn giá: KHÔNG CHẶN"),
    31: ("đã sửa", "vòng lặp & mốc neo quy mô", "v0.37", ""),
    32: ("chấp nhận làm hạn chế", "thể chế & khởi tạo", "", "người dùng chọn (A) 2026-09-28"),
    33: ("không sửa vì đóng băng", "công cụ & thiết kế benchmark", "", ""),
    34: ("không sửa vì đóng băng", "công cụ & thiết kế benchmark", "", ""),
    35: ("không sửa vì đóng băng", "khuyến khích", "", ""),
    36: ("không sửa vì đóng băng", "công cụ & thiết kế benchmark", "", "bench-v2 (tag bench-v2) xử lý ở công cụ đánh giá, không sửa mô hình"),
}

# ---------------- test -> loai, benh ly, mo ta 1 cau (Claude Code tom tat tu docstring/ma test)
T = {
    "test_sfc_conservation_single_bank": ("SFC heuristic", [18, 19, 28], "Bảo toàn dòng tiền từng bước (ΔTổng = −khấu hao) + 2 bất biến sổ cái, 1 ngân hàng, 3 seed, policy heuristic"),
    "test_sfc_conservation_multi_bank": ("SFC heuristic", [18, 19, 28], "Như trên với nhiều ngân hàng (định tuyến creditor/depository bank không rò hoặc nhân đôi tiền)"),
    "test_sfc_conservation_random_policy": ("SFC policy ngẫu nhiên", [3, 18, 19, 28], "Bảo toàn dòng tiền dưới hành động ngẫu nhiên kiểu RL; bắt buộc kích hoạt bơm cầu, tiền phạt, chi mua G"),
    "test_sfc_conservation_when_deficit_creates_public_debt": ("SFC policy ngẫu nhiên", [30], "Ép Kho bạc âm → nợ công; đẳng thức có trừ nợ công vẫn đúng từng bước, kênh phải thật sự kích hoạt"),
    "test_train_and_simulate_paths_are_identical": ("hợp đồng môi trường", [5], "Đường train (RLlib wrapper) và simulate cho cùng obs/reward/terminated từng bước"),
    "test_death_penalty_scales_with_age_after_reward_pipeline": ("hợp đồng môi trường", [4, 5], "Sau pipeline reward, phạt tử vong người trẻ nặng hơn người già và không chạm trần clip"),
    "test_reward_clip_too_small_is_rejected": ("hợp đồng môi trường", [5], "reward_clip quá nhỏ bị từ chối khi khởi tạo"),
    "test_sanitizers_neutralize_non_finite_values": ("hợp đồng môi trường", [], "Bộ làm sạch action/obs thay NaN/Inf bằng giá trị hữu hạn"),
    "test_economy_receives_positive_trade_volume": ("hợp đồng môi trường", [6], "Economy nhận khối lượng giao dịch > 0 (khoá total_market_turnover được ghi)"),
    "test_train_and_server_share_one_ppo_config": ("hợp đồng môi trường", [12], "main.py và server.py dùng chung cấu hình PPO (mạng, vf_clip_param, danh sách policy)"),
    "test_price_indexation_loop_gain_equals_theta_above_ceiling": ("hợp đồng môi trường (đo khuếch đại)", [2, 8, 17], "Trên trần chỉ số hoá, dP_t/dP_{t-1} = θ_Calvo ở 4 mức cung (nhiễu +2% trên 2 bản sao)"),
    "test_hiring_is_not_dominated_by_firing": ("hợp đồng môi trường (ép hành động)", [7, 16, 17, 35], "Ép thuê/sa thải: reward Firm khi thuê > khi sa thải, việc làm, giá trong [0.25×, 3×], ≥4/5 firm sống, không nợ công"),
    "test_government_purchases_follow_taxes_and_treasury_cap": ("hợp đồng môi trường", [7, 17], "G_t = min(ρ·(thuế+phạt kỳ trước), số dư Kho bạc)"),
    "test_household_consumption_draws_on_deposits": ("hợp đồng môi trường", [7], "Hộ có tiền gửi nhưng 0 tiền mặt vẫn chi tiêu được, tiền gửi giảm tương ứng"),
    "test_government_reward_is_based_on_real_gdp": ("hợp đồng môi trường", [11], "Reward Government theo GDP thực: kinh tế lành mạnh > siêu lạm phát"),
    "test_initial_conditions_avoid_artificial_crisis_window": ("hợp đồng môi trường", [8, 10], "Dưới policy ngẫu nhiên: phần lớn có việc từ đầu, giá không sốc quá 5×, gần như không chết trong 60 bước đầu"),
    "test_initial_employment_assignment_is_capital_weighted_and_shuffled": ("hợp đồng môi trường", [10], "Gán việc ban đầu đúng 90%, mỗi người 1 firm, theo trọng số vốn (Hamilton)"),
    "test_wage_ratchet_converges_near_ceiling_with_mrpl_scale_fix": ("hợp đồng môi trường", [8, 16, 30], "Lương hội tụ quanh trần với mrpl_scale_constant; bản cũ (scale 1.0) tái hiện cóc lương"),
    "test_wage_renegotiation_does_not_open_single_step_price_loop": ("hợp đồng môi trường (đo khuếch đại)", [24], "Đàm phán lại lương không làm khuếch đại 1 bước vượt θ ở 3 xác suất đàm phán"),
    "test_wage_renegotiation_reduces_ratchet_on_price_reversal": ("hợp đồng môi trường", [24], "Giá lên rồi xuống: prob 0.12 giảm cóc lương rõ so với prob 0 (logic cũ tái hiện được)"),
    "test_firm_bankruptcy_via_rule_engine_is_not_a_zombie": ("hợp đồng môi trường", [13], "Firm phá sản theo rule_engine nhận BANKRUPT/terminated, không thành zombie"),
    "test_last_bank_bankruptcy_triggers_bailout_not_crash": ("hợp đồng môi trường", [14], "Ngân hàng duy nhất phá sản → cứu trợ, không crash episode"),
    "test_bailout_is_not_free_money_bagehot_penalty_rate": ("hợp đồng môi trường", [14], "Cứu trợ tạo khoản nợ có lãi phạt (Bagehot), không phải tiền miễn phí"),
    "test_second_bank_bankruptcy_is_cleaned_up_when_another_survives": ("hợp đồng môi trường", [14], "Có ≥2 ngân hàng: ngân hàng chết được dọn, không cứu trợ, terminated đúng 1 lần"),
    "test_parquet_read_dataset_round_trips_with_installed_pyarrow": ("hợp đồng môi trường", [15], "read_dataset đọc lại đúng dữ liệu đã ghi với pyarrow đang cài"),
    "test_employee_credit_loop_gain_equals_theta": ("hợp đồng môi trường (đo khuếch đại)", [1], "Kênh tín dụng tiêu dùng không mở lại vòng lặp: khuếch đại 1 bước = θ ở 4 mức cung"),
    "test_employee_liquidity_constrained_borrows_and_bank_reflects_it": ("hợp đồng môi trường", [], "Người thiếu thanh khoản vay được và khoản vay hiện trong dư nợ ngân hàng"),
    "test_employee_never_borrows_beyond_subsistence_shortfall": ("hợp đồng môi trường", [], "Vay không vượt phần thiếu hụt sinh tồn"),
    "test_employee_death_with_debt_is_written_off_not_ghost_debt": ("hợp đồng môi trường", [13], "Người chết có nợ → ghi nợ xấu đúng ngân hàng, không nợ ma, SFC vẫn đúng"),
    "test_economy_buffer_stock_loop_gain_equals_theta": ("hợp đồng môi trường (đo khuếch đại)", [19], "Buffer-stock mua/bán tối đa không làm khuếch đại 1 bước vượt θ ở 4 mức cung"),
    "test_economy_buffer_stock_conserves_sfc_on_forced_sell": ("hợp đồng môi trường", [19], "Ép Economy bán: tổng giá trị hệ thống bảo toàn qua 1 bước"),
    "test_economy_buffer_stock_never_exceeds_balance_sheet_constraints": ("hợp đồng môi trường", [19], "Quỹ và kho của Economy không bao giờ âm"),
    "test_default_loss_is_recognized_once_in_bank_equity": ("hợp đồng môi trường", [28], "Vỡ nợ D làm vốn ngân hàng giảm ~D (không ~2D)"),
    "test_bank_failure_uses_equity_not_gross_reserves": ("hợp đồng môi trường", [28], "Phá sản ngân hàng theo vốn chủ sở hữu; tiêu chí cũ (reserves) tái hiện được"),
    "test_deposit_ledger_stays_consistent_after_death_and_inheritance": ("hợp đồng môi trường", [28], "Sổ cái tiền gửi khớp sau chết và thừa kế"),
    "test_shirking_is_detected_only_below_effort_threshold": ("hợp đồng môi trường", [27], "Chỉ effort dưới ngưỡng mới bị phát hiện lười và sa thải (q = 1)"),
    "test_shirking_monitor_has_no_wage_effect_at_compliant_effort": ("hợp đồng môi trường", [27], "Khi effort tuân thủ, giám sát không ảnh hưởng lương (trùng q = 0)"),
    "test_effort_has_private_benefit_only_with_monitoring": ("hợp đồng môi trường (ép hành động)", [27], "Ép effort 0.1 vs 0.6: q=0 lười thắng, q=0.05 tuân thủ thắng (tương thích khuyến khích)"),
    "test_dismissed_shirker_cannot_be_rehired_until_lockout_expires": ("hợp đồng môi trường", [27], "Người bị sa thải vì lười không được tuyển lại trước khi hết lockout"),
    "test_government_action_space_has_unemployment_relief_dimension": ("hợp đồng môi trường", [29], "Action Government có chiều thứ 5 (mức trợ cấp)"),
    "test_government_relief_action_scales_subsidies_linearly_and_can_be_zero": ("hợp đồng môi trường", [29], "Trợ cấp tuyến tính theo action[4], bằng 0 khi action = 0"),
    "test_government_reward_is_equality_times_productivity_and_legacy_is_reproducible": ("hợp đồng môi trường", [29], "Reward Government = 0.02·(1−Gini)·GDP thực − phạt tử vong; dạng cũ tái hiện được"),
    "test_initial_treasury_is_operating_balance_and_legacy_endowment_funding_reproducible": ("hợp đồng môi trường", [29], "Kho bạc ban đầu = số dư vận hành; logic cũ (Kho bạc tài trợ vốn khởi tạo) tái hiện được"),
    "test_market_demand_factor_is_an_informative_observation_not_a_constant": ("hợp đồng môi trường", [29], "market_demand_factor trong obs Firm có phương sai thật, không dính sàn"),
    "test_treasury_outflow_counters_track_offbook_spending_channels": ("hợp đồng môi trường", [29, 30], "3 bộ đếm chi Kho bạc ngoài sổ khớp đúng tổng chi từng kênh"),
    "test_initial_employees_have_the_wage_they_are_actually_paid": ("hợp đồng môi trường", [30], "Lao động có việc lúc reset có wage > 0 đúng mức được trả"),
    "test_unemployment_relief_is_paid_within_treasury_budget": ("hợp đồng môi trường", [30], "Trợ cấp không bị cổng tuyệt đối 1000 chặn nhưng không vượt số dư; logic cũ tái hiện được"),
    "test_buffer_band_blocks_procyclical_intervention": ("hợp đồng môi trường", [31], "Dải ±band chặn mua khi giá cao/bán khi giá thấp; logic cũ (không dải) tái hiện được"),
    "test_buffer_buy_is_bounded_by_current_household_market": ("hợp đồng môi trường", [31], "Lượng mua ≤ κ × chi tiêu hộ của bước; logic cũ (mốc cố định) tái hiện được"),
    "test_late_buffer_buy_after_deflation_no_longer_triples_prices": ("hợp đồng môi trường (ép hành động)", [31], "Mua muộn sau giảm phát: logic mới giá ≤1.5× trong 10 bước, logic cũ >2×"),
}
# test co nhanh tu kiem logic LOI qua cong tac/tham so cu (doc ma test 2026-10-07)
LEGACY_IN_TEST = {
    "test_wage_ratchet_converges_near_ceiling_with_mrpl_scale_fix": "mrpl_scale=1.0",
    "test_wage_renegotiation_reduces_ratchet_on_price_reversal": "wage_renegotiation_prob=0.0",
    "test_bank_failure_uses_equity_not_gross_reserves": "bank_failure_criterion='reserves'",
    "test_government_reward_is_equality_times_productivity_and_legacy_is_reproducible": "reward_mode='legacy'",
    "test_initial_treasury_is_operating_balance_and_legacy_endowment_funding_reproducible": "initial_treasury=1e6 + treasury_funds_initial_endowments=True",
    "test_unemployment_relief_is_paid_within_treasury_budget": "relief gate 'legacy_gate'",
    "test_buffer_band_blocks_procyclical_intervention": "buffer_stock_rule='legacy_fixed_anchor'",
    "test_buffer_buy_is_bounded_by_current_household_market": "buffer_stock_rule='legacy_fixed_anchor'",
    "test_late_buffer_buy_after_deflation_no_longer_triples_prices": "buffer_stock_rule='legacy_fixed_anchor'",
    "test_effort_has_private_benefit_only_with_monitoring": "shirking_monitor_prob=0 (đối chứng)",
}

# so truong hop moi ham test: dem tu pytest --collect-only (tham so co the la bien, khong doc duoc bang ast)
_col = subprocess.run([sys.executable, "-m", "pytest", "be/tests", "--collect-only", "-q"], capture_output=True, text=True,
                      encoding="utf-8").stdout.splitlines()
N_CASES = collections.Counter(l.split("::")[1].split("[")[0] for l in _col if "::" in l)
rows = []
for f in ("be/tests/test_sfc_accounting.py", "be/tests/test_sfc_random_policy.py", "be/tests/test_env_contracts.py"):
    tree = ast.parse(open(f, encoding="utf-8").read())
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"):
            n_param = N_CASES[n.name]
            kind, pats, desc = T[n.name]
            rows.append([os.path.basename(f), kind, n.name, n.lineno, n_param, desc, ";".join(map(str, pats)),
                         LEGACY_IN_TEST.get(n.name, "")])
assert sum(r[4] for r in rows) == 73, sum(r[4] for r in rows)
with open(f"{OUT}/B5_tests.csv", "w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo)
    w.writerow(["file", "kind", "test", "line", "n_cases", "checks", "pathology_ids", "self_checks_buggy_logic_via"])
    w.writerows(rows)

# ---------------- danh muc benh ly
KP = open("KNOWN_PATHOLOGIES.md", encoding="utf-8").read().splitlines()
heads = [(i, int(m.group(1)), l[3:].strip()) for i, l in enumerate(KP) for m in [re.match(r"^## (\d+)\.", l)] if m]
heads.append((len(KP), None, ""))
tests_by_pat = collections.defaultdict(list)
for r in rows:
    for p in filter(None, r[6].split(";")):
        tests_by_pat[int(p)].append(r[2])
with open(f"{OUT}/B4_pathology_catalog.csv", "w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo)
    w.writerow(["id", "title", "status_category", "group_proposed", "fixed_version", "fixed_date", "commit_upper_bound",
                "status_src_line", "tests_guarding", "note"])
    for (i, pid, title), (j, _, _) in zip(heads, heads[1:]):
        st = next((k + 1 for k in range(i, j) if "Trạng thái" in KP[k]), "")
        cat, grp, ver, note = CAT[pid]
        date = ver_date.get(ver, "")
        w.writerow([pid, title, cat, grp, ver, date, commit_upper(date) if date else "", st,
                    ";".join(tests_by_pat.get(pid, [])), note])
cnt = collections.Counter(v[0] for v in CAT.values())
grp = collections.Counter(v[1] for v in CAT.values())
with open(f"{OUT}/B4_status_counts.csv", "w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo)
    w.writerow(["kind", "value", "count"])
    for k, v in sorted(cnt.items()):
        w.writerow(["status", k, v])
    for k, v in sorted(grp.items()):
        w.writerow(["group", k, v])
print(dict(cnt))
print(dict(grp))
print("tests:", len(rows), "functions /", sum(r[4] for r in rows), "cases")
print("no pathology:", [r[2] for r in rows if not r[6]])
print("pathologies without test:", [p for p in CAT if p not in tests_by_pat])
