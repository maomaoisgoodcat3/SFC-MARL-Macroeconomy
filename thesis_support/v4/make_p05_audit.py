"""P0.5 (YEU_CAU v4): kiem toan cong thuc chuong Environment. Trich NGUYEN VAN dong ma tu file (khong chep tay) de tranh sai.
Ket luan KHOP/KHONG KHOP do Claude Code doc ma (2026-10-08) — cot 'verdict'/'note'. Dau ra: thesis_support/v4/P0_5_formula_audit.md
Dung: python thesis_support/v4/make_p05_audit.py"""
import os
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO)


def lines(path, a, b):
    src = open(path, encoding="utf-8").read().splitlines()
    return "\n".join(f"{i:>5}: {src[i - 1]}" for i in range(a, b + 1))


ITEMS = [
    (1, "Cobb–Douglas, số mũ lao động 0.6, hằng số quy mô c = 0.38",
     [("be/rule_engine.py", 280, 280), ("be/rule_engine.py", 286, 286), ("be/rule_engine.py", 574, 574), ("be/rule_engine.py", 604, 604),
      ("scenarios/em_baseline.yaml", 64, 64)],
     "KHỚP", "Q_f = c·A_f·K^0.3·L_eff^0.6, c = mrpl_scale_constant = 0.38, A_f = productivity_factor, L_eff = Σ effort·skill của nhân viên. "
             "Khoá luận nên ghi thêm số mũ vốn 0.3 và định nghĩa L_eff."),
    (2, "Thu nhập phi chính thức khi thất nghiệp = 0.35·s_i·P",
     [("be/rule_engine.py", 616, 618), ("be/rule_engine.py", 1047, 1047), ("be/rule_engine.py", 1066, 1066), ("be/rule_engine.py", 668, 668)],
     "KHÔNG KHỚP (một phần)", "0.35·s_i là SẢN LƯỢNG vật chất phi chính thức. Thu nhập = phần chi tiêu hộ theo tỷ trọng sản lượng: "
             "total_consumer_spending × (0.35·s_i / total_real_supply). Công thức 0.35·s_i·P_idx chỉ dùng làm THU NHẬP KỲ VỌNG để tính hạn mức vay (dòng 668)."),
    (3, "Kỹ năng ban đầu = clip(0.5 + 0.45·LogNormal(0, 0.35), 0.60, 4.0)", [("be/env.py", 332, 338)], "KHỚP", ""),
    (4, "Con thừa hưởng kỹ năng cha mẹ + nhiễu Gauss σ = 0.05 × kỹ năng cha mẹ, clip [0.60, 4.0]",
     [("be/env.py", 852, 855), ("scenarios/em_baseline.yaml", 37, 37)], "KHỚP", "trait_mutation_sigma = 0.05 (nhân với kỹ năng cha/mẹ cho kỹ năng; cộng thẳng cho risk_aversion/tax_morale)."),
    (5, "Giá Calvo θ = 0.70 (và 0.30)", [("be/rule_engine.py", 1005, 1007), ("be/rule_engine.py", 530, 530)],
     "KHỚP", "P_t = 0.70·giá kỳ vọng + 0.30·giá thanh toán bù trừ tức thời; 0.30 = 1 − θ (tỷ trọng giá cân bằng tức thời), không phải hằng số riêng."),
    (6, "P_idx = min(giá kỳ vọng, 3.0 × chi phí sinh hoạt ban đầu)", [("be/rule_engine.py", 276, 277), ("scenarios/em_baseline.yaml", 61, 62)], "KHỚP", ""),
    (7, "Giám sát: q = 0.05; nỗ lực < 0.5 bị sa thải; không thuê lại 9 tháng", [("be/scenario_config.py", 146, 147), ("be/scenario_config.py", 163, 163)],
     "KHỚP", "Giá trị mặc định ScenarioConfig; em_baseline.yaml không ghi đè."),
    (8, "Firm thuê: thanh khoản > 1.2·P, tín hiệu thuê > −0.1, tối đa max(2, ⌈0.25·headcount⌉)/tháng",
     [("be/rule_engine.py", 320, 320), ("be/rule_engine.py", 340, 343)], "KHỚP",
     "P ở đây là expected_price (giá kỳ vọng, KHÔNG chặn trần); thanh khoản = tiền mặt firm − dự trữ an toàn."),
    (9, "Firm vỡ nợ khi tài sản ròng < 0; haircut 0.30; nợ xấu xoá sau 6 tháng",
     [("be/rule_engine.py", 1420, 1420), ("be/rule_engine.py", 1429, 1435), ("be/agents/bank.py", 280, 280), ("scenarios/em_baseline.yaml", 51, 51)],
     "KHÔNG KHỚP (điều kiện vỡ nợ)", "Vỡ nợ khi (0 nhân viên VÀ tiền mặt dự kiến ≤ 0 VÀ tuổi firm > 6) HOẶC (tài sản ròng < 0 VÀ tiền mặt dự kiến < −chi phí cố định). "
             "Tài sản ròng = tiền mặt + 0.70·vốn − nợ. Haircut 0.30 và xoá NPL sau 6 tháng (npl_writeoff_months) KHỚP."),
    (10, "Buffer stock: κ = 0.15, dải ±5%, EMA bán rã 12 tháng", [("be/rule_engine.py", 97, 98), ("be/rule_engine.py", 987, 998)], "KHỚP", ""),
    (11, "Trợ cấp: relief × chi phí sinh hoạt tháng 1–3; 0.375×relief tháng 4–6; 0 sau 6; thiếu tiền trả theo tỷ lệ",
     [("be/rule_engine.py", 1126, 1127), ("be/rule_engine.py", 1132, 1132), ("be/rule_engine.py", 1141, 1145), ("be/rule_engine.py", 1072, 1072)],
     "KHỚP", "Điều kiện nhận (khoá luận chưa ghi): tài sản thanh khoản < 0.8 × actual_living_cost; actual_living_cost = giỏ sinh tồn × giá thị trường bước này (KHÔNG chặn trần). "
             "Ngân sách khả dụng = Kho bạc − (chi mua G + bơm cầu dương) của bước."),
    (12, "Lãi suất cho vay/huy động 0.12/0.05; chi phí sinh hoạt ban đầu 6.0", [("scenarios/em_baseline.yaml", 44, 45), ("scenarios/em_baseline.yaml", 62, 62)], "KHỚP", ""),
    (13, "Reward Worker: 2ln(max(0.01,c)) − e² − 1.5·m·(1−d)² − min(2+0.5·streak, 8)·1[thất nghiệp], clip ±50; chết −100(1+horizon ratio)",
     [("be/agents/employee.py", 264, 282), ("be/env.py", 492, 495), ("scenarios/em_baseline.yaml", 52, 54)], "KHỚP",
     "m = tax_morale. Sau đó env nhân reward_scale_employee = 0.045 rồi clip ±100 (env.py:492-495)."),
    (14, "Reward Firm: 0.1·profit + 0.5·headcount·1[profit>0] − 2m(1−d)² − 3·max(0, debt/K − 1.5), clip ±50; phá sản −100",
     [("be/agents/firm.py", 194, 196), ("be/agents/firm.py", 199, 199), ("be/agents/firm.py", 215, 228)], "KHỚP",
     "−100 khi BANKRUPT hoặc TERMINATED. Env nhân reward_scale_firm = 0.018 rồi clip ±100."),
    (15, "Reward Government: 0.02·(1−Gini)·GDP thực − 20·deaths − 0.00005·nợ công, clip",
     [("be/agents/government.py", 320, 325), ("be/agents/government.py", 336, 338), ("scenarios/em_baseline.yaml", 48, 48), ("scenarios/em_baseline.yaml", 56, 56)], "KHỚP", "clip ±100 (government.py), sau đó env nhân reward_scale_government = 0.04 và clip ±100. 20 = death_penalty_coef (em_baseline.yaml)."),
    (16, "Reward Bank, Supervisor, Economy (nguyên văn)",
     [("be/agents/bank.py", 365, 366), ("be/agents/bank.py", 375, 386), ("be/agents/supervisor.py", 152, 164), ("be/agents/economy.py", 271, 273), ("be/agents/economy.py", 278, 278),
      ("be/agents/economy.py", 294, 298), ("be/agents/economy.py", 335, 346), ("scenarios/em_baseline.yaml", 54, 60)], "(dán nguyên văn)",
     "Bank: 0.01·(lãi thu − lãi trả) − 0.06·tổn thất vỡ nợ mới − 50·NPL/dư nợ − 0.05·max(0, dự trữ bắt buộc − dự trữ), clip ±100 (×0.04). "
     "Supervisor: 0.01·tiền phạt + 2·bắt đúng − 0.5·kiểm sai − 0.005·chi phí, clip ±50 (×0.012). "
     "Economy: 2·ln(1 + khối lượng thực) − 50·(π − 0.0016)² − 1700·chết/max(dân số, sàn) − 4000·(tỷ lệ sinh − 0.0033)² − 2·max(0, 5 − tồn kho nhà), clip ±50 (×0.02)."),
    (17, "Thâm hụt Kho bạc chuyển thành nợ công", [("be/agents/government.py", 261, 266), ("be/agents/government.py", 273, 275)], "KHỚP",
     "Kho bạc âm cuối bước → cộng |Kho bạc| vào nợ công, Kho bạc = 0; khi thặng dư trả tối đa 50% Kho bạc mỗi bước."),
    (18, "Gini tài sản âm: tịnh tiến w − min + 1e-5", [("be/rule_engine.py", 1750, 1768), ("be/rule_engine.py", 1710, 1716)], "KHỚP",
     "Chuẩn hoá theo N. Docstring trích Raffinetti et al. (2015) là SAI (bài không tịnh tiến). Tập: Employee ACTIVE/INITIALIZED không chết trong bước; tài sản = tiền mặt + tiền gửi (không trừ nợ)."),
    (19, "Baselines: us_federal/saez ρ=1, injection 0, relief 0.40; free_market (0,0,1,0,0)", [("be/baselines.py", 104, 121)], "KHỚP", "Thuế phẳng quy đổi (#34)."),
    (20, "Thưởng headcount Firm (firm.py:215)", [("be/agents/firm.py", 215, 215)], "KHỚP", ""),
    (21, "Không gian quan sát/hành động (tab:agents)", [("be/rllib_wrapper.py", 18, 23), ("be/rllib_wrapper.py", 30, 70)], "(bảng)",
     "Quan sát: Worker 15, Firm 14, Government 10, Bank 11, Supervisor 8, Economy 11. Hành động: Worker 4 [0,1]; Firm 3 ([−1,1],[0,1],[0,1]); "
     "Government 5 (PIT [0,0.5], CIT [0,0.5], ρ [0,1], bơm cầu [−0.2,0.2], relief [0,1]); Bank 3; Supervisor 3; Economy 3 (chỉ chiều 0 có tác dụng)."),
    (22, "Số tác tử, độ dài episode, PPO", [("scenarios/em_baseline.yaml", 27, 30), ("be/rllib_wrapper.py", 243, 243), ("be/rllib_wrapper.py", 252, 255)], "(bảng)",
     "50 worker, 5 firm, 1 bank; 240 bước. Ray 2.34.0 PPO: γ=0.99, λ(GAE)=1.0, clip=0.3, kl_coeff=0.2, kl_target=0.01, entropy=0, vf_loss_coeff=1.0, "
     "vf_clip=2000, grad_clip=0.5, lr=3e-4, 10 epoch SGD, train batch 4000, minibatch 256, 4 env runner, mạng [64,64] tanh, 100 iteration "
     "(đọc từ build_ppo_config bằng mã, 08/10; các giá trị γ/λ/clip/kl/entropy là mặc định RLlib 2.34)."),
]

head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
out = [f"# P0.5 — Kiểm toán công thức chương Environment\n\nHEAD {head}; mã mô hình đóng băng (be/ không đổi từ ed57a19). "
       "Dòng mã trích tự động từ file (không chép tay). Tạo bởi `python thesis_support/v4/make_p05_audit.py`.\n"]
for n, claim, refs, verdict, note in ITEMS:
    out.append(f"\n## {n}. {claim}\n\n**{verdict}**" + (f" — {note}" if note else "") + "\n")
    for f, a, b in refs:
        out.append(f"\n`{f}:{a}-{b}`\n```\n{lines(f, a, b)}\n```")
open("thesis_support/v4/P0_5_formula_audit.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("ok", len(ITEMS))
