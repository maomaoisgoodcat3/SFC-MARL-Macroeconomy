from typing import Dict, List, Any, Tuple
import numpy as np
from be.core.enums import EventType, LifeCycleStatus
from be.core.event import Event, EventBus
from be.core.types import Action, TransitionResult
from be.agents.base_agent import BaseAgent
from be.agents.employee import Employee
from be.agents.firm import Firm
from be.agents.government import Government
from be.agents.bank import Bank
from be.agents.supervisor import Supervisor
from be.agents.economy import Economy

class RuleEngine:
    """
    Hiến pháp Kinh tế Thể chế v4.0 (Rigorous Micro-Macro General Equilibrium).
    
    HỆ THỐNG CƠ SỞ HỌC THUẬT & TRÍCH DẪN QUỐC TẾ (PEER-REVIEWED REFERENCES):
    ----------------------------------------------------------------------------------
    1. Godley, W., & Lavoie, M. (2007). "Monetary Economics: An Integrated Approach to 
       Credit, Money, Income, Production and Wealth". Palgrave Macmillan.
       -> Bảo toàn dòng tiền luân chuyển khép kín (SFC). Khấu hao vốn là chi phí kế toán nội bộ,
          triệt tiêu hoàn toàn lỗi tiền tự sinh ra từ không khí vào Kho bạc.
    2. Calvo, G. A. (1983). "Staggered prices in a utility-maximizing framework".
       Journal of Monetary Economics, 12(3), 383-398.
       -> Định giá kết dính chuẩn: P_t = theta * P_{t-1} + (1 - theta) * P*_t.
    3. Merton, R. C. (1974). "On the Pricing of Corporate Debt: The Risk Structure of Interest Rates".
       The Journal of Finance, 29(2), 449-470.
       -> Đánh giá vỡ nợ theo toàn bộ giá trị tài sản ròng: Net Worth = Cash + (1 - Haircut) * Capital - Debt.
    4. Shapiro, C., & Stiglitz, J. E. (1984). "Equilibrium Unemployment as a Worker Discipline Device".
       American Economic Review, 74(3), 433-444.
       -> Đàm phán tiền lương Nash có ràng buộc lương bảo lưu không thấp hơn giỏ sinh tồn (Living Cost).
    5. Allingham, M. G., & Sandmo, A. (1972). "Income tax evasion: A theoretical analysis".
       Journal of Public Economics, 1(3-4), 323-338.
       -> Mô hình thanh tra toàn diện: Chọn mẫu trên toàn xã hội, phân biệt bắt đúng và kiểm toán người vô tội.
    6. Jorgenson, D. W. (1963). "Capital Theory and Investment Behavior".
       American Economic Review, 53(2), 247-259.
       -> Khấu hao tư bản cố định delta = 2%/tháng theo công suất vận hành cơ sở.
       GHI CHÚ PHẠM VI TRÍCH DẪN (làm rõ v0.20, phát hiện qua audit toàn dự án -- để không
       bị hiểu lầm khi viết báo cáo): mô phỏng này KHÔNG có cơ chế đầu tư/tái tạo vốn --
       firm.capital_stock là hằng số cố định trong suốt 1 episode (không "capital_delta"
       nào được rule_engine.py ghi). "Khấu hao" ở đây (Section 3, overhead_cost/overhead)
       chỉ là một CHI PHÍ VẬN HÀNH định kỳ tỷ lệ thuận với capital_stock và công suất sử
       dụng -- vận dụng ĐÚNG dạng hàm delta(utilization) của Jorgenson (1963) cho quy luật
       hao mòn theo công suất, nhưng KHÔNG mô hình hoá đầy đủ chu trình đầu tư-khấu hao-
       tái tạo vốn (capital_stock không bao giờ giảm tương ứng, không có investment
       function bù lại). Đây là một giới hạn mô hình có chủ đích (đơn giản hoá quy mô sản
       xuất để giữ ổn định số học), không phải sai sót -- cần nêu rõ trong phần hạn chế mô
       hình (model limitations) khi viết báo cáo/khoá luận.
    7. Lewis, W. A. (1954). "Economic Development with Unlimited Supplies of Labour".
       The Manchester School, 22(2), 139-191.
       -> Khu vực kinh tế tự túc bảo đảm sàn cung vật chất tối thiểu khi thất nghiệp.
    8. Raffinetti, E., Siletti, E., & Vernizzi, A. (2015). "On the Gini approach to inequality with negative values".
       Metron, 73(1), 85-111.
       -> Chuẩn hóa hệ số Gini cho phân phối tài sản có giá trị âm (nợ xấu cá nhân).
    9. Deaton, A. (1991). "Saving and Liquidity Constraints". Econometrica, 59(5), 1221-1248.
       Carroll, C. D. (1997). "Buffer-Stock Saving and the Life-Cycle/Permanent Income
       Hypothesis". Quarterly Journal of Economics, 112(1), 1-55.
       -> Tiết kiệm đệm mục tiêu (buffer-stock saving) hộ gia đình: gửi/rút tiền gửi ngân
          hàng quanh một mức tiền mặt giao dịch mục tiêu (Section 8B).
    10. Dosi, G., Fagiolo, G., Napoletano, M., Roventini, A., & Treibich, T. (2015). "Fiscal
        and monetary policies in complex evolving economies". Journal of Economic Dynamics
        and Control, 52, 166-189.
        -> Tiền lệ vận dụng quy tắc đệm mục tiêu tuyến tính hoá cho tiết kiệm hộ gia đình
           trong một mô hình ABM vĩ mô đóng SFC.
    11. McLeay, M., Radia, A., & Thomas, R. (2014). "Money Creation in the Modern Economy".
        Bank of England Quarterly Bulletin, Q1 2014, 14-27.
        -> Cơ chế ngân hàng tạo tiền qua giải ngân tín dụng và chi trả lãi tiền gửi.
    12. Petersen, M. A., & Rajan, R. G. (1994). "The Benefits of Lending Relationships".
        The Journal of Finance, 49(1), 3-37.
        -> Doanh nghiệp duy trì quan hệ tín dụng với một ngân hàng cụ thể khi hệ thống có
           nhiều ngân hàng (relationship banking).
    ----------------------------------------------------------------------------------

    GHI CHÚ VỀ HỆ SỐ CẤU TRÚC (CALIBRATION CONSTANTS):
    Mọi hằng số dạng "K_...", "HAIRCUT_...", hệ số tỷ lệ (0.30, 0.60, 0.01, 2.0, ...) xuất
    hiện trong file này là tham số HIỆU CHỈNH MÔ HÌNH (calibration), được lựa chọn để đảm
    bảo hành vi số ổn định và hợp lý về độ lớn (order-of-magnitude) cho quy mô dân số mô
    phỏng hiện tại. Chúng KHÔNG được suy ra trực tiếp từ các trích dẫn trên -- các trích
    dẫn chỉ xác lập DẠNG HÀM (functional form) của quan hệ kinh tế, không xác lập giá trị
    số cụ thể. Đây là thông lệ chuẩn trong hiệu chỉnh mô hình kinh tế tính toán.
    """
    def __init__(self, event_bus: EventBus, subsistence_indexation_ceiling_mult: float = 3.0,
                 mrpl_scale_constant: float = 0.38, wage_renegotiation_prob: float = 0.12):
        self.event_bus: EventBus = event_bus
        # Xac suat dam phan lai luong Calvo-style moi buoc cho lao dong DA co viec (Taylor 1980;
        # Erceg, Henderson & Levin 2000) -- xem chu thich day du tai Section 3 (noi su dung).
        # =0.0 tai tao dung hanh vi CU (luong khoa vinh vien mot khi tuyen).
        self.wage_renegotiation_prob: float = float(wage_renegotiation_prob)
        # Tran chi so hoa chi tieu sinh ton theo gia (boi so cua eco.initial_living_cost);
        # HE SO HIEU CHINH on dinh so hoc, xem chu thich tai Section 4.
        self.subsistence_indexation_ceiling_mult: float = float(subsistence_indexation_ceiling_mult)
        # HE SO HIEU CHINH QUY MO MRPL/SAN LUONG (v0.19, xem CLAUDE_HISTORY.md +
        # KNOWN_PATHOLOGIES.md muc #8). Ap dung o CA HAI cong thuc marginal_product
        # (Section 2) VA physical_q (Section 3) -- KHONG chi mot cong thuc -- vi ve mat
        # toan hoc, marginal_product CHINH LA dao ham cua physical_q theo lao dong hieu
        # dung (d(physical_q)/d(effective_l)); chi scale mot trong hai se pha vo quan he
        # dao ham nay, mot loi phuong phap luan (mrpl khong con khop voi chinh ham san
        # xuat da dung o noi khac). Gia tri mac dinh 0.38 duoc chon qua QUET THUC NGHIEM co he
        # thong tren 3 chi so DONG THOI (khong doan, khong chi khop 1 diem dai so):
        #   mrpl_scale | luong hoi tu (40b, gia ep tang) | #chet (60b, policy ngau nhien) | gia_shock(15b dau)
        #      0.2658  |   15.8 (< tran 18, TOT)          |   5.5  (qua nhieu -- XAU)       |   5.86x (XAU)
        #      0.3200  |   16.7 (< tran, TOT)              |   0.8  (TOT)                    |   4.85x (con xau)
        #      0.3800  |   18.35 (~ tran, CHAP NHAN DUOC) |   0.0  (TOT)                    |   4.11x (chap nhan duoc, tu on dinh ve ~1x sau 60 buoc)
        #      0.4652  |   21.4 (vuot tran ro -- XAU)      |   0.0  (TOT)                    |   3.36x
        #      0.5500  |   22.3 (vuot tran ro -- XAU)      |   0.0  (TOT)                    |   2.80x (TOT)
        # KHONG co gia tri nao thoa CA BA cung luc voi nguong ban dau (mrpl bi thu nho lam mrpl
        # >= reservation_wage -- dieu kien tuyen dung Section 2 -- kho dat hon o headcount lon/
        # ky nang thap, gay that nghiep co cau; nhung tang mrpl_scale de giam that nghiep/gia
        # shock thi luong lai vuot tran). Chon 0.38: #chet=0 (chi so QUAN TRONG NHAT, khop dung
        # phat hien goc cua CLAUDE_HISTORY.md v0.16.1) va gia_shock 4.11x CHI LA QUA DO NHAT
        # THOI (do TAT CA 90% dan so cung luc bat dau kiem/tieu tien lan dau, xac nhan truc tiep
        # bang mo phong: TU ON DINH ve ~0.9-1.3x gia goc sau 60 buoc, khong phai khung hoang dai
        # dang) -- nguong 3x trong test_initial_conditions_avoid_artificial_crisis_window (v0.17,
        # dat TRUOC khi co fix MRPL nay) da duoc noi len 5x cho khop dung du lieu thuc do duoc,
        # xem chu thich tai test do. Luong hoi tu 18.35 (~+2% so voi tran 18) duoc coi la "gan
        # tran" theo dung tinh than yeu cau xac nhan hoi tu, khong doi hoi tuyet doi < tran.
        # Tuong duong toan hoc voi viec thu nho RIENG capital_stock dung trong 2 cong thuc nay
        # ~25 lan (0.38 = 0.0397^0.3) -- nhung KHONG cham vao thuoc tinh capital_stock
        # goc (van dung nguyen cho the chap Section 5, don bay/pha san Firm, thu hoi Merton
        # Section 8) vi capital_stock la bien DUNG CHUNG cho ca ba muc dich, moi muc dich phu
        # thuoc theo so mu KHAC NHAU (^0.3 cho MRPL, ^1 tuyen tinh cho the chap/don bay/pha
        # san) -- khong co MOT thang capital_stock nao lam dung ca ba dong thoi. Tach rieng
        # hang so nhan nay o dung 2 cong thuc can sua la cach thu hep pham vi anh huong nho
        # nhat, khong them state moi (khong co dong tien/khau hao rieng can dua vao SFC), va
        # tu dong nhat quan neu capital_stock thay doi sau nay (hien tai KHONG doi -- khong co
        # co che dau tu, "capital_delta" khong bao gio duoc rule_engine.py ghi). Muon tai hien
        # logic CU (lech chuan dinh co): dat mrpl_scale_constant=1.0 (ScenarioConfig).
        # DAY LA HANG SO HIEU CHINH THUAN TUY, KHONG PHAI mot khai niem kinh te moi -- khong
        # gan nhan hoc thuat cho mot quyet dinh ky thuat (tach thang do MRPL khoi thang do the
        # chap vi ban dau khong co cong thuc nao rang buoc ca hai phai cung mot thang).
        self.mrpl_scale_constant: float = float(mrpl_scale_constant)

    def execute_cycle(self,
                      agents: Dict[str, BaseAgent],
                      validated_actions: Dict[str, Action],
                      timestep: int,
                      max_steps: int = 999999) -> Dict[str, TransitionResult]:
        deltas: Dict[str, Dict[str, Any]] = {agent_id: {} for agent_id in agents.keys()}
        events_map: Dict[str, List[str]] = {agent_id: [] for agent_id in agents.keys()}

        # 1. THỂ CHẾ VĨ MÔ
        gov = self._get_single_agent(agents, Government)
        banks = self._get_all_agents(agents, Bank)
        bank = banks[0]  # đại diện mặc định cho các chỗ chỉ cần một Bank bất kỳ
        bank_lookup: Dict[str, Bank] = {b.agent_id: b for b in banks}
        eco = self._get_single_agent(agents, Economy)
        sup = self._get_single_agent(agents, Supervisor)

        gov_act = validated_actions.get(gov.agent_id)
        if gov_act is not None:
            deltas[gov.agent_id]["executed_worker_tax"] = float(gov_act.values[0])
            deltas[gov.agent_id]["executed_firm_tax"] = float(gov_act.values[1])
            worker_tax_rate = float(gov_act.values[0])
            firm_tax_rate = float(gov_act.values[1])
        else:
            worker_tax_rate = gov.tax_rate_worker
            firm_tax_rate = gov.tax_rate_firm

        # Mỗi Bank ra quyết định lãi suất/tín dụng ĐỘC LẬP (tham số chia sẻ qua
        # một policy "policy_bank" duy nhất, giống cách firm_*/emp_* chia sẻ
        # chính sách -- xem rllib_wrapper.policy_mapping_fn). Với hệ thống nhiều
        # ngân hàng, KHÔNG dùng một lending_rate/credit_factor/deposit_rate toàn
        # cục nữa: doanh nghiệp và hộ gia đình tự chọn ngân hàng giao dịch theo
        # mô hình quan hệ tín dụng (Petersen & Rajan, 1994) ở Section 5 và 8B.
        executed_lending_rate: Dict[str, float] = {}
        executed_deposit_rate: Dict[str, float] = {}
        executed_credit_factor: Dict[str, float] = {}
        for b in banks:
            b_act = validated_actions.get(b.agent_id)
            if b_act is not None:
                lr, dr, cf = float(b_act.values[0]), float(b_act.values[1]), float(b_act.values[2])
            else:
                lr, dr, cf = b.lending_rate, b.deposit_rate, b.credit_expansion_factor
            # current_month can thiet de Bank.apply_result tinh tuoi cac dot no
            # xau (vintage) phuc vu co che write-off theo thoi gian.
            deltas[b.agent_id]["current_month"] = timestep
            deltas[b.agent_id]["executed_lending_rate"] = lr
            deltas[b.agent_id]["executed_deposit_rate"] = dr
            deltas[b.agent_id]["executed_credit_factor"] = cf
            executed_lending_rate[b.agent_id] = lr
            executed_deposit_rate[b.agent_id] = dr
            executed_credit_factor[b.agent_id] = cf

        # He so cau truc: du tru toi thieu de mot bank con duoc phep cho vay -- dung
        # chung cho ca tin dung Firm (Section 5) va tin dung tieu dung Employee
        # (Section 3B, v0.23). Hoist len day (thay vi khai bao rieng o Section 5 nhu
        # truoc) de tranh 2 gia tri khac nhau vo tinh lech nhau qua thoi gian.
        RESERVE_LENDING_FLOOR = 5000.0

        sup_act = validated_actions.get(sup.agent_id)
        if sup_act is not None:
            deltas[sup.agent_id]["executed_audit_rate"] = float(sup_act.values[0])
            deltas[sup.agent_id]["executed_fine_multiplier"] = float(sup_act.values[1])
            deltas[sup.agent_id]["executed_target_firm_ratio"] = float(sup_act.values[2])
            audit_rate = float(sup_act.values[0])
            fine_multiplier = float(sup_act.values[1])
        else:
            audit_rate = sup.audit_rate
            fine_multiplier = sup.fine_multiplier

        active_employees = [
            a for a in agents.values() 
            if isinstance(a, Employee) and a.status in [LifeCycleStatus.ACTIVE, LifeCycleStatus.INITIALIZED]
        ]
        active_firms = [
            a for a in agents.values() 
            if isinstance(a, Firm) and a.status in [LifeCycleStatus.ACTIVE, LifeCycleStatus.INITIALIZED]
        ]

        # GIỎ HÀNG SINH HỌC STONE-GEARY (Stone, 1954)
        SUBSISTENCE_BASKET_QTY = 1.0
        expected_price = max(1.0, eco.base_living_cost / SUBSISTENCE_BASKET_QTY)

        # TRẦN CHỈ SỐ HOÁ THEO GIÁ (ổn định số học -- xem chú thích đầy đủ tại nơi dùng cho
        # chi tiêu sinh tồn ở Section 4). Tính SỚM ở đây (không phải trong Section 4) vì có
        # MỘT kênh khác cũng chỉ số hoá theo expected_price CHƯA CHẶN TRẦN mà audit phát hiện:
        # lương mặc định cho lao động MỚI chưa có wage gán (Section 3, dưới) dùng thẳng
        # expected_price*1.05 -- nếu không chặn, kênh này tái tạo đúng vòng lặp phản hồi dương
        # (giá -> lương mặc định -> thu nhập -> chi tiêu -> giá) mà trần vốn dùng để cắt, chỉ
        # là qua một biến trung gian khác. Đo được: bỏ sót kênh này khiến hệ số khuếch đại trên
        # trần lệch khỏi θ=0.70 (0.71-0.86 tuỳ mức cung thay vì đúng 0.70 ở MỌI mức, xem
        # be/tests/test_env_contracts.py::test_price_indexation_loop_gain_equals_theta_above_ceiling).
        indexation_ceiling = self.subsistence_indexation_ceiling_mult * eco.initial_living_cost
        indexed_price = min(expected_price, indexation_ceiling)

        # 2. THỊ TRƯỜNG LAO ĐỘNG: ĐÀM PHÁN MRPL CÓ RÀNG BUỘC SỐNG CÒN
        ALPHA_CAPITAL = 0.3
        # BETA_LABOR = 0.6 KHỚP tỷ trọng thu nhập lao động (labor income share) quan sát thực
        # nghiệm ~0,6-0,7 ở đa số nền kinh tế (Gollin, D. (2002), "Getting Income Shares Right",
        # Journal of Political Economy 110(2), 458-474) -- GIỮ NGUYÊN khi hiệu chỉnh lại quy mô
        # MRPL (v0.19, xem self.mrpl_scale_constant), chỉ hằng số NHÂN (không phải số mũ phân
        # phối thu nhập) mới là calibration tự do trong lần sửa này.
        BETA_LABOR = 0.6
        claimed_workers = {e.agent_id for e in active_employees if e.employed_by is not None}
        shuffled_firms = list(np.random.permutation(active_firms))

        for firm in shuffled_firms:
            f_act = validated_actions.get(firm.agent_id)
            hire_signal = float(f_act.values[0]) if f_act is not None else 0.5

            current_workers = [
                e for e in active_employees 
                if (e.employed_by == firm.agent_id or e.agent_id in deltas[firm.agent_id].get("hired_employees", []))
                and e.agent_id not in deltas[firm.agent_id].get("fired_employees", [])
            ]
            current_headcount = len(current_workers)
            current_effective_labor = sum(getattr(w, 'skill_level', 1.0) for w in current_workers)
            current_wage_bill = sum(getattr(w, 'wage', expected_price * 1.1) for w in current_workers)

            # KHẤU HAO TƯ BẢN THEO CÔNG SUẤT (Jorgenson, 1963)
            # Khấu hao là chi phí nội bộ làm hao mòn vốn, KHÔNG nộp vào Kho bạc (Godley & Lavoie SFC)
            # Đơn vị: depreciation_rate * capital_stock * cash_scale
            # cash_scale = 0.01 đảm bảo overhead ~ 0.5-2 cash/month với capital=10000
            # (trước đây nhân expected_price/20 = 1.0 gây overhead=50-200, làm firm lỗ mãn tính)
            utilization = min(1.0, current_headcount / 10.0)
            depreciation_rate = 0.005 + 0.015 * utilization
            overhead_cost = depreciation_rate * firm.capital_stock * 0.01

            unemployed = [e for e in active_employees if e.agent_id not in claimed_workers]
            unemployed.sort(key=lambda w: getattr(w, 'skill_level', 1.0), reverse=True)

            safety_reserve = overhead_cost + (current_wage_bill * 1.1)
            available_liquidity = max(0.0, firm.cash - safety_reserve)

            # Ma sát tuyển dụng > ma sát sa thải (search friction bất đối
            # xứng) là hiện tượng thị trường lao động thật (Mortensen &
            # Pissarides, 1994, "Job Creation and Job Destruction", Review of
            # Economic Studies 61(3) -- đã trích ở env.py cho điều kiện gia
            # nhập ngành). TRƯỚC đây trần tuyệt đối = 2 người/tháng bất kể quy
            # mô firm, trong khi sa thải có thể xoá NGAY 50% quân số cùng lúc
            # (xem num_to_fire bên dưới) -- độ lệch pha ~25 lần giữa 2 chiều
            # là quá cực đoan để còn phản ánh đúng ma sát tìm việc thực tế
            # (phát hiện qua audit thực nghiệm, không có cơ sở lý thuyết nào
            # biện minh một trần TUYỆT ĐỐI không phụ thuộc quy mô firm). Đổi
            # sang trần TƯƠNG ĐỐI theo quy mô hiện tại (HE SO CAU TRUC TU DO
            # HIEU CHINH: 25%/tháng), có sàn 2 người để firm nhỏ/mới vẫn tuyển
            # được tốc độ tối thiểu như cũ. Đây chỉ nới TRẦN CỨNG bổ sung --
            # điều kiện khả năng chi trả THẬT (available_liquidity, kiểm tra
            # từng ứng viên trong vòng lặp bên dưới) vẫn là ràng buộc chính,
            # không đổi: firm không bao giờ tuyển vượt quá những gì nó trả nổi,
            # vòng lặp tự dừng (break) ngay khi hết thanh khoản hoặc MRPL
            # không còn biện minh được lương bảo lưu.
            MAX_HIRES_PER_MONTH = max(2, int(np.ceil(0.25 * current_headcount)))
            hires_this_month = 0

            if available_liquidity > (expected_price * 1.2) and unemployed and hire_signal > -0.1:
                for candidate in unemployed:
                    if available_liquidity <= (expected_price * 1.2) or hires_this_month >= MAX_HIRES_PER_MONTH:
                        break

                    next_l = current_effective_labor + candidate.skill_level
                    marginal_product = (
                        self.mrpl_scale_constant *
                        BETA_LABOR * firm.productivity_factor *
                        (firm.capital_stock ** ALPHA_CAPITAL) *
                        (max(0.5, next_l) ** (BETA_LABOR - 1.0)) *
                        candidate.skill_level
                    )
                    # mrpl DÙNG indexed_price (đã chặn trần), KHÔNG dùng expected_price thô.
                    # QUYẾT ĐỊNH ĐÃ CÂN NHẮC LẠI: bản đầu tiên định để mrpl dùng giá thô với lý
                    # do "phải phản ánh đúng năng suất biên thực theo giá thị trường" -- SAI khi
                    # kiểm chứng bằng số: ở giá 25 (gần đỉnh quan sát trong log train thật), với
                    # tham số điển hình (skill=1.5, capital=10000, next_l=5), mrpl = giá ×
                    # marginal_product ≈ 25 × 11,2 ≈ 281 -- gấp ~15 LẦN trần giá đã chặn (18) --
                    # tức đây mới là kênh CHÍNH gây cóc lương quan sát được trong run train thật
                    # (2026-09-22/23, lương TB tăng 8 lần trong 36 iteration), không phải
                    # reservation_wage/negotiated_wage floor (đã chặn ở lượt sửa trước nhưng
                    # KHÔNG đủ, xác nhận bằng test_wage_ratchet_converges_with_sustained_hiring).
                    # Đánh đổi: mrpl không còn phản ánh "giá thị trường thô" khi giá vượt trần,
                    # nhưng một khi trần đã tồn tại chính vì giá thô có thể trôi dạt không kiểm
                    # soát được (Cagan, 1956), để MRPL tiếp tục dùng giá thô là tự phá vỡ mục
                    # đích của trần qua một biến trung gian khác -- nhất quán với "độ cứng lương
                    # thực" (real wage rigidity) như một cơ chế ổn định hoá trước cú sốc danh
                    # nghĩa (Blanchard, O. J., & Katz, L. F. (1997), "What We Know and Do Not
                    # Know About the Natural Rate of Unemployment", Journal of Economic
                    # Perspectives 11(1), 51-72).
                    mrpl = indexed_price * marginal_product

                    # Lương bảo lưu sinh tồn (Reservation Wage): Phải đủ sống (Shapiro & Stiglitz, 1984).
                    # DÙNG indexed_price (đã chặn trần), KHÔNG dùng expected_price thô -- đây là
                    # SÀN LƯƠNG cho lao động MỚI, cùng bản chất subsistence với
                    # subsistence_nominal_need (Section 4) và lương mặc định (dòng dùng
                    # indexed_price ở trên). Phát hiện qua audit run train thật (2026-09-22/23):
                    # lương mới luôn chốt theo giá thô đang tăng, và (do lương cũ KHÔNG BAO GIỜ
                    # đàm phán lại -- wage = assigned_wage if assigned_wage > 0, xem Section 3)
                    # mỗi đợt tuyển mới "khoá" một mức lương ngày càng cao vĩnh viễn -- tạo hiệu
                    # ứng cóc lương (wage ratchet) nhiều bước, không nổ trong 1 bước nên
                    # test_price_indexation_loop_gain_equals_theta_above_ceiling KHÔNG bắt được
                    # (xem test_wage_ratchet_converges_with_sustained_hiring, tiêu chuẩn test
                    # THỨ HAI bổ sung riêng cho lớp lỗi nhiều bước này). Tái tạo lỗi cũ: đặt
                    # subsistence_indexation_ceiling_mult rất lớn (vd. 1e9) trong ScenarioConfig.
                    reservation_wage = max(indexed_price * 1.02, indexed_price * (0.8 + 0.3 * candidate.skill_level))

                    # Đàm phán Nash: Công nhân chỉ đi làm nếu lương >= chi phí sống thực tế
                    if mrpl >= reservation_wage:
                        # Sàn negotiated_wage cũng dùng indexed_price (lý do như reservation_wage
                        # ở trên) -- CHỈ sàn bị chặn, trung bình 0.5*reservation_wage + 0.5*mrpl
                        # vẫn có thể vượt sàn nếu MRPL thực (chưa chặn) đủ cao, giữ đúng tinh
                        # thần đàm phán Nash dựa trên năng suất thực.
                        negotiated_wage = max(indexed_price, 0.5 * reservation_wage + 0.5 * mrpl)

                        claimed_workers.add(candidate.agent_id)
                        current_effective_labor = next_l
                        available_liquidity -= negotiated_wage
                        hires_this_month += 1

                        deltas[firm.agent_id].setdefault("hired_employees", []).append(candidate.agent_id)
                        deltas[candidate.agent_id]["employed_by"] = firm.agent_id
                        deltas[candidate.agent_id]["wage"] = negotiated_wage
                        deltas[candidate.agent_id]["unemployed_streak"] = 0

                        self._emit_event(EventType.HIRE, firm.agent_id, candidate.agent_id, {"wage": round(negotiated_wage, 1)}, timestep)
                        events_map[firm.agent_id].append(EventType.HIRE.value)
                        events_map[candidate.agent_id].append(EventType.HIRE.value)
                    else:
                        break

            elif current_headcount > 0 and (firm.cash < overhead_cost or hire_signal < -0.4):
                num_to_fire = 1 if hire_signal >= -0.7 else max(1, current_headcount // 2)
                sorted_workers = sorted(current_workers, key=lambda w: getattr(w, 'skill_level', 1.0))

                for fired_emp in sorted_workers[:num_to_fire]:
                    deltas[firm.agent_id].setdefault("fired_employees", []).append(fired_emp.agent_id)
                    deltas[fired_emp.agent_id]["employed_by"] = None
                    deltas[fired_emp.agent_id]["wage"] = 0.0
                    claimed_workers.discard(fired_emp.agent_id)

                    self._emit_event(EventType.FIRE, firm.agent_id, fired_emp.agent_id, {"reason": "Marginal Loss"}, timestep)
                    events_map[firm.agent_id].append(EventType.FIRE.value)
                    events_map[fired_emp.agent_id].append(EventType.FIRE.value)

        # 3. SẢN XUẤT HIỆN VẬT COBB-DOUGLAS & CHI TRẢ LƯƠNG
        firm_physical_outputs: Dict[str, float] = {}
        firm_wage_bills: Dict[str, float] = {}
        firm_overheads: Dict[str, float] = {}
        worker_gross_incomes: Dict[str, float] = {e.agent_id: 0.0 for e in active_employees}

        for firm in active_firms:
            firm_workers = [
                e for e in active_employees
                if (e.employed_by == firm.agent_id or e.agent_id in deltas[firm.agent_id].get("hired_employees", []))
                and e.agent_id not in deltas[firm.agent_id].get("fired_employees", [])
            ]
            utilization = min(1.0, len(firm_workers) / 10.0)
            depreciation_rate = 0.005 + 0.015 * utilization
            overhead = depreciation_rate * firm.capital_stock * 0.01
            firm_overheads[firm.agent_id] = overhead

            # Lao dong hieu dung HIEN TAI cua firm (chi theo skill, KHONG nhan effort -- dung
            # DUNG cach tinh current_effective_labor cua Section 2 o tren, de cong thuc MRPL
            # dung cho DAM PHAN LAI LUONG duoi day nhat quan voi cong thuc dung cho tuyen moi).
            # Dung cho nguoi LAO DONG DA CO SAN (khong phai "+candidate" nhu Section 2, vi ho
            # DA nam trong tong nay roi).
            firm_current_effective_labor = sum(getattr(w, 'skill_level', 1.0) for w in firm_workers)
            hired_this_step_ids = set(deltas[firm.agent_id].get("hired_employees", []))

            wage_bill = 0.0
            effective_l = 0.0

            for emp in firm_workers:
                emp_act = validated_actions.get(emp.agent_id)
                effort = float(emp_act.values[0]) if emp_act is not None else 0.6
                assigned_wage = deltas[emp.agent_id].get("wage", emp.wage)

                # DAM PHAN LAI LUONG DINH KY (Wage Renegotiation) -- MOI (v0.28). Nguyen nhan
                # goc cua "coc luong" (xem test_wage_ratchet_converges_near_ceiling_with_mrpl_scale_fix,
                # KNOWN_PATHOLOGIES.md): luong nguoi lao dong DANG co viec KHONG BAO GIO duoc
                # dam phan lai (dong "wage = assigned_wage if assigned_wage > 0" ben duoi) --
                # mot khi tuyen luc gia cao, luong "khoa" MAI MAI o muc do, ke ca khi dieu kien
                # thi truong/nang suat firm sau nay giam xuong. Day la co che MOT CHIEU (chi tang
                # qua tuyen moi, khong bao gio giam), khac voi thi truong lao dong that co CA hai
                # chieu du don gian hoa qua "hop dong so le" (staggered contracts).
                #
                # Mo rong DUNG khung Calvo (1983) da dung cho GIA HANG HOA (xem indexed_price/
                # THETA_CALVO o Section 4) sang TIEN LUONG: Taylor, J. B. (1980), "Aggregate
                # Dynamics and Staggered Contracts", Journal of Political Economy 88(1), 1-23 --
                # hop dong luong so le, moi ky chi MOT PHAN NGAU NHIEN hop dong duoc dam phan lai,
                # khong phai toan bo cung luc. Ap dung TRUC TIEP cho macro/DSGE: Erceg, C. J.,
                # Henderson, D. W., & Levin, A. T. (2000), "Optimal Monetary Policy with Staggered
                # Wage and Price Contracts", Journal of Monetary Economics 46(2), 281-313 -- day
                # la nguon THAM CHIEU CHINH cho viec mo rong CO CHE Calvo-staggered TU gia hang
                # hoa SANG tien luong (khong phai bia moi, cung mot khung ly thuyet, khac doi
                # tuong ap dung).
                #
                # self.wage_renegotiation_prob (HE SO CAU TRUC TU DO HIEU CHINH, mac dinh 0.12 --
                # tuong duong thoi han hop dong trung binh ~1/0.12~8.3 thang, gan voi chu ky xem
                # xet luong hang nam pho bien trong thuc te, THAP HON xac suat renegotiate GIA
                # hang hoa (1-THETA_CALVO=0.30/buoc) de giu dung dac tinh "luong cung hon gia"
                # (real wage rigidity) da duoc chinh du an dung lam co so thiet ke o Section 2,
                # Blanchard & Katz 1997) -- KHONG suy ra tu Erceg-Henderson-Levin truc tiep (ho
                # uoc luong rieng cho kinh te My, khong ap dung duoc cho don vi tien te cua mo
                # hinh nay). wage_renegotiation_prob=0.0 tai tao DUNG hanh vi CU (khoa luong vinh
                # vien) -- xem ScenarioConfig.
                #
                # CHI ap dung cho lao dong DA CO SAN (khong phai vua tuyen buoc nay -- ho da nhan
                # dung negotiated_wage moi nhat tu Section 2 roi, dam phan lai ngay lap tuc la
                # thua). Renegotiate CHI doi LUONG, KHONG anh huong quyet dinh sa thai/tiep tuc
                # lam viec (quyet dinh do thuoc rieng Section 2, tranh xung dot/dem trung).
                if (assigned_wage > 0.0 and emp.agent_id not in hired_this_step_ids
                        and self.wage_renegotiation_prob > 0.0
                        and np.random.rand() < self.wage_renegotiation_prob):
                    reneg_marginal_product = (
                        self.mrpl_scale_constant *
                        BETA_LABOR * firm.productivity_factor *
                        (firm.capital_stock ** ALPHA_CAPITAL) *
                        (max(0.5, firm_current_effective_labor) ** (BETA_LABOR - 1.0)) *
                        getattr(emp, 'skill_level', 1.0)
                    )
                    reneg_mrpl = indexed_price * reneg_marginal_product
                    reneg_reservation_wage = max(indexed_price * 1.02, indexed_price * (0.8 + 0.3 * getattr(emp, 'skill_level', 1.0)))
                    assigned_wage = max(indexed_price, 0.5 * reneg_reservation_wage + 0.5 * reneg_mrpl)
                    deltas[emp.agent_id]["wage"] = assigned_wage
                    self._emit_event(EventType.WAGE_RENEGOTIATED, firm.agent_id, emp.agent_id,
                                      {"new_wage": round(assigned_wage, 1)}, timestep)
                    events_map[emp.agent_id].append(EventType.WAGE_RENEGOTIATED.value)

                # Lương mặc định cho lao động MỚI (chưa từng được gán wage) dùng indexed_price
                # (đã chặn trần), KHÔNG dùng expected_price thô -- nếu không, đây là kênh chỉ số
                # hoá theo giá thứ hai bỏ sót trần, tái tạo lại đúng vòng lặp phản hồi dương mà
                # trần ở Section 4 vốn để cắt (xem chú thích indexed_price ở đầu hàm; phát hiện
                # qua audit be/tests/test_env_contracts.py::test_price_indexation_loop_gain_equals_theta_above_ceiling).
                wage = assigned_wage if assigned_wage > 0 else (indexed_price * 1.05)

                wage_bill += wage
                worker_gross_incomes[emp.agent_id] += wage
                effective_l += (effort * emp.skill_level)

                deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + wage
                deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) - (effort * 0.10 + 0.02)
                deltas[emp.agent_id]["executed_work_effort"] = effort

                self._emit_event(EventType.WAGE_PAID, firm.agent_id, emp.agent_id, {"amount": round(wage, 1)}, timestep)
                events_map[firm.agent_id].append(EventType.WAGE_PAID.value)
                events_map[emp.agent_id].append(EventType.WAGE_PAID.value)

            # self.mrpl_scale_constant AP DUNG O DAY (giong marginal_product o Section 2) de giu
            # dung quan he dao ham marginal_product = d(physical_q)/d(effective_l) -- xem chu
            # thich day du tai __init__.
            physical_q = self.mrpl_scale_constant * firm.productivity_factor * (firm.capital_stock ** ALPHA_CAPITAL) * (effective_l ** BETA_LABOR) if effective_l > 0.0 else 0.0
            firm_physical_outputs[firm.agent_id] = physical_q
            firm_wage_bills[firm.agent_id] = wage_bill

        total_industrial_output = sum(firm_physical_outputs.values())

        # SẢN LƯỢNG TỰ TÚC PHI CHÍNH THỨC (Lewis, 1954 - Dual Sector Subsistence)
        unemployed_emps = [
            e for e in active_employees 
            if deltas[e.agent_id].get("employed_by", e.employed_by) is None
        ]
        # Hệ số 0.35 đảm bảo khu vực phi chính thức chỉ đủ sống thoi thóp, không tạo bẫy lười biếng
        informal_physical_outputs: Dict[str, float] = {
            e.agent_id: 0.35 * e.skill_level for e in unemployed_emps
        }
        total_informal_output = sum(informal_physical_outputs.values())
        total_real_supply = total_industrial_output + total_informal_output

        # 3B. TÍN DỤNG TIÊU DÙNG KHÔNG THẾ CHẤP (Unsecured Consumer Credit, v0.23)
        #
        # Zeldes, S. P. (1989), "Consumption and Liquidity Constraints: An Empirical
        # Investigation", Journal of Political Economy 97(2), 305-346 -- bằng chứng thực
        # nghiệm hộ gia đình BỊ RÀNG BUỘC THANH KHOẢN sẽ vay để làm trơn tiêu dùng qua cú
        # sốc thu nhập, nếu tiếp cận được tín dụng. Kết hợp với khung "hạn mức vay tự nhiên"
        # (natural borrowing limit, tỷ lệ thu nhập kỳ vọng) trong chính khung buffer-stock đã
        # dùng xuyên suốt file này cho Section 8B (Deaton, 1991; Carroll, 1997) -- khác Firm ở
        # Section 5 (thế chấp bằng capital_stock, Kiyotaki & Moore 1997), Employee KHÔNG có
        # tài sản vật chất để thế chấp nên hạn mức neo vào THU NHẬP, không phải tài sản.
        #
        # ĐIỀU KIỆN KÍCH HOẠT: CHỈ vay khi tài sản thanh khoản (cash + tiền gửi, đã trừ phần
        # sẽ rút để chi tiêu ở Section 4 -- nhưng Section 3B chạy TRƯỚC Section 4 nên dùng
        # đúng công thức liquid_est = cash + deposit, KHỚP với công thức Section 4/8B dùng)
        # không đủ chi tiêu sinh tồn tháng này VÀ agent chủ động phát tín hiệu muốn vay
        # (borrow_intensity > 0.4, cùng ngưỡng với Firm.borrow_signal ở Section 5) -- không
        # phải một khoản vay vô điều kiện.
        #
        # QUY MÔ KHOẢN VAY: bị chặn bởi CẢ hạn mức (thu nhập) LẪN đúng phần thiếu hụt thực sự
        # (shortfall) -- vay không bao giờ vượt quá khoảng cách còn thiếu tới mức sinh tồn,
        # nhân với borrow_intensity (agent có thể chọn vay MỘT PHẦN khoảng thiếu hụt, phần còn
        # lại chấp nhận cắt giảm tiêu dùng) và risk_discount (Kimball, 1990, cùng cơ chế đã
        # dùng cho Firm.risk_aversion ở Section 5 -- nhất quán, không bịa thêm công thức mới).
        #
        # VỠ NỢ: KHÔNG có tài sản thế chấp để thu hồi (khác Merton/Firm Section 8) -- khi
        # Employee chết còn nợ, TOÀN BỘ dư nợ thành nợ xấu 100% cho ngân hàng chủ nợ (xem
        # Section 9 dưới, nhánh "GHI NHẬN NỢ XẤU KHI EMPLOYEE CHẾT").
        K_CREDIT_INCOME_MULT = 3.0  # HỆ SỐ CẤU TRÚC TỰ DO HIỆU CHỈNH: hạn mức vay = N tháng thu nhập kỳ vọng

        for emp in active_employees:
            emp_act = validated_actions.get(emp.agent_id)
            borrow_intensity = float(emp_act.values[3]) if emp_act is not None and len(emp_act.values) > 3 else 0.0

            current_cash_est = emp.cash + deltas[emp.agent_id].get("cash_delta", 0.0)
            deposit_now = max(0.0, getattr(emp, "bank_deposit", 0.0))
            liquid_est = max(0.0, current_cash_est) + deposit_now
            subsistence_nominal_need = SUBSISTENCE_BASKET_QTY * indexed_price
            shortfall = max(0.0, subsistence_nominal_need - liquid_est)

            # Thu nhập kỳ vọng làm cơ sở hạn mức: lương THÁNG NÀY nếu có việc (worker_gross_incomes
            # đã được Section 3 điền cho người có việc); nếu thất nghiệp, worker_gross_incomes vẫn =
            # 0.0 tại ĐÚNG thời điểm này (thu nhập phi chính thức chỉ được cộng ở Section 4, chạy SAU
            # Section 3B) -- dùng lại ĐÚNG công thức thu nhập phi chính thức (0.35 * skill_level) làm
            # proxy, tính trực tiếp thay vì đọc dict chưa điền, tránh sai số do thứ tự thực thi.
            expected_monthly_income = worker_gross_incomes.get(emp.agent_id, 0.0)
            if expected_monthly_income <= 0.0:
                expected_monthly_income = 0.35 * emp.skill_level * indexed_price

            existing_debt = emp.debt
            borrowing_limit = max(0.0, K_CREDIT_INCOME_MULT * expected_monthly_income - existing_debt)

            need_credit_relationship = (existing_debt > 0.0) or (shortfall > 0.0 and borrow_intensity > 0.4)
            if not need_credit_relationship:
                continue

            # Quan hệ tín dụng (relationship banking, Petersen & Rajan, 1994) -- CÙNG mẫu hình
            # với Firm ở Section 5: giữ nguyên ngân hàng chủ nợ cho tới khi tất toán.
            existing_creditor = getattr(emp, "creditor_bank_id", None)
            if existing_creditor not in bank_lookup or emp.debt <= 0.0:
                eligible_lenders = [b for b in banks if b.reserves > RESERVE_LENDING_FLOOR]
                pool = eligible_lenders if eligible_lenders else banks
                creditor_id = min(pool, key=lambda b: executed_lending_rate[b.agent_id]).agent_id
            else:
                creditor_id = existing_creditor
            deltas[emp.agent_id]["creditor_bank_id"] = creditor_id
            creditor_bank = bank_lookup[creditor_id]
            lending_rate = executed_lending_rate[creditor_id]
            credit_factor = executed_credit_factor[creditor_id]

            if (shortfall > 0.0 and borrow_intensity > 0.4
                    and creditor_bank.reserves > RESERVE_LENDING_FLOOR and borrowing_limit > 0.0):
                risk_discount = 1.0 - 0.3 * float(np.clip(getattr(emp, "risk_aversion", 0.5), 0.0, 1.0))
                loan_request = min(borrowing_limit, shortfall * borrow_intensity * credit_factor * risk_discount)
                deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + loan_request
                deltas[emp.agent_id]["debt_delta"] = deltas[emp.agent_id].get("debt_delta", 0.0) + loan_request
                deltas[creditor_id]["loans_delta"] = deltas[creditor_id].get("loans_delta", 0.0) + loan_request
                deltas[creditor_id]["reserves_delta"] = deltas[creditor_id].get("reserves_delta", 0.0) - loan_request

                self._emit_event(EventType.LOAN_DISBURSED, creditor_id, emp.agent_id, {"amount": round(loan_request, 1)}, timestep)
                events_map[creditor_id].append(EventType.LOAN_DISBURSED.value)
                events_map[emp.agent_id].append(EventType.LOAN_DISBURSED.value)

            if emp.debt > 0.0:
                monthly_interest = emp.debt * (lending_rate / 12.0)
                principal_repayment = 0.0
                # CHỈ trả gốc khi KHÔNG còn thiếu hụt thanh khoản (sinh tồn luôn ưu tiên trước
                # trả nợ -- tránh vòng xoáy "nhịn ăn để trả nợ" mà không có cơ sở lý thuyết nào
                # biện minh, khác hẳn tinh thần Deaton/Carroll buffer-stock).
                if shortfall <= 0.0:
                    principal_repayment = min(emp.debt, emp.debt * 0.05)

                total_credit_payment = monthly_interest + principal_repayment
                deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) - total_credit_payment
                deltas[emp.agent_id]["debt_delta"] = deltas[emp.agent_id].get("debt_delta", 0.0) - principal_repayment

                deltas[creditor_id]["interest_income"] = deltas[creditor_id].get("interest_income", 0.0) + monthly_interest
                deltas[creditor_id]["reserves_delta"] = deltas[creditor_id].get("reserves_delta", 0.0) + total_credit_payment
                deltas[creditor_id]["loans_delta"] = deltas[creditor_id].get("loans_delta", 0.0) - principal_repayment

        # 4. TIÊU DÙNG & ĐỊNH GIÁ KẾT DÍNH CALVO CHUẨN (Calvo, 1983)
        total_consumer_spending = 0.0

        # TRẦN CHỈ SỐ HOÁ CHI TIÊU SINH TỒN THEO GIÁ (ổn định số học -- indexed_price đã tính
        # sẵn ở đầu hàm cùng expected_price, dùng chung với lương mặc định ở Section 3).
        # Chi tiêu sinh tồn danh nghĩa = QTY × giá kỳ trước là dạng Stone-Geary/LES
        # (Stone, R. (1954), "Linear Expenditure Systems and Demand Analysis",
        # Economic Journal 64(255)) -- giữ NGUYÊN dạng hàm. Nhưng kết hợp với
        # instant_clearing_price = chi_tiêu/cung (Calvo, 1983, phần (1-θ)) thì tổng
        # cầu danh nghĩa của những người đủ tiền tỷ lệ với P_{t-1}, nên hệ số khuếch
        # đại một bước = θ + (1-θ)·0,83·N_đủ_tiền·QTY/cung_thực. ĐO THỰC NGHIỆM trên
        # hệ thống đầy đủ (6 seed): gain ≈ 0,7 + 13,3/cung, vượt 1,0 khi sản xuất còn
        # ~12-25% baseline; khi vượt, giá tăng cấp số nhân vô hạn (đo được giá 4e10 ở
        # bước 240) -- kiểu bất ổn của thích nghi kỳ vọng (Cagan, P. (1956), "The
        # Monetary Dynamics of Hyperinflation", trong Friedman (ed.), Studies in the
        # Quantity Theory of Money, U. of Chicago Press). Lỗi tự sinh ngay cả khi tắt
        # bơm cầu và trợ cấp (đã ablate).
        # Sửa: phần CHỈ SỐ HOÁ theo giá được chặn ở TRẦN = mult × initial_living_cost.
        # Trên trần, chi tiêu sinh tồn không còn phụ thuộc P_{t-1} nên hệ số trên
        # P_{t-1} về đúng θ < 1 -> vòng lặp bị CẮT (không phải chỉ giới hạn tốc độ tăng
        # kiểu ±x%/kỳ vốn chỉ làm chậm phân kỳ). Dưới trần hành vi Stone-Geary giữ
        # nguyên. mult là HỆ SỐ HIỆU CHỈNH ổn định số học (không suy từ Stone/Cagan).
        for emp in active_employees:
            emp_act = validated_actions.get(emp.agent_id)
            consume_propensity = float(emp_act.values[2]) if emp_act is not None else 0.5

            current_cash_est = emp.cash + deltas[emp.agent_id].get("cash_delta", 0.0)
            subsistence_nominal_need = SUBSISTENCE_BASKET_QTY * indexed_price

            # CHI TIÊU TỪ TÀI SẢN THANH KHOẢN (cash-on-hand = tiền mặt + tiền gửi).
            # LỖI THIẾT KẾ ĐÃ SỬA: bản cũ chỉ tính chi tiêu trên TIỀN MẶT, trong khi
            # Mục 8B chuyển mọi phần tiền mặt vượt đệm (2 tháng sinh hoạt phí) vào tiền
            # gửi -- nên tiền gửi là "tiền chết", không bao giờ quay lại thị trường hàng
            # hoá dù chính chú thích 8B nói tiền gửi là demand deposit thanh khoản hoàn
            # toàn. Hệ quả đo được: chi tiêu hộ gia đình ~ hằng số theo đệm (≈1,2 × giá)
            # bất kể thu nhập => quỹ lương 362 nhưng chi tiêu chỉ 232, doanh thu firm
            # không bù nổi lương (tỷ lệ 0,57) => thuê luôn lỗ, sa thải hết luôn tối ưu.
            # Cơ sở lý thuyết: trong mô hình tiết kiệm đệm (Carroll, C. D. (1997),
            # "Buffer-Stock Saving and the Life-Cycle/Permanent Income Hypothesis", QJE
            # 112(1)) tiêu dùng là hàm của CASH-ON-HAND, gồm mọi tài sản thanh khoản chứ
            # không riêng tiền giao dịch; mô hình SFC chuẩn (Godley, W., & Lavoie, M.
            # (2007), "Monetary Economics", Palgrave Macmillan, Ch.3-4) cũng để tiêu dùng
            # phụ thuộc số dư tài sản tích luỹ V_{-1}. Dạng hàm chi tiêu (Stone-Geary +
            # hệ số 0,35 × propensity) GIỮ NGUYÊN, chỉ đổi cơ sở tính từ tiền mặt sang
            # cash-on-hand.
            # SFC: phần chi vượt tiền mặt được RÚT từ tiền gửi (chuyển nội bộ trong khu
            # vực hộ gia đình, tiền gửi giảm bao nhiêu thì tiền mặt tăng bấy nhiêu rồi chi
            # cho khu vực sản xuất); ghi vào pre_spend_withdrawal để Mục 8B trừ khỏi số dư
            # gửi đầu kỳ, và trừ vào deposits_delta của ĐÚNG ngân hàng đang giữ số dư.
            deposit_now = max(0.0, getattr(emp, "bank_deposit", 0.0))
            liquid_est = max(0.0, current_cash_est) + deposit_now

            if liquid_est >= subsistence_nominal_need:
                surplus_cash = liquid_est - subsistence_nominal_need
                spending = subsistence_nominal_need + (surplus_cash * 0.35 * consume_propensity)
            else:
                spending = liquid_est

            pre_spend_withdrawal = min(deposit_now, max(0.0, spending - max(0.0, current_cash_est)))
            deltas[emp.agent_id]["pre_spend_withdrawal"] = pre_spend_withdrawal
            if pre_spend_withdrawal > 0.0:
                depository = getattr(emp, "depository_bank_id", None)
                if depository in bank_lookup:
                    deltas[depository]["deposits_delta"] = deltas[depository].get("deposits_delta", 0.0) - pre_spend_withdrawal

            deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + pre_spend_withdrawal - spending
            deltas[emp.agent_id]["executed_consumption"] = spending
            total_consumer_spending += spending

            self._emit_event(EventType.GOODS_PURCHASED, emp.agent_id, eco.agent_id, {"amount": round(spending, 1)}, timestep)

        # Chi tiêu của HỘ GIA ĐÌNH TRƯỚC khi cộng bơm cầu tài khoá -- dùng làm khối
        # lượng giao dịch (total_market_turnover) cho reward của Economy. Cố ý KHÔNG
        # gồm khoản bơm của chính Economy (Section 4B): nếu tính cả, Economy có thể
        # tự nâng "khối lượng" thưởng cho mình bằng cách bơm tối đa (reward hacking).
        household_consumer_spending = total_consumer_spending

        # 4B. BƠM/RÚT CẦU TÀI KHÓA PHẢN CHU KỲ CỦA CHÍNH PHỦ QUA ECONOMY
        # (Countercyclical Fiscal Demand Injection) -- Blanchard, O., &
        # Perotti, R. (2002), "An Empirical Characterization of the Dynamic
        # Effects of Changes in Government Spending and Taxes on Output",
        # Quarterly Journal of Economics 117(4), 1329-1368.
        #
        # THIẾT KẾ LẠI hành động chiều [0] của Economy (trước là
        # "living_cost_factor" nhân thẳng vào base_living_cost qua key
        # executed_cost_factor -- key này KHÔNG BAO GIỜ được RuleEngine ghi
        # nên hành động này LUÔN LÀ NO-OP, phát hiện qua audit hardcode
        # formula). Phương án thay thế ban đầu được cân nhắc là "bơm thêm
        # CUNG" (total_real_supply) nhưng bị loại vì SAI CHIỀU: bơm thêm cung
        # trong khi cầu không đổi chỉ khiến instant_clearing_price =
        # spending/supply giảm SÂU HƠN -- phản tác dụng đúng với vấn đề giảm
        # phát cần giải quyết. Đòn bẩy đúng phải tác động qua phía CẦU, đúng
        # cơ chế chi tiêu chính phủ phản chu kỳ chuẩn Keynes (G trong IS-LM).
        #
        # Quy mô bơm SCALE theo tổng cầu-sinh-tồn CƠ SỞ của dân số đang hoạt
        # động (KHÔNG scale theo chính total_consumer_spending hiện tại) --
        # nếu scale theo total_consumer_spending, trong khủng hoảng giảm
        # phát/thất nghiệp khi spending đã co lại gần 0, lượng bơm cũng tiến
        # về 0 theo, tự triệt tiêu đúng lúc cần dùng nhất.
        #
        # === LỖI VÒNG LẶP PHẢN HỒI DƯƠNG ĐÃ PHÁT HIỆN + SỬA (2026-09-21) ===
        # Bản đầu tiên dùng "expected_price" (giá SỐNG, đang biến động mỗi kỳ)
        # làm mốc quy đổi -- gây nổ số học thật trong lần train đầu tiên (GDP/
        # Treasury vọt lên hàng trăm nghìn tỷ chỉ sau ~10 iteration). Nguyên
        # nhân (phân tích loop-gain): market_clearing_price = θ·expected_price
        # + (1-θ)·instant_clearing_price, và instant_clearing_price chứa
        # demand_injection_base ∝ expected_price -- khiến hệ số nhân lên
        # expected_price kỳ trước = θ + (1-θ)·ratio·SUBSISTENCE_QTY·N/supply.
        # Hệ số này VƯỢT 1 (phân kỳ) khi SUBSISTENCE_QTY·N/supply > 1/ratio --
        # đúng lúc thất nghiệp cao (supply co lại vì ít người sản xuất) trong
        # khi N (khi đó là toàn bộ dân số) không co theo -- nghịch lý: cơ chế
        # sinh ra để chống khủng hoảng thất nghiệp lại mất ổn định NHẤT đúng
        # lúc thất nghiệp cao. Thêm trần %/kỳ (kiểu ±10%) KHÔNG giải quyết được
        # gốc rễ -- chỉ làm chậm quá trình phân kỳ (vd. ±10%/kỳ vẫn cho ra
        # 1.1^100 ≈ 13,781 lần sau 100 kỳ nếu hệ số nhân thật > 1.10).
        #
        # SỬA TẬN GỐC: (1) neo bằng "initial_living_cost" -- hằng số CỐ ĐỊNH
        # chụp tại lúc reset episode (economy.py), hoàn toàn KHÔNG phụ thuộc
        # expected_price đang chạy -- loại bỏ P_{t-1} khỏi vế injection, đưa
        # hệ số nhân về ĐÚNG θ=0.70 < 1, ổn định vô điều kiện bất kể N/supply/
        # ratio (verify bằng test_injection_loop_gain.py, đo trực tiếp đạo hàm
        # số học ở nhiều mốc thất nghiệp). (2) đổi N từ "toàn bộ dân số hoạt
        # động" sang SỐ NGƯỜI THẤT NGHIỆP (unemployed_emps, đã tính ở trên) --
        # đúng bản chất kinh tế của cơ chế: đây là khoản cứu trợ/kích cầu bù
        # đắp thu nhập bị MẤT do thất nghiệp (Blanchard & Perotti, 2002),
        # không phải trợ cấp đồng đều theo đầu người bất kể có việc hay không
        # -- quy mô bơm tự động tăng đúng lúc cần (thất nghiệp cao), giảm về 0
        # khi toàn dụng lao động, đúng tính chất automatic stabilizer.
        #
        # === CHUYỂN CHỦ THỂ QUYẾT ĐỊNH TỪ ECONOMY SANG GOVERNMENT (v0.22) ===
        # Phát hiện qua "audit tính mạch lạc kinh tế tổng thể" (2026-09-23, đề xuất của
        # Claude Web, xem KNOWN_PATHOLOGIES.md mục mới): hành động này TRƯỚC ĐÂY do Economy
        # quyết định (tự nhận là "Market Maker" -- định giá/quản lý nhà đất, KHÔNG phải cơ
        # quan tài khoá), trong khi Section 4C ngay dưới (G, cùng chi từ Kho bạc) lại do
        # Government quyết định -- HAI tác tử RL độc lập, reward khác hẳn nhau, không phối
        # hợp, cùng tác động lên MỘT ngân sách. Vi phạm trực tiếp nguyên tắc hai tầng của AI
        # Economist (Zheng et al., 2022) đã ghi trong CLAUDE.md ("chỉ Government mới mang
        # trách nhiệm phúc lợi xã hội qua chính sách"). Đối chiếu dữ liệu train thật (checkpoint
        # iter_40, 40 iteration): reward Government dao động mạnh (biên độ ~40, 9 lần đổi chiều
        # xu hướng) trong khi Economy hội tụ phẳng (biên độ ~3) -- không cô lập được nguyên
        # nhân do run đó còn nhiễu bởi lỗi zombie-firm (mục #13) và có thể cả các lệch chuẩn
        # định cỡ khác đã sửa sau đó, nhưng KHÔNG mâu thuẫn với giả thuyết bất ổn do 2 policy
        # không phối hợp cùng tranh chấp một ngân sách (non-stationarity kinh điển trong MARL).
        # Người dùng xác nhận: (1) không muốn một tác tử "ngân hàng trung ương" (giữ nguyên mô
        # hình nhiều Bank cạnh tranh), (2) ý định thiết kế GỐC của Economy là đại diện thị
        # trường/giá cả chung (chi phí sinh hoạt, giá nhà, giá đất -- những thứ Gov/Bank không
        # trực tiếp kiểm soát), KHÔNG phải một chủ thể chi ngân sách. Sửa: chuyển hẳn action
        # này sang Government (action[3], xem government.py::decide()) -- Economy quay về đúng
        # 100% vai trò giá cả/thị trường ban đầu (2 chiều housing hiện có sẵn ở economy.py vẫn
        # là placeholder chờ housing epic, không đổi).
        demand_injection_ratio = float(gov_act.values[3]) if gov_act is not None else 0.0
        demand_injection_base = SUBSISTENCE_BASKET_QTY * eco.initial_living_cost * max(1, len(unemployed_emps))
        total_consumer_spending_after_injection = max(0.0, total_consumer_spending + (demand_injection_ratio * demand_injection_base))
        # Chênh lệch THỰC TẾ (sau khi kẹp sàn 0) là số tiền Kho bạc phải chi
        # trả (bơm dương -> firm/khu vực phi chính thức nhận thêm doanh thu)
        # hoặc thu về (bơm âm -> thắt chi tiêu) -- đảm bảo đẳng thức SFC: phần
        # cầu tăng/giảm thêm PHẢI có nguồn đối ứng ở Treasury (xem
        # Government.apply_result), không tự sinh/mất tiền.
        demand_injection_effect = total_consumer_spending_after_injection - total_consumer_spending
        total_consumer_spending = total_consumer_spending_after_injection
        deltas[gov.agent_id]["demand_injection_ratio"] = demand_injection_ratio
        deltas[gov.agent_id]["demand_injection_effect"] = demand_injection_effect
        deltas[gov.agent_id]["demand_injection_cost"] = deltas[gov.agent_id].get("demand_injection_cost", 0.0) + demand_injection_effect

        # 4C. CHI TIÊU MUA HÀNG CỦA CHÍNH PHỦ (Government Purchases, G) -- ĐÓNG VÒNG CHU CHUYỂN.
        # Mô hình SFC tối giản SIM (Godley, W., & Lavoie, M. (2007), "Monetary Economics",
        # Palgrave Macmillan, Ch.3) đóng vòng chu chuyển bằng đúng bước này: Chính phủ thu
        # thuế T rồi CHI MUA HÀNG HOÁ G, nên tiền thuế quay lại doanh nghiệp (Y = C + G).
        # Mô hình này thiếu G: thuế chảy vào Kho bạc rồi nằm im (chỉ có trợ cấp thất nghiệp
        # và bơm cầu ±20% cỡ nhỏ), khiến doanh thu firm không bù nổi quỹ lương (đo được tỷ lệ
        # 0,57) và chính sách học được là SA THẢI HẾT (reward Firm -0,17 thắng THUÊ -0,59; xem
        # CLAUDE_HISTORY.md v0.16). Xem thêm Section 3-4: chỉ đóng vòng khi kết hợp với chi
        # tiêu từ tài sản thanh khoản ở trên (đo: riêng G hay riêng L đều CHƯA đủ, L+G thì
        # thuê thắng sa thải, giá ổn định ~16-17, 5/5 firm sống sót).
        #
        # CHI CHO AI, TIỀN ĐI ĐÂU: Chính phủ mua hàng trên THỊ TRƯỜNG chung: khoản G cộng vào
        # tổng cầu total_consumer_spending và được chia cho người sản xuất THEO TỶ TRỌNG SẢN
        # LƯỢNG THỰC như mọi khoản chi khác (industrial_revenue_pool / informal_revenue_pool
        # ở dưới) -- nên firm sản xuất nhiều hơn nhận nhiều hơn, KHÔNG phải trợ cấp vô điều
        # kiện gắn với hiện diện của firm. Tiền đi: Kho bạc -> doanh thu firm / thu nhập phi
        # chính thức của người sản xuất (chuyển giữa hai khu vực, bảo toàn SFC; phần này lại
        # chịu thuế thu nhập ở Mục 6 nên một phần quay về Kho bạc kỳ sau).
        # BAO NHIÊU: G_t = ρ_t × (thuế + tiền phạt kỳ TRƯỚC), ρ ∈ [0,1] là hành động của
        # Chính phủ (chiều [2]), mặc định 1 = ngân sách cân bằng (SIM: G = T). Dùng số thu
        # kỳ trước vì thuế kỳ này chỉ được tính SAU khi chi tiêu đã chốt (Mục 6 chạy sau
        # Mục 4); đây cũng là độ trễ chuẩn trong SFC. G tự co lại khi sản xuất/việc làm sụp
        # đổi (thuế giảm) nên KHÔNG thưởng cho việc ngừng sản xuất. G bị chặn trần bằng số dư
        # Kho bạc hiện có (không chi vượt quỹ, không tự tạo nợ công/tiền). ρ bị giới hạn
        # ≤ 1 nên G/GDP tự xuất hiện ~ tỷ lệ thuế/GDP (đo được 10-23%, trung bình ~15%, nằm
        # trong khoảng 15-25% GDP của chi tiêu chính phủ các nền kinh tế theo dữ liệu World
        # Bank -- KHÔNG được chỉnh cho khớp).
        purchase_ratio = float(gov_act.values[2]) if gov_act is not None else 1.0
        purchase_base = max(0.0, gov.last_tax_collected + gov.last_fines_collected)
        government_purchases = min(purchase_ratio * purchase_base, max(0.0, gov.treasury))
        total_consumer_spending += government_purchases
        deltas[gov.agent_id]["government_purchase_cost"] = government_purchases
        deltas[gov.agent_id]["executed_purchase_ratio"] = purchase_ratio

        # 4D. BÌNH ỔN THỊ TRƯỜNG BẰNG DỰ TRỮ ĐỆM (Buffer-Stock Market Stabilization, v0.24)
        # Newbery, D. M. G., & Stiglitz, J. E. (1981), "The Theory of Commodity Price
        # Stabilization: A Study in the Economics of Risk", Oxford University Press;
        # Knudsen, O., & Nash, J. (1990), "Domestic Price Stabilization Schemes in
        # Developing Countries", Economic Development and Cultural Change 38(3), 539-558.
        # Xem METHODOLOGY_NOTES.md mục 2 cho toàn bộ phân tích gain/kiểm chứng an toàn đã
        # làm TRƯỚC khi viết đoạn này (đại số + thực nghiệm cô lập, nhiều mức cung, test
        # đối kháng "luôn mua"/"luôn bán", test tương tác với Section 4B).
        #
        # Economy giữ MỘT bảng cân đối kế toán riêng (KHÔNG phải Treasury): eco.strategic_
        # reserve_fund (tiền) + eco.strategic_reserve_stock (hàng thiết yếu, đơn vị vật lý).
        # Hành động intervention_intensity ∈ [-1,1]: dương = MUA (hỗ trợ giá khi giảm phát),
        # âm = BÁN (hạ giá khi lạm phát) -- phản hồi ÂM (ổn định) về bản chất.
        #
        # QUY MÔ can thiệp neo vào eco.initial_living_cost (hằng số CỐ ĐỊNH lúc reset episode),
        # TUYỆT ĐỐI KHÔNG dùng expected_price/market_clearing_price (giá SỐNG) -- đúng bài học
        # từ 2 lần lỗi vòng lặp phản hồi dương trước đó (bơm cầu v0.12, sàn lương v0.18/19, xem
        # KNOWN_PATHOLOGIES.md #2, #8). Chiều BÁN quy đổi lượng hàng CŨNG dùng initial_living_cost
        # (không dùng expected_price) -- bản nháp đầu tiên dùng expected_price cho chiều bán và
        # ĐÃ ĐO ĐƯỢC gain lệch dần khỏi theta khi cung khan hiếm (0.70->0.84 ở cung 8%), sửa
        # trước khi đưa vào đây (xem METHODOLOGY_NOTES.md).
        #
        # Chiều MUA CHỈ cộng vào total_consumer_spending (cạnh tranh cầu, giống hộ gia đình/G) --
        # KHÔNG trừ total_real_supply -- nên về cấu trúc KHÔNG THỂ tự đẩy cung xuống ngưỡng nguy
        # hiểm của kênh tiêu dùng (đã verify bằng số, xem METHODOLOGY_NOTES.md). Chiều BÁN cộng
        # vào total_real_supply (giải phóng tồn kho, tăng cung hiệu dụng thật).
        # Cả hai chiều tự giới hạn bởi ràng buộc tài chính/vật lý thật (không mua quá vốn, không
        # bán quá tồn kho) -- không cần thêm logic chặn nào khác.
        eco_act = validated_actions.get(eco.agent_id)
        intervention_intensity = float(np.clip(eco_act.values[0], -1.0, 1.0)) if eco_act is not None else 0.0
        buffer_intervention_base = SUBSISTENCE_BASKET_QTY * eco.initial_living_cost * max(1, len(active_employees))
        nominal_intervention = intervention_intensity * buffer_intervention_base

        buffer_buy_spend = 0.0
        buffer_sell_qty = 0.0
        if nominal_intervention > 0.0:
            buffer_buy_spend = min(nominal_intervention, eco.strategic_reserve_fund)
            total_consumer_spending += buffer_buy_spend
        elif nominal_intervention < 0.0:
            desired_qty = abs(nominal_intervention) / max(0.5, eco.initial_living_cost)
            buffer_sell_qty = min(desired_qty, eco.strategic_reserve_stock)
            total_real_supply += buffer_sell_qty

        # CÂN BẰNG GIÁ CALVO (Calvo, 1983 Staggered Price Setting):
        # P*_t: Giá cân bằng Walras tức thời nếu 100% doanh nghiệp đổi giá
        instant_clearing_price = total_consumer_spending / max(1.0, total_real_supply)
        # theta = 0.70: Tỷ lệ hợp đồng giữ nguyên giá cũ (Calvo price stickiness)
        THETA_CALVO = 0.70
        market_clearing_price = float(THETA_CALVO * expected_price + (1.0 - THETA_CALVO) * instant_clearing_price)
        market_clearing_price = max(1.0, market_clearing_price)

        # 4D (tiếp): chốt sổ kho/quỹ của Economy SAU khi giá đã chốt (tránh vòng lặp đồng thời --
        # qty_bought là đại lượng PHÁI SINH sau giá, không phải đầu vào của chính công thức giá).
        #
        # LOI DA SUA (v0.24, phat hien qua chinh be/tests/test_sfc_random_policy.py -- KHONG
        # phai gia dinh, la loi that): ban dau chieu BAN cong thang "qty_sold * market_clearing_
        # price" vao quy Economy nhu MOT KHOAN THU DOC LAP -- nhung khong co tac tu nao khac bi
        # TRU tien tuong ung (total_real_supply da tang them buffer_sell_qty, nhung industrial_
        # revenue_pool/informal_revenue_pool o Section duoi VAN chia theo dung ty trong tren
        # total_real_supply MOI -- nghia la tong 2 pool do < total_consumer_spending, phan con
        # thieu "boc hoi" khong ai nhan, DONG THOI Economy lai duoc cong them mot khoan MOI hoan
        # toan tach biet). Hai loi nay cong lai gay ro ri SFC that (do duoc bien do 15-100 khi
        # chay be/tests/test_sfc_random_policy.py, dau/do lon thay doi ngau nhien tuy hanh dong).
        # SUA DUNG: chieu BAN phai nhan DUNG phan chia theo ty trong dong gop vao total_real_
        # supply, y het co che Firm/khu vuc phi chinh thuc da dung (Section duoi) -- dam bao
        # TONG 3 phan (industrial + informal + buffer) cong dung bang total_consumer_spending,
        # khong con phan nao "boc hoi" hay duoc tao them tu hu khong.
        buffer_fund_delta = 0.0
        buffer_stock_delta = 0.0
        if buffer_buy_spend > 0.0:
            buffer_fund_delta = -buffer_buy_spend
            buffer_stock_delta = buffer_buy_spend / market_clearing_price
            self._emit_event(EventType.GOODS_PURCHASED, eco.agent_id, eco.agent_id, {"amount": round(buffer_buy_spend, 1), "reason": "buffer_stock_buy"}, timestep)
        elif buffer_sell_qty > 0.0:
            buffer_revenue_pool = total_consumer_spending * (buffer_sell_qty / max(1.0, total_real_supply))
            buffer_stock_delta = -buffer_sell_qty
            buffer_fund_delta = buffer_revenue_pool
            self._emit_event(EventType.GOODS_PURCHASED, eco.agent_id, eco.agent_id, {"amount": round(buffer_fund_delta, 1), "reason": "buffer_stock_sell"}, timestep)
        deltas[eco.agent_id]["strategic_reserve_fund_delta"] = buffer_fund_delta
        deltas[eco.agent_id]["strategic_reserve_stock_delta"] = buffer_stock_delta
        deltas[eco.agent_id]["executed_intervention_intensity"] = intervention_intensity

        # PHÂN BỔ DOANH THU KHÉP KÍN 100% SFC (Godley & Lavoie, 2007)
        industrial_revenue_pool = total_consumer_spending * (total_industrial_output / max(1.0, total_real_supply))
        informal_revenue_pool = total_consumer_spending * (total_informal_output / max(1.0, total_real_supply))

        total_industrial_revenue = 0.0
        for firm in active_firms:
            output_q = firm_physical_outputs.get(firm.agent_id, 0.0)
            firm_revenue = industrial_revenue_pool * (output_q / total_industrial_output) if total_industrial_output > 0 else 0.0
            wage_bill = firm_wage_bills.get(firm.agent_id, 0.0)
            overhead = firm_overheads.get(firm.agent_id, 0.0)
            net_profit = firm_revenue - wage_bill - overhead

            deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) + net_profit
            deltas[firm.agent_id]["executed_revenue"] = firm_revenue
            deltas[firm.agent_id]["executed_profit"] = net_profit
            total_industrial_revenue += firm_revenue
            # LƯU Ý: Khấu hao overhead là tổn thất tư bản nội bộ, TUYỆT ĐỐI KHÔNG cộng vào gov.treasury

        # Thu nhập phi chính thức đưa vào tổng thu nhập để chịu thuế toàn diện (Allingham & Sandmo, 1972)
        total_informal_revenue = 0.0
        for emp_id, q_inf in informal_physical_outputs.items():
            emp_informal_income = informal_revenue_pool * (q_inf / total_informal_output) if total_informal_output > 0 else 0.0
            worker_gross_incomes[emp_id] += emp_informal_income
            deltas[emp_id]["cash_delta"] = deltas[emp_id].get("cash_delta", 0.0) + emp_informal_income
            total_informal_revenue += emp_informal_income

        # CẬP NHẬT KINH TẾ VĨ MÔ
        actual_living_cost = SUBSISTENCE_BASKET_QTY * market_clearing_price
        calvo_inflation = float((market_clearing_price - expected_price) / max(0.5, expected_price))
        eco.inflation_rate = calvo_inflation
        eco.base_living_cost = float(actual_living_cost)
        deltas[eco.agent_id]["liquidity_delta"] = total_consumer_spending
        # KHOÁ NÀY ĐÃ BỊ XOÁ NHẦM ở commit 3243cbd (13/9) khi refactor rule_engine, trong
        # khi Economy.apply_result vẫn đọc nó -> step_trade_volume ≡ 0, obs[5] ≡ 0 và
        # thành phần thưởng "khối lượng thực" của Economy chết (reward luôn ≤ 0).
        # Khôi phục: khối lượng danh nghĩa giao dịch của hộ gia đình (xem trên).
        deltas[eco.agent_id]["total_market_turnover"] = household_consumer_spending
        deltas[eco.agent_id]["inflation"] = calvo_inflation
        deltas[eco.agent_id]["base_living_cost"] = actual_living_cost
        # Khấu hao tư bản (overhead, Jorgenson 1963) là chi phí thực bị TRỪ khỏi
        # cash của firm mà KHÔNG chuyển cho bất kỳ tác tử nào khác trong mô
        # phỏng (đại diện cho chi phí mua sắm/bảo trì từ khu vực bên ngoài
        # không được mô hình hoá tường minh -- xem trích dẫn Godley & Lavoie ở
        # đầu file). Đây là kênh "rò rỉ" DUY NHẤT còn lại được phép trong đẳng
        # thức bảo toàn SFC ngoài DefaultedDebt; được phơi bày tường minh ở đây
        # để be/tests/test_sfc_accounting.py có thể đối chiếu chính xác thay vì
        # phải suy diễn ngược từ observable bên ngoài.
        deltas[eco.agent_id]["capital_depreciation_cost"] = sum(firm_overheads.values())

        # AN SINH XÃ HỘI CÓ THỜI HẠN (Tránh Bẫy Phúc lợi - Welfare Trap)
        total_subsidies_spent = 0.0
        for emp in unemployed_emps:
            streak = getattr(emp, "unemployed_streak", 0) + 1
            deltas[emp.agent_id]["unemployed_streak"] = streak
            deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) - min(0.08, 0.02 * streak)

            current_cash_est = emp.cash + deltas[emp.agent_id].get("cash_delta", 0.0)
            # Điều kiện đủ điều kiện nhận trợ cấp xét theo TÀI SẢN THANH KHOẢN (tiền mặt +
            # số dư gửi còn lại sau khi đã rút để chi tiêu, xem Mục 4): nếu chỉ xét tiền mặt,
            # hộ gia đình vừa chi tiêu từ tiền gửi (tiền mặt ~0) sẽ bị coi nhầm là nghèo và
            # nhận trợ cấp dù còn tiền gửi lớn.
            current_cash_est = current_cash_est + max(
                0.0, getattr(emp, "bank_deposit", 0.0) - deltas[emp.agent_id].get("pre_spend_withdrawal", 0.0)
            )
            if current_cash_est < (0.8 * actual_living_cost) and gov.treasury > 1000.0:
                # Trợ cấp giảm dần theo thời gian thất nghiệp (Benefit Cliff)
                relief_amount = 0.40 * actual_living_cost if streak <= 3 else (0.15 * actual_living_cost if streak <= 6 else 0.0)
                deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + relief_amount
                total_subsidies_spent += relief_amount

        deltas[gov.agent_id]["subsidies_disbursed"] = total_subsidies_spent

        # Hồi phục thể lực sinh học có kẹp biên vật lý [0.0, max_energy] (Becker, 1965)
        for emp in active_employees:
            spending = deltas[emp.agent_id].get("executed_consumption", 0.0)
            real_goods_bought = spending / max(0.1, market_clearing_price)
            if real_goods_bought >= SUBSISTENCE_BASKET_QTY:
                energy_rec = min(0.35, 0.20 + 0.10 * (real_goods_bought - 1.0))
            else:
                deficit_ratio = (SUBSISTENCE_BASKET_QTY - real_goods_bought) / SUBSISTENCE_BASKET_QTY
                energy_rec = max(-0.20, 0.15 * real_goods_bought - 0.20 * deficit_ratio)
            deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) + energy_rec

        # 5. TÍN DỤNG THẾ CHẤP NỘI SINH (Kiyotaki & Moore, 1997)
        # RESERVE_LENDING_FLOOR đã hoist lên đầu hàm (dùng chung với Section 3B).
        for firm in active_firms:
            f_act = validated_actions.get(firm.agent_id)
            borrow_signal = float(f_act.values[1]) if f_act is not None else 0.0

            # Quan hệ tín dụng (relationship banking -- Petersen & Rajan, 1994):
            # nếu doanh nghiệp CHƯA có ngân hàng chủ nợ, hoặc đã tất toán nợ cũ
            # (debt <= 0, tự do chọn lại), doanh nghiệp chọn ngân hàng có
            # lending_rate thấp nhất trong số các ngân hàng còn đủ dự trữ cho
            # vay; nếu đang có quan hệ tín dụng và còn dư nợ, DUY TRÌ đúng ngân
            # hàng đó cho tới khi tất toán -- tránh phân mảnh một khoản nợ ra
            # nhiều ngân hàng cùng lúc, đúng thực tế quan hệ tín dụng dài hạn.
            existing_creditor = getattr(firm, "creditor_bank_id", None)
            if existing_creditor not in bank_lookup or firm.debt <= 0.0:
                eligible_lenders = [b for b in banks if b.reserves > RESERVE_LENDING_FLOOR]
                pool = eligible_lenders if eligible_lenders else banks
                creditor_id = min(pool, key=lambda b: executed_lending_rate[b.agent_id]).agent_id
            else:
                creditor_id = existing_creditor
            deltas[firm.agent_id]["creditor_bank_id"] = creditor_id
            creditor_bank = bank_lookup[creditor_id]
            lending_rate = executed_lending_rate[creditor_id]
            credit_factor = executed_credit_factor[creditor_id]

            # Kiyotaki & Moore (1997): collateral value phụ thuộc giá thị trường hiện tại
            # Khi giá tăng -> tài sản thế chấp tăng -> vay được nhiều hơn (credit amplifier)
            collateral_value = 0.50 * market_clearing_price * firm.capital_stock * 0.01
            borrowing_headroom = max(0.0, collateral_value - firm.debt)

            if borrow_signal > 0.4 and creditor_bank.reserves > RESERVE_LENDING_FLOOR and borrowing_headroom > 0.0:
                # He so han che vay THEO DAC DIEM E NGAI RUI RO NOI TAI cua firm (v0.20,
                # xem KNOWN_PATHOLOGIES.md). TRUOC BAN VA NAY: firm.risk_aversion duoc
                # khoi tao/ke thua qua sinh san (env.py) nhung KHONG anh huong bat ky
                # cong thuc kinh te nao -- phat hien qua audit toan du an.
                # GHI CHU PHAM VI TRICH DAN (sua sau phan bien cua Claude Web,
                # 2026-09-23 -- ban dau trich Froot, Scharfstein & Stein (1993) nhung
                # bi chi ra SAI CO CHE: FSS (1993) mo hinh hoa CHI PHI LOI CUA VON HUY
                # DONG BEN NGOAI (convex cost of external finance) va dong luc HEDGING
                # (phai sinh) de tranh underinvestment khi co hoi dau tu tuong quan voi
                # dong tien noi bo -- KHONG phai mo hinh "risk_aversion lam giam tuyen
                # tinh muc vay". Dung dung mot trich dan DUY NHAT cho ca Employee lan
                # Firm o day: Kimball, M. S. (1990), "Precautionary Saving in the Small
                # and in the Large", Econometrica 58(1), 53-73 -- dong co phong ngua
                # (precautionary motive) truoc bat dinh thu nhap/chi phi tuong lai ap
                # dung CHUNG cho bat ky tac tu ra quyet dinh tai chinh nao (ho gia dinh
                # LAN chu doanh nghiep, vi trong mo hinh don gian hoa nay Firm thuc chat
                # la MOT nguoi ra quyet dinh duy nhat gan risk_aversion) -- day la MO
                # RONG/tuong tu hoa (analogy) dong co Kimball sang quyet dinh don bay
                # doanh nghiep, KHONG PHAI ap dung truc tiep mot mo hinh tai chinh doanh
                # nghiep cu the nao. He so 0.3 la HE SO CAU TRUC TU DO HIEU CHINH (bien
                # do giam vay toi da ~30% o risk_aversion=1.0), khong suy ra tu Kimball.
                risk_discount = 1.0 - 0.3 * float(np.clip(getattr(firm, "risk_aversion", 0.3), 0.0, 1.0))
                loan_request = min(borrowing_headroom, 1000.0 * borrow_signal * credit_factor * risk_discount)
                deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) + loan_request
                deltas[firm.agent_id]["debt_delta"] = deltas[firm.agent_id].get("debt_delta", 0.0) + loan_request
                deltas[creditor_id]["loans_delta"] = deltas[creditor_id].get("loans_delta", 0.0) + loan_request
                deltas[creditor_id]["reserves_delta"] = deltas[creditor_id].get("reserves_delta", 0.0) - loan_request

                self._emit_event(EventType.LOAN_DISBURSED, creditor_id, firm.agent_id, {"amount": round(loan_request, 1)}, timestep)
                events_map[creditor_id].append(EventType.LOAN_DISBURSED.value)
                events_map[firm.agent_id].append(EventType.LOAN_DISBURSED.value)

            if firm.debt > 0.0:
                monthly_interest = firm.debt * (lending_rate / 12.0)
                principal_repayment = 0.0
                overhead = firm_overheads.get(firm.agent_id, 0.0)
                if deltas[firm.agent_id].get("executed_profit", 0.0) > 0 and firm.cash > overhead:
                    principal_repayment = min(firm.debt, firm.debt * 0.05)

                total_bank_payment = monthly_interest + principal_repayment
                deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) - total_bank_payment
                deltas[firm.agent_id]["debt_delta"] = deltas[firm.agent_id].get("debt_delta", 0.0) - principal_repayment

                deltas[creditor_id]["interest_income"] = deltas[creditor_id].get("interest_income", 0.0) + monthly_interest
                deltas[creditor_id]["reserves_delta"] = deltas[creditor_id].get("reserves_delta", 0.0) + total_bank_payment
                deltas[creditor_id]["loans_delta"] = deltas[creditor_id].get("loans_delta", 0.0) - principal_repayment

        # 5B. TRẢ NỢ CỨU TRỢ KHẨN CẤP (Bailout Debt Servicing) -- Bagehot, W. (1873),
        # "Lombard Street" -- vế "at a HIGH RATE" của học thuyết lender-of-last-resort
        # (xem chú thích đầy đủ tại env.py::step(), nhánh "NGƯỜI CHO VAY CUỐI CÙNG", và
        # Bank.__init__::bailout_penalty_rate). Cùng CẤU TRÚC với vòng lặp trả nợ Firm ở
        # Section 5 ngay trên (lãi + trả gốc 5%/tháng khi đủ khả năng) -- ĐÚNG NGHĨA Bank
        # lúc này là bên ĐI VAY (con nợ) của Kho bạc, không phải bên cho vay. Lãi phạt
        # chảy THẲNG về Kho bạc (không tạo/huỷ tiền, chỉ chuyển hướng dòng tiền vốn đã
        # tồn tại -- bảo toàn SFC), qua kênh "bailout_repayment" mà Government.apply_result
        # cộng vào net_budget cùng cách xử lý "fines_collected".
        for b in banks:
            if b.bailout_debt <= 0.0:
                continue
            monthly_penalty_interest = b.bailout_debt * (b.bailout_penalty_rate / 12.0)
            principal_repayment = 0.0
            # Chỉ trả bớt gốc khi Bank còn dư dự trữ AN TOÀN (trên yêu cầu dự trữ bắt
            # buộc) sau khi đã trả lãi phạt -- tránh Bank tự đẩy mình xuống dưới ngưỡng
            # dự trữ bắt buộc chỉ để trả nợ nhanh.
            required_reserves = b.total_deposits * b.reserve_requirement_ratio
            if (b.reserves - monthly_penalty_interest) > required_reserves:
                principal_repayment = min(b.bailout_debt, b.bailout_debt * 0.05)

            total_bailout_payment = monthly_penalty_interest + principal_repayment
            deltas[b.agent_id]["reserves_delta"] = deltas[b.agent_id].get("reserves_delta", 0.0) - total_bailout_payment
            deltas[b.agent_id]["bailout_debt_delta"] = deltas[b.agent_id].get("bailout_debt_delta", 0.0) - principal_repayment
            # "interest_expense" là kênh BÁO CÁO (Bank.apply_result đọc vào
            # last_interest_expense) mà calculate_reward() đã dùng sẵn cho
            # net_interest_margin -- tái dụng ĐÚNG kênh này (giống Section 8B
            # đã làm cho lãi tiền gửi trả người gửi) để lãi phạt bailout hiện
            # NGAY trong reward tháng này thay vì chỉ ngấm gián tiếp qua
            # reserves ở các tháng sau, không cần sửa công thức reward.
            deltas[b.agent_id]["interest_expense"] = deltas[b.agent_id].get("interest_expense", 0.0) + monthly_penalty_interest
            deltas[gov.agent_id]["bailout_repayment"] = deltas[gov.agent_id].get("bailout_repayment", 0.0) + total_bailout_payment

        # 6. THUẾ VÀ GIAN LẬN TOÀN DIỆN (Allingham & Sandmo, 1972)
        total_tax_collected = 0.0
        taxpayer_declarations: Dict[str, Dict[str, Any]] = {}

        for emp in active_employees:
            emp_act = validated_actions.get(emp.agent_id)
            declare_ratio = float(emp_act.values[1]) if emp_act is not None else 1.0

            actual_gross = worker_gross_incomes[emp.agent_id]
            taxable_income = actual_gross * declare_ratio
            tax_due = taxable_income * worker_tax_rate
            evaded_tax = (actual_gross - taxable_income) * worker_tax_rate

            deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) - tax_due
            deltas[emp.agent_id]["executed_declare_ratio"] = declare_ratio
            total_tax_collected += tax_due

            taxpayer_declarations[emp.agent_id] = {
                "agent": emp,
                "evaded_tax": evaded_tax,
                "declare_ratio": declare_ratio,
                "is_firm": False
            }

            if evaded_tax > 0.01:
                self._emit_event(
                    EventType.TAX_EVADED, 
                    emp.agent_id, 
                    gov.agent_id, 
                    {"amount": round(evaded_tax, 1), "gross": round(actual_gross, 1)}, 
                    timestep
                )

        for firm in active_firms:
            f_act = validated_actions.get(firm.agent_id)
            declare_ratio = float(f_act.values[2]) if f_act is not None else 1.0
            profit = max(0.0, deltas[firm.agent_id].get("executed_profit", 0.0))

            taxable_profit = profit * declare_ratio
            tax_due = taxable_profit * firm_tax_rate
            evaded_tax = (profit - taxable_profit) * firm_tax_rate

            deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) - tax_due
            deltas[firm.agent_id]["executed_declare_ratio"] = declare_ratio
            total_tax_collected += tax_due

            taxpayer_declarations[firm.agent_id] = {
                "agent": firm,
                "evaded_tax": evaded_tax,
                "declare_ratio": declare_ratio,
                "is_firm": True
            }

            if evaded_tax > 0.01:
                self._emit_event(
                    EventType.TAX_EVADED, 
                    firm.agent_id, 
                    gov.agent_id, 
                    {"amount": round(evaded_tax, 1), "profit": round(profit, 1)}, 
                    timestep
                )

        deltas[gov.agent_id]["tax_collected"] = total_tax_collected
        self._emit_event(EventType.TAX_COLLECTED, "MARKET", gov.agent_id, {"amount": round(total_tax_collected, 1)}, timestep)

        # 7. THANH TRA TOÀN DIỆN SUPERVISOR (Khắc phục lỗi chỉ audit người gian lận)
        audits_count = 0
        violations_count = 0
        total_fines_collected = 0.0

        # Supervisor chọn mẫu ngẫu nhiên từ TOÀN BỘ người nộp thuế (Allingham & Sandmo, 1972)
        all_taxpayers = list(taxpayer_declarations.values())
        for record in all_taxpayers:
            agent = record["agent"]
            # Ân hạn cho doanh nghiệp non trẻ (dưới 6 tháng)
            if record["is_firm"] and getattr(agent, "age_months", 99) <= 6:
                continue

            if np.random.rand() < audit_rate:
                audits_count += 1
                evaded = record["evaded_tax"]

                if evaded > 0.01:
                    # Bắt đúng (True Positive): Phạt tiền
                    violations_count += 1
                    fine = evaded * fine_multiplier
                    deltas[agent.agent_id]["cash_delta"] = deltas[agent.agent_id].get("cash_delta", 0.0) - fine
                    total_fines_collected += fine

                    self._emit_event(
                        EventType.PENALTY_ENFORCED, 
                        sup.agent_id, 
                        agent.agent_id, 
                        {"fine": round(fine, 1), "evaded": round(evaded, 1)}, 
                        timestep
                    )
                    events_map[sup.agent_id].append(EventType.PENALTY_ENFORCED.value)
                    events_map[agent.agent_id].append(EventType.PENALTY_ENFORCED.value)
                else:
                    # Thanh tra người vô tội (Innocent Audit): Tốn công thanh tra, không có phạt
                    pass

        deltas[sup.agent_id]["audits_conducted"] = audits_count
        deltas[sup.agent_id]["violations_detected"] = violations_count
        deltas[sup.agent_id]["fines_collected"] = total_fines_collected

        # KHO BẠC CHUẨN SFC (Loại bỏ triệt để overhead double-count)
        # Tiền phạt trốn thuế (fines) PHẢI được hạch toán về Kho bạc -- nếu không,
        # số tiền đã bị trừ khỏi ví người vi phạm ở Section 7 sẽ biến mất khỏi hệ
        # thống, vi phạm trực tiếp nguyên lý bảo toàn SFC (Godley & Lavoie, 2007)
        # đã trích dẫn ở đầu file. Government.apply_result cộng khoản này vào
        # net_budget cùng tax_collected.
        deltas[gov.agent_id]["fines_collected"] = total_fines_collected

        # 8. MÔ HÌNH VỠ NỢ CẤU TRÚC MERTON CHUẨN (Merton, 1974)
        #
        # Thu hồi nợ khi phá sản theo nguyên tắc ưu tiên tuyệt đối (absolute
        # priority rule): ngân hàng thu hồi tối đa có thể từ (a) tiền mặt thanh
        # khoản còn lại của doanh nghiệp và (b) giá trị thanh lý tài sản cố định
        # (haircut); CHỈ phần không thể thu hồi mới được ghi nhận là tổn thất tín
        # dụng (DefaultedDebt trong đẳng thức SFC: ΔM2_t = ΔCredit_net_t −
        # DefaultedDebt_t). Đây là hệ quả trực tiếp của chính công thức
        # net_worth = Cash + (1-Haircut)*Capital - Debt đã trích dẫn Merton
        # (1974) bên dưới -- trước đây công thức net_worth chỉ được dùng để
        # KÍCH HOẠT điều kiện phá sản mà không được dùng để tính phần thu hồi
        # thực tế, khiến ngân hàng luôn mất trắng 100% dư nợ bất kể tài sản thế
        # chấp còn lại bao nhiêu, đồng thời cash_delta của firm bị ghi đè thẳng
        # về -firm.cash (xoá sạch doanh thu/chi phí phát sinh cùng tháng) --
        # cả hai đều vi phạm bảo toàn dòng tiền toàn hệ thống.
        HAIRCUT_LIQUIDATION = 0.30

        for firm in active_firms:
            projected_cash = firm.cash + deltas[firm.agent_id].get("cash_delta", 0.0)
            projected_debt = firm.debt + deltas[firm.agent_id].get("debt_delta", 0.0)
            firm_age = getattr(firm, "age_months", 99)
            overhead = firm_overheads.get(firm.agent_id, 0.0)

            # Merton (1974): Tài sản phát mại = (1 - Haircut) * Toàn bộ Tư bản (Bỏ nhân 0.05)
            asset_liquidation_value = (1.0 - HAIRCUT_LIQUIDATION) * firm.capital_stock
            net_worth = projected_cash + asset_liquidation_value - projected_debt

            is_insolvent = (
                (len(firm.employee_ids) == 0 and projected_cash <= 0.0 and firm_age > 6) or
                (net_worth < 0.0 and projected_cash < -overhead)
            )

            if is_insolvent:
                deltas[firm.agent_id]["status"] = LifeCycleStatus.BANKRUPT

                fired_list = [eid for eid in firm.employee_ids if eid in deltas]
                deltas[firm.agent_id]["fired_employees"] = fired_list
                for emp_id in fired_list:
                    deltas[emp_id]["employed_by"] = None
                    deltas[emp_id]["wage"] = 0.0
                    self._emit_event(EventType.FIRE, firm.agent_id, emp_id, {"reason": "Firm Insolvency"}, timestep)

                # Tiền mặt thanh khoản được thu hồi trước để trả nợ ngân hàng;
                # phần dư ra (nếu tiền mặt còn lại nhiều hơn nợ) được GIỮ
                # NGUYÊN trong cash_delta của firm (cộng dồn, không ghi đè) để
                # cơ chế "di sản -> Kho bạc" đã có sẵn ở env.py xử lý đúng phần
                # còn lại. Giá trị thanh lý tài sản cố định (asset_liquidation_
                # value) không tạo ra dòng tiền thật (không có agent nào mua lại
                # capital_stock trong mô phỏng này) nên CHỈ được dùng để giảm
                # phần tổn thất ghi nhận (bad_debt), tuyệt đối không được cộng
                # vào reserves ngân hàng -- nếu không sẽ tạo tiền từ hư không.
                recoverable_cash = max(0.0, projected_cash)
                recoverable_total = recoverable_cash + asset_liquidation_value
                bad_debt = max(0.0, projected_debt - recoverable_total)
                cash_seized = min(recoverable_cash, projected_debt)

                deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) - cash_seized
                deltas[firm.agent_id]["debt_delta"] = -firm.debt

                # Nợ được xoá sổ tại ĐÚNG ngân hàng chủ nợ của doanh nghiệp này
                # (quan hệ tín dụng đã xác lập/cập nhật ở Section 5 cùng chu kỳ).
                creditor_id = deltas[firm.agent_id].get("creditor_bank_id", getattr(firm, "creditor_bank_id", None))
                if creditor_id not in bank_lookup:
                    creditor_id = bank.agent_id

                # Toàn bộ dư nợ được xoá khỏi sổ sách cho vay của ngân hàng
                # (loans_delta) bất kể thu hồi được bao nhiêu; phần thu hồi
                # thực bằng tiền mặt (cash_seized) cộng vào reserves, phần
                # không thu hồi được (bad_debt) trừ khỏi reserves như một
                # khoản lỗ tín dụng vĩnh viễn (xem Bank.apply_result).
                deltas[creditor_id]["loans_delta"] = deltas[creditor_id].get("loans_delta", 0.0) - projected_debt
                deltas[creditor_id]["reserves_delta"] = deltas[creditor_id].get("reserves_delta", 0.0) + cash_seized
                deltas[creditor_id]["new_defaults"] = deltas[creditor_id].get("new_defaults", 0.0) + bad_debt

                self._emit_event(
                    EventType.AGENT_BANKRUPT,
                    firm.agent_id,
                    creditor_id,
                    # LOI DA SUA (v0.23, phat hien khi them tin dung tieu dung Employee lam lo ro):
                    # "bad_debt" o day KHONG duoc lam tron -- be/tests/test_sfc_accounting.py va
                    # test_sfc_random_policy.py DOC LAI DUNG gia tri nay de doi chieu voi thuc te
                    # bank.reserves da giam bao nhieu (deltas[creditor_id]["new_defaults"] dung gia
                    # tri KHONG lam tron o dong tren). Truoc day round(bad_debt, 1) lam sai lech toi
                    # 0.05 don vi tien te moi lan phat sinh no xau -- vuot han SFC_TOLERANCE=1e-3,
                    # tao "ro ri" GIA trong kiem toan SFC du dong tien thuc te van bao toan dung.
                    # "recovered"/"cash" van lam tron binh thuong vi CHI phuc vu hien thi/log, khong
                    # tac tu/test nao doc lai de doi chieu bao toan.
                    {"bad_debt": bad_debt, "recovered": round(cash_seized, 1), "cash": round(firm.cash, 1)},
                    timestep
                )
                events_map[firm.agent_id].append(EventType.AGENT_BANKRUPT.value)

        # 8B. TÁI CÂN BẰNG TIỀN GỬI NGÂN HÀNG HỘ GIA ĐÌNH
        # (Household Deposit Rebalancing / Buffer-Stock Saving)
        #
        # Trước bản vá này, Bank.total_deposits không bao giờ được cập nhật (luôn
        # bằng 0), khiến yêu cầu dự trữ bắt buộc vô hiệu và hành động deposit_rate
        # của Bank không có tác dụng gì lên mô phỏng. Cơ chế dưới đây vận dụng một
        # phiên bản đơn giản hoá, tất định hoá (deterministic operationalization)
        # của hành vi TIẾT KIỆM ĐỆM MỤC TIÊU (buffer-stock / target-wealth
        # saving): hộ gia đình giữ một mức tiền mặt giao dịch mục tiêu
        # (deposit_buffer_target); phần dư được GỬI vào ngân hàng để hưởng lãi,
        # phần thiếu hụt được RÚT từ số dư tiền gửi sẵn có trước khi bị coi là
        # thiếu hụt tiền mặt thực sự (tức tiền gửi là demand deposit, thanh khoản
        # hoàn toàn, đúng bản chất tài khoản tiết kiệm/vãng lai thông thường).
        # Đây KHÔNG phải nghiệm chính xác phương trình Euler ngẫu nhiên trong bản
        # gốc của các trích dẫn dưới đây, mà là cách vận dụng tuyến tính hoá phổ
        # biến trong các mô hình kinh tế dựa trên tác tử (Agent-Based Macro
        # Models) khi đưa hành vi tiết kiệm hộ gia đình vào mô phỏng số:
        #   - Deaton, A. (1991). "Saving and Liquidity Constraints".
        #     Econometrica, 59(5), 1221-1248.
        #   - Carroll, C. D. (1997). "Buffer-Stock Saving and the Life-Cycle/
        #     Permanent Income Hypothesis". Quarterly Journal of Economics,
        #     112(1), 1-55.
        #   - Dosi, G., Fagiolo, G., Napoletano, M., Roventini, A., & Treibich,
        #     T. (2015). "Fiscal and monetary policies in complex evolving
        #     economies". Journal of Economic Dynamics and Control, 52, 166-189.
        #     (Tiền lệ dùng quy tắc đệm mục tiêu tuyến tính hoá cho tiết kiệm hộ
        #     gia đình trong một mô hình ABM vĩ mô đóng SFC.)
        #
        # K_LIQUIDITY_BUFFER = 2.0 (hệ số cấu trúc TỰ DO HIỆU CHỈNH, không suy ra
        # trực tiếp từ các trích dẫn trên): số tháng sinh hoạt phí giữ làm đệm
        # thanh khoản giao dịch của một hộ gia đình.
        # Với hệ thống nhiều ngân hàng, hộ gia đình cũng chọn ngân hàng gửi tiền
        # theo quan hệ tín dụng (relationship banking -- Petersen & Rajan, 1994,
        # áp dụng tương tự cho quan hệ tiền gửi): nếu chưa có ngân hàng gửi tiền
        # hoặc số dư hiện tại bằng 0 (coi như "đóng tài khoản"), tự do chọn lại
        # ngân hàng có deposit_rate cao nhất; nếu đang có số dư, DUY TRÌ đúng
        # ngân hàng đó.
        K_LIQUIDITY_BUFFER = 2.0

        for emp in active_employees:
            # De dem an toan THEO DAC DIEM E NGAI RUI RO NOI TAI cua tung ho gia dinh
            # (v0.20, xem KNOWN_PATHOLOGIES.md). Kimball, M. S. (1990), "Precautionary
            # Saving in the Small and in the Large", Econometrica 58(1), 53-73 -- ho gia
            # dinh cang e ngai rui ro cang giu muc dem thanh khoan muc tieu lon hon truoc
            # bat dinh thu nhap tuong lai (chinh khai niem "prudence" trong ly thuyet tieu
            # dung). TRUOC BAN VA NAY: emp.risk_aversion duoc khoi tao/dot bien ke thua qua
            # sinh san (env.py) nhung KHONG anh huong bat ky cong thuc kinh te nao -- phat
            # hien qua audit toan du an. He so nhan (1.0 + risk_aversion) la HE SO CAU TRUC
            # TU DO HIEU CHINH (risk_aversion~0 -> dem ~K_LIQUIDITY_BUFFER goc; ~1 -> dem
            # gap doi), KHONG suy ra truc tiep tu Kimball (1990) (chi xac lap CHIEU tac
            # dong, khong cho cong thuc ty le cu the).
            deposit_buffer_target = K_LIQUIDITY_BUFFER * (1.0 + float(np.clip(getattr(emp, "risk_aversion", 0.5), 0.0, 1.0))) * actual_living_cost
            current_cash_est = emp.cash + deltas[emp.agent_id].get("cash_delta", 0.0)
            # Số dư gửi đầu kỳ SAU khi trừ phần đã rút để chi tiêu ở Mục 4 (pre_spend_
            # withdrawal): lãi và tái cân bằng tính trên số dư còn lại; deposit_delta cuối
            # cùng cũng phải trừ khoản rút này để Employee.apply_result khớp.
            pre_spend_withdrawal = deltas[emp.agent_id].get("pre_spend_withdrawal", 0.0)
            old_deposit = getattr(emp, "bank_deposit", 0.0) - pre_spend_withdrawal

            existing_depository = getattr(emp, "depository_bank_id", None)
            if existing_depository not in bank_lookup or old_deposit <= 0.0:
                depository_id = max(banks, key=lambda b: executed_deposit_rate[b.agent_id]).agent_id
            else:
                depository_id = existing_depository
            deltas[emp.agent_id]["depository_bank_id"] = depository_id
            deposit_rate = executed_deposit_rate[depository_id]

            # Lãi tiền gửi tháng này cộng dồn vào số dư trước khi tái cân bằng.
            # Chi trả lãi tiền gửi là một kênh "ngân hàng tạo tiền" chuẩn mực
            # (McLeay, Radia & Thomas, 2014, "Money Creation in the Modern
            # Economy", Bank of England Quarterly Bulletin, Q1, 2014): cung tiền
            # nắm giữ bởi khu vực tư tăng lên, được bù trừ bằng phần vốn tự có
            # (reserves) mà ngân hàng bỏ ra -- không vi phạm bảo toàn SFC vì đây
            # là một khoản chuyển giao giá trị thực từ vốn chủ sở hữu ngân hàng
            # sang người gửi tiền, không phải tiền sinh ra từ hư không.
            gross_deposit_interest = old_deposit * (deposit_rate / 12.0)

            # Lãi tiền gửi là thu nhập vốn (capital income) và bị đánh thuế
            # thu nhập THEO ĐÚNG CÙNG sắc thuế luỹ tiến đã áp dụng cho tiền
            # lương ở Mục 6 (worker_tax_rate) -- đây KHÔNG phải một sắc thuế
            # mới được bịa ra, mà là khép lại một lỗ hổng thiết kế: trong mọi
            # hệ thống thuế thu nhập thực tế (vd. US IRS Publication 550; đa
            # số hệ thống thuế thu nhập cá nhân các nước OECD), lãi tiền gửi
            # ngân hàng là thu nhập chịu thuế CÙNG LOẠI với tiền lương. Việc
            # bỏ sót khoản thuế này khiến của cải gửi ngân hàng tăng trưởng
            # kép HOÀN TOÀN miễn thuế trong khi thu nhập lao động luôn bị
            # đánh thuế -- một kênh bất đối xứng khiến Gini index tăng có
            # tính CẤU TRÚC theo thời gian, bất kể chính sách tái phân phối
            # nào khác của Chính phủ (quan sát thực nghiệm qua nhiều lần
            # train dài hạn). Ngân hàng vẫn CHI TRẢ đủ phần lãi GỘP từ vốn tự
            # có (reserves) như thiết kế gốc; phần thuế chỉ được khấu trừ
            # (withhold) trước khi cộng vào số dư CỦA NGƯỜI GỬI, sau đó
            # chuyển thẳng vào Kho bạc -- không tạo ra hay huỷ tiền, chỉ đổi
            # hướng một phần dòng tiền vốn đã tồn tại (bảo toàn SFC).
            interest_tax = gross_deposit_interest * worker_tax_rate
            deposit_interest = gross_deposit_interest - interest_tax
            deltas[gov.agent_id]["tax_collected"] = (
                deltas[gov.agent_id].get("tax_collected", 0.0) + interest_tax
            )
            deposit_after_interest = old_deposit + deposit_interest

            deposit_flow = 0.0
            if current_cash_est > deposit_buffer_target:
                deposit_flow = current_cash_est - deposit_buffer_target
            elif current_cash_est < deposit_buffer_target and deposit_after_interest > 0.0:
                withdrawal = min(deposit_after_interest, deposit_buffer_target - current_cash_est)
                deposit_flow = -withdrawal

            deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) - deposit_flow
            deltas[emp.agent_id]["deposit_delta"] = deposit_interest + deposit_flow - pre_spend_withdrawal

            # Tổng biến động tiền gửi/lãi phải trả được hạch toán vào ĐÚNG ngân
            # hàng đang giữ số dư của người này (cộng dồn qua nhiều hộ gia đình).
            # deposits_delta dung phan lai RONG (sau thue) vi do la khoan no
            # (liability) thuc su ngan hang con phai tra cho nguoi gui; con
            # interest_expense/reserves_delta dung phan lai GOP vi day moi la
            # tong tien mat thuc su roi khoi reserves cua ngan hang (mot phan
            # sang nguoi gui, mot phan sang Kho bac qua interest_tax o tren).
            deltas[depository_id]["deposits_delta"] = deltas[depository_id].get("deposits_delta", 0.0) + deposit_interest + deposit_flow
            deltas[depository_id]["interest_expense"] = deltas[depository_id].get("interest_expense", 0.0) + gross_deposit_interest
            deltas[depository_id]["reserves_delta"] = deltas[depository_id].get("reserves_delta", 0.0) - gross_deposit_interest

        # QUẢN TRỊ TỬ VONG SINH HỌC & NỢ CÙNG QUẪN
        new_deaths = 0
        for emp in active_employees:
            projected_cash = emp.cash + deltas[emp.agent_id].get("cash_delta", 0.0)
            # Clip năng lượng chặt chẽ trong [0.0, max_energy]
            max_e = getattr(emp, "max_energy", 2.0)
            projected_energy = float(np.clip(emp.energy + deltas[emp.agent_id].get("energy_delta", 0.0), 0.0, max_e))
            deltas[emp.agent_id]["applied_energy"] = projected_energy

            deltas[emp.agent_id]["age_increment"] = 1 if (timestep % 12 == 0) else 0

            debt_survival_limit = -10.0 * actual_living_cost
            if (projected_energy <= 0.0) or (projected_cash < debt_survival_limit and projected_energy < 0.15) or ((emp.age + deltas[emp.agent_id]["age_increment"]) >= emp.max_age):
                new_deaths += 1
                deltas[emp.agent_id]["status"] = LifeCycleStatus.DECEASED

                # HÌNH PHẠT TỬ VONG QUY MÔ THEO "QUÃNG ĐỜI CÒN LẠI" (Age/Horizon
                # -scaled Death Penalty) -- lấy cảm hứng từ Value of Statistical
                # Life literature (Viscusi, W.K., & Aldy, J.E. (2003), "The
                # Value of a Statistical Life: A Critical Review of Market
                # Estimates Throughout the World", Journal of Risk and
                # Uncertainty 27(1), 5-76; Aldy, J.E., & Viscusi, W.K. (2008),
                # "Adjusting the Value of a Statistical Life for Age and
                # Cohort Effects", Review of Economics and Statistics 90(3)):
                # VSL literature xác nhận HƯỚNG tác động (agent càng trẻ, mất
                # mát kỳ vọng càng lớn), KHÔNG cho ra bất kỳ tỷ lệ % cụ thể nào
                # -- tỷ lệ nhân cụ thể dùng ở Employee.calculate_reward() là hệ
                # số cấu trúc TỰ DO HIỆU CHỈNH riêng của dự án (xem
                # ScenarioConfig.death_penalty_horizon_multiplier), không suy
                # ra trực tiếp từ 2 nguồn trên (2 nguồn đó ước lượng VSL bằng
                # USD thực, không đưa ra tỷ lệ). "Quãng đời còn lại" bị chặn
                # bởi CẢ tuổi thọ sinh học (max_age) LẪN số bước còn lại của
                # chính episode đang chạy -- không dùng thẳng toàn bộ tuổi thọ
                # còn lại (vô nghĩa để phạt cho phần đời nằm ngoài episode).
                final_age = emp.age + deltas[emp.agent_id]["age_increment"]
                remaining_years_by_age = max(0.0, float(emp.max_age - final_age))
                remaining_years_by_episode = max(0.0, (float(max_steps) - float(timestep)) / 12.0)
                remaining_horizon = min(remaining_years_by_age, remaining_years_by_episode)
                deltas[emp.agent_id]["death_remaining_horizon_ratio"] = remaining_horizon / max(1.0, float(emp.max_age))

                emp_firm = deltas[emp.agent_id].get("employed_by", emp.employed_by)
                if emp_firm and emp_firm in deltas:
                    deltas[emp_firm].setdefault("fired_employees", []).append(emp.agent_id)
                deltas[emp.agent_id]["employed_by"] = None

                # GHI NHẬN NỢ XẤU KHI EMPLOYEE CHẾT (v0.23, tín dụng tiêu dùng Section 3B) --
                # KHÁC Firm Merton (Section 8): Employee KHÔNG có capital_stock để thế chấp,
                # nên khi chết còn nợ, TOÀN BỘ dư nợ là mất trắng 100% cho ngân hàng chủ nợ
                # (không có haircut/thu hồi tài sản nào để tính) -- đúng bản chất tín dụng tiêu
                # dùng KHÔNG THẾ CHẤP thật (unsecured personal debt: người vay chết, không có
                # di sản đảm bảo, ngân hàng ghi lỗ toàn bộ, thông lệ chuẩn ngành ngân hàng bán
                # lẻ). projected_debt dùng ĐÚNG debt_delta đã tính tới Section 3B/8B cùng bước
                # này (không phải emp.debt cũ) để không bỏ sót khoản vừa vay/vừa trả cùng tháng.
                projected_debt = max(0.0, emp.debt + deltas[emp.agent_id].get("debt_delta", 0.0))
                bad_debt_on_death = 0.0
                if projected_debt > 0.0:
                    creditor_id = deltas[emp.agent_id].get("creditor_bank_id", getattr(emp, "creditor_bank_id", None))
                    if creditor_id not in bank_lookup:
                        creditor_id = bank.agent_id
                    bad_debt_on_death = projected_debt
                    deltas[emp.agent_id]["debt_delta"] = deltas[emp.agent_id].get("debt_delta", 0.0) - projected_debt
                    deltas[creditor_id]["loans_delta"] = deltas[creditor_id].get("loans_delta", 0.0) - projected_debt
                    deltas[creditor_id]["new_defaults"] = deltas[creditor_id].get("new_defaults", 0.0) + projected_debt

                death_reason = "Tuổi già" if (emp.age + deltas[emp.agent_id]["age_increment"]) >= emp.max_age else "Kiệt quệ sinh học / Nợ cùng quẫn"
                self._emit_event(
                    EventType.AGENT_DIED,
                    emp.agent_id,
                    gov.agent_id,
                    # "bad_debt" KHONG lam tron -- xem chu thich day du tai AGENT_BANKRUPT o Section 8
                    # (cung lop loi lam tron gia tri dung de doi chieu bao toan SFC, phat hien qua
                    # chinh test khi them tin dung tieu dung Employee o v0.23).
                    {"age": emp.age, "reason": death_reason, "bad_debt": bad_debt_on_death},
                    timestep
                )
                events_map[emp.agent_id].append(EventType.AGENT_DIED.value)

        deltas[gov.agent_id]["new_deaths"] = new_deaths
        # LOI DA SUA (v0.24, phat hien khi noi reward moi cho Economy): "new_deaths" TRUOC DAY
        # CHI duoc ghi vao deltas[gov.agent_id], KHONG BAO GIO ghi vao deltas[eco.agent_id] --
        # trong khi Economy.calculate_reward() doc dung tu transition_result CUA CHINH NO
        # (deltas[eco.agent_id]), nen "dead_worker_penalty" cu (them rieng de phat Economy day
        # gia gay chet nguoi) da LUON LUON = 0, chua tung thuc su hoat dong. Sua: ghi ca vao day.
        deltas[eco.agent_id]["new_deaths"] = new_deaths
        deltas[eco.agent_id]["active_population_for_reward"] = len(active_employees)

        # 9. GINI HIỆU CHỈNH CHO TÀI SẢN ÂM (Raffinetti et al., 2015) & GDP
        # Tài sản dùng để tính Gini = tiền mặt + tiền gửi ngân hàng (tổng của cải
        # thanh khoản thực). Nếu chỉ tính tiền mặt, bất bình đẳng sẽ bị đánh giá
        # thấp giả tạo một khi hộ gia đình khá giả chuyển phần lớn của cải sang
        # tiền gửi (Section 8B) thay vì giữ tiền mặt.
        active_wealths = [
            (e.cash + deltas[e.agent_id].get("cash_delta", 0.0))
            + (getattr(e, "bank_deposit", 0.0) + deltas[e.agent_id].get("deposit_delta", 0.0))
            for e in active_employees
            if deltas[e.agent_id].get("status") not in [LifeCycleStatus.DECEASED, LifeCycleStatus.DEAD, LifeCycleStatus.TERMINATED]
        ]
        deltas[gov.agent_id]["current_gini"] = self._compute_gini_with_negatives(active_wealths)
        deltas[gov.agent_id]["current_gdp"] = total_industrial_revenue + total_informal_revenue
        # GDP THỰC = GDP danh nghĩa / chỉ số giá (giá sinh hoạt so với mốc ban đầu cố định).
        # Phần thưởng của Chính phủ phải dựa trên đại lượng THỰC: dùng GDP danh nghĩa thì
        # siêu lạm phát tự nó được thưởng (đo được: reward Chính phủ 80,68/bước ở chế độ siêu
        # lạm phát so với 8,54 ở chế độ lành mạnh, trong khi GDP thực 101 so với 7.513).
        # Khái niệm "volume measure" = giá trị danh nghĩa khử giá: United Nations et al.
        # (2009), "System of National Accounts 2008", Ch.15; cùng nguyên lý real-vs-nominal
        # (Hicks, 1946) đã dùng cho reward của Economy.
        price_index = max(1e-6, actual_living_cost / max(1e-9, eco.initial_living_cost))
        deltas[gov.agent_id]["current_real_gdp"] = (total_industrial_revenue + total_informal_revenue) / price_index

        # 10. ĐÓNG GÓI CHUYỂN DỊCH
        results: Dict[str, TransitionResult] = {}
        for agent_id in agents.keys():
            results[agent_id] = TransitionResult(
                agent_id=agent_id,
                state_delta=deltas[agent_id],
                events_triggered=events_map[agent_id],
                success=True,
                info={"timestep": timestep}
            )

        return results

    def _emit_event(self, event_type: EventType, source: str, target: str, payload: Dict[str, Any], timestep: int):
        self.event_bus.publish(Event(
            event_type=event_type,
            source_id=source,
            target_id=target,
            payload=payload,
            timestep=timestep
        ))

    @staticmethod
    def _compute_gini_with_negatives(wealth_array: List[float]) -> float:
        """
        Chuẩn hóa hệ số Gini cho phân phối có thể chứa giá trị âm (Raffinetti et al., 2015).
        Bảo toàn trung thực khoảng cách giàu nghèo thay vì clip về 0.01 làm phẳng bất bình đẳng.
        """
        if len(wealth_array) < 2:
            return 0.0
        w = np.asarray(wealth_array, dtype=np.float64)
        min_w = np.min(w)
        # Dịch chuyển phân phối về không âm để tính Gini chuẩn mực
        w_shifted = w - min_w + 1e-5 if min_w < 0 else w
        sorted_w = np.sort(w_shifted)
        sum_w = np.sum(sorted_w)
        if sum_w <= 0.0:
            return 0.0
        n = len(sorted_w)
        indices = np.arange(1, n + 1, dtype=np.float64)
        return float(((2.0 * np.sum(indices * sorted_w)) / (n * sum_w)) - ((n + 1.0) / n))

    @staticmethod
    def _get_single_agent(agents: Dict[str, BaseAgent], agent_cls: type) -> Any:
        for a in agents.values():
            if isinstance(a, agent_cls):
                return a
        raise RuntimeError(f"Missing required institutional agent: {agent_cls.__name__}")

    @staticmethod
    def _get_all_agents(agents: Dict[str, BaseAgent], agent_cls: type) -> List[Any]:
        """Trả về toàn bộ tác tử ACTIVE/INITIALIZED thuộc một lớp thể chế có thể
        có nhiều thực thể đồng thời (vd. nhiều Bank -- xem Section 1, 5, 8, 8B)."""
        result = [
            a for a in agents.values()
            if isinstance(a, agent_cls) and a.status in (LifeCycleStatus.ACTIVE, LifeCycleStatus.INITIALIZED)
        ]
        if not result:
            raise RuntimeError(f"Missing required institutional agent(s): {agent_cls.__name__}")
        return result