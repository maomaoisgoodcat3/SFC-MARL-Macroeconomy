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
    def __init__(self, event_bus: EventBus, subsistence_indexation_ceiling_mult: float = 3.0):
        self.event_bus: EventBus = event_bus
        # Tran chi so hoa chi tieu sinh ton theo gia (boi so cua eco.initial_living_cost);
        # HE SO HIEU CHINH on dinh so hoc, xem chu thich tai Section 4.
        self.subsistence_indexation_ceiling_mult: float = float(subsistence_indexation_ceiling_mult)

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

        # 2. THỊ TRƯỜNG LAO ĐỘNG: ĐÀM PHÁN MRPL CÓ RÀNG BUỘC SỐNG CÒN
        ALPHA_CAPITAL = 0.3
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
                        BETA_LABOR * firm.productivity_factor * 
                        (firm.capital_stock ** ALPHA_CAPITAL) * 
                        (max(0.5, next_l) ** (BETA_LABOR - 1.0)) * 
                        candidate.skill_level
                    )
                    mrpl = expected_price * marginal_product

                    # Lương bảo lưu sinh tồn (Reservation Wage): Phải đủ sống (Shapiro & Stiglitz, 1984)
                    reservation_wage = max(expected_price * 1.02, expected_price * (0.8 + 0.3 * candidate.skill_level))

                    # Đàm phán Nash: Công nhân chỉ đi làm nếu lương >= chi phí sống thực tế
                    if mrpl >= reservation_wage:
                        negotiated_wage = max(expected_price, 0.5 * reservation_wage + 0.5 * mrpl)

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

            wage_bill = 0.0
            effective_l = 0.0

            for emp in firm_workers:
                emp_act = validated_actions.get(emp.agent_id)
                effort = float(emp_act.values[0]) if emp_act is not None else 0.6
                assigned_wage = deltas[emp.agent_id].get("wage", emp.wage)
                wage = assigned_wage if assigned_wage > 0 else (expected_price * 1.05)

                wage_bill += wage
                worker_gross_incomes[emp.agent_id] += wage
                effective_l += (effort * emp.skill_level)

                deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + wage
                deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) - (effort * 0.10 + 0.02)
                deltas[emp.agent_id]["executed_work_effort"] = effort

                self._emit_event(EventType.WAGE_PAID, firm.agent_id, emp.agent_id, {"amount": round(wage, 1)}, timestep)
                events_map[firm.agent_id].append(EventType.WAGE_PAID.value)
                events_map[emp.agent_id].append(EventType.WAGE_PAID.value)

            physical_q = firm.productivity_factor * (firm.capital_stock ** ALPHA_CAPITAL) * (effective_l ** BETA_LABOR) if effective_l > 0.0 else 0.0
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

        # 4. TIÊU DÙNG & ĐỊNH GIÁ KẾT DÍNH CALVO CHUẨN (Calvo, 1983)
        total_consumer_spending = 0.0

        # TRẦN CHỈ SỐ HOÁ CHI TIÊU SINH TỒN THEO GIÁ (ổn định số học).
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
        indexation_ceiling = self.subsistence_indexation_ceiling_mult * eco.initial_living_cost
        indexed_price = min(expected_price, indexation_ceiling)

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
        eco_act = validated_actions.get(eco.agent_id)
        # Action đã được Economy.validate_action() clip vào [-0.20, 0.20],
        # không clip lại ở đây (tránh trùng lặp logic biên).
        demand_injection_ratio = float(eco_act.values[0]) if eco_act is not None else 0.0
        demand_injection_base = SUBSISTENCE_BASKET_QTY * eco.initial_living_cost * max(1, len(unemployed_emps))
        total_consumer_spending_after_injection = max(0.0, total_consumer_spending + (demand_injection_ratio * demand_injection_base))
        # Chênh lệch THỰC TẾ (sau khi kẹp sàn 0) là số tiền Kho bạc phải chi
        # trả (bơm dương -> firm/khu vực phi chính thức nhận thêm doanh thu)
        # hoặc thu về (bơm âm -> thắt chi tiêu) -- đảm bảo đẳng thức SFC: phần
        # cầu tăng/giảm thêm PHẢI có nguồn đối ứng ở Treasury (xem
        # Government.apply_result), không tự sinh/mất tiền.
        demand_injection_effect = total_consumer_spending_after_injection - total_consumer_spending
        total_consumer_spending = total_consumer_spending_after_injection
        deltas[eco.agent_id]["demand_injection_ratio"] = demand_injection_ratio
        deltas[eco.agent_id]["demand_injection_effect"] = demand_injection_effect
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

        # CÂN BẰNG GIÁ CALVO (Calvo, 1983 Staggered Price Setting):
        # P*_t: Giá cân bằng Walras tức thời nếu 100% doanh nghiệp đổi giá
        instant_clearing_price = total_consumer_spending / max(1.0, total_real_supply)
        # theta = 0.70: Tỷ lệ hợp đồng giữ nguyên giá cũ (Calvo price stickiness)
        THETA_CALVO = 0.70
        market_clearing_price = float(THETA_CALVO * expected_price + (1.0 - THETA_CALVO) * instant_clearing_price)
        market_clearing_price = max(1.0, market_clearing_price)

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
        RESERVE_LENDING_FLOOR = 5000.0  # hệ số cấu trúc: dự trữ tối thiểu để còn được phép cho vay
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
                loan_request = min(borrowing_headroom, 1000.0 * borrow_signal * credit_factor)
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
                    {"bad_debt": round(bad_debt, 1), "recovered": round(cash_seized, 1), "cash": round(firm.cash, 1)},
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
        deposit_buffer_target = K_LIQUIDITY_BUFFER * actual_living_cost

        for emp in active_employees:
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

                death_reason = "Tuổi già" if (emp.age + deltas[emp.agent_id]["age_increment"]) >= emp.max_age else "Kiệt quệ sinh học / Nợ cùng quẫn"
                self._emit_event(
                    EventType.AGENT_DIED, 
                    emp.agent_id, 
                    gov.agent_id, 
                    {"age": emp.age, "reason": death_reason}, 
                    timestep
                )
                events_map[emp.agent_id].append(EventType.AGENT_DIED.value)

        deltas[gov.agent_id]["new_deaths"] = new_deaths

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