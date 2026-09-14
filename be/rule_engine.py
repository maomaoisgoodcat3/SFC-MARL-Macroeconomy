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
    Hiến pháp Kinh tế Thể chế v3.0 (Endogenous General Equilibrium).
    - Cân bằng thị trường hàng hóa Walras (Godley & Lavoie, 2007).
    - Giỏ hàng sinh hoạt Stone-Geary LES (Stone, 1954).
    - Hàm sản xuất Cobb-Douglas DRS (Cobb & Douglas, 1928).
    - Phân bổ tiền lương theo MRPL (Shapiro & Stiglitz, 1984).
    - Mô hình mất khả năng thanh toán cấu trúc Merton (Merton, 1974).
    """
    def __init__(self, event_bus: EventBus):
        self.event_bus: EventBus = event_bus

    def execute_cycle(self, 
                      agents: Dict[str, BaseAgent], 
                      validated_actions: Dict[str, Action], 
                      timestep: int) -> Dict[str, TransitionResult]:
        deltas: Dict[str, Dict[str, Any]] = {agent_id: {} for agent_id in agents.keys()}
        events_map: Dict[str, List[str]] = {agent_id: [] for agent_id in agents.keys()}

        # 1. THỂ CHẾ VĨ MÔ & THAM SỐ THỊ TRƯỜNG CƠ SỞ
        gov = self._get_single_agent(agents, Government)
        bank = self._get_single_agent(agents, Bank)
        eco = self._get_single_agent(agents, Economy)
        sup = self._get_single_agent(agents, Supervisor)

        gov_act = validated_actions.get(gov.agent_id)
        if gov_act is not None:
            deltas[gov.agent_id]["executed_worker_tax"] = float(gov_act.values[0])
            deltas[gov.agent_id]["executed_firm_tax"] = float(gov_act.values[1])
            deltas[gov.agent_id]["executed_subsidy_ratio"] = float(gov_act.values[2])
            worker_tax_rate = float(gov_act.values[0])
            firm_tax_rate = float(gov_act.values[1])
        else:
            worker_tax_rate = gov.tax_rate_worker
            firm_tax_rate = gov.tax_rate_firm

        bank_act = validated_actions.get(bank.agent_id)
        if bank_act is not None:
            deltas[bank.agent_id]["executed_lending_rate"] = float(bank_act.values[0])
            deltas[bank.agent_id]["executed_deposit_rate"] = float(bank_act.values[1])
            deltas[bank.agent_id]["executed_credit_factor"] = float(bank_act.values[2])
            lending_rate = float(bank_act.values[0])
            credit_factor = float(bank_act.values[2])
        else:
            lending_rate = bank.lending_rate
            credit_factor = bank.credit_expansion_factor

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

        # GIÁ KỲ VỌNG NỘI SINH (Stone-Geary Subsistence: gamma = 1.0 đơn vị hàng hóa)
        SUBSISTENCE_BASKET_QTY = 1.0
        expected_price = max(0.5, eco.base_living_cost / SUBSISTENCE_BASKET_QTY)

        # 2. THỊ TRƯỜNG LAO ĐỘNG: ĐÀM PHÁN MRPL & HIỆU SUẤT GIẢM DẦN COBB-DOUGLAS
        # Y_j = A_j * K_j^alpha * L_j^beta (alpha = 0.3, beta = 0.6 => DRS alpha + beta = 0.9)
        ALPHA_CAPITAL = 0.3
        BETA_LABOR = 0.6
        CAPITAL_DEPRECIATION_RATE = 0.02  # Jorgenson (1963): 2% khấu hao tư bản/tháng

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
            current_effective_labor = sum(getattr(w, 'skill_level', 1.0) for w in current_workers)
            current_wage_bill = sum(getattr(w, 'wage', expected_price * 1.1) for w in current_workers)

            # Khấu hao tư bản thực tế theo Jorgenson (1963)
            firm_capital = max(100.0, firm.capital_stock)
            overhead_cost = CAPITAL_DEPRECIATION_RATE * expected_price * (firm_capital * 0.05)

            unemployed = [e for e in active_employees if e.agent_id not in claimed_workers]
            unemployed.sort(key=lambda w: getattr(w, 'skill_level', 1.0), reverse=True)

            safety_reserve = overhead_cost + (current_wage_bill * 1.15)
            available_liquidity = max(0.0, firm.cash - safety_reserve)

            # Tuyển dụng dựa trên Doanh thu Sản phẩm Cận biên (MRPL)
            if available_liquidity > (expected_price * 1.5) and unemployed and hire_signal > -0.1:
                for candidate in unemployed:
                    if available_liquidity <= (expected_price * 1.5):
                        break

                    # Sản phẩm cận biên dự phóng khi có thêm lao động mới
                    next_l = current_effective_labor + candidate.skill_level
                    marginal_product = BETA_LABOR * firm.productivity_factor * (firm_capital ** ALPHA_CAPITAL) * (max(0.5, next_l) ** (BETA_LABOR - 1.0)) * candidate.skill_level
                    mrpl = expected_price * marginal_product

                    # Lương bảo lưu (Reservation Wage): Bù đắp giỏ hàng sinh tồn Stone-Geary
                    reservation_wage = expected_price * SUBSISTENCE_BASKET_QTY * (0.8 + 0.3 * candidate.skill_level)

                    # Doanh nghiệp chỉ tuyển khi giá trị cận biên lớn hơn lương bảo lưu
                    if mrpl >= reservation_wage:
                        negotiated_wage = 0.5 * reservation_wage + 0.5 * mrpl

                        claimed_workers.add(candidate.agent_id)
                        current_effective_labor = next_l
                        available_liquidity -= negotiated_wage

                        deltas[firm.agent_id].setdefault("hired_employees", []).append(candidate.agent_id)
                        deltas[candidate.agent_id]["employed_by"] = firm.agent_id
                        deltas[candidate.agent_id]["wage"] = negotiated_wage
                        deltas[candidate.agent_id]["unemployed_streak"] = 0

                        self._emit_event(EventType.HIRE, firm.agent_id, candidate.agent_id, {"wage": round(negotiated_wage, 1)}, timestep)
                        events_map[firm.agent_id].append(EventType.HIRE.value)
                        events_map[candidate.agent_id].append(EventType.HIRE.value)
                    else:
                        # Hiệu suất giảm dần triệt tiêu động lực mở rộng thêm (Tự nhiên hóa quy mô)
                        break

            # Sa thải tự nhiên nếu biên lợi nhuận cận biên âm hoặc kiệt quệ tiền mặt
            elif len(current_workers) > 0 and (firm.cash < overhead_cost or hire_signal < -0.4):
                num_to_fire = 1 if hire_signal >= -0.7 else max(1, len(current_workers) // 2)
                sorted_workers = sorted(current_workers, key=lambda w: getattr(w, 'skill_level', 1.0))

                for fired_emp in sorted_workers[:num_to_fire]:
                    deltas[firm.agent_id].setdefault("fired_employees", []).append(fired_emp.agent_id)
                    deltas[fired_emp.agent_id]["employed_by"] = None
                    deltas[fired_emp.agent_id]["wage"] = 0.0
                    claimed_workers.discard(fired_emp.agent_id)

                    self._emit_event(EventType.FIRE, firm.agent_id, fired_emp.agent_id, {"reason": "Marginal Loss"}, timestep)
                    events_map[firm.agent_id].append(EventType.FIRE.value)
                    events_map[fired_emp.agent_id].append(EventType.FIRE.value)

        # 3. SẢN XUẤT HIỆN VẬT COBB-DOUGLAS & CHI TRẢ TIỀN LƯƠNG
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
            firm_k = max(100.0, firm.capital_stock)
            overhead = CAPITAL_DEPRECIATION_RATE * expected_price * (firm_k * 0.05)
            firm_overheads[firm.agent_id] = overhead

            wage_bill = 0.0
            effective_l = 0.0

            for emp in firm_workers:
                emp_act = validated_actions.get(emp.agent_id)
                effort = float(emp_act.values[0]) if emp_act is not None else 0.6
                assigned_wage = deltas[emp.agent_id].get("wage", emp.wage)
                wage = assigned_wage if assigned_wage > 0 else (expected_price * 1.1)

                wage_bill += wage
                worker_gross_incomes[emp.agent_id] += wage
                effective_l += (effort * emp.skill_level)

                deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + wage
                deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) - (effort * 0.10 + 0.02)
                deltas[emp.agent_id]["executed_work_effort"] = effort

                self._emit_event(EventType.WAGE_PAID, firm.agent_id, emp.agent_id, {"amount": round(wage, 1)}, timestep)
                events_map[firm.agent_id].append(EventType.WAGE_PAID.value)
                events_map[emp.agent_id].append(EventType.WAGE_PAID.value)

            # Hàm sản xuất thực tế: Cobb-Douglas DRS
            if effective_l > 0.0:
                physical_q = firm.productivity_factor * (firm_k ** ALPHA_CAPITAL) * (effective_l ** BETA_LABOR)
            else:
                physical_q = 0.0

            firm_physical_outputs[firm.agent_id] = physical_q
            firm_wage_bills[firm.agent_id] = wage_bill

        total_physical_supply = sum(firm_physical_outputs.values())

        # 4. TIÊU DÙNG STONE-GEARY & CÂN BẰNG THỊ TRƯỜNG WALRAS (GODLEY & LAVOIE SFC)
        total_consumer_spending = 0.0
        for emp in active_employees:
            emp_act = validated_actions.get(emp.agent_id)
            consume_propensity = float(emp_act.values[2]) if emp_act is not None else 0.5

            current_cash_est = emp.cash + deltas[emp.agent_id].get("cash_delta", 0.0)
            subsistence_nominal_need = SUBSISTENCE_BASKET_QTY * expected_price

            # Tiêu dùng theo Hệ thống Chi tiêu Tuyến tính (Stone 1954):
            # 1. Bắt buộc mua giỏ sinh tồn: E_sub
            # 2. Tiêu dùng thặng dư dựa trên xu hướng biên tiêu dùng (MPC)
            if current_cash_est >= subsistence_nominal_need:
                surplus_cash = current_cash_est - subsistence_nominal_need
                spending = subsistence_nominal_need + (surplus_cash * 0.35 * consume_propensity)
            else:
                spending = max(0.0, current_cash_est)

            deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) - spending
            deltas[emp.agent_id]["executed_consumption"] = spending
            total_consumer_spending += spending

            self._emit_event(EventType.GOODS_PURCHASED, emp.agent_id, eco.agent_id, {"amount": round(spending, 1)}, timestep)

        # GIÁ CÂN BẰNG THỊ TRƯỜNG WALRAS: P_t = E_t / Y_t (USD / Đơn vị sản phẩm)
        market_clearing_price = total_consumer_spending / max(1.0, total_physical_supply)

        # Cập nhật mức thỏa mãn sinh học thực tế theo lượng hàng hóa mua được: q_i = E_i / P_t
        for emp in active_employees:
            spending = deltas[emp.agent_id].get("executed_consumption", 0.0)
            real_goods_bought = spending / max(0.01, market_clearing_price)
            # Ăn đủ giỏ hàng sinh tồn (>= 1.0) hồi phục thể lực, thiếu hụt bị suy nhược
            if real_goods_bought >= SUBSISTENCE_BASKET_QTY:
                energy_rec = min(0.50, 0.35 + 0.10 * (real_goods_bought - 1.0))
            else:
                energy_rec = max(-0.25, 0.35 * real_goods_bought - 0.20 * (1.0 - real_goods_bought))
            deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) + energy_rec

        # PHÂN BỔ DOANH THU KHÉP KÍN 100% SFC VỀ CÁC DOANH NGHIỆP: Revenue_j = Q_j * P_t
        total_industrial_revenue = 0.0
        for firm in active_firms:
            output_q = firm_physical_outputs.get(firm.agent_id, 0.0)
            firm_revenue = output_q * market_clearing_price
            wage_bill = firm_wage_bills.get(firm.agent_id, 0.0)
            overhead = firm_overheads.get(firm.agent_id, 0.0)
            net_profit = firm_revenue - wage_bill - overhead

            deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) + net_profit
            deltas[firm.agent_id]["executed_revenue"] = firm_revenue
            deltas[firm.agent_id]["executed_profit"] = net_profit
            total_industrial_revenue += firm_revenue

            deltas[gov.agent_id]["treasury_overhead"] = deltas[gov.agent_id].get("treasury_overhead", 0.0) + overhead

        # CẬP NHẬT KỲ VỌNG THÍCH NGHI (Friedman Adaptive Expectations)
        actual_living_cost = SUBSISTENCE_BASKET_QTY * market_clearing_price
        prev_living_cost = eco.base_living_cost
        updated_living_cost = float(0.80 * prev_living_cost + 0.20 * actual_living_cost)
        monthly_inflation = float((market_clearing_price - expected_price) / max(0.01, expected_price))

        eco.inflation_rate = monthly_inflation
        eco.base_living_cost = updated_living_cost
        deltas[eco.agent_id]["liquidity_delta"] = total_consumer_spending
        deltas[eco.agent_id]["inflation"] = monthly_inflation
        deltas[eco.agent_id]["base_living_cost"] = updated_living_cost

        # XỬ LÝ KINH TẾ PHI CHÍNH THỨC & AN SINH XÃ HỘI NỘI SINH
        total_subsidies_spent = 0.0
        total_informal_production = 0.0
        for emp in active_employees:
            effective_employer = deltas[emp.agent_id].get("employed_by", emp.employed_by)
            if effective_employer is None:
                streak = getattr(emp, "unemployed_streak", 0) + 1
                deltas[emp.agent_id]["unemployed_streak"] = streak
                deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) - min(0.10, 0.02 * streak)

                # Sản lượng tự túc phi chính thức được định giá theo thời giá thị trường
                informal_income = emp.skill_level * (0.35 * updated_living_cost)
                worker_gross_incomes[emp.agent_id] += informal_income
                deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + informal_income
                total_informal_production += informal_income

                current_estimated_cash = emp.cash + deltas[emp.agent_id].get("cash_delta", 0.0)
                if current_estimated_cash < (0.5 * updated_living_cost) and gov.treasury > 1000.0:
                    relief_amount = 0.40 * updated_living_cost if streak <= 3 else (0.20 * updated_living_cost)
                    deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + relief_amount
                    total_subsidies_spent += relief_amount
            else:
                deltas[emp.agent_id]["unemployed_streak"] = 0

        deltas[gov.agent_id]["subsidies_disbursed"] = total_subsidies_spent

        # 5. TÍN DỤNG THẾ CHẤP NỘI SINH (Kiyotaki & Moore, 1997)
        for firm in active_firms:
            f_act = validated_actions.get(firm.agent_id)
            borrow_signal = float(f_act.values[1]) if f_act is not None else 0.0

            # Hạn mức tín dụng dựa trên giá trị thế chấp tài sản tư bản (Collateral Headroom)
            firm_k = max(100.0, firm.capital_stock)
            collateral_value = 0.50 * market_clearing_price * firm_k
            borrowing_headroom = max(0.0, collateral_value - firm.debt)

            if borrow_signal > 0.4 and bank.reserves > 5000.0 and borrowing_headroom > 0.0:
                loan_request = min(borrowing_headroom, 1000.0 * borrow_signal * credit_factor)
                deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) + loan_request
                deltas[firm.agent_id]["debt_delta"] = deltas[firm.agent_id].get("debt_delta", 0.0) + loan_request
                deltas[bank.agent_id]["loans_delta"] = deltas[bank.agent_id].get("loans_delta", 0.0) + loan_request
                deltas[bank.agent_id]["reserves_delta"] = deltas[bank.agent_id].get("reserves_delta", 0.0) - loan_request

                self._emit_event(EventType.LOAN_DISBURSED, bank.agent_id, firm.agent_id, {"amount": round(loan_request, 1)}, timestep)
                events_map[bank.agent_id].append(EventType.LOAN_DISBURSED.value)
                events_map[firm.agent_id].append(EventType.LOAN_DISBURSED.value)

            if firm.debt > 0.0:
                monthly_interest = firm.debt * (lending_rate / 12.0)
                principal_repayment = 0.0
                if deltas[firm.agent_id].get("executed_profit", 0.0) > 0 and firm.cash > overhead_cost:
                    principal_repayment = min(firm.debt, firm.debt * 0.05)

                total_bank_payment = monthly_interest + principal_repayment
                deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) - total_bank_payment
                deltas[firm.agent_id]["debt_delta"] = deltas[firm.agent_id].get("debt_delta", 0.0) - principal_repayment

                deltas[bank.agent_id]["interest_income"] = deltas[bank.agent_id].get("interest_income", 0.0) + monthly_interest
                deltas[bank.agent_id]["reserves_delta"] = deltas[bank.agent_id].get("reserves_delta", 0.0) + total_bank_payment
                deltas[bank.agent_id]["loans_delta"] = deltas[bank.agent_id].get("loans_delta", 0.0) - principal_repayment

        # 6. THUẾ VÀ GIAN LẬN NỘI SINH
        total_tax_collected = 0.0
        tax_evasion_records: List[Tuple[BaseAgent, float, float]] = []

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

            if evaded_tax > 0.01:
                hidden_pct = round((1.0 - declare_ratio) * 100, 1)
                tax_evasion_records.append((emp, evaded_tax, declare_ratio))
                self._emit_event(
                    EventType.TAX_EVADED, 
                    emp.agent_id, 
                    gov.agent_id, 
                    {"amount": round(evaded_tax, 1), "gross": round(actual_gross, 1), "hidden_pct": hidden_pct}, 
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

            if evaded_tax > 0.01:
                hidden_pct = round((1.0 - declare_ratio) * 100, 1)
                tax_evasion_records.append((firm, evaded_tax, declare_ratio))
                self._emit_event(
                    EventType.TAX_EVADED, 
                    firm.agent_id, 
                    gov.agent_id, 
                    {"amount": round(evaded_tax, 1), "profit": round(profit, 1), "hidden_pct": hidden_pct}, 
                    timestep
                )

        deltas[gov.agent_id]["tax_collected"] = total_tax_collected
        self._emit_event(EventType.TAX_COLLECTED, "MARKET", gov.agent_id, {"amount": round(total_tax_collected, 1)}, timestep)

        # 7. THANH TRA & CHẾ TÀI
        audits_count = 0
        violations_count = 0
        total_fines_collected = 0.0

        for agent, evaded_amount, _ in tax_evasion_records:
            if isinstance(agent, Firm) and getattr(agent, "age_months", 99) <= 6:
                continue

            if np.random.rand() < audit_rate:
                audits_count += 1
                violations_count += 1
                fine = evaded_amount * fine_multiplier
                deltas[agent.agent_id]["cash_delta"] = deltas[agent.agent_id].get("cash_delta", 0.0) - fine
                total_fines_collected += fine

                self._emit_event(
                    EventType.PENALTY_ENFORCED, 
                    sup.agent_id, 
                    agent.agent_id, 
                    {"fine": round(fine, 1), "evaded": round(evaded_amount, 1)}, 
                    timestep
                )
                events_map[sup.agent_id].append(EventType.PENALTY_ENFORCED.value)
                events_map[agent.agent_id].append(EventType.PENALTY_ENFORCED.value)

        deltas[sup.agent_id]["audits_conducted"] = audits_count
        deltas[sup.agent_id]["violations_detected"] = violations_count
        deltas[sup.agent_id]["fines_collected"] = total_fines_collected

        net_gov_inflow = total_tax_collected + total_fines_collected + deltas[gov.agent_id].get("treasury_overhead", 0.0) - total_subsidies_spent
        deltas[gov.agent_id]["treasury_delta"] = net_gov_inflow

        # 8. VỠ NỢ CẤU TRÚC MERTON (1974) & SINH HỌC LAO ĐỘNG
        total_defaults = 0.0
        HAIRCUT_LIQUIDATION = 0.30

        for firm in active_firms:
            projected_cash = firm.cash + deltas[firm.agent_id].get("cash_delta", 0.0)
            projected_debt = firm.debt + deltas[firm.agent_id].get("debt_delta", 0.0)
            firm_k = max(100.0, firm.capital_stock)
            firm_age = getattr(firm, "age_months", 99)

            # Mô hình Merton: Giá trị ròng tài sản (Net Worth) = Tiền mặt + Giá trị tư bản phát mại - Nợ
            asset_liquidation_value = (1.0 - HAIRCUT_LIQUIDATION) * market_clearing_price * (firm_k * 0.1)
            net_worth = projected_cash + asset_liquidation_value - projected_debt

            is_insolvent = (
                (len(firm.employee_ids) == 0 and projected_cash <= 0.0 and firm_age > 6) or
                (net_worth < 0.0 and projected_cash < -overhead_cost)
            )

            if is_insolvent:
                deltas[firm.agent_id]["status"] = LifeCycleStatus.BANKRUPT

                fired_list = [eid for eid in firm.employee_ids if eid in deltas]
                deltas[firm.agent_id]["fired_employees"] = fired_list
                for emp_id in fired_list:
                    deltas[emp_id]["employed_by"] = None
                    deltas[emp_id]["wage"] = 0.0
                    self._emit_event(EventType.FIRE, firm.agent_id, emp_id, {"reason": "Firm Insolvency"}, timestep)

                total_defaults += projected_debt
                deltas[firm.agent_id]["debt_delta"] = -firm.debt
                deltas[firm.agent_id]["cash_delta"] = -firm.cash
                self._emit_event(
                    EventType.AGENT_BANKRUPT, 
                    firm.agent_id, 
                    bank.agent_id, 
                    {"bad_debt": round(projected_debt, 1), "cash": round(firm.cash, 1)}, 
                    timestep
                )
                events_map[firm.agent_id].append(EventType.AGENT_BANKRUPT.value)

        deltas[bank.agent_id]["new_defaults"] = total_defaults

        new_deaths = 0
        for emp in active_employees:
            projected_cash = emp.cash + deltas[emp.agent_id].get("cash_delta", 0.0)
            projected_energy = emp.energy + deltas[emp.agent_id].get("energy_delta", 0.0)
            deltas[emp.agent_id]["age_increment"] = 1 if (timestep % 12 == 0) else 0

            # Ngưỡng tử vong: Kiệt quệ sinh học (Energy <= 0) hoặc Nợ vượt khả năng sinh tồn (Thâm hụt > 10 tháng lương thực)
            debt_survival_limit = -10.0 * updated_living_cost
            if (projected_energy <= 0.0) or (projected_cash < debt_survival_limit and projected_energy < 0.15) or ((emp.age + deltas[emp.agent_id]["age_increment"]) >= emp.max_age):
                new_deaths += 1
                deltas[emp.agent_id]["status"] = LifeCycleStatus.DECEASED

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

        # 9. ĐO LƯỜNG VĨ MÔ CHUẨN HÓA (NOMINAL GDP & GINI)
        active_wealths = [
            max(0.01, e.cash + deltas[e.agent_id].get("cash_delta", 0.0)) 
            for e in active_employees 
            if deltas[e.agent_id].get("status") not in [LifeCycleStatus.DECEASED, LifeCycleStatus.DEAD, LifeCycleStatus.TERMINATED]
        ]
        deltas[gov.agent_id]["current_gini"] = self._compute_gini(active_wealths)
        deltas[gov.agent_id]["current_gdp"] = total_industrial_revenue + total_informal_production

        # 10. ĐÓNG GÓI CHUYỂN DỊCH HỢP LỆ
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
    def _compute_gini(wealth_array: List[float]) -> float:
        if len(wealth_array) < 2:
            return 0.0
        sorted_w = np.sort(np.asarray(wealth_array, dtype=np.float64))
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