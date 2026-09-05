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
    Hien phap cua he thong mo phong (Constitution & Physical Laws).
    Dieu phoi chuyen dich trang thai toan cuc, dam bao bao toan dong tien khep kin,
    thuc thi luat thue, pha san, tin dung, kiem toan va sinh tu.
    """
    def __init__(self, event_bus: EventBus):
        self.event_bus: EventBus = event_bus

    def execute_cycle(self, 
                      agents: Dict[str, BaseAgent], 
                      validated_actions: Dict[str, Action], 
                      timestep: int) -> Dict[str, TransitionResult]:
        """
        Thuc thi toan bo chu ky theo cac giai doan nghiem ngat:
        1. Cap nhat the che vi mo (Gov, Bank, Economy, Supervisor)
        2. Thi truong lao dong & San xuat (Employment, Wages, Production)
        3. Tieu dung & Sinh hoat phi (Living cost, Subsidies)
        4. Thi truong tin dung & Tai chinh (Borrowing, Repayment)
        5. Thu thue & Gian lan (Taxation, Evasion tracking)
        6. Thanh tra & Xu phat (Audits, Fines, Confiscation)
        7. Giai the, Pha san & Sinh tu (Bankruptcy, Default, Deaths)
        """
        # Khoi tao bo dem ket qua chuyen dich cho tung tac tu
        deltas: Dict[str, Dict[str, Any]] = {agent_id: {} for agent_id in agents.keys()}
        events_map: Dict[str, List[str]] = {agent_id: [] for agent_id in agents.keys()}

        # 1. CAP NHAT THE CHE VI MO
        gov = self._get_single_agent(agents, Government)
        bank = self._get_single_agent(agents, Bank)
        eco = self._get_single_agent(agents, Economy)
        sup = self._get_single_agent(agents, Supervisor)

        # Chinh phu ban hanh thue suat va tro cap
        gov_act = validated_actions.get(gov.agent_id)
        if gov_act is not None:
            deltas[gov.agent_id]["executed_worker_tax"] = float(gov_act.values[0])
            deltas[gov.agent_id]["executed_firm_tax"] = float(gov_act.values[1])
            deltas[gov.agent_id]["executed_subsidy_ratio"] = float(gov_act.values[2])
            worker_tax_rate = float(gov_act.values[0])
            firm_tax_rate = float(gov_act.values[1])
            subsidy_ratio = float(gov_act.values[2])
        else:
            worker_tax_rate = gov.tax_rate_worker
            firm_tax_rate = gov.tax_rate_firm
            subsidy_ratio = gov.subsidy_budget_ratio

        # Ngan hang thiet lap lai suat
        bank_act = validated_actions.get(bank.agent_id)
        if bank_act is not None:
            deltas[bank.agent_id]["executed_lending_rate"] = float(bank_act.values[0])
            deltas[bank.agent_id]["executed_deposit_rate"] = float(bank_act.values[1])
            deltas[bank.agent_id]["executed_credit_factor"] = float(bank_act.values[2])
            lending_rate = float(bank_act.values[0])
            deposit_rate = float(bank_act.values[1])
            credit_factor = float(bank_act.values[2])
        else:
            lending_rate = bank.lending_rate
            deposit_rate = bank.deposit_rate
            credit_factor = bank.credit_expansion_factor

        # Economy dinh gia ro hang hoa
        eco_act = validated_actions.get(eco.agent_id)
        if eco_act is not None:
            deltas[eco.agent_id]["executed_cost_factor"] = float(eco_act.values[0])
            deltas[eco.agent_id]["executed_housing_factor"] = float(eco_act.values[1])
            deltas[eco.agent_id]["executed_housing_supply"] = int(eco_act.values[2])
            living_cost = eco.base_living_cost * float(eco_act.values[0])
        else:
            living_cost = eco.base_living_cost

        # Supervisor thiet lap muc do thanh tra
        sup_act = validated_actions.get(sup.agent_id)
        if sup_act is not None:
            deltas[sup.agent_id]["executed_audit_rate"] = float(sup_act.values[0])
            deltas[sup.agent_id]["executed_fine_multiplier"] = float(sup_act.values[1])
            deltas[sup.agent_id]["executed_target_firm_ratio"] = float(sup_act.values[2])
            audit_rate = float(sup_act.values[0])
            fine_multiplier = float(sup_act.values[1])
            target_firm_ratio = float(sup_act.values[2])
        else:
            audit_rate = sup.audit_rate
            fine_multiplier = sup.fine_multiplier
            target_firm_ratio = sup.target_firm_ratio

        # 2. THI TRUONG LAO DONG & SAN XUAT
        active_employees = [a for a in agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE]
        active_firms = [a for a in agents.values() if isinstance(a, Firm) and a.status == LifeCycleStatus.ACTIVE]

        # Tuyen dung / Sa thai theo quyet dinh cua Doanh nghiep
        for firm in active_firms:
            f_act = validated_actions.get(firm.agent_id)
            if f_act is None:
                continue

            hire_signal = float(f_act.values[0])
            if hire_signal > 0.3:
                # Tim kiem ung vien chua co viec lam
                unemployed = [e for e in active_employees if e.employed_by is None and e.agent_id not in deltas[firm.agent_id].get("hired_employees", [])]
                if unemployed:
                    target_emp = unemployed[0]
                    deltas[firm.agent_id].setdefault("hired_employees", []).append(target_emp.agent_id)
                    deltas[target_emp.agent_id]["employed_by"] = firm.agent_id
                    deltas[target_emp.agent_id]["wage"] = 35.0 * target_emp.skill_level
                    
                    self._emit_event(EventType.HIRE, firm.agent_id, target_emp.agent_id, {"wage": deltas[target_emp.agent_id]["wage"]}, timestep)
                    events_map[firm.agent_id].append(EventType.HIRE.value)
                    events_map[target_emp.agent_id].append(EventType.HIRE.value)

            elif hire_signal < -0.3 and len(firm.employee_ids) > 0:
                # Sa thai bot nhan vien de cat giam chi phi
                fired_emp_id = firm.employee_ids[-1]
                deltas[firm.agent_id].setdefault("fired_employees", []).append(fired_emp_id)
                deltas[fired_emp_id]["employed_by"] = None
                deltas[fired_emp_id]["wage"] = 0.0

                self._emit_event(EventType.FIRE, firm.agent_id, fired_emp_id, {}, timestep)
                events_map[firm.agent_id].append(EventType.FIRE.value)
                events_map[fired_emp_id].append(EventType.FIRE.value)

        # Lao dong thuc hien cong viec, Doanh nghiep tra luong va san xuat
        total_market_production = 0.0
        for firm in active_firms:
            firm_employees = [e for e in active_employees if e.employed_by == firm.agent_id or e.agent_id in deltas[firm.agent_id].get("hired_employees", [])]
            firm_wage_bill = 0.0
            firm_output = 0.0

            for emp in firm_employees:
                emp_act = validated_actions.get(emp.agent_id)
                effort = float(emp_act.values[0]) if emp_act is not None else 0.5
                wage = emp.wage if emp.wage > 0 else 35.0

                # Trừ quỹ tiền của Firm, trả lương cho Employee
                firm_wage_bill += wage
                deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + wage
                deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) - ((effort * 0.25) + 0.05)
                deltas[emp.agent_id]["executed_work_effort"] = effort
                
                firm_output += (effort * emp.skill_level * 50.0 * firm.productivity_factor)

                self._emit_event(EventType.WAGE_PAID, firm.agent_id, emp.agent_id, {"amount": wage}, timestep)
                events_map[firm.agent_id].append(EventType.WAGE_PAID.value)
                events_map[emp.agent_id].append(EventType.WAGE_PAID.value)

            revenue = firm_output * 1.10
            deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) + revenue - firm_wage_bill
            deltas[firm.agent_id]["executed_revenue"] = revenue
            deltas[firm.agent_id]["executed_profit"] = revenue - firm_wage_bill
            total_market_production += revenue

        # Lao dong tu do (khong co cong ty tuyen dung)
        for emp in active_employees:
            if emp.employed_by is None and "employed_by" not in deltas[emp.agent_id]:
                emp_act = validated_actions.get(emp.agent_id)
                effort = float(emp_act.values[0]) if emp_act is not None else 0.3
                informal_income = effort * emp.skill_level * 18.0

                deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + informal_income
                deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) - ((effort * 0.15) + 0.05)
                deltas[emp.agent_id]["executed_work_effort"] = effort

        # 3. TIEU DUNG & SINH HOAT PHI
        total_market_turnover = 0.0
        for emp in active_employees:
            emp_act = validated_actions.get(emp.agent_id)
            consume_ratio = float(emp_act.values[2]) if emp_act is not None else 0.5
            
            # Sinh hoat phi bat buoc nop cho Economy
            cost = living_cost * (0.8 + 0.4 * consume_ratio)
            deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) - cost
            deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) + (0.35 if emp.cash > cost else 0.05)
            deltas[emp.agent_id]["executed_consumption"] = cost
            total_market_turnover += cost

            self._emit_event(EventType.GOODS_PURCHASED, emp.agent_id, eco.agent_id, {"amount": cost}, timestep)

        deltas[eco.agent_id]["liquidity_delta"] = total_market_turnover
        deltas[eco.agent_id]["total_market_turnover"] = total_market_turnover + total_market_production

        # Trợ cấp Chính phủ
        subsidy_pool = gov.treasury * subsidy_ratio
        subsidy_per_citizen = (subsidy_pool / len(active_employees)) if active_employees else 0.0
        deltas[gov.agent_id]["subsidies_disbursed"] = subsidy_pool
        for emp in active_employees:
            deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + subsidy_per_citizen

        # 4. THI TRUONG TIN DUNG & TAI CHINH
        total_loans_requested = 0.0
        for firm in active_firms:
            f_act = validated_actions.get(firm.agent_id)
            borrow_signal = float(f_act.values[1]) if f_act is not None else 0.0

            if borrow_signal > 0.4 and bank.reserves > 5000.0:
                loan_request = 500.0 * borrow_signal * credit_factor
                total_loans_requested += loan_request

                deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) + loan_request
                deltas[firm.agent_id]["debt_delta"] = deltas[firm.agent_id].get("debt_delta", 0.0) + loan_request
                
                deltas[bank.agent_id]["loans_delta"] = deltas[bank.agent_id].get("loans_delta", 0.0) + loan_request
                deltas[bank.agent_id]["reserves_delta"] = deltas[bank.agent_id].get("reserves_delta", 0.0) - loan_request

                self._emit_event(EventType.LOAN_DISBURSED, bank.agent_id, firm.agent_id, {"amount": loan_request}, timestep)
                events_map[bank.agent_id].append(EventType.LOAN_DISBURSED.value)
                events_map[firm.agent_id].append(EventType.LOAN_DISBURSED.value)

            # Thu lai vay hang thang
            if firm.debt > 0.0:
                monthly_interest = firm.debt * (lending_rate / 12.0)
                deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) - monthly_interest
                deltas[bank.agent_id]["interest_income"] = deltas[bank.agent_id].get("interest_income", 0.0) + monthly_interest
                deltas[bank.agent_id]["reserves_delta"] = deltas[bank.agent_id].get("reserves_delta", 0.0) + monthly_interest

        # 5. THU THUE & GIAN LAN
        total_tax_collected = 0.0
        tax_evasion_records: List[Tuple[BaseAgent, float, float]] = []

        # Thu thuế Người lao động
        for emp in active_employees:
            emp_act = validated_actions.get(emp.agent_id)
            declare_ratio = float(emp_act.values[1]) if emp_act is not None else 1.0
            actual_income = max(0.0, deltas[emp.agent_id].get("cash_delta", 0.0))
            
            taxable_income = actual_income * declare_ratio
            tax_due = taxable_income * worker_tax_rate
            evaded_tax = (actual_income - taxable_income) * worker_tax_rate

            deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) - tax_due
            deltas[emp.agent_id]["executed_declare_ratio"] = declare_ratio
            total_tax_collected += tax_due

            if evaded_tax > 0.01:
                tax_evasion_records.append((emp, evaded_tax, declare_ratio))
                self._emit_event(EventType.TAX_EVADED, emp.agent_id, gov.agent_id, {"amount": evaded_tax}, timestep)

        # Thu thuế Doanh nghiệp
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
                tax_evasion_records.append((firm, evaded_tax, declare_ratio))
                self._emit_event(EventType.TAX_EVADED, firm.agent_id, gov.agent_id, {"amount": evaded_tax}, timestep)

        deltas[gov.agent_id]["tax_collected"] = total_tax_collected
        self._emit_event(EventType.TAX_COLLECTED, "MARKET", gov.agent_id, {"amount": total_tax_collected}, timestep)

        # 6. THANH TRA & XU PHAT
        audits_count = 0
        violations_count = 0
        total_fines_collected = 0.0

        # Lấy mẫu kiểm tra dựa trên audit_rate
        for agent, evaded_amount, _ in tax_evasion_records:
            if np.random.rand() < audit_rate:
                audits_count += 1
                violations_count += 1
                fine = evaded_amount * fine_multiplier
                
                # Cưỡng chế trừ tiền phạt
                deltas[agent.agent_id]["cash_delta"] = deltas[agent.agent_id].get("cash_delta", 0.0) - fine
                total_fines_collected += fine

                self._emit_event(EventType.AUDIT_CONDUCTED, sup.agent_id, agent.agent_id, {}, timestep)
                self._emit_event(EventType.PENALTY_ENFORCED, sup.agent_id, agent.agent_id, {"fine": fine}, timestep)
                events_map[sup.agent_id].append(EventType.PENALTY_ENFORCED.value)
                events_map[agent.agent_id].append(EventType.PENALTY_ENFORCED.value)

        deltas[sup.agent_id]["audits_conducted"] = audits_count
        deltas[sup.agent_id]["violations_detected"] = violations_count
        deltas[sup.agent_id]["fines_collected"] = total_fines_collected
        # Tiền phạt chuyển vào ngân khố Chính phủ
        deltas[gov.agent_id]["tax_collected"] = deltas[gov.agent_id].get("tax_collected", 0.0) + total_fines_collected

        # 7. GIAI THE, PHA SAN & SINH TU
        # Doanh nghiệp vỡ nợ
        total_defaults = 0.0
        for firm in active_firms:
            projected_cash = firm.cash + deltas[firm.agent_id].get("cash_delta", 0.0)
            projected_debt = firm.debt + deltas[firm.agent_id].get("debt_delta", 0.0)

            if projected_cash < 0.0 and projected_debt > (firm.capital_stock * 2.0 + 500.0):
                # Sa thải toàn bộ nhân viên ngay lập tức
                for emp_id in firm.employee_ids:
                    deltas[emp_id]["employed_by"] = None
                    deltas[emp_id]["wage"] = 0.0
                    self._emit_event(EventType.FIRE, firm.agent_id, emp_id, {"reason": "Firm bankruptcy"}, timestep)

                # Nợ của Firm biến thành tổn thất nợ xấu cho Bank
                total_defaults += projected_debt
                deltas[firm.agent_id]["debt_delta"] = -firm.debt
                
                self._emit_event(EventType.AGENT_BANKRUPT, firm.agent_id, bank.agent_id, {"bad_debt": projected_debt}, timestep)
                events_map[firm.agent_id].append(EventType.AGENT_BANKRUPT.value)

        deltas[bank.agent_id]["new_defaults"] = total_defaults

        # Người lao động tử vong (do cạn kiệt tài nguyên hoặc già yếu)
        new_deaths = 0
        for emp in active_employees:
            projected_cash = emp.cash + deltas[emp.agent_id].get("cash_delta", 0.0)
            projected_energy = emp.energy + deltas[emp.agent_id].get("energy_delta", 0.0)

            # Tăng 1 tháng tuổi
            deltas[emp.agent_id]["age_increment"] = 1

            if projected_cash < -300.0 or projected_energy <= 0.0 or (emp.age + 1) >= emp.max_age:
                new_deaths += 1
                # Nếu đang có việc làm, giải phóng vị trí
                if emp.employed_by and emp.employed_by in deltas:
                    deltas[emp.employed_by].setdefault("fired_employees", []).append(emp.agent_id)

                self._emit_event(EventType.AGENT_DIED, emp.agent_id, gov.agent_id, {}, timestep)
                events_map[emp.agent_id].append(EventType.AGENT_DIED.value)

        deltas[gov.agent_id]["new_deaths"] = new_deaths

        # 8. CAP NHAT CHI SO VI MO CHUNG
        active_wealths = [max(0.01, e.cash + deltas[e.agent_id].get("cash_delta", 0.0)) for e in active_employees if (e.cash + deltas[e.agent_id].get("cash_delta", 0.0)) > -300.0]
        gini = self._compute_gini(active_wealths)
        deltas[gov.agent_id]["current_gini"] = gini
        deltas[gov.agent_id]["current_gdp"] = gov.treasury + total_tax_collected - subsidy_pool + total_market_production

        # 9. DONG GOI TRANSITION RESULTS
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