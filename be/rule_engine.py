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
    Hien phap kinh te the che v1.0.
    Tuan thu nghiem ngat: Tach bach GDP khoi Kho bac, ap dat chi phi co dinh
    cho Doanh nghiep, va ap dung co che tro cap an sinh co dieu kien.
    """
    def __init__(self, event_bus: EventBus):
        self.event_bus: EventBus = event_bus

    def execute_cycle(self, 
                      agents: Dict[str, BaseAgent], 
                      validated_actions: Dict[str, Action], 
                      timestep: int) -> Dict[str, TransitionResult]:
        deltas: Dict[str, Dict[str, Any]] = {agent_id: {} for agent_id in agents.keys()}
        events_map: Dict[str, List[str]] = {agent_id: [] for agent_id in agents.keys()}

        # 1. THE CHE VI MO
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

        eco_act = validated_actions.get(eco.agent_id)
        if eco_act is not None:
            deltas[eco.agent_id]["executed_cost_factor"] = float(eco_act.values[0])
            deltas[eco.agent_id]["executed_housing_factor"] = float(eco_act.values[1])
            deltas[eco.agent_id]["executed_housing_supply"] = int(eco_act.values[2])
            living_cost = eco.base_living_cost * float(eco_act.values[0])
        else:
            living_cost = eco.base_living_cost

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

        active_employees = [a for a in agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE]
        active_firms = [a for a in agents.values() if isinstance(a, Firm) and a.status == LifeCycleStatus.ACTIVE]

        # 2. THI TRUONG LAO DONG, SAN XUAT & CHI PHI CO DINH
        # Tuyen dung / Sa thai
        for firm in active_firms:
            f_act = validated_actions.get(firm.agent_id)
            if f_act is None:
                continue
            hire_signal = float(f_act.values[0])
            if hire_signal > 0.15:
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
                fired_emp_id = firm.employee_ids[-1]
                deltas[firm.agent_id].setdefault("fired_employees", []).append(fired_emp_id)
                deltas[fired_emp_id]["employed_by"] = None
                deltas[fired_emp_id]["wage"] = 0.0
                self._emit_event(EventType.FIRE, firm.agent_id, fired_emp_id, {}, timestep)
                events_map[firm.agent_id].append(EventType.FIRE.value)
                events_map[fired_emp_id].append(EventType.FIRE.value)

        total_market_production = 0.0
        for firm in active_firms:
            firm_employees = [e for e in active_employees if e.employed_by == firm.agent_id or e.agent_id in deltas[firm.agent_id].get("hired_employees", [])]
            firm_wage_bill = 0.0
            firm_output = 0.0

            for emp in firm_employees:
                emp_act = validated_actions.get(emp.agent_id)
                effort = float(emp_act.values[0]) if emp_act is not None else 0.5
                wage = emp.wage if emp.wage > 0 else 35.0

                firm_wage_bill += wage
                deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + wage
                deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) - ((effort * 0.25) + 0.05)
                deltas[emp.agent_id]["executed_work_effort"] = effort
                
                firm_output += (effort * emp.skill_level * 50.0 * firm.productivity_factor)
                self._emit_event(EventType.WAGE_PAID, firm.agent_id, emp.agent_id, {"amount": wage}, timestep)
                events_map[firm.agent_id].append(EventType.WAGE_PAID.value)
                events_map[emp.agent_id].append(EventType.WAGE_PAID.value)

            # Chi phi van hanh co dinh hang thang (Fixed Overhead)
            fixed_overhead = 35.0
            revenue = firm_output * 1.10
            net_firm_income = revenue - firm_wage_bill - fixed_overhead

            deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) + net_firm_income
            deltas[firm.agent_id]["executed_revenue"] = revenue
            deltas[firm.agent_id]["executed_profit"] = net_firm_income
            total_market_production += revenue

        # Lao dong tu do
        for emp in active_employees:
            if emp.employed_by is None and "employed_by" not in deltas[emp.agent_id]:
                emp_act = validated_actions.get(emp.agent_id)
                effort = float(emp_act.values[0]) if emp_act is not None else 0.3
                informal_income = effort * emp.skill_level * 18.0
                deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + informal_income
                deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) - ((effort * 0.15) + 0.05)
                deltas[emp.agent_id]["executed_work_effort"] = effort
                total_market_production += informal_income

        # 3. TIEU DUNG & AN SINH XA HOI CO DIEU KIEN
        total_consumer_spending = 0.0
        for emp in active_employees:
            emp_act = validated_actions.get(emp.agent_id)
            consume_ratio = float(emp_act.values[2]) if emp_act is not None else 0.5
            cost = living_cost * (0.8 + 0.4 * consume_ratio)
            
            deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) - cost
            deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) + (0.35 if emp.cash > cost else 0.05)
            deltas[emp.agent_id]["executed_consumption"] = cost
            total_consumer_spending += cost

            self._emit_event(EventType.GOODS_PURCHASED, emp.agent_id, eco.agent_id, {"amount": cost}, timestep)

        deltas[eco.agent_id]["liquidity_delta"] = total_consumer_spending
        deltas[eco.agent_id]["total_market_turnover"] = total_consumer_spending + total_market_production

        # Goi cuu tro an sinh co muc tieu (Targeted Welfare)
        total_subsidies_spent = 0.0
        for emp in active_employees:
            current_estimated_cash = emp.cash + deltas[emp.agent_id].get("cash_delta", 0.0)
            current_estimated_energy = emp.energy + deltas[emp.agent_id].get("energy_delta", 0.0)
            
            # Chi tro cap neu can ke pha san hoac kiet suc
            if (current_estimated_cash < 30.0 or current_estimated_energy < 0.3) and gov.treasury > 1000.0:
                relief_amount = 30.0
                deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + relief_amount
                total_subsidies_spent += relief_amount

        deltas[gov.agent_id]["subsidies_disbursed"] = total_subsidies_spent

        # 4. TIN DUNG NGAN HANG
        for firm in active_firms:
            f_act = validated_actions.get(firm.agent_id)
            borrow_signal = float(f_act.values[1]) if f_act is not None else 0.0

            if borrow_signal > 0.4 and bank.reserves > 5000.0:
                loan_request = 500.0 * borrow_signal * credit_factor
                deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) + loan_request
                deltas[firm.agent_id]["debt_delta"] = deltas[firm.agent_id].get("debt_delta", 0.0) + loan_request
                deltas[bank.agent_id]["loans_delta"] = deltas[bank.agent_id].get("loans_delta", 0.0) + loan_request
                deltas[bank.agent_id]["reserves_delta"] = deltas[bank.agent_id].get("reserves_delta", 0.0) - loan_request

                self._emit_event(EventType.LOAN_DISBURSED, bank.agent_id, firm.agent_id, {"amount": loan_request}, timestep)
                events_map[bank.agent_id].append(EventType.LOAN_DISBURSED.value)
                events_map[firm.agent_id].append(EventType.LOAN_DISBURSED.value)

            if firm.debt > 0.0:
                monthly_interest = firm.debt * (lending_rate / 12.0)
                deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) - monthly_interest
                deltas[bank.agent_id]["interest_income"] = deltas[bank.agent_id].get("interest_income", 0.0) + monthly_interest
                deltas[bank.agent_id]["reserves_delta"] = deltas[bank.agent_id].get("reserves_delta", 0.0) + monthly_interest

        # 5. THUE VA GIAN LAN
        total_tax_collected = 0.0
        tax_evasion_records: List[Tuple[BaseAgent, float, float]] = []

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

        # 6. THANH TRA
        audits_count = 0
        violations_count = 0
        total_fines_collected = 0.0

        for agent, evaded_amount, _ in tax_evasion_records:
            if np.random.rand() < audit_rate:
                audits_count += 1
                violations_count += 1
                fine = evaded_amount * fine_multiplier
                deltas[agent.agent_id]["cash_delta"] = deltas[agent.agent_id].get("cash_delta", 0.0) - fine
                total_fines_collected += fine
                self._emit_event(EventType.PENALTY_ENFORCED, sup.agent_id, agent.agent_id, {"fine": fine}, timestep)
                events_map[sup.agent_id].append(EventType.PENALTY_ENFORCED.value)
                events_map[agent.agent_id].append(EventType.PENALTY_ENFORCED.value)

        deltas[sup.agent_id]["audits_conducted"] = audits_count
        deltas[sup.agent_id]["violations_detected"] = violations_count
        deltas[sup.agent_id]["fines_collected"] = total_fines_collected
        deltas[gov.agent_id]["tax_collected"] = deltas[gov.agent_id].get("tax_collected", 0.0) + total_fines_collected

        # 7. PHA SAN VA SINH TU
        total_defaults = 0.0
        for firm in active_firms:
            projected_cash = firm.cash + deltas[firm.agent_id].get("cash_delta", 0.0)
            projected_debt = firm.debt + deltas[firm.agent_id].get("debt_delta", 0.0)

            if projected_cash < 0.0 and (projected_debt > (firm.capital_stock * 1.5) or projected_cash < -500.0):
                for emp_id in firm.employee_ids:
                    deltas[emp_id]["employed_by"] = None
                    deltas[emp_id]["wage"] = 0.0
                    self._emit_event(EventType.FIRE, firm.agent_id, emp_id, {"reason": "Firm bankruptcy"}, timestep)

                total_defaults += projected_debt
                deltas[firm.agent_id]["debt_delta"] = -firm.debt
                self._emit_event(EventType.AGENT_BANKRUPT, firm.agent_id, bank.agent_id, {"bad_debt": projected_debt}, timestep)
                events_map[firm.agent_id].append(EventType.AGENT_BANKRUPT.value)

        deltas[bank.agent_id]["new_defaults"] = total_defaults

        new_deaths = 0
        for emp in active_employees:
            projected_cash = emp.cash + deltas[emp.agent_id].get("cash_delta", 0.0)
            projected_energy = emp.energy + deltas[emp.agent_id].get("energy_delta", 0.0)
            deltas[emp.agent_id]["age_increment"] = 1 if (timestep % 12 == 0) else 0

            if projected_cash < -300.0 or projected_energy <= 0.0 or (emp.age + deltas[emp.agent_id]["age_increment"]) >= emp.max_age:
                new_deaths += 1
                if emp.employed_by and emp.employed_by in deltas:
                    deltas[emp.employed_by].setdefault("fired_employees", []).append(emp.agent_id)
                self._emit_event(EventType.AGENT_DIED, emp.agent_id, gov.agent_id, {}, timestep)
                events_map[emp.agent_id].append(EventType.AGENT_DIED.value)

        deltas[gov.agent_id]["new_deaths"] = new_deaths

        # 8. GDP CHUAN VA HE SO GINI
        active_wealths = [max(0.01, e.cash + deltas[e.agent_id].get("cash_delta", 0.0)) for e in active_employees if (e.cash + deltas[e.agent_id].get("cash_delta", 0.0)) > -300.0]
        deltas[gov.agent_id]["current_gini"] = self._compute_gini(active_wealths)
        
        # GDP thuc te = Tong gia tri san pham tao ra + Tieu dung
        deltas[gov.agent_id]["current_gdp"] = total_market_production + total_consumer_spending

        # 9. DONG GOI TRANSITIONS
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