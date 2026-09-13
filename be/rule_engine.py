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
    Hien phap kinh te the che v1.5.
    - Dinh gia thi truong dong (Supply-Demand Inflation Engine).
    - Luong toi thieu bao toan the luc dua tren chi phi sinh hoat (Efficiency Wage).
    - Triet tieu hoan toan Multi-Hiring & Bao toan ke toan vi mo SFC.
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
        eco_cost_factor = float(eco_act.values[0]) if eco_act is not None else 1.0
        living_cost = max(15.0, min(120.0, eco.base_living_cost * eco_cost_factor))

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

        # 2. THI TRUONG LAO DONG: LUONG HIEU DUNG THEO CHI PHI SINH HOAT
        FIXED_OVERHEAD = 35.0
        # Luong co ban dam bao du bu dap chi phi sinh hoat va thue
        BASE_WAGE_ESTIMATE = max(35.0, living_cost * 1.15)

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
            current_wage_bill = sum(
                e.wage if getattr(e, "wage", 0.0) > 0.0 else BASE_WAGE_ESTIMATE 
                for e in current_workers
            )

            unemployed = [e for e in active_employees if e.agent_id not in claimed_workers]
            unemployed.sort(key=lambda w: getattr(w, 'skill_level', 1.0), reverse=True)

            safety_reserve = FIXED_OVERHEAD + (current_wage_bill * 1.2)
            available_cash = max(0.0, firm.cash - safety_reserve)

            if available_cash > (BASE_WAGE_ESTIMATE * 1.5) and unemployed and hire_signal > -0.1:
                target_hire_count = min(len(unemployed), max(1, int(available_cash // (BASE_WAGE_ESTIMATE * 1.5))))

                for target_emp in unemployed[:target_hire_count]:
                    claimed_workers.add(target_emp.agent_id)
                    competitive_wage = BASE_WAGE_ESTIMATE * (0.8 + 0.4 * target_emp.skill_level)
                    
                    deltas[firm.agent_id].setdefault("hired_employees", []).append(target_emp.agent_id)
                    deltas[target_emp.agent_id]["employed_by"] = firm.agent_id
                    deltas[target_emp.agent_id]["wage"] = competitive_wage
                    deltas[target_emp.agent_id]["unemployed_streak"] = 0

                    self._emit_event(EventType.HIRE, firm.agent_id, target_emp.agent_id, {"wage": competitive_wage}, timestep)
                    events_map[firm.agent_id].append(EventType.HIRE.value)
                    events_map[target_emp.agent_id].append(EventType.HIRE.value)

            elif current_headcount > 0 and (firm.cash < 50.0 or hire_signal < -0.4):
                num_to_fire = 1 if hire_signal >= -0.7 else max(1, current_headcount // 2)
                sorted_workers = sorted(current_workers, key=lambda w: getattr(w, 'skill_level', 1.0))

                for fired_emp in sorted_workers[:num_to_fire]:
                    deltas[firm.agent_id].setdefault("fired_employees", []).append(fired_emp.agent_id)
                    deltas[fired_emp.agent_id]["employed_by"] = None
                    deltas[fired_emp.agent_id]["wage"] = 0.0
                    claimed_workers.discard(fired_emp.agent_id)
                    
                    self._emit_event(EventType.FIRE, firm.agent_id, fired_emp.agent_id, {"reason": "Downsizing"}, timestep)
                    events_map[firm.agent_id].append(EventType.FIRE.value)
                    events_map[fired_emp.agent_id].append(EventType.FIRE.value)

        # 3. SAN XUAT TAO GIA TRI THUC & TINH TONG SUC CUNG
        total_market_production = 0.0
        worker_gross_incomes: Dict[str, float] = {e.agent_id: 0.0 for e in active_employees}

        for firm in active_firms:
            firm_workers = [
                e for e in active_employees 
                if (e.employed_by == firm.agent_id or e.agent_id in deltas[firm.agent_id].get("hired_employees", []))
                and e.agent_id not in deltas[firm.agent_id].get("fired_employees", [])
            ]
            firm_wage_bill = 0.0
            firm_output = 0.0

            for emp in firm_workers:
                emp_act = validated_actions.get(emp.agent_id)
                effort = float(emp_act.values[0]) if emp_act is not None else 0.6
                assigned_wage = deltas[emp.agent_id].get("wage", emp.wage)
                wage = assigned_wage if assigned_wage > 0 else BASE_WAGE_ESTIMATE

                firm_wage_bill += wage
                worker_gross_incomes[emp.agent_id] += wage
                deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + wage
                deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) - (effort * 0.12 + 0.03)
                deltas[emp.agent_id]["executed_work_effort"] = effort

                firm_output += (effort * emp.skill_level * 80.0 * firm.productivity_factor)

                self._emit_event(EventType.WAGE_PAID, firm.agent_id, emp.agent_id, {"amount": wage}, timestep)
                events_map[firm.agent_id].append(EventType.WAGE_PAID.value)
                events_map[emp.agent_id].append(EventType.WAGE_PAID.value)

            revenue = firm_output * 1.25
            net_profit = revenue - firm_wage_bill - FIXED_OVERHEAD

            deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) + net_profit
            deltas[firm.agent_id]["executed_revenue"] = revenue
            deltas[firm.agent_id]["executed_profit"] = net_profit
            total_market_production += revenue

            deltas[gov.agent_id]["treasury_overhead"] = deltas[gov.agent_id].get("treasury_overhead", 0.0) + FIXED_OVERHEAD

        # 4. TIEU DUNG & DONG LUC LAM PHAT
        total_consumer_spending = 0.0
        for emp in active_employees:
            emp_act = validated_actions.get(emp.agent_id)
            consume_ratio = float(emp_act.values[2]) if emp_act is not None else 0.5
            cost = living_cost * (0.85 + 0.3 * consume_ratio)

            deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) - cost
            # Tang cuong kha nang phuc hoi the luc khi an uong day du
            available_cash_estimate = emp.cash + deltas[emp.agent_id].get("cash_delta", 0.0)
            energy_recovery = 0.45 if available_cash_estimate >= cost else 0.10
            deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) + energy_recovery
            deltas[emp.agent_id]["executed_consumption"] = cost
            total_consumer_spending += cost

            self._emit_event(EventType.GOODS_PURCHASED, emp.agent_id, eco.agent_id, {"amount": cost}, timestep)

        # CAP NHAT DONG LAM PHAT DUA TREN TI LE CUNG - CAU (SUPPLY-DEMAND ENGINE)
        aggregate_demand = total_consumer_spending
        aggregate_supply = max(100.0, total_market_production)
        demand_supply_gap = (aggregate_demand - aggregate_supply) / aggregate_supply
        
        # Lam phat bien dong mem tu -5% den +15% moi thang
        monthly_inflation = float(np.clip(demand_supply_gap * 0.25, -0.05, 0.15))
        eco.inflation_rate = monthly_inflation
        eco.base_living_cost = float(np.clip(eco.base_living_cost * (1.0 + monthly_inflation), 15.0, 110.0))
        
        deltas[eco.agent_id]["liquidity_delta"] = total_consumer_spending
        deltas[eco.agent_id]["inflation"] = monthly_inflation
        deltas[eco.agent_id]["base_living_cost"] = eco.base_living_cost

        # Xu ly Lao dong That nghiep
        total_subsidies_spent = 0.0
        for emp in active_employees:
            effective_employer = deltas[emp.agent_id].get("employed_by", emp.employed_by)
            is_employed = effective_employer is not None

            if not is_employed:
                streak = getattr(emp, "unemployed_streak", 0) + 1
                deltas[emp.agent_id]["unemployed_streak"] = streak
                deltas[emp.agent_id]["energy_delta"] = deltas[emp.agent_id].get("energy_delta", 0.0) - min(0.15, 0.04 * streak)

                informal_income = emp.skill_level * 15.0
                worker_gross_incomes[emp.agent_id] += informal_income
                deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + informal_income
                total_market_production += informal_income

                current_estimated_cash = emp.cash + deltas[emp.agent_id].get("cash_delta", 0.0)
                if current_estimated_cash < 50.0 and gov.treasury > 1000.0:
                    relief_amount = 40.0 if streak <= 3 else 15.0
                    deltas[emp.agent_id]["cash_delta"] = deltas[emp.agent_id].get("cash_delta", 0.0) + relief_amount
                    total_subsidies_spent += relief_amount
            else:
                deltas[emp.agent_id]["unemployed_streak"] = 0

        deltas[gov.agent_id]["subsidies_disbursed"] = total_subsidies_spent

        # 5. TIN DUNG NGAN HANG & TRA NO GOC
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
                principal_repayment = 0.0
                if deltas[firm.agent_id].get("executed_profit", 0.0) > 0 and firm.cash > 200.0:
                    principal_repayment = min(firm.debt, firm.debt * 0.05)

                total_bank_payment = monthly_interest + principal_repayment
                deltas[firm.agent_id]["cash_delta"] = deltas[firm.agent_id].get("cash_delta", 0.0) - total_bank_payment
                deltas[firm.agent_id]["debt_delta"] = deltas[firm.agent_id].get("debt_delta", 0.0) - principal_repayment
                
                deltas[bank.agent_id]["interest_income"] = deltas[bank.agent_id].get("interest_income", 0.0) + monthly_interest
                deltas[bank.agent_id]["reserves_delta"] = deltas[bank.agent_id].get("reserves_delta", 0.0) + total_bank_payment
                deltas[bank.agent_id]["loans_delta"] = deltas[bank.agent_id].get("loans_delta", 0.0) - principal_repayment

        # 6. THUE VA GIAN LAN
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
        self._emit_event(EventType.TAX_COLLECTED, "MARKET", gov.agent_id, {"amount": total_tax_collected}, timestep)

        # 7. THANH TRA
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
                
                # Payload chi tiet: So tien phat va so tien da tron
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

        # 8. PHA SAN VA TU VONG
        total_defaults = 0.0
        for firm in active_firms:
            projected_cash = firm.cash + deltas[firm.agent_id].get("cash_delta", 0.0)
            projected_debt = firm.debt + deltas[firm.agent_id].get("debt_delta", 0.0)

            is_insolvent = (
                (len(firm.employee_ids) == 0 and projected_cash <= 0.0 and timestep > 12) or
                (projected_cash < -200.0) or
                (projected_debt > firm.capital_stock * 2.0 and projected_cash < 0.0)
            )

            if is_insolvent:
                deltas[firm.agent_id]["status"] = LifeCycleStatus.BANKRUPT

                fired_list = [eid for eid in firm.employee_ids if eid in deltas]
                deltas[firm.agent_id]["fired_employees"] = fired_list
                for emp_id in fired_list:
                    deltas[emp_id]["employed_by"] = None
                    deltas[emp_id]["wage"] = 0.0
                    self._emit_event(EventType.FIRE, firm.agent_id, emp_id, {"reason": "Firm bankruptcy"}, timestep)

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

            # Ngưỡng tử vong sinh học hợp lý
            if projected_cash < -400.0 or projected_energy <= 0.0 or (emp.age + deltas[emp.agent_id]["age_increment"]) >= emp.max_age:
                new_deaths += 1
                deltas[emp.agent_id]["status"] = LifeCycleStatus.DECEASED

                emp_firm = deltas[emp.agent_id].get("employed_by", emp.employed_by)
                if emp_firm and emp_firm in deltas:
                    deltas[emp_firm].setdefault("fired_employees", []).append(emp.agent_id)
                deltas[emp.agent_id]["employed_by"] = None

                death_reason = "Tuổi già" if (emp.age + deltas[emp.agent_id]["age_increment"]) >= emp.max_age else "Kiệt sức / Thâm hụt tài chính"
                self._emit_event(
                    EventType.AGENT_DIED, 
                    emp.agent_id, 
                    gov.agent_id, 
                    {"age": emp.age, "reason": death_reason}, 
                    timestep
                )
                events_map[emp.agent_id].append(EventType.AGENT_DIED.value)

        deltas[gov.agent_id]["new_deaths"] = new_deaths

        # 9. GDP VA HE SO GINI
        active_wealths = [
            max(0.01, e.cash + deltas[e.agent_id].get("cash_delta", 0.0)) 
            for e in active_employees 
            if deltas[e.agent_id].get("status") not in [LifeCycleStatus.DECEASED, LifeCycleStatus.DEAD, LifeCycleStatus.TERMINATED]
        ]
        deltas[gov.agent_id]["current_gini"] = self._compute_gini(active_wealths)
        deltas[gov.agent_id]["current_gdp"] = total_market_production

        # 10. DONG GOI TRANSITIONS
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