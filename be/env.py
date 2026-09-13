from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from be.core.enums import LifeCycleStatus, EventType
from be.core.event import Event, EventBus
from be.core.types import Action, TransitionResult, ValidationResult
from be.agents.base_agent import BaseAgent
from be.agents.employee import Employee
from be.agents.firm import Firm
from be.agents.government import Government
from be.agents.bank import Bank
from be.agents.supervisor import Supervisor
from be.agents.economy import Economy
from be.rule_engine import RuleEngine

class MacroEnvironment:
    """
    Mo phong Kinh te vi mo the che da tac tu.
    Bao toan dong tien (SFC) & Can bang dong dan so.
    """
    def __init__(self, num_employees: int = 50, num_firms: int = 5, max_steps: int = 240):
        self.num_employees: int = num_employees
        self.num_firms: int = num_firms
        self.max_steps: int = max_steps
        self.timestep: int = 0

        self.event_bus: EventBus = EventBus()
        self.rule_engine: RuleEngine = RuleEngine(event_bus=self.event_bus)
        self.agents: Dict[str, BaseAgent] = {}

        self.next_emp_id: int = self.num_employees
        self.next_firm_id: int = self.num_firms
        self.reported_dead_agents: set = set()
        self._create_world()

    def _create_world(self) -> None:
        self.agents.clear()
        self.gov = Government(agent_id="gov_1")
        self.bank = Bank(agent_id="bank_1")
        self.eco = Economy(agent_id="eco_1")
        self.sup = Supervisor(agent_id="sup_1")
        
        self.agents[self.gov.agent_id] = self.gov
        self.agents[self.bank.agent_id] = self.bank
        self.agents[self.eco.agent_id] = self.eco
        self.agents[self.sup.agent_id] = self.sup

        for i in range(self.num_firms):
            f_id = f"firm_{i}"
            self.agents[f_id] = Firm(agent_id=f_id)

        for j in range(self.num_employees):
            e_id = f"emp_{j}"
            self.agents[e_id] = Employee(agent_id=e_id)

    def _sample_skill(self) -> float:
        """Phan hoa 4 bac ky nang ro ret."""
        roll = np.random.rand()
        if roll < 0.35:
            return float(np.random.uniform(0.6, 0.9))   # Pho thong
        elif roll < 0.75:
            return float(np.random.uniform(1.1, 1.6))   # Ban chuyen
        elif roll < 0.95:
            return float(np.random.uniform(1.8, 2.5))   # Ky su
        else:
            return float(np.random.uniform(2.8, 3.8))   # Chuyen gia

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        self.timestep = 0
        self.next_emp_id = self.num_employees
        self.next_firm_id = self.num_firms
        self.reported_dead_agents.clear()
        self.event_bus.clear()
        self._create_world()

        # GDP ban dau luon la 0.0 de khong lam hong truc bieu do
        self.gov.initialize(initial_treasury=1000000.0, initial_worker_tax=0.15, initial_firm_tax=0.20)
        self.gov.current_gdp = 0.0
        self.bank.initialize(initial_reserves=500000.0, initial_lending_rate=0.06, initial_deposit_rate=0.02)
        self.eco.initialize(initial_living_cost=15.0, initial_housing_inventory=100, initial_house_price=1000.0)
        self.sup.initialize(initial_budget=50000.0, initial_audit_rate=0.05, initial_fine_multiplier=1.5)

        # KHOI PHUC DUNG NHANH EMPLOYEE VA FIRM
        for agent in self.agents.values():
            if isinstance(agent, Firm):
                init_cap = float(np.random.uniform(8000.0, 14000.0))
                agent.initialize(
                    productivity_factor=float(np.random.uniform(1.3, 1.8)),
                    risk_aversion=float(np.random.uniform(0.2, 0.4)),
                    tax_morale=float(np.random.uniform(0.5, 0.85)),
                    initial_capital=init_cap
                )
                agent.age_months = 0
                seed_cash = float(np.random.uniform(3000.0, 5000.0))
                self.gov.treasury -= seed_cash
                agent.cash = seed_cash

            elif isinstance(agent, Employee):
                start_cash = float(np.random.uniform(200.0, 400.0))
                self.gov.treasury -= start_cash # Trich tu Kho bac sang Emp
                agent.initialize(
                    skill_level=self._sample_skill(),
                    risk_aversion=float(np.random.uniform(0.3, 0.8)),
                    tax_morale=float(np.random.uniform(0.5, 0.95)),
                    initial_cash=start_cash,
                    initial_energy=float(np.random.uniform(0.8, 1.2)),
                    age=int(np.random.randint(20, 50))
                )

        raw_state = self.get_raw_environment_state()
        initial_obs = {aid: a.observe(raw_state).vector for aid, a in self.agents.items()}
        infos = {aid: {"status": a.status.name} for aid, a in self.agents.items()}
        return initial_obs, infos

    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]
    ]:
        self.timestep += 1
        for a in self.agents.values():
            if isinstance(a, Firm):
                a.age_months = getattr(a, "age_months", 0) + 1

        validated_actions: Dict[str, Action] = {}
        for agent_id, raw_vals in action_dict.items():
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if agent.status == LifeCycleStatus.ACTIVE:
                    act = Action(agent_id=agent_id, action_type="STEP_ACTION", values=np.asarray(raw_vals, dtype=np.float32))
                    val_res: ValidationResult = agent.validate_action(act)
                    validated_actions[agent_id] = Action(
                        agent_id=agent_id,
                        action_type=act.action_type,
                        values=val_res.sanitized_values,
                        metadata={"is_valid": val_res.is_valid}
                    )

        transition_results: Dict[str, TransitionResult] = self.rule_engine.execute_cycle(
            agents=self.agents,
            validated_actions=validated_actions,
            timestep=self.timestep
        )

        rewards_all: Dict[str, float] = {}
        for agent_id, trans_res in transition_results.items():
            if agent_id in self.agents:
                self.agents[agent_id].apply_result(trans_res)
                if "status" in trans_res.state_delta:
                    self.agents[agent_id].status = trans_res.state_delta["status"]
                rewards_all[agent_id] = self.agents[agent_id].calculate_reward(trans_res)

        # DEMOGRAPHIC TURNOVER DONG BO KHO BAC
        dead_emps = [aid for aid in self.agents.items() if isinstance(aid[1], Employee) and aid[1].status in [LifeCycleStatus.DECEASED, LifeCycleStatus.DEAD, LifeCycleStatus.TERMINATED]]
        bankrupt_firms = [aid for aid in self.agents.items() if isinstance(aid[1], Firm) and aid[1].status == LifeCycleStatus.BANKRUPT]

        for d_id, emp in dead_emps:
            self.reported_dead_agents.add(d_id)
            self.agents.pop(d_id, None)
            if emp.cash > 0:
                self.gov.treasury += emp.cash
            if emp.employed_by and emp.employed_by in self.agents:
                employer = self.agents[emp.employed_by]
                if hasattr(employer, "employee_ids") and d_id in employer.employee_ids:
                    employer.employee_ids.remove(d_id)

        for b_id, firm in bankrupt_firms:
            self.reported_dead_agents.add(b_id)
            self.agents.pop(b_id, None)
            if firm.cash > 0:
                self.gov.treasury += firm.cash
            for a in self.agents.values():
                if isinstance(a, Employee) and a.employed_by == b_id:
                    a.employed_by = None
                    a.wage = 0.0

        # A. DIEU TIET DAN SO THEO SUC TAI KINH TE (ENDOGENOUS DEMOGRAPHIC CAPACITY)
        active_emps_list = [a for a in self.agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE]
        current_emp_count = len(active_emps_list)
        unemployed_count = sum(1 for e in active_emps_list if e.employed_by is None)
        unemployment_rate = (unemployed_count / current_emp_count) if current_emp_count > 0 else 0.0

        employed_emps = [e for e in active_emps_list if e.employed_by is not None]
        avg_wage = (sum(e.wage for e in employed_emps) / len(employed_emps)) if employed_emps else 25.0
        living_standard_ratio = avg_wage / max(1.0, self.eco.base_living_cost)

        # Chot chan ky thuat an toan phan cung (30 <= Emp <= 85)
        HARD_MIN_EMP = 30
        HARD_MAX_EMP = 85

        num_newborns = 0
        if current_emp_count < HARD_MIN_EMP:
            # Cuu nguy dan so khan cap neu tut duoi nguong an toan
            num_newborns = 2 if current_emp_count < 20 else 1
        elif current_emp_count < HARD_MAX_EMP:
            # Xac suat sinh phu thuoc vao doi song kinh te:
            # Thuong neu ty le that nghiep thap (<10%) va luong cao hon muc song (>1.2)
            p_birth = 0.12
            if unemployment_rate < 0.10:
                p_birth += 0.15
            elif unemployment_rate > 0.30:
                p_birth -= 0.10

            if living_standard_ratio > 1.25:
                p_birth += 0.15
            elif living_standard_ratio < 0.90:
                p_birth -= 0.08

            p_birth = float(np.clip(p_birth, 0.02, 0.50))
            if np.random.rand() < p_birth and self.gov.treasury >= 600.0:
                num_newborns = 1

        for _ in range(num_newborns):
            new_eid = f"emp_{self.next_emp_id}"
            self.next_emp_id += 1
            grant = 300.0 if self.gov.treasury >= 600.0 else 50.0
            self.gov.treasury -= grant
            
            new_emp = Employee(agent_id=new_eid)
            new_emp.initialize(
                skill_level=self._sample_skill(),
                risk_aversion=float(np.random.uniform(0.3, 0.7)),
                tax_morale=float(np.random.uniform(0.5, 0.95)),
                initial_cash=grant,
                initial_energy=1.0,
                age=18
            )
            self.agents[new_eid] = new_emp
            self.event_bus.publish(Event(
                event_type=EventType.AGENT_BORN,
                source_id="SOCIETY",
                target_id=new_eid,
                payload={"cash": new_emp.cash, "skill": new_emp.skill_level},
                timestep=self.timestep
            ))

        # B. KHOI NGHIEP THEO CO HOI THI TRUONG (ENDOGENOUS MARKET ENTRY)
        active_firms_list = [a for a in self.agents.values() if isinstance(a, Firm) and a.status == LifeCycleStatus.ACTIVE]
        current_firm_count = len(active_firms_list)
        HARD_MAX_FIRMS = 7

        # Cuu ho khẩn cấp: Thi truong chi con 1 firm va con nguoi de tuyen dung
        need_emergency_firm = (current_firm_count < 2 and unemployed_count >= 2)
        
        # Mo rong noi sinh: Con cho trong (< 7 firms), co du lao dong roi ranh de thue, va Kho bac du ngan sach
        avg_firm_profit = float(np.mean([getattr(f, "last_profit", 0.0) for f in active_firms_list])) if active_firms_list else 0.0
        is_market_healthy = (avg_firm_profit >= -15.0) # Thi truong khong trong trang thai sup do hang loat
        market_expansion = (
            current_firm_count < HARD_MAX_FIRMS and 
            unemployed_count >= 3 and 
            unemployment_rate >= 0.12 and 
            is_market_healthy and 
            self.gov.treasury >= 4000.0 and 
            np.random.rand() < 0.12
        )

        if need_emergency_firm or market_expansion:
            new_fid = f"firm_{self.next_firm_id}"
            self.next_firm_id += 1
            grant_firm = 3500.0 if self.gov.treasury >= 8000.0 else 1500.0
            self.gov.treasury -= grant_firm

            new_firm = Firm(agent_id=new_fid)
            new_firm.initialize(
                productivity_factor=float(np.random.uniform(1.3, 1.8)),
                risk_aversion=float(np.random.uniform(0.2, 0.4)),
                tax_morale=float(np.random.uniform(0.5, 0.85)),
                initial_capital=8000.0
            )
            new_firm.age_months = 0
            new_firm.cash = grant_firm
            self.agents[new_fid] = new_firm
            self.event_bus.publish(Event(
                event_type=EventType.HIRE,
                source_id="MARKET",
                target_id=new_fid,
                payload={"capital": new_firm.capital_stock, "status": "INCORPORATED"},
                timestep=self.timestep
            ))

        raw_state = self.get_raw_environment_state()
        is_time_up = self.timestep >= self.max_steps

        observations = {aid: a.observe(raw_state).vector for aid, a in self.agents.items()}
        rewards = {aid: rewards_all.get(aid, 0.0) for aid in self.agents.keys()}
        terminateds = {"__all__": False}
        truncateds = {"__all__": is_time_up}
        infos = {aid: {"status": a.status.name} for aid, a in self.agents.items()}

        for aid in self.agents.keys():
            terminateds[aid] = False
            truncateds[aid] = is_time_up

        return observations, rewards, terminateds, truncateds, infos

    def get_raw_environment_state(self) -> Dict[str, Any]:
        active_employees = [a for a in self.agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE]
        unemployed_count = sum(1 for e in active_employees if e.employed_by is None)
        unemployment_rate = (unemployed_count / len(active_employees)) if active_employees else 0.0

        employed = [e for e in active_employees if e.employed_by is not None]
        avg_wage = (sum(e.wage for e in employed) / len(employed)) if employed else 25.0

        active_firms = [a for a in self.agents.values() if isinstance(a, Firm) and a.status == LifeCycleStatus.ACTIVE]
        total_credit_demand = sum(f.debt for f in active_firms)

        return {
            "timestep": self.timestep,
            "macro_indicators": {
                "inflation": self.eco.inflation_rate,
                "base_living_cost": self.eco.base_living_cost,
                "worker_tax_rate": self.gov.tax_rate_worker,
                "firm_tax_rate": self.gov.tax_rate_firm,
                "base_interest_rate": self.bank.deposit_rate,
                "bank_lending_rate": self.bank.lending_rate,
                "unemployment_rate": unemployment_rate,
                "average_wage": avg_wage,
                "market_demand_factor": 1.0,
                "total_credit_demand": total_credit_demand,
                "estimated_tax_evasion": float(self.sup.violations_detected * 50.0),
                "aggregate_demand": self.eco.step_trade_volume,
                "aggregate_supply": sum(f.capital_stock for f in active_firms)
            }
        }

    def export_full_world_state(self) -> Dict[str, Any]:
        total_emp_cash = sum(a.cash for a in self.agents.values() if isinstance(a, Employee))
        total_firm_cash = sum(a.cash for a in self.agents.values() if isinstance(a, Firm))
        m2_supply = self.gov.treasury + self.bank.reserves + total_emp_cash + total_firm_cash

        return {
            "timestep": self.timestep,
            "agents": {agent_id: agent.export_state() for agent_id, agent in self.agents.items()},
            "macro": {
                "gdp": self.gov.current_gdp,
                "gini": self.gov.current_gini,
                "treasury": self.gov.treasury,
                "bank_reserves": self.bank.reserves,
                "npl": self.bank.non_performing_loans,
                "living_cost": self.eco.base_living_cost,
                "housing_price": self.eco.housing_price,
                "inflation": self.eco.inflation_rate,
                "m2_supply": m2_supply,
                "active_employees": sum(1 for a in self.agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE),
                "active_firms": sum(1 for a in self.agents.values() if isinstance(a, Firm) and a.status == LifeCycleStatus.ACTIVE)
            }
        }