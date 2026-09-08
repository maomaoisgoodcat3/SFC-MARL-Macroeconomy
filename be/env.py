from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from be.core.enums import LifeCycleStatus, AgentType
from be.core.event import EventBus
from be.core.types import Action, Observation, TransitionResult, ValidationResult
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
    Trai tim cua he thong mo phong (Simulation Layer).
    Quan ly the gioi, khoi tao tac tu, thuc thi buoc mo phong hang thang (1 step = 1 thang).
    Tuan thu nghiem ngat SAS v1.0:
    - Khong tu tinh toan reward (Reward thuoc trach nhiem cua Agent).
    - Khong tu sua doi luat (Luat thuoc trach nhiem cua RuleEngine).
    """
    def __init__(self, 
                 num_employees: int = 50, 
                 num_firms: int = 5, 
                 max_steps: int = 240):
        self.num_employees: int = num_employees
        self.num_firms: int = num_firms
        self.max_steps: int = max_steps
        self.timestep: int = 0

        # Quan ly tap hop tac tu da duoc bao cao tu vong de cach ly khoi RLlib
        self.reported_dead_agents: set = set()

        # He thong su kien va hien phap
        self.event_bus: EventBus = EventBus()
        self.rule_engine: RuleEngine = RuleEngine(event_bus=self.event_bus)

        self.agents: Dict[str, BaseAgent] = {}
        self._create_world()

    def _create_world(self) -> None:
        """Khoi tao toan bo thuc the the che va vi mo."""
        self.agents.clear()
        
        # 4 Tac tu the che vi mo
        self.gov = Government(agent_id="gov_1")
        self.bank = Bank(agent_id="bank_1")
        self.eco = Economy(agent_id="eco_1")
        self.sup = Supervisor(agent_id="sup_1")
        
        self.agents[self.gov.agent_id] = self.gov
        self.agents[self.bank.agent_id] = self.bank
        self.agents[self.eco.agent_id] = self.eco
        self.agents[self.sup.agent_id] = self.sup

        # Cac Doanh nghiep (Firms)
        for i in range(self.num_firms):
            f_id = f"firm_{i}"
            self.agents[f_id] = Firm(agent_id=f_id)

        # Cac Nguoi lao dong (Employees)
        for j in range(self.num_employees):
            e_id = f"emp_{j}"
            self.agents[e_id] = Employee(agent_id=e_id)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        self.timestep = 0
        self.reported_dead_agents.clear()
        self.event_bus.clear()

        # Khoi tao the che vi mo
        self.gov.initialize(initial_treasury=1000000.0, initial_worker_tax=0.15, initial_firm_tax=0.20)
        self.bank.initialize(initial_reserves=500000.0, initial_lending_rate=0.06, initial_deposit_rate=0.02)
        self.eco.initialize(initial_living_cost=15.0, initial_housing_inventory=100, initial_house_price=1000.0)
        self.sup.initialize(initial_budget=50000.0, initial_audit_rate=0.05, initial_fine_multiplier=1.5)

        for agent in self.agents.values():
            if isinstance(agent, Firm):
                agent.initialize(
                    productivity_factor=float(np.random.uniform(1.0, 1.5)),
                    risk_aversion=float(np.random.uniform(0.2, 0.6)),
                    tax_morale=float(np.random.uniform(0.4, 0.9)),
                    initial_capital=float(np.random.uniform(3000.0, 5000.0))
                )
            elif isinstance(agent, Employee):
                agent.initialize(
                    skill_level=float(np.random.uniform(0.8, 2.0)),
                    risk_aversion=float(np.random.uniform(0.3, 0.8)),
                    tax_morale=float(np.random.uniform(0.5, 0.95)),
                    initial_cash=float(np.random.uniform(300.0, 800.0)),
                    initial_energy=float(np.random.uniform(0.8, 1.2)),
                    age=int(np.random.randint(20, 50))
                )

        raw_state = self.get_raw_environment_state()
        initial_observations: Dict[str, np.ndarray] = {}
        infos: Dict[str, Any] = {}

        for agent_id, agent in self.agents.items():
            obs = agent.observe(raw_state)
            initial_observations[agent_id] = obs.vector
            infos[agent_id] = {"status": agent.status.name}

        return initial_observations, infos

    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray], 
        Dict[str, float], 
        Dict[str, bool], 
        Dict[str, bool], 
        Dict[str, Any]
    ]:
        self.timestep += 1

        # 1. Chi tiep nhan hanh dong tu cac tac tu con song
        actions: Dict[str, Action] = {}
        for agent_id, raw_vals in action_dict.items():
            if agent_id in self.agents and agent_id not in self.reported_dead_agents:
                actions[agent_id] = Action(
                    agent_id=agent_id,
                    action_type="STEP_ACTION",
                    values=np.asarray(raw_vals, dtype=np.float32)
                )

        # 2. Xac thuc hanh dong
        validated_actions: Dict[str, Action] = {}
        for agent_id, action in actions.items():
            agent = self.agents[agent_id]
            if agent.status == LifeCycleStatus.ACTIVE:
                val_res: ValidationResult = agent.validate_action(action)
                validated_actions[agent_id] = Action(
                    agent_id=agent_id,
                    action_type=action.action_type,
                    values=val_res.sanitized_values,
                    metadata={"is_valid": val_res.is_valid, "reason": val_res.reason}
                )

        # 3. RuleEngine thuc thi chu ky
        transition_results: Dict[str, TransitionResult] = self.rule_engine.execute_cycle(
            agents=self.agents,
            validated_actions=validated_actions,
            timestep=self.timestep
        )

        # 4. Cap nhat ket qua noi tai
        for agent_id, trans_res in transition_results.items():
            self.agents[agent_id].apply_result(trans_res)

        # 5. Tinh toan Reward
        rewards_all: Dict[str, float] = {}
        for agent_id, trans_res in transition_results.items():
            rewards_all[agent_id] = self.agents[agent_id].calculate_reward(trans_res)

        # 6. Dong goi quan sat va xu ly che do MultiAgentEnv chuan
        raw_state = self.get_raw_environment_state()
        is_time_up = self.timestep >= self.max_steps

        observations: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        terminateds: Dict[str, bool] = {}
        truncateds: Dict[str, bool] = {}
        infos: Dict[str, Any] = {}

        for agent_id, agent in self.agents.items():
            # Neu tac tu da duoc bao cao chet tu truoc, bo qua hoan toan
            if agent_id in self.reported_dead_agents:
                continue

            is_dead_now = agent.status in [LifeCycleStatus.TERMINATED, LifeCycleStatus.BANKRUPT, LifeCycleStatus.DECEASED]

            if is_dead_now:
                # Bao cao lan cuoi cung va ghi danh vao danh sach dead
                self.reported_dead_agents.add(agent_id)
                observations[agent_id] = agent.observe(raw_state).vector
                rewards[agent_id] = rewards_all[agent_id]
                terminateds[agent_id] = True
                truncateds[agent_id] = False
                infos[agent_id] = {"status": agent.status.name, "events": transition_results[agent_id].events_triggered}
            else:
                # Tac tu dang song binh thuong
                observations[agent_id] = agent.observe(raw_state).vector
                rewards[agent_id] = rewards_all[agent_id]
                terminateds[agent_id] = False
                truncateds[agent_id] = is_time_up
                infos[agent_id] = {"status": agent.status.name, "events": transition_results[agent_id].events_triggered}

        # Dieu kien dung toan bo the gioi
        terminateds["__all__"] = False
        truncateds["__all__"] = is_time_up

        return observations, rewards, terminateds, truncateds, infos

    def get_raw_environment_state(self) -> Dict[str, Any]:
        """
        Gom cac chi so kinh te vi mo tong the de cung cap cho phuong thuc observe() cua cac Agent.
        """
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
        """
        Xuat toan bo trang thai cac thuc the de phuc vu Logger va WebSocket Server.
        """
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
                "active_employees": sum(1 for a in self.agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE),
                "active_firms": sum(1 for a in self.agents.values() if isinstance(a, Firm) and a.status == LifeCycleStatus.ACTIVE)
            }
        }