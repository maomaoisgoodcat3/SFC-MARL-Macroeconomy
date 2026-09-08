import os
import json
import logging
from typing import Dict, List, Any
from be.core.enums import LifeCycleStatus
from be.core.event import Event
from be.agents.base_agent import BaseAgent
from be.agents.employee import Employee
from be.agents.firm import Firm
from be.agents.government import Government
from be.agents.bank import Bank
from be.agents.economy import Economy
from be.agents.supervisor import Supervisor
from be.parquet_io import ParquetIO

logger = logging.getLogger("InstitutionalEconomist.Logger")

class InstitutionalLogger:
    """
    He thong ghi log kep (Dual Logger) phan tach Micro, Macro va Event Stream.
    Quan ly bo dem tren RAM va xả dinh ky xuong Parquet de chong tran bo nho OOM.
    """
    def __init__(self, 
                 run_dir: str = "be/logs", 
                 export_dir: str = "be/exports", 
                 flush_interval: int = 100):
        self.run_dir = run_dir
        self.export_dir = export_dir
        self.flush_interval = flush_interval

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)

        self.parquet_io = ParquetIO(base_export_dir=self.export_dir)

        # Bo dem du lieu tren bo nho RAM
        self.macro_buffer: List[Dict[str, Any]] = []
        self.micro_buffer: List[Dict[str, Any]] = []
        self.ui_event_queue: List[Dict[str, Any]] = []

    def log_macro_step(self, 
                       timestep: int, 
                       gov: Government, 
                       bank: Bank, 
                       eco: Economy, 
                       sup: Supervisor, 
                       active_workers: int, 
                       active_firms: int) -> None:
        """Ghi nhan trang thai Vi mo toan xa hoi theo tung thang."""
        record = {
            "month": timestep,
            "gdp": float(gov.current_gdp),
            "inflation": float(eco.inflation_rate),
            "gini": float(gov.current_gini),
            "active_population": int(active_workers),
            "active_firms": int(active_firms),
            "cumulative_deaths": int(gov.dead_citizens_count),
            "housing_index": float(eco.housing_price),
            "housing_inventory": int(eco.housing_inventory),
            "living_cost": float(eco.base_living_cost),
            "government_cash": float(gov.treasury),
            "public_debt": float(gov.public_debt),
            "bank_reserves": float(bank.reserves),
            "bank_total_loans": float(bank.total_loans),
            "bank_npl": float(bank.non_performing_loans),
            "audit_violations": int(sup.violations_detected)
        }
        self.macro_buffer.append(record)

    def log_micro_step(self, timestep: int, agents: Dict[str, BaseAgent]) -> None:
        """Ghi nhan chi tiet tung ca nhan Employee va Firm theo tung thang."""
        for agent_id, agent in agents.items():
            if isinstance(agent, Employee):
                is_active = agent.status == LifeCycleStatus.ACTIVE
                record = {
                    "month": timestep,
                    "agent_id": agent.agent_id,
                    "agent_type": "employee",
                    "status": agent.status.name,
                    "income": float(agent.wage if agent.employed_by else (agent.last_work_effort * agent.skill_level * 18.0)),
                    "living_cost_paid": float(agent.last_consumption),
                    "tax_declare_ratio": float(agent.last_declare_ratio),
                    "savings_cash": float(agent.cash),
                    "net_worth": float(agent.cash - agent.debt),
                    "energy": float(agent.energy),
                    "skill_level": float(agent.skill_level),
                    "employed_by": str(agent.employed_by) if agent.employed_by else "None",
                    "debt": float(agent.debt),
                    "age": int(agent.age)
                }
                self.micro_buffer.append(record)

            elif isinstance(agent, Firm):
                record = {
                    "month": timestep,
                    "agent_id": agent.agent_id,
                    "agent_type": "firm",
                    "status": agent.status.name,
                    "revenue": float(agent.last_revenue),
                    "profit": float(agent.last_profit),
                    "tax_declare_ratio": float(agent.last_declare_ratio),
                    "capital_stock": float(agent.capital_stock),
                    "cash": float(agent.cash),
                    "debt": float(agent.debt),
                    "headcount": int(len(agent.employee_ids))
                }
                self.micro_buffer.append(record)

    def log_event_for_ui(self, event: Event) -> None:
        """Bien doi Event Bus thanh cau truc du lieu JSON cho Frontend Force Graph."""
        ui_event = {
            "type": event.event_type.value,
            "source": event.source_id,
            "target": event.target_id,
            "payload": event.payload,
            "timestep": event.timestep,
            "event_id": event.event_id
        }
        self.ui_event_queue.append(ui_event)

    def flush_ui_events(self, output_path: str = "be/exports/ui_events.json") -> None:
        """
        Ghi file JSON phuc vu Frontend.
        Ghi de truc tiep de tranh loi khoa file WinError 5 cua os.replace tren Windows.
        """
        if not self.ui_event_queue:
            return

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.ui_event_queue, f, indent=2)
        except PermissionError:
            # Bo qua nhip ghi neu file dang bi tien trinh khac tren Windows khoa tam thoi
            pass
        except Exception as exc:
            logger.warning(f"[WARNING] Could not flush UI events to {output_path}: {str(exc)}")
        finally:
            self.ui_event_queue.clear()

    def step_end(self, timestep: int) -> None:
        """Kiem tra va xa buffer xuong dia dinh ky khi cham nguong flush_interval."""
        if timestep % self.flush_interval == 0:
            self.flush_to_disk()

    def flush_to_disk(self) -> None:
        """Xa toan bo Macro va Micro buffer ra Parquet va giai phong RAM."""
        if self.macro_buffer:
            self.parquet_io.write_batch(self.macro_buffer, sub_category="macro_logs", prefix="macro")
            self.macro_buffer.clear()

        if self.micro_buffer:
            self.parquet_io.write_batch(self.micro_buffer, sub_category="micro_logs", prefix="micro")
            self.micro_buffer.clear()