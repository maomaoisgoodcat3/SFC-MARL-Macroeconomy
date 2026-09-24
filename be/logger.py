import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Sequence, Optional
from be.core.enums import LifeCycleStatus
from be.core.event import Event
from be.agents.base_agent import BaseAgent
from be.agents.employee import Employee
from be.agents.firm import Firm
from be.agents.government import Government
from be.agents.bank import Bank, compute_npl_ratio_pct
from be.agents.economy import Economy
from be.agents.supervisor import Supervisor
from be.parquet_io import ParquetIO

logger = logging.getLogger("InstitutionalEconomist.Logger")

class InstitutionalLogger:
    """
    He thong ghi log nghien cuu: 5 category Parquet tach bach ro rang theo
    dung mang thong ke (macro / micro_employee / micro_firm / micro_bank /
    events), moi lan simulate/train sinh DUNG 1 file/category (xem
    ParquetIO) thay vi hang tram file UUID roi rac nhu thiet ke cu -- va mot
    hang doi JSON rieng phuc vu Frontend real-time (khong lien quan Parquet).

    Ly do tach micro_bank rieng khoi macro: so luong ngan hang nho (1-5) nhung
    dong luc canh tranh tin dung giua cac ngan hang (relationship banking,
    Petersen & Rajan, 1994) la mot dong gop rieng cua du an nay so voi cac
    paper doi chieu (khong paper nao trong nhom da doc co ngan hang RL agent
    dong thoi co NPL) -- xung dang co 1 bang rieng de phan tich thay vi bi
    gop lan vao 1 dong macro moi thang.
    """
    def __init__(self,
                 run_dir: str = "be/logs",
                 export_dir: str = "be/exports",
                 flush_interval: int = 100,
                 run_id: Optional[str] = None):
        self.run_dir = run_dir
        self.export_dir = export_dir
        self.flush_interval = flush_interval
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)

        self.parquet_io = ParquetIO(base_export_dir=self.export_dir, run_id=self.run_id)

        # Bo dem tren RAM, xa dinh ky xuong Parquet de chong tran bo nho OOM
        self.macro_buffer: List[Dict[str, Any]] = []
        self.micro_employee_buffer: List[Dict[str, Any]] = []
        self.micro_firm_buffer: List[Dict[str, Any]] = []
        self.micro_bank_buffer: List[Dict[str, Any]] = []
        self.events_buffer: List[Dict[str, Any]] = []
        self.ui_event_queue: List[Dict[str, Any]] = []

    def log_step(self,
                 timestep: int,
                 gov: Government,
                 banks: Sequence[Bank],
                 eco: Economy,
                 sup: Supervisor,
                 agents: Dict[str, BaseAgent],
                 births_this_step: int = 0) -> None:
        """
        Diem vao DUY NHAT ghi nhan toan bo trang thai he thong tai 1 timestep
        -- gop macro + micro_employee + micro_firm + micro_bank vao 1 lan duyet
        qua agents (truoc day log_macro_step/log_micro_step duyet 2 lan rieng
        biet, macro_step con khong the tu tinh unemployment_rate/avg_wage vi
        khong nhan agents lam tham so).
        """
        active_employees = [a for a in agents.values() if isinstance(a, Employee) and a.status == LifeCycleStatus.ACTIVE]
        active_firms_list = [a for a in agents.values() if isinstance(a, Firm) and a.status == LifeCycleStatus.ACTIVE]

        employed = [e for e in active_employees if e.employed_by is not None]
        unemployment_rate = 1.0 - (len(employed) / max(1, len(active_employees)))
        avg_wage = (sum(e.wage for e in employed) / len(employed)) if employed else 0.0

        total_loans = sum(b.total_loans for b in banks)
        total_npl = sum(b.non_performing_loans for b in banks)

        # --- 1. MACRO (1 dong/thang) ---
        self.macro_buffer.append({
            "month": timestep,
            "gdp": float(gov.current_gdp),
            # GDP thuc (khu gia) + chi mua hang cua Chinh phu (Section 4C) va ty trong G/GDP,
            # de theo doi vong chu chuyen da dong hay chua (xem CLAUDE_HISTORY.md v0.16).
            "real_gdp": float(gov.current_real_gdp),
            "government_purchases": float(gov.last_purchase),
            "government_purchase_share_gdp": float(gov.last_purchase / gov.current_gdp) if gov.current_gdp > 0 else 0.0,
            "gini": float(gov.current_gini),
            # equality = 1 - gini, dung dinh nghia Eq.7 cua Zheng et al. (2022),
            # "The AI Economist", de ket qua co the doi chieu truc tiep voi
            # benchmark cua chinh bai bao goc va cac paper mo rong (TaxAI,
            # ABIDES-Economist...) da doc.
            "equality": float(1.0 - gov.current_gini),
            "inflation_pct": float(eco.inflation_rate * 100.0),
            "cpi_index": float(eco.cpi_index),
            "living_cost": float(eco.base_living_cost),
            "housing_price": float(eco.housing_price),
            "housing_inventory": int(eco.housing_inventory),
            "active_population": len(active_employees),
            "employed_count": len(employed),
            # NPL ratio dung TOTAL LOANS lam mau so (khong phai reserves) --
            # da tu phat hien va sua sai lam nay o phien lam viec truoc.
            "unemployment_rate_pct": float(unemployment_rate * 100.0),
            "avg_wage": float(avg_wage),
            "active_firms": len(active_firms_list),
            "cumulative_deaths": int(gov.dead_citizens_count),
            "births_this_step": int(births_this_step),
            "government_treasury": float(gov.treasury),
            "public_debt": float(gov.public_debt),
            "tax_collected": float(gov.last_tax_collected),
            "subsidies_disbursed": float(gov.last_subsidies_paid),
            "worker_tax_rate_pct": float(gov.tax_rate_worker * 100.0),
            "firm_tax_rate_pct": float(gov.tax_rate_firm * 100.0),
            "bank_count": len(banks),
            "bank_reserves_total": float(sum(b.reserves for b in banks)),
            "bank_deposits_total": float(sum(b.total_deposits for b in banks)),
            "bank_loans_total": float(total_loans),
            "bank_npl_total": float(total_npl),
            "npl_ratio_pct": compute_npl_ratio_pct(total_npl, total_loans),
            "audit_violations": int(sup.violations_detected),
        })

        # --- 2. MICRO_EMPLOYEE (1 dong/employee dang ACTIVE/thang) ---
        for emp in active_employees:
            self.micro_employee_buffer.append({
                "month": timestep,
                "agent_id": emp.agent_id,
                "status": emp.status.name,
                # HE SO CAU TRUC TU DO HIEU CHINH (bo sung nhan con thieu, v0.20, phat hien
                # qua audit toan du an): 18.0 CHI phuc vu HIEN THI/LOG (uoc luong thu nhap
                # phi chinh thuc cho ca nhan chua co viec lam chinh thuc de nguoi doc log de
                # hinh dung ty le so voi luong chinh thuc) -- KHONG duoc dung trong reward
                # hay bat ky cong thuc kinh te nao trong rule_engine.py (khu vuc phi chinh
                # thuc that su dung "0.35 * skill_level" o rule_engine.py Section 3, khong
                # phai gia tri nay).
                "income": float(emp.wage if emp.employed_by else (emp.last_work_effort * emp.skill_level * 18.0)),
                "living_cost_paid": float(emp.last_consumption),
                "tax_declare_ratio": float(emp.last_declare_ratio),
                "savings_cash": float(emp.cash),
                "bank_deposit": float(getattr(emp, "bank_deposit", 0.0)),
                # net_worth PHAI cong ca bank_deposit -- neu khong, cua cai ho
                # gia dinh bi danh gia thap gia tao mot khi ho chuyen phan lon
                # tien mat sang tien gui ngan hang (Section 8B, rule_engine.py).
                "net_worth": float(emp.cash + getattr(emp, "bank_deposit", 0.0) - emp.debt),
                "energy": float(emp.energy),
                "skill_level": float(emp.skill_level),
                "employed_by": str(emp.employed_by) if emp.employed_by else "",
                "depository_bank_id": str(getattr(emp, "depository_bank_id", "") or ""),
                "debt": float(emp.debt),
                "age": int(emp.age),
                # parent_id: truy vet pha he cho phan tich thua ke lien the he
                # (Piketty, 2014, r>g) -- rong neu la the he goc hoac sinh qua
                # nhanh an sinh khan cap (xem env.py Section A).
                "parent_id": str(getattr(emp, "parent_id", "") or ""),
            })

        # --- 3. MICRO_FIRM (1 dong/firm dang ACTIVE/thang) ---
        for firm in active_firms_list:
            self.micro_firm_buffer.append({
                "month": timestep,
                "agent_id": firm.agent_id,
                "status": firm.status.name,
                "revenue": float(firm.last_revenue),
                "profit": float(firm.last_profit),
                "tax_declare_ratio": float(firm.last_declare_ratio),
                "capital_stock": float(firm.capital_stock),
                "productivity_factor": float(firm.productivity_factor),
                "cash": float(firm.cash),
                "debt": float(firm.debt),
                "creditor_bank_id": str(getattr(firm, "creditor_bank_id", "") or ""),
                "headcount": int(len(firm.employee_ids)),
                "age_months": int(getattr(firm, "age_months", 0)),
            })

        # --- 4. MICRO_BANK (1 dong/bank dang hoat dong/thang) ---
        for bank in banks:
            bank_npl_ratio = compute_npl_ratio_pct(bank.non_performing_loans, bank.total_loans)
            self.micro_bank_buffer.append({
                "month": timestep,
                "agent_id": bank.agent_id,
                "reserves": float(bank.reserves),
                "total_deposits": float(bank.total_deposits),
                "total_loans": float(bank.total_loans),
                "non_performing_loans": float(bank.non_performing_loans),
                "npl_ratio_pct": float(bank_npl_ratio),
                "lending_rate_annual_pct": float(bank.lending_rate * 100.0),
                "deposit_rate_annual_pct": float(bank.deposit_rate * 100.0),
                "credit_expansion_factor": float(bank.credit_expansion_factor),
                "interest_income": float(bank.last_interest_income),
                "interest_expense": float(bank.last_interest_expense),
            })

    def log_event_for_ui(self, event: Event) -> None:
        """Bien doi Event Bus thanh cau truc du lieu JSON cho Frontend, DONG
        THOI dem vao buffer Parquet 'events' de phan tich hau ky (vd. dem so
        lan audit that bai, truy vet HIRE/FIRE, doi chieu AGENT_BORN.parent_id
        voi micro_employee.parent_id). Payload duoc gop thanh 1 cot JSON string
        (payload_json) vi cau truc payload khac nhau tuy loai su kien -- tranh
        schema qua thua (sparse) neu tach tung field payload thanh 1 cot rieng."""
        ui_event = {
            "type": event.event_type.value,
            "source": event.source_id,
            "target": event.target_id,
            "payload": event.payload,
            "timestep": event.timestep,
            "event_id": event.event_id
        }
        self.ui_event_queue.append(ui_event)

        self.events_buffer.append({
            "month": int(event.timestep),
            "event_type": str(event.event_type.value),
            "source_id": str(event.source_id) if event.source_id else "",
            "target_id": str(event.target_id) if event.target_id else "",
            "payload_json": json.dumps(event.payload, default=str),
            "event_id": str(event.event_id),
        })

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
        """Xa toan bo 5 buffer ra dung 5 category Parquet va giai phong RAM."""
        self._flush_buffer(self.macro_buffer, "macro")
        self._flush_buffer(self.micro_employee_buffer, "micro_employee")
        self._flush_buffer(self.micro_firm_buffer, "micro_firm")
        self._flush_buffer(self.micro_bank_buffer, "micro_bank")
        self._flush_buffer(self.events_buffer, "events")

    def _flush_buffer(self, buffer: List[Dict[str, Any]], category: str) -> None:
        if buffer:
            self.parquet_io.write_batch(buffer, category=category)
            buffer.clear()

    def close(self) -> None:
        """Xa not du lieu con lai va DONG toan bo ParquetWriter -- BAT BUOC
        goi ham nay khi ket thuc mot lan simulate/train (thay cho goi rieng
        flush_to_disk() nhu truoc), neu khong footer Parquet se chua duoc ghi
        va cac file .parquet vua tao se KHONG doc duoc."""
        self.flush_to_disk()
        self.parquet_io.close()
