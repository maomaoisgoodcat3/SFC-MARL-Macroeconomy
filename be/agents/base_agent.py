from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from be.core.enums import LifeCycleStatus
from be.core.types import Observation, Action, ValidationResult, TransitionResult

class BaseAgent(ABC):
    """
    Contract chuan cua Agent theo Software Architecture Specification v1.0.
    Khong chua dinh danh tinh, khong chua luat kinh te, chi quan ly vong doi.
    """
    def __init__(self, agent_id: str):
        self.agent_id: str = agent_id
        self.status: LifeCycleStatus = LifeCycleStatus.INITIALIZED
        self.internal_state: Dict[str, Any] = {}
        self.history: list = []

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Thiet lap thong so ban dau truoc khi bat dau chu ky."""
        self.status = LifeCycleStatus.ACTIVE

    @abstractmethod
    def observe(self, raw_environment_state: Dict[str, Any]) -> Observation:
        """Chuyen doi trang thai the gioi thanh vector quan sat cuc bo."""
        raise NotImplementedError

    @abstractmethod
    def decide(self, observation: Observation) -> Action:
        """Hanh vi quyet dinh (Policy inference / Heuristic)."""
        raise NotImplementedError

    @abstractmethod
    def validate_action(self, action: Action) -> ValidationResult:
        """Tu kiem tra xem hanh dong co vuot qua nguong the ly/noi tai khong."""
        raise NotImplementedError

    @abstractmethod
    def apply_result(self, transition_result: TransitionResult) -> None:
        """Cap nhat trang thai noi tai sau khi RuleEngine da phe duyet va thuc thi."""
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self, transition_result: TransitionResult) -> float:
        """Tinh toan utility/reward dua tren ket qua thuc thi trang thai moi."""
        raise NotImplementedError

    @abstractmethod
    def export_state(self) -> Dict[str, Any]:
        """Xuat trang thai hien tai de phuc vu he thong Log va Frontend."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Khoi phuc trang thai ban dau cho Episode moi."""
        self.internal_state.clear()
        self.history.clear()
        self.status = LifeCycleStatus.INITIALIZED

    @abstractmethod
    def terminate(self, reason: str = "") -> None:
        """Ket thuc vong doi (pha san hoac tu vong)."""
        self.status = LifeCycleStatus.TERMINATED