from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

@dataclass
class Observation:
    agent_id: str
    timestep: int
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Action:
    agent_id: str
    action_type: str
    values: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    is_valid: bool
    sanitized_values: np.ndarray
    reason: Optional[str] = None

@dataclass
class TransitionResult:
    agent_id: str
    state_delta: Dict[str, Any]
    events_triggered: List[str]
    success: bool
    info: Dict[str, Any] = field(default_factory=dict)