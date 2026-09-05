import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable
from be.core.enums import EventType

@dataclass(frozen=True)
class Event:
    event_type: EventType
    source_id: str
    target_id: str
    payload: Dict[str, Any]
    timestep: int
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

class EventBus:
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {
            e_type: [] for e_type in EventType
        }
        self._history: List[Event] = []

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        self._subscribers[event_type].append(handler)

    def publish(self, event: Event):
        self._history.append(event)
        for handler in self._subscribers.get(event.event_type, []):
            handler(event)

    def get_events(self, timestep: int = None) -> List[Event]:
        if timestep is None:
            return list(self._history)
        return [e for e in self._history if e.timestep == timestep]

    def clear(self):
        self._history.clear()