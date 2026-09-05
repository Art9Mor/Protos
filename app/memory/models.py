from dataclasses import dataclass, field
from enum import Enum, auto
import time

import numpy as np


class MemoryEventType(Enum):
    """
    Тип события, которое хранится в памяти.
    """

    MESSAGE = auto()
    THOUGHT = auto()
    ACTION = auto()
    LEARNING = auto()
    SYSTEM = auto()


@dataclass(slots=True, frozen=True)
class MemoryEvent:
    """
    Одно событие из жизни Протоса.
    """

    event_type: MemoryEventType
    content: str

    importance: float = 0.1

    timestamp: float = field(default_factory=time.time)

    embedding: np.ndarray | None = None