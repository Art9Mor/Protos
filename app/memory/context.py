import time
from collections import deque

import numpy as np

from .models import MemoryEvent, MemoryEventType


class ContextMemory:
    """
    Управление кратковременной памятью Протоса.
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.events: deque[MemoryEvent] = deque(maxlen=max_size)

    def record(self, event: MemoryEvent) -> None:
        """
        Записать событие в память.
        """
        self.events.append(event)

    def remember_message(
            self,
            text: str,
            importance: float = 0.1,
            embedding: np.ndarray | None = None,
    ) -> None:
        """
        Сохранить сообщение в память.
        """

        self.record(
            MemoryEvent(
                event_type=MemoryEventType.MESSAGE,
                content=text,
                importance=importance,
                embedding=embedding,
            )
        )

    def remember_thought(
            self,
            text: str,
            importance: float = 0.3,
            embedding: np.ndarray | None = None,
    ) -> None:
        """
        Сохранить мысль в память.
        """

        self.record(
            MemoryEvent(
                event_type=MemoryEventType.THOUGHT,
                content=text,
                importance=importance,
                embedding=embedding,
            )
        )

    def remember_action(
            self,
            text: str,
            importance: float = 0.2,
    ) -> None:
        """
        Сохранить действие Протоса.
        """

        self.record(
            MemoryEvent(
                event_type=MemoryEventType.ACTION,
                content=text,
                importance=importance,
            )
        )

    def remember_learning(
            self,
            text: str,
            importance: float = 0.4,
    ) -> None:
        """
        Сохранить результат обучения.
        """

        self.record(
            MemoryEvent(
                event_type=MemoryEventType.LEARNING,
                content=text,
                importance=importance,
            )
        )

    @staticmethod
    def _score(event: MemoryEvent) -> float:
        """
        Рассчитать приоритет события.

        Пока учитываются только:
        - важность
        - давность
        """

        age = time.time() - event.timestamp

        # Через час значимость уменьшается примерно вдвое.
        freshness = 1 / (1 + age / 3600)

        return event.importance * freshness

    def get_context(self, limit: int = 10) -> list[MemoryEvent]:
        """
        Получить наиболее значимые события.
        """

        if len(self.events) <= limit:
            return list(self.events)

        ranked = sorted(
            self.events,
            key=self._score,
            reverse=True,
        )

        return ranked[:limit]

    def get_context_with_embeddings(
        self,
        limit: int = 10,
    ) -> list[tuple[MemoryEvent, np.ndarray]]:
        """
        Получить события вместе с embedding.
        """

        return [
            (event, event.embedding)
            for event in self.get_context(limit)
            if event.embedding is not None
        ]

    def summarize(self) -> str:
        """
        Краткая сводка памяти.
        """

        if not self.events:
            return "Память пока пуста."

        avg_importance = (
            sum(event.importance for event in self.events)
            / len(self.events)
        )

        last = self.events[-1]

        return (
            f"Память содержит {len(self.events)} событий. "
            f"Средняя важность: {avg_importance:.2f}. "
            f"Последнее событие: {last.content[:50]}..."
        )

    def get_recent(self, n: int = 5) -> list[MemoryEvent]:
        """
        Получить последние события.
        """

        return list(self.events)[-n:]

    def clear(self) -> None:
        """
        Очистить память.
        """

        self.events.clear()

    def __len__(self) -> int:
        """
        Количество событий.
        """

        return len(self.events)