# app/memory/context.py

import time
from collections import deque

import numpy as np


class ContextMemory:
    """
    Управление контекстом диалога с приоритезацией.
    """

    def __init__(self, max_size: int = 50, embedding_dim: int = 10):
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.messages = deque(maxlen=max_size)
        self.importance_scores = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.embeddings = deque(maxlen=max_size)

    def add_mes(self, message: str, embedding: np.ndarray | None = None, importance: float = 0.1) -> None:
        """
        Добавление сообщения в память.
        """
        self.messages.append(message)

        # Если эмбеддинг не передан, создаем случайный
        if embedding is None:
            embedding = np.random.randn(self.embedding_dim) * 0.01
        self.embeddings.append(embedding)

        self.importance_scores.append(importance)
        self.timestamps.append(time.time())

    def get_context(self, limit: int = 10) -> list[str]:
        """
        Получение наиболее важного контекста.
        """
        if len(self.messages) <= limit:
            return list(self.messages)

        combined = list(zip(self.messages, self.importance_scores, self.timestamps))
        current_time = time.time()
        combined.sort(
            key=lambda x: x[1] * (1 + 0.1 * (current_time - x[2])),
            reverse=True
        )
        return [msg for msg, _, _ in combined[:limit]]

    def get_context_with_embeddings(self, limit: int = 10) -> list[tuple[str, np.ndarray]]:
        """
        Получение контекста с эмбеддингами.
        """
        if len(self.messages) <= limit:
            return list(zip(self.messages, self.embeddings))

        combined = list(zip(self.messages, self.embeddings, self.importance_scores, self.timestamps))
        current_time = time.time()
        combined.sort(
            key=lambda x: x[2] * (1 + 0.1 * (current_time - x[3])),
            reverse=True
        )
        return [(msg, emb) for msg, emb, _, _ in combined[:limit]]

    def summarize(self) -> str:
        """
        Создание краткой сводки памяти.
        """
        if not self.messages:
            return "Диалог еще не начат"

        avg_importance = sum(self.importance_scores) / len(self.importance_scores)
        return f"Память содержит {len(self.messages)} сообщений. Средняя важность: {avg_importance:.2f}. Последнее: {self.messages[-1][:50]}..."

    def get_recent(self, n: int = 5) -> list[str]:
        """
        Получение последних N сообщений.
        """
        return list(self.messages)[-n:]

    def clear(self) -> None:
        """
        Очистка памяти.
        """
        self.messages.clear()
        self.embeddings.clear()
        self.importance_scores.clear()
        self.timestamps.clear()

    def __len__(self) -> int:
        """
        Количество сообщений в памяти.
        """
        return len(self.messages)