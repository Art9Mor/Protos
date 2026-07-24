# app/training/book_processor.py

import re


class BookProcessor:
    """
    Обработка книг для извлечения знаний.
    """

    def __init__(self):
        self.knowledge_graph = {}  # {понятие: [связанные_понятия]}
        self.facts = []  # Список извлеченных фактов

    def process_book(self, text: str) -> dict:
        """
        Обработка книги и извлечение знаний.
        """
        # 1. Разбиваем на предложения
        sentences = self._split_sentences(text)

        # 2. Извлекаем факты
        for sentence in sentences:
            facts = self._extract_facts(sentence)
            if facts:
                self.facts.extend(facts)

        # 3. Строим граф знаний
        self._build_knowledge_graph()

        # 4. Создаем структурированное знание
        return {
            'facts': self.facts,
            'knowledge_graph': self.knowledge_graph,
            'summary': self._create_summary()
        }

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Разбиение текста на предложения.
        """
        # Простое разбиение по точкам
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    @staticmethod
    def _extract_facts(sentence: str) -> list[str]:
        """
        Извлечение фактов из предложения.
        """
        facts = []

        # Паттерны фактов
        patterns = [
            r'([А-Яа-я]+)\s+—\s+это\s+([А-Яа-я\s,]+)',  # "X — это Y"
            r'([А-Яа-я]+)\s+является\s+([А-Яа-я\s,]+)',  # "X является Y"
            r'([А-Яа-я]+)\s+называется\s+([А-Яа-я\s,]+)',  # "X называется Y"
            r'([А-Яа-я]+)\s+имеет\s+([А-Яа-я\s,]+)',  # "X имеет Y"
            r'([А-Яа-я]+)\s+состоит\s+из\s+([А-Яа-я\s,]+)',  # "X состоит из Y"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, sentence)
            for match in matches:
                if len(match) == 2:
                    facts.append(f"{match[0]} → {match[1]}")

        return facts

    def _build_knowledge_graph(self) -> None:
        """
        Построение графа знаний.
        """
        for fact in self.facts:
            if '→' in fact:
                subject, obj = fact.split('→')
                subject = subject.strip()
                obj = obj.strip()

                if subject not in self.knowledge_graph:
                    self.knowledge_graph[subject] = set()
                self.knowledge_graph[subject].add(obj)

    def _create_summary(self) -> str:
        """
        Создание суммаризации знаний.
        """
        if not self.facts:
            return "В книге не найдено структурированных знаний"

        summary = []
        for subject, objects in list(self.knowledge_graph.items())[:10]:
            summary.append(f"{subject} связано с: {', '.join(list(objects)[:5])}")

        return '\n'.join(summary)