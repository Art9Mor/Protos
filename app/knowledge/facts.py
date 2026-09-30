import json
import os
import re

from app.utils.logger import debug_logger, error_logger, assistant_logger


class FactsStore:
    """
    Хранение и поиск коротких фактов (data/facts.json).
    """

    def __init__(self, path: str = 'data/facts.json'):
        self.path = path
        self.facts: list[dict] = []
        self.load()

    def load(self) -> None:
        """
        Загрузка фактов из файла.
        """

        self.facts = []

        if not os.path.exists(self.path):
            debug_logger.debug('Файл фактов не найден: {}', self.path)
            return
        try:
            with open(self.path, 'r', encoding='utf-8')as f:
                data = json.load(f)
            self.facts = data if isinstance(data, list) else []
            debug_logger.debug(f'Загружено фактов: {len(self.facts)}')
            assistant_logger.info(f'Загружено фактов: {len(self.facts)}')
        except (json.JSONDecodeError, IOError) as e:
            error_logger.error(f'Ошибка загрузки фактов: {e}')
            self.facts = []

    def save(self) -> None:
        """
        Сохранение фактов.
        """

        os.makedirs(os.path.dirname(self.path) or 'data', exist_ok=True)
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.facts, f, ensure_ascii=False, indent=2)
            debug_logger.debug(f'Сохранено фактов: {len(self.facts)}')
        except IOError as e:
            error_logger.error(f'Ошибка сохранения фактов: {e}')

    def add(
            self,
            subject: str,
            fact: str,
            source: str = 'manual',
            aliases: list[str] | None = None,
    ) -> None:
        """
        Добавление или обновление факта.
        """

        sub = subject.strip().lower()
        if not sub or not fact.strip():
            return

        for item in self.facts:
            if item.get('subject', '').lower() == sub:
                item['fact'] = fact.strip()
                item['source'] = source
                if aliases is not None:
                    item['aliases'] = aliases
                self.save()
                return

        self.facts.append({
            'subject': sub,
            'aliases': aliases or [],
            'fact': fact.strip(),
            'source': source,
        })
        self.save()

    def find(self, text: str) -> str | None:
        """
        Поиск факта по фразе пользователя.
        """

        if not self.facts:
            return None

        low = text.lower().strip()
        m = re.search(
            r'(?:кто\s+так(?:ой|ая|ое|ие)|что\s+так(?:ое|ая|ой)|что\s+значит|зачем\s+нужн\w*)\s+(.+?)(?:\?|$)',
            low,
        )
        query = m.group(1).strip(' .!?…') if m else low
        query = re.sub(r'^(такое|такой|такая)\s+', '', query).strip()

        best = None
        best_score = 0.0

        for item in self.facts:
            keys = [item.get('subject', '')] + list(item.get('aliases') or [])
            for key in keys:
                key = (key or '').lower().strip()
                if not key:
                    continue
                if key == query:
                    return item.get('fact')
                if key in query or query in key:
                    score = len(key) / max(len(query), 1)
                    if score > best_score:
                        best_score = score
                        best = item.get('fact')

        return best if best_score >= 0.3 else None
