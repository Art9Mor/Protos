from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from app.utils.logger import debug_logger, error_logger

if TYPE_CHECKING:
    from app.interface.assistant import Assistant


class CreatorSecret:
    """
    Подтверждение роли создателя по секретной фразе.
    """

    def __init__(self, assistant: Assistant) -> None:
        self.a = assistant


    def check_secret_phrase(self, text: str) -> bool:
        """
        Проверка секретной фразы.
        """

        normalized = (text or '').lower().strip()
        for prefix in (
                'код:', 'код доступа:', 'фраза:', 'секрет:', 'пароль:',
        ):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        normalized = normalized.rstrip('.,!?;:…')
        return normalized == (self.a.creator_secret or '').lower().strip()


    def request_auth(self, lang: str = 'ru') -> str:
        """
        Запрос секретной фразы; включает pending_auth.
        """

        a = self.a
        a.pending_auth = True
        hint = a.creator_secret_hint or ''
        if lang == 'ru':
            return (
                'Чтобы подтвердить, что ты создатель, назови секретную фразу.\n'
                f'Подсказка: {hint}'
            )
        return (
            'To confirm you are the creator, say the secret phrase.\n'
            f'Hint: {hint}'
        )


    def load(self) -> None:
        """
        Загрузка секрета из data/creator_secret.json.
        """

        path = 'data/creator_secret.json'
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            a = self.a
            a.creator_secret = data.get('secret', a.creator_secret).lower().strip()
            a.creator_secret_hint = data.get('hint', a.creator_secret_hint)
            debug_logger.debug('Секретная фраза загружена')
        except (json.JSONDecodeError, IOError) as e:
            error_logger.error(f'Ошибка загрузки секрета: {e}')


    def save(self) -> None:
        """
        Сохранение секрета на диск.
        """

        path = 'data/creator_secret.json'
        os.makedirs('data', exist_ok=True)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'secret': self.a.creator_secret,
                        'hint': self.a.creator_secret_hint,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            debug_logger.debug('Секретная фраза сохранена')
        except IOError as e:
            error_logger.error(f'Ошибка сохранения секрета: {e}')