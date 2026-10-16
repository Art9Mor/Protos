from __future__ import annotations

import hashlib
import secrets
import uuid

from typing import TYPE_CHECKING

from app.utils.logger import assistant_logger, debug_logger

if TYPE_CHECKING:
    from app.interface.assistant import Assistant
    from app.auth.service import AuthService


class PasswordAuth:
    """
    Учётные записи с паролем для пользователей.
    """

    def __init__(self, assistant: Assistant, auth: AuthService) -> None:
        self.a = assistant
        self.auth = auth


    @staticmethod
    def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
        """
        Возвращает (salt, hash_hex).
        """

        if salt is None:
            salt = secrets.token_hex(16)
        raw = hashlib.sha256((salt + password).encode('utf-8')).hexdigest()
        return salt, raw


    def verify_password(self, password: str, salt: str, password_hash: str) -> bool:
        """
        Проверка пароля по salt и hash.
        """

        _, got = self.hash_password(password, salt=salt)
        return secrets.compare_digest(got, password_hash)


    def login_with_password(
            self,
            name: str,
            status: str,
            password: str,
            lang: str = 'ru',
    ) -> str:
        """
        Проверка пароля и открытие сессии.
        """

        a = self.a
        rec = None
        key = name
        for k, v in a.known_users.items():
            if k.lower() == name.lower():
                rec = v
                key = k
                break

        if not rec:
            a.pending_login = {'step': 'ask_name'}
            return 'Учётная запись не найдена. Идентифицируйте себя.'

        salt = rec.get('salt') or ''
        ph = rec.get('password_hash') or ''
        if not ph or not self.verify_password(password, salt, ph):
            debug_logger.debug(f'Неверный пароль для {key}')
            assistant_logger.info(f'В доступе отказано: неверный пароль ({key})')
            return 'Неверный пароль. Попробуйте ещё раз или напишите «логаут».'

        self.auth.open_session(key, rec.get('status', status))
        a.pending_login = None
        assistant_logger.info(f'Вход выполнен: {key} ({a.user_role})')
        address = a._address_name() if hasattr(a, '_address_name') else key
        return f'Вход выполнен. Привет, {address}!'


    def register_user(self, name: str, password: str, lang: str = 'ru') -> str:
        """
        Регистрация user с паролем и открытие сессии.
        """

        a = self.a
        salt, ph = self.hash_password(password)
        uid = str(uuid.uuid4())[:8]

        a.known_users[name] = {
            'id': uid,
            'status': 'user',
            'relations': {},
            'salt': salt,
            'password_hash': ph,
            'aliases': [],
            'preferred_name': None,
            'logged_in': False,
        }
        a.user_statuses[name] = 'user'
        self.auth.open_session(name, 'user')
        a.pending_login = None
        assistant_logger.info(f'Зарегистрирован user: {name}')
        return (
            f'Готово, {name}. Ты зарегистрирован как пользователь. '
            f'Можем общаться.'
        )