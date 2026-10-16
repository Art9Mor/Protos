from __future__ import annotations

from typing import TYPE_CHECKING

from app.utils.logger import debug_logger, assistant_logger

if TYPE_CHECKING:
    from app.interface.assistant import Assistant


class SessionManager:
    """
    Управление активной сессией пользователя.
    """


    def __init__(self, assistant: Assistant) -> None:
        self.a = assistant


    def open_session(self, name: str, status: str) -> None:
        """
        Открытие сессии после успешной проверки.
        """

        a = self.a
        a.current_speaker = name
        a.user_name = name
        a.user_role = status if status in ('user', 'owner', 'creator') else 'user'
        a.session_authenticated = True
        a.pending_auth = False
        a.pending_login = None
        self.set_logged_in(name)
        debug_logger.debug(f'Сессия открыта: {name}, role={a.user_role}')
        assistant_logger.info(f'Сессия открыта: {name} ({a.user_role})')


    def set_logged_in(self, name: str | None) -> None:
        """
        Ровно один пользователь с logged_in=True, либо никто (логаут).
        """

        a = self.a
        for n, rec in list(a.known_users.items()):
            if not isinstance(rec, dict):
                a.known_users[n] = {
                    'status': a.user_statuses.get(n, 'user'),
                    'relations': {},
                    'logged_in': False,
                }
                rec = a.known_users[n]
            rec['logged_in'] = bool(name) and (n == name)

        if name and name not in a.known_users:
            status = a.user_statuses.get(name, 'user')
            if name == a.creator_name or a._is_creator_name(name):
                status = 'creator'
            a.known_users[name] = {
                'status': status,
                'relations': {},
                'aliases': list(a.creator_aliases) if status == 'creator' else [],
                'logged_in': True,
            }

        a._save_user_statuses()
        debug_logger.debug(f'logged_in → {name!r}')


    def restore_session(self) -> bool:
        """
        После перезапуска: один logged_in=True → сессия без нового логина.
        """
        a = self.a
        active = [
            n for n, r in (a.known_users or {}).items()
            if isinstance(r, dict) and r.get('logged_in')
        ]

        if len(active) == 0:
            return False

        if len(active) > 1:
            assistant_logger.warning(
                f'Несколько logged_in={active}, сбрасываю все'
            )
            self.set_logged_in(None)
            return False

        name = active[0]
        rec = a.known_users.get(name) or {}
        role = rec.get('status') or a.user_statuses.get(name) or 'user'

        if role == 'creator' or a._is_creator_name(name):
            name = a._canonical_creator_name()
            role = 'creator'

        a.current_speaker = name
        a.user_name = name
        a.user_role = role
        a.session_authenticated = True
        a.pending_login = None
        a.pending_auth = False

        assistant_logger.info(f'Сессия восстановлена: {name} ({role})')
        return True


    def logout(self, lang: str = 'ru') -> str:
        """
        Сброс текущей сессии (учётки с диска не удаляет).
        """

        a = self.a
        prev = a.current_speaker or '?'
        debug_logger.debug(f'Logout: был {prev}, role={a.user_role}')
        assistant_logger.info(f'Выход из сессии: {prev}')

        a.session_authenticated = False
        a.user_role = 'user'
        a.current_speaker = None
        a.user_name = None
        a.pending_auth = False
        a.pending_teach = None
        a.pending_rename = None
        a.pending_login = None
        self.set_logged_in(None)

        if lang == 'ru':
            return (
                'Сессия была завершена. '
                'Инициализируйте новую сессию, чтобы продолжить.'
            )
        return (
            'The session has been completed. '
            'Initialize a new session to continue.'
        )
