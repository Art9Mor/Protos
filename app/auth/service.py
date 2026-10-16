from typing import TYPE_CHECKING

from app.auth.login_flow import LoginFlow
from app.auth.passwords import PasswordAuth
from app.auth.secret import CreatorSecret
from app.auth.session import SessionManager


if TYPE_CHECKING:
    from app.interface.assistant import Assistant


class AuthService:
    """
    Единая точка входа в подсистему аутентификации.
    """

    def __init__(self, assistant: 'Assistant') -> None:
        self.a = assistant
        self.session = SessionManager(assistant)
        self.secret = CreatorSecret(assistant)
        self.passwords = PasswordAuth(assistant, self)
        self.login = LoginFlow(assistant, self)


    # --- session ---
    def open_session(self, name: str, status: str) -> None:
        self.session.open_session(name, status)


    def set_logged_in(self, name: str | None) -> None:
        self.session.set_logged_in(name)


    def restore_session(self) -> bool:
        return self.session.restore_session()


    def logout(self, lang: str = 'ru') -> str:
        return self.session.logout(lang)


    # --- secret ---
    def check_secret_phrase(self, text: str) -> bool:
        return self.secret.check_secret_phrase(text)


    def request_auth(self, lang: str = 'ru') -> str:
        return self.secret.request_auth(lang)


    def load_creator_secret(self) -> None:
        self.secret.load()


    def save_creator_secret(self) -> None:
        self.secret.save()


    # --- passwords ---
    def login_with_password(
            self, name: str, status: str, password: str, lang: str = 'ru',
    ) -> str:
        return self.passwords.login_with_password(name, status, password, lang)


    def register_user(self, name: str, password: str, lang: str = 'ru') -> str:
        return self.passwords.register_user(name, password, lang)


    def hash_password(self, password: str, salt: str | None = None) -> tuple[str, str]:
        return self.passwords.hash_password(password, salt)


    def verify_password(self, password: str, salt: str, password_hash: str) -> bool:
        return self.passwords.verify_password(password, salt, password_hash)


    # --- login flow ---
    def start_login_flow(self, lang: str = 'ru') -> str:
        return self.login.start_login_flow(lang)


    def handle_login_step(self, user_input: str, lang: str = 'ru') -> str:
        return self.login.handle_login_step(user_input, lang)


    def login_after_status(self, name: str, status: str, lang: str = 'ru') -> str:
        return self.login.login_after_status(name, status, lang)


    # def handle_login_step(self, text: str, lang: str) -> str:
    #     """
    #     Машина состояний: имя → статус → пароль / регистрация / creator auth.
    #     """
    #
    #     text = (user_input or '').strip()
    #     low = text.lower()
    #     info = self.a.pending_login or {}
    #     step = info.get('step', 'ask_name')
    #
    #     debug_logger.debug(f'Login step={step}, input={text[:40]!r}')
    #
    #     farewell = {
    #         'пока', 'до свидания', 'мне пора', 'увидимся', 'до встречи',
    #         'goodbye', 'bye', 'выход', 'стоп', 'exit', 'quit', 'конец сеанса',
    #     }
    #     low_full = re.sub(r'[.!?…]+$', '', text.lower().strip()).strip()
    #
    #     if step == 'ask_name':
    #         if low_full in farewell:
    #             self.a.pending_login = None
    #             return '__SHUTDOWN__До встречи!'
    #
    #         name = text.strip().strip('.!')
    #         if len(name) < 2:
    #             return 'Назовите Ваше имя.'
    #         name = name[0].upper() + name[1:] if name else name
    #         self.a.pending_login = {'step': 'ask_status', 'name': name}
    #         assistant_logger.info(f'Login: имя={name}')
    #         return 'Принято имя: {}. Укажите Ваш статус.'.format(name)
    #
    #     name = info.get('name') or ''
    #
    #     if step == 'ask_status':
    #         if low_full in farewell:
    #             self.a.pending_login = None
    #             return '__SHUTDOWN__До встречи!'
    #
    #         if any(w in low for w in ('создатель', 'creator')):
    #             status = 'creator'
    #         elif any(w in low for w in ('владелец', 'owner')):
    #             status = 'owner'
    #         elif any(w in low for w in (
    #                 'пользователь', 'user', 'юзер',
    #                 'developer', 'разработчик',
    #                 'researcher', 'исследователь',
    #         )):
    #             status = 'user'
    #         else:
    #             return (
    #                 'Укажите Ваш статус.\n'
    #                 'Или введите «конец сеанса», чтобы выйти.'
    #             )
    #
    #         self.a.pending_login = {'step': 'after_status', 'name': name, 'status': status}
    #         return self.a._login_after_status(name, status, lang)
    #
    #     if step == 'ask_password':
    #         return self.a._login_with_password(name, info.get('status', 'user'), text, lang)
    #
    #     if step == 'register_password':
    #         if len(text) < 4:
    #             return 'Пароль слишком короткий (минимум 4 символа).'
    #         self.a.pending_login = {
    #             'step': 'register_password2',
    #             'name': name,
    #             'status': 'user',
    #             'password_once': text,
    #         }
    #         return 'Повторите пароль ещё раз.'
    #
    #     if step == 'register_password2':
    #         if text != info.get('password_once'):
    #             self.a.pending_login = {'step': 'register_password', 'name': name, 'status': 'user'}
    #             return 'Пароли не совпали. Придумайте пароль снова.'
    #         return self.a._register_user(name, text, lang)
    #
    #     if step == 'register_confirm':
    #         if any(w in low for w in ('да', 'yes', 'ага', 'регистрируй')):
    #             self.a.pending_login = {'step': 'register_password', 'name': name, 'status': 'user'}
    #             return 'Придумайте пароль для входа.'
    #         if any(w in low for w in ('нет', 'no', 'не надо')):
    #             self.a.pending_login = None
    #             return '__SHUTDOWN__Хорошо. До встречи!'
    #         return 'Ответьте «да» (зарегистрировать) или «нет» (выйти).'
    #
    #     if step == 'confirm_existing':
    #         if any(w in low for w in ('да', 'yes', 'это я')):
    #             status = info.get('status') or self.a.known_users.get(name, {}).get('status', 'user')
    #             self.a.pending_login = {'step': 'after_status', 'name': name, 'status': status}
    #             return self.a._login_after_status(name, status, lang)
    #         if any(w in low for w in ('нет', 'no', 'новый')):
    #             self.a.pending_login = {'step': 'register_confirm', 'name': name}
    #             return f'Зарегистрировать нового пользователя «{name}»? Да / нет.'
    #         return 'Ответьте «да» (это я) или «нет» (я другой человек).'
    #
    #     self.a.pending_login = {'step': 'ask_name'}
    #     return 'Повторная идентификация: Назовите Ваше имя...'


    # def logout(self, lang: str = 'ru') -> str:
    #     """
    #     Сброс текущей сессии (не удаляет creator с диска).
    #     """
    #
    #     return self.session.logout(lang)
