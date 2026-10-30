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
        """
        Выход из сессии и сразу новый сценарий входа.
        """

        head = self.session.logout(lang)
        tail = self.start_login_flow(lang)
        return f'{head}\n{tail}'


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


    def register_owner(self, name: str, password: str, lang: str = 'ru') -> str:
        return self.passwords.register_owner(name, password, lang)
