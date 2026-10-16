from __future__ import annotations

import re
from typing import TYPE_CHECKING


from app.utils.logger import assistant_logger, debug_logger

if TYPE_CHECKING:
    from app.interface.assistant import Assistant
    from app.auth.service import AuthService

class LoginFlow:
    """
    Пошаговый сценарий входа.
    """

    def __init__(self, assistant: Assistant, auth: AuthService) -> None:
        self.a = assistant
        self.auth = auth


    def start_login_flow(self, lang: str = 'ru') -> str:
        """
        Начало входа: ждём имя.
        """

        a = self.a
        a.pending_login = {'step': 'ask_name'}
        a.session_authenticated = False
        debug_logger.debug('Старт login flow: ask_name')
        assistant_logger.info('Ожидание имени для входа')
        if lang == 'ru':
            return 'Выполняется процесс распознания... Назовите Ваше имя.'
        return 'The recognition process is underway... Please state your name.'


    def handle_login_step(self, user_input: str, lang: str = 'ru') -> str:
        """
        Машина состояний: имя → статус → пароль / регистрация / creator auth.
        """

        a = self.a
        text = (user_input or '').strip()
        low = text.lower()
        info = a.pending_login or {}
        step = info.get('step', 'ask_name')

        debug_logger.debug(f'Login step={step}, input={text[:40]!r}')

        farewell = {
            'пока', 'до свидания', 'мне пора', 'увидимся', 'до встречи',
            'goodbye', 'bye', 'выход', 'стоп', 'exit', 'quit', 'конец сеанса',
        }
        low_full = re.sub(r'[.!?…]+$', '', text.lower().strip()).strip()

        if step == 'ask_name':
            if low_full in farewell:
                a.pending_login = None
                return '__SHUTDOWN__До встречи!'

            name = text.strip().strip('.!')
            if len(name) < 2:
                return 'Назовите Ваше имя.'
            name = name[0].upper() + name[1:] if name else name
            a.pending_login = {'step': 'ask_status', 'name': name}
            assistant_logger.info(f'Login: имя={name}')
            return f'Принято имя: {name}. Укажите Ваш статус.'

        name = info.get('name') or ''

        if step == 'ask_status':
            if low_full in farewell:
                a.pending_login = None
                return '__SHUTDOWN__До встречи!'

            if any(w in low for w in ('создатель', 'creator')):
                status = 'creator'
            elif any(w in low for w in ('владелец', 'owner')):
                status = 'owner'
            elif any(w in low for w in (
                    'пользователь', 'user', 'юзер',
                    'developer', 'разработчик',
                    'researcher', 'исследователь',
            )):
                status = 'user'
            else:
                return (
                    'Укажите Ваш статус.\n'
                    'Или введите «конец сеанса», чтобы выйти.'
                )

            a.pending_login = {
                'step': 'after_status',
                'name': name,
                'status': status,
            }
            return self.login_after_status(name, status, lang)

        if step == 'ask_password':
            return self.auth.login_with_password(
                name, info.get('status', 'user'), text, lang,
            )

        if step == 'register_password':
            if len(text) < 4:
                return 'Пароль слишком короткий (минимум 4 символа).'
            a.pending_login = {
                'step': 'register_password2',
                'name': name,
                'status': 'user',
                'password_once': text,
            }
            return 'Повторите пароль ещё раз.'

        if step == 'register_password2':
            if text != info.get('password_once'):
                a.pending_login = {
                    'step': 'register_password',
                    'name': name,
                    'status': 'user',
                }
                return 'Пароли не совпали. Придумайте пароль снова.'
            return self.auth.register_user(name, text, lang)

        if step == 'register_confirm':
            if any(w in low for w in ('да', 'yes', 'ага', 'регистрируй')):
                a.pending_login = {
                    'step': 'register_password',
                    'name': name,
                    'status': 'user',
                }
                return 'Придумайте пароль для входа.'
            if any(w in low for w in ('нет', 'no', 'не надо')):
                a.pending_login = None
                return '__SHUTDOWN__Хорошо. До встречи!'
            return 'Ответьте «да» (зарегистрировать) или «нет» (выйти).'

        if step == 'confirm_existing':
            if any(w in low for w in ('да', 'yes', 'это я')):
                status = (
                        info.get('status')
                        or a.known_users.get(name, {}).get('status', 'user')
                )
                a.pending_login = {
                    'step': 'after_status',
                    'name': name,
                    'status': status,
                }
                return self.login_after_status(name, status, lang)
            if any(w in low for w in ('нет', 'no', 'новый')):
                a.pending_login = {'step': 'register_confirm', 'name': name}
                return f'Зарегистрировать нового пользователя «{name}»? Да / нет.'
            return 'Ответьте «да» (это я) или «нет» (я другой человек).'

        a.pending_login = {'step': 'ask_name'}
        return 'Повторная идентификация: Назовите Ваше имя...'


    def login_after_status(self, name: str, status: str, lang: str = 'ru') -> str:
        """
        После имени и статуса: ветка creator / known / unknown.
        """

        a = self.a

        if status == 'creator':
            a.pending_login = None
            a.pending_auth = True
            a.current_speaker = name
            a.user_name = name
            assistant_logger.info(f'Login: запрос секретной фразы для {name}')
            return self.auth.request_auth(lang)

        rec = a.known_users.get(name) or a.known_users.get(name.capitalize())
        if rec is None:
            for k, v in a.known_users.items():
                if k.lower() == name.lower():
                    rec = v
                    name = k
                    break

        if rec is None:
            a.pending_login = {'step': 'register_confirm', 'name': name}
            return f'Пользователь «{name}» не найден. Зарегистрировать? Да / нет.'

        rec_status = rec.get('status', 'user')
        if status == 'owner' and rec_status != 'owner':
            return (
                'В базе эта учётная запись не владелец. '
                'Укажите другой статус или имя.'
            )

        if rec_status in ('user', 'owner') or status in ('user', 'owner'):
            if not rec.get('password_hash'):
                a.pending_login = {
                    'step': 'register_password',
                    'name': name,
                    'status': rec_status,
                }
                return f'У «{name}» ещё нет пароля. Придумайте пароль.'
            a.pending_login = {
                'step': 'ask_password',
                'name': name,
                'status': rec_status,
            }
            return 'Введите пароль.'

        a.pending_login = {'step': 'register_confirm', 'name': name}
        return (
            f'Не могу войти как {status}. '
            f'Зарегистрировать как пользователя? Да / нет.'
        )