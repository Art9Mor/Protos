from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils.logger import (
    assistant_logger,
    debug_logger,
    error_logger,
    fs_logger,
    memory_logger,
    training_logger,
)

if TYPE_CHECKING:
    from .assistant import Assistant

CommandCallback = Callable[[list[str]], str]


class CommandHandler:
    """
    Slash-команды и узкие NL-команды Протоса.

    Обучение и модель доступны creator и owner.
    Личность, секрет и отладка — только creator.
    """


    def __init__(self, assistant: Assistant) -> None:
        """
        Привязка к экземпляру Assistant.
        """

        self.assistant = assistant


    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def handle(self, command: str) -> str | None:
        """
        Разбор и выполнение команды вида /name [args].
        """

        raw = (command or '').strip()
        if not raw.startswith('/'):
            raw = '/' + raw

        parts = raw[1:].split()
        command_name = (parts[0] if parts else '').lower()
        arguments = parts[1:]

        debug_logger.debug(
            f'Обработка команды: {command_name}, аргументы: {arguments}'
        )
        assistant_logger.debug(
            f'Команда: {command_name}, аргументы: {arguments}'
        )

        handlers: dict[str, CommandCallback] = {
            'help': self._cmd_help,
            'commands': self._cmd_help,
            'команды': self._cmd_help,
            'clear_memory': self._cmd_clear_memory,
            'reset': self._cmd_reset,
            'debug': self._cmd_debug,
            'train': self._cmd_train,
            'learn_from_file': self._cmd_learn_from_file,
            'learn_text': self._cmd_learn_text,
            'save_model': self._cmd_save_model,
            'load_model': self._cmd_load_model,
            'set_status': self._cmd_set_status,
            'users': self._cmd_users,
            'register_owner': self._cmd_register_owner,
            'delete_user': self._cmd_delete_user,
            'relations': self._cmd_relations,
            'knowledge': self._cmd_knowledge,
            'learning_status': self._cmd_learning_status,
            'train_on_books': self._cmd_train_on_books,
            'continue_training': self._cmd_continue_training,
            'check_books': self._cmd_check_books,
            'download_books': self._cmd_download_books,
            'auth': self._cmd_auth,
            'set_secret': self._cmd_set_secret,
            'gen_only': self._cmd_gen_only,
            'logout': self._cmd_logout,
            'exit': self._cmd_exit,
            'quit': self._cmd_exit,
        }

        handler = handlers.get(command_name)
        if handler is None:
            return None
        return handler(arguments)


    @staticmethod
    def match_nl(text: str) -> str | None:
        """
        Сопоставление естественной фразы с именем команды (без /).
        """

        normalized = text.lower().strip()
        rules = [
            (r'обуч(ись|итесь|ение).*(книг|books)|обучись\s+на\s+книгах', 'train_on_books'),
            (r'сохрани\s+модель|сохранить\s+модель', 'save_model'),
            (r'загрузи\s+модель|загрузить\s+модель', 'load_model'),
            (r'статус\s+обучения|как\s+идёт\s+обучение|как\s+идет\s+обучение', 'learning_status'),
            (r'очисти\s+память|очистить\s+память', 'clear_memory'),
            (r'какие\s+команды|список\s+команд|что\s+ты\s+умеешь', 'help'),
            (r'известн\w*\s+пользовател|список\s+пользовател|кто\s+мне\s+известен', 'users'),
            (r'логаут|выйти из аккаунта|конец сессии|разлогин', 'logout'),
            (r'^выход$|^стоп$|выключись|заверши работу', 'exit'),
        ]
        for pattern, command_name in rules:
            if re.search(pattern, normalized):
                return command_name
        return None


    def help_text(self) -> str:
        """
        Текст справки в зависимости от роли текущей сессии.
        """

        role = self.assistant.user_role

        if role == 'creator':
            return (
                'Creator commands / Команды создателя:\n'
                '/train — train on accumulated data\n'
                '/train_on_books — train on data/books\n'
                '/continue_training — continue book training\n'
                '/save_model — save model\n'
                '/load_model — load model\n'
                '/learn_from_file <path> — load file for training\n'
                '/learn_text <text> — add text for training\n'
                '/learning_status — training data status\n'
                '/check_books — list books in data/books\n'
                '/download_books — download sample books\n'
                '/auth <phrase> — confirm creator secret\n'
                '/set_secret <phrase> — change secret (creator only)\n'
                '/set_status <name> <status> — set user status\n'
                '/debug — debug info\n'
                '/clear_memory — clear dialogue memory\n'
                '/reset — reset session memory + learning flag\n'
                '/users — known users\n'
                '/'
                '/'
                '/knowledge <query> — search knowledge\n'
                '/relations — your relations\n'
                '/gen_only — toggle LSTM-only replies\n'
                '/logout — end of session\n'
                '/exit, /quit - shutdown\n'
                '\n'
                'Without / : «обучись на книгах», «сохрани модель»,\n'
                '«статус обучения», «какие команды», «задай вопрос».'
            )

        if role == 'owner':
            return (
                'Owner commands / Команды владельца:\n'
                '/train — train on accumulated data\n'
                '/train_on_books — train on data/books\n'
                '/continue_training — continue book training\n'
                '/save_model — save model\n'
                '/load_model — load model\n'
                '/learn_from_file <path> — load file for training\n'
                '/learn_text <text> — add text for training\n'
                '/learning_status — training data status\n'
                '/check_books — list books\n'
                '/download_books — download sample books\n'
                '/set_status <name> <status> — set user status\n'
                '/knowledge <query> — search knowledge\n'
                '/relations — your relations\n'
                '/help — this help\n'
                '• /logout — end of session\n'
                '•/exit, /quit - shutdown\n'
                '\n'
                'Personality / secret: creator only.\n'
                'Личность и секрет — только создатель.'
            )

        return (
            'Available / Доступно:\n'
            '• talk / поговорить\n'
            '• /knowledge <query> — search knowledge\n'
            '• /relations — your relations\n'
            '• /help — this help\n'
            '• /logout — end of session\n'
            '•/ exit, / quit - shutdown\n'
        )


    def system_command(self, command: str) -> str:
        """
        Файловые команды ls / cd (только owner или creator).
        """

        denied = self._require_owner_or_creator()
        if denied:
            return denied

        assistant = self.assistant
        if command.startswith('ls'):
            path = command[2:].strip()
            debug_logger.debug(f'Команда ls: {path or "."}')
            fs_logger.info(f'Команда ls: {path or "."}')
            return assistant.fs_manager.list_dir(path if path else None)
        if command.startswith('cd'):
            path = command[2:].strip()
            debug_logger.debug(f'Команда cd: {path or "~"}')
            fs_logger.info(f'Команда cd: {path or "~"}')
            return assistant.fs_manager.change_dir(path if path else None)
        return 'Unknown command / Неизвестная команда'


    def info(self) -> str:
        """
        Краткая статистика текущей сессии.
        """

        assistant = self.assistant
        debug_logger.debug('Вывод информации о состоянии')
        status = (
            assistant.user_statuses.get(assistant.current_speaker, '—')
            if assistant.current_speaker
            else '—'
        )
        return (
            f'Protos stats / Статистика Протоса:\n'
            f'  Conversations / Разговоров: {assistant.conversation_count}\n'
            f'  Memory messages / Сообщений в памяти: {len(assistant.memory)}\n'
            f'  User / Пользователь: {assistant.current_speaker or "—"}\n'
            f'  Status / Статус: {status}\n'
            f'  Role / Роль: {assistant.user_role}\n'
            f'  Creator / Создатель: {assistant.creator_name or "—"}'
        )


    def debug_info(self) -> str:
        """
        Расширенная отладочная сводка (только после проверки роли creator).
        """

        assistant = self.assistant
        debug_logger.debug('Вывод отладочной информации')
        learning_data_count = (
            len(assistant.learning_manager.learning_data)
            if assistant.learning_manager
            else 0
        )
        status = (
            assistant.user_statuses.get(assistant.current_speaker, '—')
            if assistant.current_speaker
            else '—'
        )
        return (
            f'DEBUG / ОТЛАДКА:\n'
            f'  User: {assistant.current_speaker or "—"}\n'
            f'  Status: {status}\n'
            f'  Role: {assistant.user_role}\n'
            f'  Creator: {assistant.creator_name or "—"}\n'
            f'  Owner: {assistant.owner_name or "—"}\n'
            f'  Conversations: {assistant.conversation_count}\n'
            f'  Memory size: {len(assistant.memory)}\n'
            f'  Learning: {assistant.is_learning}\n'
            f'  gen_only: {getattr(assistant, "gen_only", False)}\n'
            f'  Known users: {list(assistant.known_users.keys())}\n'
            f'  Relations: {assistant.relationships}\n'
            f'  Learning examples: {learning_data_count}'
        )


    # ------------------------------------------------------------------
    # Права
    # ------------------------------------------------------------------

    def _require_creator(self) -> str | None:
        """
        Проверка роли creator.
        """

        if self.assistant.user_role != 'creator':
            debug_logger.debug('Отказ: нужна роль creator')
            return (
                'Access denied. Creator only. / '
                'Доступ запрещён. Только создатель.'
            )
        return None


    def _require_owner_or_creator(self) -> str | None:
        """
        Проверка роли owner или creator.
        """

        if self.assistant.user_role not in ('owner', 'creator'):
            debug_logger.debug('Отказ: нужна роль owner или creator')
            return (
                'Access denied. Owner or creator only. / '
                'Доступ запрещён. Только владелец или создатель.'
            )
        return None


    # ------------------------------------------------------------------
    # Реализации /команд
    # ------------------------------------------------------------------

    def _cmd_help(self, _arguments: list[str]) -> str:
        """
        /help — справка по доступным командам.
        """

        return self.help_text()


    def _cmd_clear_memory(self, _arguments: list[str]) -> str:
        """
        /clear_memory — очистка диалоговой памяти (только creator).
        """

        denied = self._require_creator()
        if denied:
            return denied
        self.assistant.memory.clear()
        debug_logger.debug('Память очищена по команде создателя')
        memory_logger.info('Память очищена по команде создателя')
        return 'Memory cleared. / Память полностью очищена.'


    def _cmd_reset(self, _arguments: list[str]) -> str:
        """
        /reset — сброс памяти и флага обучения (только creator).
        """

        denied = self._require_creator()
        if denied:
            return denied
        self.assistant.memory.clear()
        self.assistant.is_learning = True
        debug_logger.debug('Система сброшена')
        memory_logger.info('Система сброшена')
        return 'System reset. / Система сброшена к начальному состоянию.'


    def _cmd_debug(self, _arguments: list[str]) -> str:
        """
        /debug — расширенная отладочная информация (только creator).
        """

        denied = self._require_creator()
        if denied:
            return denied
        return self.debug_info()


    def _cmd_train(self, _arguments: list[str]) -> str:
        """
        /train — обучение на накопленных данных (owner или creator).
        """

        denied = self._require_owner_or_creator()
        if denied:
            return denied
        debug_logger.debug('Запуск обучения по команде')
        training_logger.info('Запуск обучения по команде')
        self.assistant.learning_manager.train(epochs=10)
        self.assistant.learning_manager.save_model('models/protos_lstm.json')
        training_logger.success('Обучение завершено')
        return 'Training finished. / Обучение завершено.'


    def _cmd_learn_from_file(self, arguments: list[str]) -> str:
        """
        /learn_from_file <path> — добавить файл в данные обучения.
        """

        denied = self._require_owner_or_creator()
        if denied:
            return denied
        if not arguments:
            return 'Usage: /learn_from_file <path>'
        filepath = arguments[0]
        debug_logger.debug(f'Загрузка файла для обучения: {filepath}')
        return self.assistant.learning_manager.learn_from_file(filepath)


    def _cmd_learn_text(self, arguments: list[str]) -> str:
        """
        /learn_text <text> — добавить произвольный текст в данные обучения.
        """

        denied = self._require_owner_or_creator()
        if denied:
            return denied
        if not arguments:
            return 'Usage: /learn_text <text>'
        text = ' '.join(arguments)
        self.assistant.learning_manager.learn_from_text(text, source='manual')
        return f'Added text for training ({len(text)} chars). / Текст добавлен.'


    def _cmd_save_model(self, _arguments: list[str]) -> str:
        """
        /save_model — сохранить LSTM-модель на диск.
        """

        denied = self._require_owner_or_creator()
        if denied:
            return denied
        self.assistant.learning_manager.save_model('models/protos_lstm.json')
        return 'Model saved. / Модель сохранена.'


    def _cmd_load_model(self, _arguments: list[str]) -> str:
        """
        /load_model — загрузить LSTM-модель с диска.
        """

        denied = self._require_owner_or_creator()
        if denied:
            return denied
        self.assistant.load_model()
        return 'Model loaded. / Модель загружена.'


    def _cmd_set_status(self, arguments: list[str]) -> str:
        """
        /set_status <name> <status> — назначить статус пользователю.
        """
        denied = self._require_owner_or_creator()
        if denied:
            return denied
        assistant = self.assistant
        if len(arguments) < 2:
            return 'Usage: /set_status <name> <status>'
        name, status = arguments[0], arguments[1]
        if status not in assistant.USER_STATUSES:
            available = ', '.join(assistant.USER_STATUSES.keys())
            return f'Unknown status. Available: {available}'
        if status != 'creator' and assistant.is_creator_name(name):
            return (
                'Cannot change creator account this way. / '
                'Учётную запись создателя так менять нельзя.'
            )
        if status == 'creator' and assistant.user_role != 'creator':
            return (
                'Only creator can assign creator status. / '
                'Назначить создателя может только создатель.'
            )
        assistant.user_statuses[name] = status
        if name in assistant.known_users:
            assistant.known_users[name]['status'] = status
        assistant.save_user_statuses()
        assistant_logger.info(f'Установлен статус {status} для {name}')
        return f'Status "{status}" set for {name}. / Статус установлен.'


    def _cmd_users(self, _arguments: list[str]) -> str:
        """
        /users — список известных пользователей (только creator).
        """

        denied = self._require_creator()
        if denied:
            return denied

        assistant = self.assistant
        if not assistant.known_users:
            return 'Пока нет известных пользователей.'

        filter_mode = (_arguments[0].lower() if _arguments else 'all')
        if filter_mode in ('owner', 'owners', 'владельцы', 'владелец'):
            filter_mode = 'owners'
        elif filter_mode in ('user', 'users', 'пользователи', 'пользователь'):
            filter_mode = 'users'
        else:
            filter_mode = 'all'

        lines = []
        for name, record in sorted(assistant.known_users.items(), key=lambda x: x[0].lower()):
            if not isinstance(record, dict):
                if filter_mode == 'all':
                    lines.append(f'  - {name}')
                continue
            status = record.get('status') or assistant.user_statuses.get(name, 'user')
            if filter_mode == 'owners' and status != 'owner':
                continue
            if filter_mode == 'users' and status != 'user':
                continue
            uid = record.get('id') or '—'
            session_state = 'in' if record.get('logged_in') else 'out'
            aliases = record.get('aliases') or []
            alias_s = f', aliases: {", ".join(aliases)}' if aliases else ''
            lines.append(
                f'  - {name} [id={uid}, {status}, session={session_state}{alias_s}]'
            )

        if not lines:
            return f'Нет записей для фильтра «{filter_mode}».'

        return (
                f'Известные пользователи ({filter_mode}):\n'
                + '\n'.join(lines)
                + '\nУдаление: /delete_user <имя>'
                + '\nВладелец: /register_owner <имя>'
        )


    def _cmd_register_owner(self, arguments: list[str]) -> str:
        """
        /register_owner <имя> — начать задание пароля владельца (только creator).
        """

        denied = self._require_creator()
        if denied:
            return denied
        if not arguments:
            return 'Использование: /register_owner <имя>'

        name = ' '.join(arguments).strip()
        name = name[0].upper() + name[1:] if name else name
        assistant = self.assistant

        if assistant.is_creator_name(name) or name == assistant.creator_name:
            return 'Это имя создателя, владельцем его сделать нельзя.'

        for k, rec in assistant.known_users.items():
            if k.lower() == name.lower():
                st = (rec or {}).get('status', 'user')
                if st == 'owner':
                    return f'«{k}» уже владелец.'
                return (
                    f'«{k}» уже есть со статусом {st}. '
                    f'Сначала /delete_user {k} или другое имя.'
                )

        assistant.pending_login = {
            'step': 'register_owner_password',
            'name': name,
            'status': 'owner',
        }
        return f'Придумайте пароль для владельца «{name}».'


    def _cmd_delete_user(self, arguments: list[str]) -> str:
        """
        /delete_user <имя> — удалить user или owner (не creator).
        """

        denied = self._require_creator()
        if denied:
            return denied
        if not arguments:
            return 'Использование: /delete_user <имя>'

        assistant = self.assistant
        raw = ' '.join(arguments).strip()
        key = None
        for name in assistant.known_users:
            if name.lower() == raw.lower():
                key = name
                break
        if key is None:
            return f'Пользователь «{raw}» не найден.'

        rec = assistant.known_users.get(key) or {}
        status = rec.get('status') or assistant.user_statuses.get(key, 'user')
        if status == 'creator' or assistant.is_creator_name(key):
            return 'Создателя удалить нельзя.'

        if rec.get('logged_in'):
            assistant.auth.session.set_logged_in(None)
            if assistant.current_speaker == key:
                assistant.session_authenticated = False
                assistant.current_speaker = None
                assistant.user_name = None
                assistant.user_role = 'user'

        del assistant.known_users[key]
        assistant.user_statuses.pop(key, None)
        if getattr(assistant, 'owner_name', None) == key:
            assistant.owner_name = None
        assistant.save_user_statuses()
        return f'Удалена учётная запись «{key}» ({status}).'


    def _cmd_relations(self, _arguments: list[str]) -> str:
        """
        /relations — связи текущего пользователя.
        """

        assistant = self.assistant
        speaker = assistant.current_speaker
        if not speaker or speaker not in assistant.relationships:
            return 'No relations for current user. / Нет связей или вы не вошли.'
        relations = assistant.relationships[speaker]
        if not relations:
            return f'No relations for {speaker}. / Связей нет.'
        parts = [f'{rel}: {person}' for rel, person in relations.items()]
        return f'Relations / Связи {speaker}: {", ".join(parts)}'


    def _cmd_knowledge(self, arguments: list[str]) -> str:
        """
        /knowledge <query> — поиск по базе знаний LearningManager.
        """

        if not arguments:
            return 'Usage: /knowledge <query>'
        query = ' '.join(arguments)
        manager = self.assistant.learning_manager
        results = manager.query_knowledge(query) if manager else []
        if not results:
            return 'Nothing found. / Ничего не найдено.'
        return 'Found / Найдено:\n' + '\n'.join(results[:3])


    def _cmd_learning_status(self, _arguments: list[str]) -> str:
        """
        /learning_status — сводка по данным обучения.
        """

        denied = self._require_owner_or_creator()
        if denied:
            return denied
        assistant = self.assistant
        if not assistant.learning_manager:
            return (
                'Learning manager not initialized. / '
                'Менеджер обучения не инициализирован.'
            )
        total = len(assistant.learning_manager.learning_data)
        history_count = len(assistant.learning_manager.training_history)
        mode = 'on' if assistant.is_learning else 'off'
        return (
            f'Learning status / Статус обучения:\n'
            f'  Examples / Примеров: {total}\n'
            f'  History / Историй: {history_count}\n'
            f'  Mode / Режим: {mode}'
        )


    def _cmd_train_on_books(self, _arguments: list[str]) -> str:
        """
        /train_on_books — обучение на текстах из data/books.
        """

        denied = self._require_owner_or_creator()
        if denied:
            return denied
        from scripts.train_on_books import train_protos_on_books

        debug_logger.debug('Запуск обучения на книгах')
        training_logger.info('Запуск обучения на книгах')
        train_protos_on_books(self.assistant)
        return 'Book training started. / Обучение на книгах запущено.'


    def _cmd_continue_training(self, _arguments: list[str]) -> str:
        """
        /continue_training — продолжение обучения на книгах с сохранённой модели.
        """

        denied = self._require_owner_or_creator()
        if denied:
            return denied
        from scripts.train_on_books import continue_training_from_saved

        continue_training_from_saved(self.assistant)
        return 'Continue training started. / Продолжение обучения запущено.'


    def _cmd_check_books(self, _arguments: list[str]) -> str:
        """
        /check_books — список .txt в data/books.
        """

        denied = self._require_owner_or_creator()
        if denied:
            return denied
        books_dir = Path('data/books')
        if not books_dir.exists():
            return (
                'Folder data/books/ missing. / '
                'Папка data/books/ не существует.'
            )
        books = list(books_dir.glob('*.txt'))
        if not books:
            return 'No .txt books in data/books/. / Книг нет.'
        listing = '\n'.join(f'  - {book.name}' for book in books)
        return f'Books ({len(books)}) / Книги:\n{listing}'


    def _cmd_download_books(self, _arguments: list[str]) -> str:
        """
        /download_books — скачать образцы книг для обучения.
        """

        denied = self._require_owner_or_creator()
        if denied:
            return denied
        try:
            from scripts.train_on_books import BookDownloader

            downloader = BookDownloader()
            russian = downloader.download_russian_books()
            english = downloader.download_english_books()
            return (
                f'Downloaded: ru={len(russian)}, en={len(english)}. / '
                f'Скачано: русских={len(russian)}, английских={len(english)}.'
            )
        except ImportError as error:
            error_logger.error(f'Ошибка импорта: {error}')
            return f'Import error: {error}'
        except Exception as error:
            error_logger.exception(f'Ошибка скачивания: {error}')
            return f'Error: {error}'


    def _cmd_auth(self, arguments: list[str]) -> str:
        """
        /auth [phrase] — проверка секретной фразы создателя (без проверки роли).
        """

        assistant = self.assistant
        phrase = ' '.join(arguments).strip()
        if not phrase:
            return assistant.auth.request_auth('ru')
        if not assistant.auth.check_secret_phrase(phrase):
            assistant.pending_auth = False
            assistant.session_authenticated = False
            assistant.current_speaker = None
            assistant.user_name = None
            assistant.user_role = 'user'
            assistant.pending_login = {'step': 'ask_name'}
            return (
                'Wrong phrase. Access denied. / '
                'Фраза неверная. Доступ запрещён. Запущена повторная аутентификация. Назовите Ваше имя.'
            )
        name = (
                assistant.current_speaker
                or assistant.user_name
                or assistant.creator_name
                or 'Создатель'
        )
        if assistant.is_creator_name(name) or name == assistant.creator_name:
            name = assistant.canonical_creator_name()
        assistant.apply_status(name, 'creator')
        assistant.auth.open_session(name, 'creator')
        assistant_logger.info(f'Сессия создателя открыта: {name}')
        return f'Phrase OK. Hello, {name}! / Фраза верная. Доступ разрешён.'


    def _cmd_set_secret(self, arguments: list[str]) -> str:
        """
        /set_secret <phrase> — смена секретной фразы (только creator).
        """

        denied = self._require_creator()
        if denied:
            return denied
        if not arguments:
            return 'Usage: /set_secret <new phrase>'
        new_secret = ' '.join(arguments).lower().strip()
        self.assistant.creator_secret = new_secret
        self.assistant.save_creator_secret()
        return 'Secret updated. / Секретная фраза обновлена.'


    def _cmd_gen_only(self, _arguments: list[str]) -> str:
        """
        /gen_only — переключение режима «только LSTM» (только creator).
        """

        denied = self._require_creator()
        if denied:
            return denied
        assistant = self.assistant
        assistant.gen_only = not getattr(assistant, 'gen_only', False)
        state = 'ON / включён' if assistant.gen_only else 'OFF / выключен'
        return f'gen_only: {state}'


    def _cmd_logout(self, _arguments: list[str]) -> str:
        """
        Сброс сессии без остановки процесса.
        """

        if not self.assistant.is_logged_in():
            return 'Сейчас никто не в системе.'
        return self.assistant.auth.logout('ru')


    def _cmd_exit(self, _arguments: list[str]) -> str:
        """
        Завершение работы Протоса (main смотрит на shutdown).
        """

        name = ''
        if hasattr(self.assistant, 'address_name'):
            name = self.assistant.address_name()
        if name:
            return f'__SHUTDOWN__До встречи, {name}!'
        return '__SHUTDOWN__До встречи!'
