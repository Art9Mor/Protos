from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.utils.logger import debug_logger, error_logger, memory_logger

if TYPE_CHECKING:
    from app.interface.assistant import Assistant


class ResponseRouter:
    """
    Маршрутизация обычного диалога к ответу.
    Жёсткие правила, факты, Q&A, знания, LSTM и fallback.
    """

    FAREWELL = (
        'пока', 'до свидания', 'мне пора', 'увидимся', 'до встречи',
        'goodbye', 'bye',
    )
    GREETING = (
        'привет', 'приветствую', 'здравствуй', 'здарова', 'добрый',
        'hello', 'hi', 'пиривет', 'приветик',
    )
    ASK_NAME = (
        'как тебя зовут', 'как тебя звать', 'твоё имя', 'твое имя',
        'what is your name', 'your name',
    )
    ASK_WHO = (
        'кто ты', 'ты кто', 'представься', 'who are you',
    )
    ASK_CREATOR = (
        'кто создал', 'кто создатель', 'кто твой создатель',
    )
    ASK_DIRECTIVE = (
        'директива', 'твоя цель', 'твоя задача', 'зачем ты',
        'твоё назначение', 'твое назначение',
    )
    ASK_COMMANDS = (
        'какие команды', 'список команд', 'что ты умеешь', 'твои команды',
    )

    def __init__(self, assistant: Assistant) -> None:
        self.assistant = assistant


    def answer(self, text: str, layer: str) -> str:
        """
        Пометка слоя ответа (для обучения или отладки).
        """

        debug_logger.debug(f'Слой ответа: {layer}')
        self.assistant._last_answer_layer = layer
        return text


    @staticmethod
    def lstm_reply_ok(text: str) -> bool:
        """
        Отсечение заведомо мусорной генерации LSTM.
        """

        t = (text or '').strip()
        if len(t) < 8:
            return False
        low = t.lower()
        if low.startswith('(lstm') or 'ничего внятного' in low:
            return False
        letters = sum(ch.isalpha() for ch in t)
        if letters < 6:
            return False
        if t.count('?') >= 3 and letters < 20:
            return False
        return True


    def greeting(self, lang: str='ru') -> str:
        """
        Приветствие для открытой сессии или запрос имени.
        """

        a = self.assistant
        debug_logger.debug(f'Язык генерации приветствия: {lang}')

        if a.current_speaker:
            status = a.user_statuses.get(a.current_speaker, 'пользователь')
            name = a.address_name()
            is_creator = status == 'creator' or a.user_role == 'creator'
            if lang == 'ru':
                if is_creator:
                    return f'Снова привет, {name}! Рад тебя видеть, создатель.'
                return f'Привет, {name}!'
            if is_creator:
                return f'Hello again, {name}! Good to see you, creator.'
            return f'Hello, {name}!'

        if lang == 'ru':
            return 'Привет! Я Протос. Как тебя зовут?'
        return "Hello! I'm Protos. What's your name?"


    @staticmethod
    def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
        return any(p in text for p in phrases)


    def route(self, user_input: str, context: list[str], lang: str) -> str:
        """
        Главный маршрут естественного ответа.
        """

        a = self.assistant

        debug_logger.debug(f'Генерация естественного ответа на: {user_input[:30]}...')
        user_lower = user_input.lower().strip()

        # --- прощание (один раз, в начале) ---
        if self._contains_any(user_lower, self.FAREWELL):
            debug_logger.debug('Прощание')
            name = a.address_name()
            msg = f'__SHUTDOWN__До встречи, {name}!' if name else '__SHUTDOWN__До встречи!'
            return self.answer(msg, 'farewell')

        # --- только LSTM ---
        if getattr(a, 'gen_only', False):
            debug_logger.debug('Режим gen_only: только LSTM')
            try:
                generated = a.generate_with_lstm(user_input, context, lang)
                if generated and self.lstm_reply_ok(generated):
                    return self.answer(generated, 'lstm_forced')
            except Exception as e:
                error_logger.exception(f'gen_only: {e}')
            return self.answer('(LSTM ничего внятного не сгенерировал)', 'lstm_empty')

        context_data = {
            'creator_name': a.creator_name or 'создатель',
            'user_name': a.current_speaker or 'ты',
            'user_status': (
                a.user_statuses.get(a.current_speaker, 'пользователь')
                if a.current_speaker else 'пользователь'
            ),
            'conversation_count': a.conversation_count,
            'lang': lang,
        }

        # --- 1. Жёсткие правила личности / идентичности ---
        is_who_is = re.search(r'кто\s+так(ой|ая|ое|ие)\s+\w+', user_lower) is not None

        if not is_who_is:
            ask_name = self._contains_any(user_lower, self.ASK_NAME)
            ask_who = self._contains_any(user_lower, self.ASK_WHO)

            if ask_name and ask_who:
                debug_logger.debug('Вопрос "кто ты" + "как зовут"')
                if lang == 'ru':
                    return self.answer(
                        'Я Протос — искусственный интеллект. '
                        'Меня зовут Протос. Я учусь понимать язык и помогать людям.',
                        'persona',
                    )
                return self.answer(
                    "I'm Protos — an artificial intelligence. My name is Protos.",
                    'persona',
                )

            if ask_name:
                debug_logger.debug('Вопрос "как тебя зовут"')
                if lang == 'ru':
                    return self.answer('Меня зовут Протос.', 'persona')
                return self.answer('My name is Protos.', 'persona')

            if ask_who:
                debug_logger.debug('Вопрос о Протосе')
                if lang == 'ru':
                    return self.answer(
                        'Я Протос — искусственный интеллект. '
                        'Я учусь понимать язык и помогать людям.',
                        'persona',
                    )
                return self.answer(
                    "I'm Protos — an artificial intelligence. "
                    "I'm learning language and how to help people.",
                    'persona',
                )

            if re.search(
                    r'(^|\s)(кто\s+я|как\s+меня\s+зовут|ты\s+знаешь\s+кто\s+я)(\s|$|[?!.])',
                    user_lower,
            ):
                debug_logger.debug('Вопрос "кто я"')
                if a.current_speaker:
                    status = a.user_statuses.get(a.current_speaker, 'пользователь')
                    main = a.current_speaker
                    call_as = a.address_name()
                    if status == 'creator' or a.is_creator_name(a.current_speaker):
                        if call_as and call_as.lower() != main.lower():
                            return self.answer(
                                f'Ты — {main}, мой создатель. '
                                f'Сейчас я обращаюсь к тебе как {call_as}.',
                                'identity',
                            )
                        return self.answer(f'Ты — {main}, мой создатель.', 'identity')
                    if call_as and call_as.lower() != main.lower():
                        return self.answer(
                            f'Ты — {main}. Сейчас я зову тебя {call_as}.',
                            'identity',
                        )
                    return self.answer(f'Ты — {main}.', 'identity')
                return self.answer(
                    'Ты пока не представился. Напиши, как тебя зовут.',
                    'identity',
                )

        if self._contains_any(user_lower, self.ASK_CREATOR):
            debug_logger.debug('Вопрос о создателе')
            if a.creator_name:
                return self.answer(f'Мой создатель — {a.creator_name}.', 'identity')
            return self.answer(
                'Мой создатель ещё не представился мне.',
                'identity',
            )

        if self._contains_any(user_lower, self.ASK_DIRECTIVE):
            debug_logger.debug('Вопрос о цели/директиве')
            if lang == 'ru':
                return self.answer(
                    'Моя главная задача — учиться понимать язык, '
                    'помнить важный контекст и помогать тебе.',
                    'persona',
                )
            return self.answer(
                'My main goal is to learn language, keep useful context, and help you.',
                'persona',
            )

        if self._contains_any(user_lower, self.ASK_COMMANDS):
            debug_logger.debug('Запрос списка команд')
            commands = getattr(self.assistant, 'commands', None)
            if not commands:
                return self.answer(
                    'Commands module not ready. / Модуль команд не готов.',
                    'error',
                )
            return self.answer(commands.help_text(), 'help')

        if self._contains_any(user_lower, self.GREETING):
            debug_logger.debug('Обнаружено приветствие')
            return self.answer(self.greeting(lang), 'greeting')

        # --- 2. Факты ---
        fact = a.facts_store.find(user_input)
        if fact:
            debug_logger.debug(f'Найден факт: {fact[:50]}...')
            return self.answer(fact, 'facts')

        # --- 3. Q&A ---
        qa_answer = a.qa_matcher.find_answer(user_input)
        if qa_answer:
            debug_logger.debug('Ответ найден в Q&A')
            return self.answer(qa_answer, 'qa')

        # --- 4. Обученные диалоги ---
        learned = a.dialogue_learner.get_response(user_input, context_data)
        if learned:
            debug_logger.debug('Найден ответ в обученных диалогах')
            return self.answer(learned, 'dialogue_learner')

        # --- 5. База знаний (только явные вопросы) ---
        if '?' in user_input:
            knowledge = (
                a.learning_manager.query_knowledge(user_input)
                if a.learning_manager else []
            )
            if knowledge:
                debug_logger.debug(f'Найдено знание: {knowledge[0][:50]}...')
                return self.answer(knowledge[0], 'learning_knowledge')

            knowledge_answer = a.query_knowledge_base(user_input)
            if knowledge_answer:
                debug_logger.debug(f'Найдено в базе знаний: {knowledge_answer[:50]}...')
                return self.answer(knowledge_answer, 'knowledge_base')

        # --- 6. Другой известный пользователь ---
        for user in a.known_users:
            if user.lower() in user_lower and user != a.current_speaker:
                debug_logger.debug(f'Вопрос о пользователе: {user}')
                status = a.known_users[user].get('status', 'user')
                relations = a.get_relationship_info(user)
                if relations:
                    return self.answer(
                        f'{user} — {status}. Связи: {relations}',
                        'rules',
                    )
                return self.answer(f'{user} — {status}', 'rules')

        # --- 7. Статус текущего ---
        if 'какой у меня статус' in user_lower or 'мой статус' in user_lower:
            debug_logger.debug('Вопрос о статусе')
            if a.current_speaker and a.current_speaker in a.known_users:
                status = a.known_users[a.current_speaker].get('status', 'user')
                return self.answer(f'Твой статус: {status}', 'rules')
            return self.answer(
                'У тебя пока нет статуса. Расскажи, кто ты?',
                'rules',
            )

        # --- 8. Связи («кто мой …») ---
        if 'кто' in user_lower and any(
                w in user_lower for w in ('мой', 'моя', 'моё', 'мое')
        ):
            for rel_type in a.RELATIONSHIP_TYPES:
                if rel_type in user_lower:
                    debug_logger.debug(f'Вопрос о связи: {rel_type}')
                    if a.current_speaker and a.current_speaker in a.relationships:
                        rels = a.relationships[a.current_speaker]
                        if rel_type in rels:
                            return self.answer(
                                f'Твой {rel_type} — {rels[rel_type]}.',
                                'rules',
                            )
                        return self.answer(
                            f'Я не знаю, кто твой {rel_type}. Расскажи мне.',
                            'rules',
                        )
                    return self.answer(
                        'Я не знаю твоих связей. Расскажи мне о своей семье и друзьях.',
                        'rules',
                    )

        # --- 9. Системные короткие команды текстом ---
        if user_input.startswith('ls') or user_input.startswith('cd'):
            debug_logger.debug('Системная команда')
            commands = getattr(a, 'commands', None)
            if not commands:
                return self.answer(
                    'Commands module not ready. / Модуль команд не готов.',
                    'error',
                )
            return self.answer(commands.system_command(user_input), 'system')

        if user_lower in ('info', 'stats'):
            debug_logger.debug('Команда info/stats')
            commands = getattr(a, 'commands', None)
            if not commands:
                return self.answer(
                    'Commands module not ready. / Модуль команд не готов.',
                    'error',
                )
            return self.answer(commands.info(), 'system')

        if user_lower == 'clear':
            if a.user_role in ('owner', 'creator'):
                a.memory.clear()
                debug_logger.debug('Память очищена')
                memory_logger.info('Память очищена')
                return self.answer('🧹 Память очищена', 'system')
            return self.answer('У вас нет прав на очистку памяти', 'system')

        # --- 10. Редкий teach по незнакомому слову ---
        if a.pending_teach is None and a.conversation_count > 2:
            unk = a.find_unknown_word(user_input)
            if unk and a.conversation_count % 5 == 0:
                debug_logger.debug(f'Незнакомое слово: {unk}')
                return self.answer(
                    a.make_teach_question('unknown_word', word=unk),
                    'teach',
                )

        # --- 11. LSTM ---
        try:
            debug_logger.debug('Генерация через LSTM')
            generated = a.generate_with_lstm(user_input, context, lang)
            if generated and self.lstm_reply_ok(generated):
                debug_logger.debug(f'LSTM сгенерировал: {generated[:30]}...')
                return self.answer(generated, 'lstm')
        except Exception as e:
            error_logger.exception(f'Ошибка генерации: {e}')
            debug_logger.debug(f'Ошибка генерации: {e}')

        # --- 12. Fallback ---
        debug_logger.debug('Использован fallback ответ')
        return self.answer(a.dialogue_learner.get_fallback(lang), 'fallback')