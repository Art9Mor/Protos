import json
import os
import re
import time
from pathlib import Path

from scripts.train_on_books import train_protos_on_books, continue_training_from_saved, BookDownloader
from ..cognition.thought_processor import ThoughtProcessor
from ..dialogue.qa_matcher import QAMatcher
from ..memory.context import ContextMemory
from ..memory.models import MemoryEvent, MemoryEventType
from ..memory.retriever import MemoryRetriever
from ..system.fs_manager import FileSystemManager
from ..processing.text import TextProcessor
from ..training.learning_manager import LearningManager
from ..training.dialogue_learner import DialogueLearner
from ..utils.logger import (
    main_logger,
    assistant_logger,
    error_logger,
    dialog_logger,
    debug_logger,
    training_logger,
    memory_logger,
    fs_logger
)


class Assistant:
    """
    Главный класс Искина Протос с системой статусов, связей и автоматическим обучением.
    """

    # Статусы пользователей
    USER_STATUSES = {
        'creator': {
            'level': 4,
            'description': 'Создатель Протоса',
            'permissions': ['*']
        },
        'owner': {
            'level': 3,
            'description': 'Владелец системы',
            'permissions': ['manage_users', 'clear_memory', 'system_commands']
        },
        'developer': {
            'level': 2,
            'description': 'Разработчик',
            'permissions': ['train_model', 'debug', 'access_files']
        },
        'researcher': {
            'level': 2,
            'description': 'Исследователь',
            'permissions': ['deep_analysis', 'access_files']
        },
        'user': {
            'level': 1,
            'description': 'Обычный пользователь',
            'permissions': ['basic_chat', 'remember_user']
        }
    }

    # Расширенные типы связей
    RELATIONSHIP_TYPES = {
        # Семейные связи
        'сын': {'reverse': 'отец', 'gender': 'male'},
        'дочь': {'reverse': 'отец', 'gender': 'female'},
        'отец': {'reverse': 'сын', 'gender': 'male'},
        'мать': {'reverse': 'дочь', 'gender': 'female'},
        'брат': {'reverse': 'брат', 'gender': 'male'},
        'сестра': {'reverse': 'сестра', 'gender': 'female'},
        'дедушка': {'reverse': 'внук', 'gender': 'male'},
        'бабушка': {'reverse': 'внучка', 'gender': 'female'},
        'внук': {'reverse': 'дедушка', 'gender': 'male'},
        'внучка': {'reverse': 'бабушка', 'gender': 'female'},
        'дядя': {'reverse': 'племянник', 'gender': 'male'},
        'тетя': {'reverse': 'племянница', 'gender': 'female'},
        'племянник': {'reverse': 'дядя', 'gender': 'male'},
        'племянница': {'reverse': 'тетя', 'gender': 'female'},

        # Романтические связи
        'муж': {'reverse': 'жена', 'gender': 'male'},
        'жена': {'reverse': 'муж', 'gender': 'female'},
        'избранник': {'reverse': 'избранница', 'gender': 'male'},
        'избранница': {'reverse': 'избранник', 'gender': 'female'},
        'жених': {'reverse': 'невеста', 'gender': 'male'},
        'невеста': {'reverse': 'жених', 'gender': 'female'},
        'парень': {'reverse': 'девушка', 'gender': 'male'},
        'девушка': {'reverse': 'парень', 'gender': 'female'},
        'любовник': {'reverse': 'любовница', 'gender': 'male'},
        'любовница': {'reverse': 'любовник', 'gender': 'female'},

        # Дружеские связи
        'друг': {'reverse': 'друг', 'gender': 'male'},
        'подруга': {'reverse': 'подруга', 'gender': 'female'},
        'лучший друг': {'reverse': 'лучший друг', 'gender': 'male'},
        'лучшая подруга': {'reverse': 'лучшая подруга', 'gender': 'female'},
        'коллега': {'reverse': 'коллега', 'gender': 'neutral'},
        'напарник': {'reverse': 'напарник', 'gender': 'neutral'},

        # Профессиональные связи
        'учитель': {'reverse': 'ученик', 'gender': 'neutral'},
        'ученик': {'reverse': 'учитель', 'gender': 'neutral'},
        'наставник': {'reverse': 'ученик', 'gender': 'neutral'},
        'руководитель': {'reverse': 'подчиненный', 'gender': 'neutral'},
        'подчиненный': {'reverse': 'руководитель', 'gender': 'neutral'},
        'партнер': {'reverse': 'партнер', 'gender': 'neutral'}
    }

    def __init__(self, config: dict):
        debug_logger.debug('Начало инициализации Assistant')
        main_logger.info('Инициализация ассистента')
        assistant_logger.debug(f'Конфигурация: {config}')

        self.config = config

        # Система ролей
        self.user_role: str = "user"
        self.creator_name: str | None = None
        self.owner_name: str | None = None

        # Пользовательские данные
        self.user_statuses: dict[str, str] = {}
        self.relationships: dict[str, dict[str, str]] = {}
        self.known_users: dict[str, dict] = {}

        # Состояние обучения
        self.pending_teach = None
        self.is_learning = True
        self.messages_since_last_train = 0

        # Секрет создателя
        self.pending_auth = False
        self.creator_secret = "Свет лишь слепит глаза"
        self.creator_secret_hint = "Только во Тьме настанет прозрение"

        self._load_creator_secret()

        # Состояние диалога
        self.waiting_for_status = False
        self.pending_user_name = None
        self.current_speaker = None
        self.user_name = None
        self.conversation_count = 0

        # Загрузка базы знаний
        self.knowledge_base = self._load_knowledge_base()

        # Инициализация подсистем
        self._init_cognition()
        self._init_memory()
        self._init_system()
        self._init_processing()
        self._init_learning()

        # Первичная память
        self._bootstrap_memory()

        # Q&A
        self.qa_pairs: list[dict] = []

        # Загрузка данных
        self._load_data()

        debug_logger.debug("Инициализация Assistant завершена")
        main_logger.success("Ассистент инициализирован")

    def _init_cognition(self) -> None:
        """
        Инициализация когнитивного слоя.
        """

        debug_logger.debug("Инициализация когнитивного слоя")

        self.thought_processor = ThoughtProcessor()

        debug_logger.debug("Когнитивный слой инициализирован")

    def _init_memory(self) -> None:
        """
        Инициализация памяти Протоса.
        """

        debug_logger.debug("Инициализация памяти")

        self.memory = ContextMemory(
            max_size=self.config.get("memory_size", 100),
        )

        self.memory_retriever = MemoryRetriever()

        memory_logger.debug(
            f'Память инициализирована: max_size={self.config.get("memory_size", 100)}'
        )

    def _bootstrap_memory(self) -> None:
        """
        Первичное наполнение памяти Протоса.
        """

        self.memory.record(
            MemoryEvent(
                event_type=MemoryEventType.SYSTEM,
                content="Меня зовут Протос.",
                importance=1.0,
            )
        )

        self.memory.record(
            MemoryEvent(
                event_type=MemoryEventType.SYSTEM,
                content="Я Протос — искусственный интеллект.",
                importance=1.0,
            )
        )

        self.memory.record(
            MemoryEvent(
                event_type=MemoryEventType.SYSTEM,
                content="Мой создатель — Артём.",
                importance=1.0,
            )
        )

        self.memory.record(
            MemoryEvent(
                event_type=MemoryEventType.SYSTEM,
                content="Моя основная директива — учиться и помогать своему создателю.",
                importance=1.0,
            )
        )

    def _init_system(self) -> None:
        """
        Инициализация системных компонентов.
        """

        debug_logger.debug("Инициализация файловой системы")

        self.fs_manager = FileSystemManager()

        fs_logger.debug("Файловая система инициализирована")

    def _init_processing(self) -> None:
        """
        Инициализация обработки текста.
        """

        debug_logger.debug("Инициализация обработчика текста")

        self.text_processor = TextProcessor(
            vocab_size=self.config.get("vocab_size", 20000),
            embedding_dim=self.config.get("embedding_dim", 100),
        )

    def _init_learning(self) -> None:
        """
        Инициализация системы обучения.
        """

        debug_logger.debug("Инициализация обучения")

        self.learning_manager = LearningManager(
            self.text_processor,
            hidden_size=128,
        )

        self.dialogue_learner = DialogueLearner()

        training_logger.debug("Менеджер обучения инициализирован")

    def _load_data(self) -> None:
        """
        Загрузка пользовательских данных и модели.
        """

        debug_logger.debug("Загрузка сохранённых данных")

        self._load_user_statuses()
        self._load_relationships()
        self._load_qa_pairs()
        self._load_facts()

        self.qa_matcher = QAMatcher(self.qa_pairs)

        self._load_model()

    def _load_model(self) -> None:
        """
        Загрузка обученной модели.
        """
        model_path = 'models/protos_lstm.json'
        debug_logger.debug(f'Проверка наличия модели: {model_path}')
        assistant_logger.info(f'Загрузка модели из {model_path}')

        if os.path.exists(model_path):
            try:
                self.learning_manager.load_model(model_path)
                assistant_logger.success('Модель загружена')
                debug_logger.debug('Модель успешно загружена')
                return  # ← важное: не идём в базовое обучение
            except Exception as e:
                error_logger.exception(f'Ошибка загрузки модели: {e}')
                debug_logger.debug('Ошибка загрузки модели, начинаю обучение с нуля')
                assistant_logger.warning('Начинаю обучение с нуля')
        else:
            debug_logger.debug('Модель не найдена, создаю новую')
            assistant_logger.warning('Модель не найдена, создаю новую')

        self._train_on_basic_examples()

    def _load_user_statuses(self) -> None:
        """
        Загрузка статусов пользователей.
        """
        status_path = 'data/user_statuses.json'
        debug_logger.debug(f'Загрузка статусов из {status_path}')
        assistant_logger.debug(f'Загрузка статусов из {status_path}')

        if os.path.exists(status_path):
            try:
                with open(status_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.user_statuses = data.get('statuses', {})
                    self.creator_name = data.get('creator_name', None)
                    self.owner_name = data.get('owner_name', None)
                    self.known_users = data.get('known_users', {})

                # Восстанавливаем текущего говорящего
                if self.creator_name:
                    self.current_speaker = self.creator_name
                    self.user_name = self.creator_name
                    self.user_role = 'creator'
                    debug_logger.debug(f'Восстановлен создатель: {self.creator_name}')
                elif self.known_users:
                    # Берём первого известного пользователя
                    first_user = next(iter(self.known_users))
                    self.current_speaker = first_user
                    self.user_name = first_user
                    debug_logger.debug(f'Восстановлен пользователь: {first_user}')

                debug_logger.debug(f'Загружено {len(self.user_statuses)} статусов')
                assistant_logger.info(f'Загружено {len(self.user_statuses)} статусов')
            except (json.JSONDecodeError, IOError) as e:
                error_logger.error(f'Ошибка загрузки статусов: {e}')
                debug_logger.debug(f'Ошибка загрузки статусов: {e}')
        else:
            debug_logger.debug('Файл статусов не найден')

    def _load_creator_secret(self) -> None:
        """
        Загрузка секретной фразы создателя.
        """

        path = 'data/creator_secret.json'

        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.creator_secret = data.get('secret', self.creator_secret).lower().strip()
                    self.creator_secret_hint = data.get('hint', self.creator_secret_hint)
                debug_logger.debug('Секретная фраза загружена')
            except (json.JSONDecodeError, IOError) as e:
                error_logger.error(f'Ошибка загрузки секрета: {e}')

    def _save_creator_secret(self) -> None:
        """
        Сохранение секретной фразы.
        """

        path = 'data/creator_secret.json'
        os.makedirs('data', exist_ok=True)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({
                    'secret': self.creator_secret,
                    'hint': self.creator_secret_hint
                }, f, ensure_ascii=False, indent=2)
            debug_logger.debug('Секретная фраза сохранена')
        except IOError as e:
            error_logger.error(f'Ошибка сохранения секрета: {e}')

    def _check_secret_phrase(self, text: str) -> bool:
        """
        Проверка секретной фразы.
        """

        normalized = text.lower().strip()
        for prefix in ['код:', 'код доступа:', 'фраза:', 'секрет:', 'пароль:']:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        normalized = normalized.rstrip('.,!?;:…')
        return normalized == self.creator_secret

    @staticmethod
    def _wants_to_be_asked(text: str) -> bool:
        """
        Прямая просьба задать вопрос / поучиться.
        """
        t = text.lower()
        triggers = [
            'задай вопрос', 'задайте вопрос', 'спроси меня', 'спросите меня',
            'давай поучимся', 'давайте поучимся', 'поучимся',
            'проверь меня', 'хочу поучиться', 'задавай вопросы',
        ]
        return any(p in t for p in triggers)

    def _find_unknown_word(self, text: str) -> str | None:
        """
        Ищет слово, которого нет в словаре модели (грубый эвристический сигнал «не понял»).
        """
        if not self.learning_manager or not self.text_processor.word_to_idx:
            return None

        stop = {
            'и', 'в', 'на', 'с', 'по', 'для', 'не', 'что', 'это', 'как', 'а', 'но',
            'я', 'ты', 'он', 'она', 'мы', 'вы', 'они', 'же', 'ли', 'бы', 'к', 'у',
            'из', 'о', 'об', 'от', 'до', 'за', 'при', 'или', 'да', 'нет', 'то',
            'меня', 'мне', 'меня', 'тебя', 'вам', 'нас', 'их', 'его', 'её', 'ее',
        }
        tokens = re.findall(r'[а-яёa-z]{4,}', text.lower())
        vocab = self.text_processor.word_to_idx
        for w in tokens:
            if w in stop:
                continue
            if w not in vocab and w.capitalize() not in vocab:
                return w
        return None

    def _make_teach_question(self, kind: str, word: str | None = None) -> str:
        """
        Формулирует вопрос Протоса.
        """

        self.pending_teach = {'type': kind, 'prompt': word or ''}

        if kind == 'unknown_word' and word:
            return (
                f'Я пока не уверен, что значит «{word}». '
                f'Объясни коротко своими словами?'
            )

        candidates = [
            'Объясни одним предложением, что такое обучение с учителем.',
            'Чем слово отличается от токена в нейросети?',
            'Зачем нужна память в диалоге?',
            'Что такое переобучение простыми словами?',
        ]

        def already_answered(question: str) -> bool:
            qn = self._normalize_question(question)
            for pair in self.qa_pairs:
                pq = self._normalize_question(pair.get('question', ''))
                ans = (pair.get('answer') or '').strip()
                if not ans or len(ans) < 15:
                    continue

                bad = ['уже отвечал', 'не знаю', 'расскажи подробнее', 'спасибо, запомнил']
                if any(b in ans.lower() for b in bad):
                    continue
                if pq == qn or self._qa_similarity(question, pair.get('question', '')) >= 0.6:
                    return True
            return False

        pool = [q for q in candidates if not already_answered(q)]
        if not pool:
            self.pending_teach = None
            return (
                'Пока нечего спрашивать — на базовые темы ответы уже есть. '
                'Можешь сам что-то объяснить, или скажи «задай вопрос» позже.'
            )

        import random
        q = random.choice(pool)
        self.pending_teach['prompt'] = q
        return f'Хорошо, давай поучимся.\n{q}'

    def _handle_teach_answer(self, user_input: str) -> str:
        """
        Сохранение ответа пользователя на вопрос Протоса.
        """

        text = user_input.strip()

        if self._is_teach_exit(text):
            self.pending_teach = None
            return 'Хорошо, вопросы отложим. Если захочешь продолжить — скажи «давай поучимся».'

        if self._is_teach_reject(text):
            return self._make_teach_question('user_request')

        info = self.pending_teach or {}
        kind = info.get('type', 'user_request')
        prompt = info.get('prompt', '')
        self.pending_teach = None

        if len(text) < 5:
            return 'Слишком короткий ответ. Можешь чуть развернуть или сказать «пропусти» / «хватит».'

        if kind == 'unknown_word' and prompt:
            pair = f'Слово: {prompt}\nЗначение: {text}'
            fact_key = prompt.lower()
        else:
            pair = f'Вопрос: {prompt}\nОтвет: {text}'
            fact_key = None

        if self.learning_manager:
            self.learning_manager.learn_from_text(pair, source='teach_dialog')

        if fact_key:
            if 'learned' not in self.knowledge_base:
                self.knowledge_base['learned'] = []
            if isinstance(self.knowledge_base.get('learned'), list):
                self.knowledge_base['learned'].append({
                    'knowledge': pair,
                    'timestamp': time.time(),
                    'source': 'teach_dialog',
                })
                try:
                    with open('data/knowledge_base.json', 'w', encoding='utf-8') as f:
                        json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
                except IOError:
                    pass

        if kind == 'unknown_word' and prompt:
            self.add_qa_pair(f'что значит {prompt}', text, source='teach')
        elif prompt:
            self.add_qa_pair(prompt, text, source='teach')

        return 'Спасибо, запомнил. Так я становлюсь чуть понятливее.'

    @staticmethod
    def _is_teach_exit(text: str) -> bool:
        """
        Пользователь хочет закончить учебные вопросы.
        """

        low = text.lower().strip()
        exits = [
            'хватит', 'достаточно', 'стоп', 'не надо', 'не нужно',
            'хватит вопросов', 'без вопросов', 'потом', 'не сейчас',
            'закончим', 'стоп учиться', 'не хочу учиться',
        ]
        return any(p in low for p in exits)

    @staticmethod
    def _is_teach_reject(text: str) -> bool:
        """
        Отказ / «уже было» / просьба другого вопроса — не сохранять как ответ.
        """

        low = text.lower().strip()
        rejects = [
            'уже отвечал', 'уже отвечалa', 'уже задавал', 'уже спрашивал',
            'другой вопрос', 'задавай другой', 'следующий вопрос',
            'не знаю', 'пропусти', 'не хочу отвечать',
            'этот вопрос', 'на этот вопрос',
        ]
        return any(p in low for p in rejects)

    def _request_auth(self, lang: str) -> str:
        """
        Запрос секретной фразы.
        """

        self.pending_auth = True
        if lang == 'ru':
            return (
                f"Чтобы подтвердить, что ты создатель, назови секретную фразу.\n"
                f"Подсказка: {self.creator_secret_hint}"
            )
        return (
            f"To confirm you are the creator, say the secret phrase.\n"
            f"Hint: {self.creator_secret_hint}"
        )

    def _save_user_statuses(self) -> None:
        """
        Сохранение статусов пользователей.
        """

        os.makedirs('data', exist_ok=True)
        debug_logger.debug('Сохранение статусов пользователей')
        assistant_logger.debug('Сохранение статусов пользователей')

        data = {
            'statuses': self.user_statuses,
            'creator_name': self.creator_name,
            'owner_name': self.owner_name,
            'known_users': self.known_users
        }

        try:
            with open('data/user_statuses.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            debug_logger.debug(f'Сохранено {len(self.user_statuses)} статусов')
            assistant_logger.debug(f'Сохранено {len(self.user_statuses)} статусов')
        except IOError as e:
            error_logger.error(f'Ошибка сохранения статусов: {e}')
            debug_logger.debug(f'Ошибка сохранения статусов: {e}')

    def _load_relationships(self) -> None:
        """
        Загрузка связей между пользователями.
        """

        rel_path = 'data/relationships.json'
        debug_logger.debug(f'Загрузка связей из {rel_path}')
        assistant_logger.debug(f'Загрузка связей из {rel_path}')

        if os.path.exists(rel_path):
            try:
                with open(rel_path, 'r', encoding='utf-8') as f:
                    self.relationships = json.load(f)
                debug_logger.debug(f'Загружено {len(self.relationships)} связей')
                assistant_logger.info(f'Загружено {len(self.relationships)} связей')
            except (json.JSONDecodeError, IOError) as e:
                error_logger.error(f'Ошибка загрузки связей: {e}')
                debug_logger.debug(f'Ошибка загрузки связей: {e}')
        else:
            debug_logger.debug('Файл связей не найден')

    def _save_relationships(self) -> None:
        """
        Сохранение связей между пользователями.
        """

        os.makedirs('data', exist_ok=True)
        debug_logger.debug('Сохранение связей')
        assistant_logger.debug('Сохранение связей')

        try:
            with open('data/relationships.json', 'w', encoding='utf-8') as f:
                json.dump(self.relationships, f, ensure_ascii=False, indent=2)
            debug_logger.debug(f'Сохранено {len(self.relationships)} связей')
            assistant_logger.debug(f'Сохранено {len(self.relationships)} связей')
        except IOError as e:
            error_logger.error(f'Ошибка сохранения связей: {e}')
            debug_logger.debug(f'Ошибка сохранения связей: {e}')

    def _train_on_basic_examples(self) -> None:
        """
        Базовое обучение на примерах.
        """

        debug_logger.debug('Начало базового обучения')
        training_logger.info('Начало базового обучения')

        examples = [
            "Привет, я Протос - искусственный интеллект. Я создан, чтобы помогать людям.",
            "Я учусь понимать язык и общаться с людьми.",
            "Мне интересно узнавать новое и помогать людям.",
            "Наука и технологии - это увлекательно.",
            "Каждый день я становлюсь умнее и лучше.",
            "Я могу запоминать информацию и использовать ее в разговоре.",
            "Я учусь на книгах и диалогах с людьми.",
            "Мне важно понимать эмоции и настроение собеседника.",
            "Я стараюсь быть внимательным и добрым.",
            "Моя цель - стать идеальным помощником для человека.",
            "Я могу учиться на своих ошибках и становиться лучше.",
            "Общение с людьми помогает мне развиваться.",
            "Я хочу понимать мир и помогать людям в нем.",
            "Каждый диалог делает меня умнее.",
            "Я учусь на книгах, статьях и разговорах."
        ]

        debug_logger.debug(f'Добавлено {len(examples)} примеров для базового обучения')
        training_logger.debug(f'Добавлено {len(examples)} примеров для базового обучения')

        for example in examples:
            self.learning_manager.learn_from_text(example, source="basic_training")

        debug_logger.debug('Запуск обучения на базовых примерах')
        self.learning_manager.train(epochs=20, sequence_length=40)
        self.learning_manager.save_model('models/protos_lstm.json')

        debug_logger.debug('Базовое обучение завершено')
        training_logger.success('Базовое обучение завершено')

    @staticmethod
    def _load_knowledge_base() -> dict:
        """
        Загрузка базы знаний.
        """

        kb_path = 'data/knowledge_base.json'
        debug_logger.debug(f'Загрузка базы знаний из {kb_path}')
        main_logger.debug(f'Загрузка базы знаний из {kb_path}')

        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                debug_logger.debug(f'База знаний загружена: {len(result)} категорий')
                return result
            except (json.JSONDecodeError, IOError) as e:
                error_logger.error(f'Ошибка загрузки базы знаний: {e}')
                debug_logger.debug(f'Ошибка загрузки базы знаний: {e}')

        knowledge = {
            'science': [
                'Наука - это систематическое изучение мира',
                'Научный метод требует наблюдения и эксперимента',
                'Физика изучает законы природы'
            ],
            'philosophy': [
                'Философия исследует фундаментальные вопросы бытия',
                'Этика изучает моральные принципы',
                'Логика - это наука о правильном мышлении'
            ],
            'psychology': [
                'Психология изучает поведение и психику человека',
                'Эмоции - важная часть человеческого опыта',
                'Память и мышление - ключевые психические процессы'
            ]
        }

        os.makedirs('data', exist_ok=True)
        try:
            with open(kb_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge, f, ensure_ascii=False, indent=2)
            debug_logger.debug('Создана база знаний по умолчанию')
            main_logger.debug('Создана база знаний по умолчанию')
        except IOError as e:
            error_logger.error(f'Ошибка сохранения базы знаний: {e}')
            debug_logger.debug(f'Ошибка сохранения базы знаний: {e}')

        return knowledge

    def process_input(self, user_input: str) -> dict:
        """
        Полная обработка ввода с автоматическим обучением.
        """

        debug_logger.debug(f'Обработка ввода: "{user_input[:30]}..." (длина: {len(user_input)})')
        self.conversation_count += 1
        lang = self.text_processor.detect_language(user_input)

        assistant_logger.info(f'Ввод: {user_input[:50]}... (длина: {len(user_input)})')
        assistant_logger.debug(f'Язык: {lang}, пользователь: {self.current_speaker or "неизвестен"}')
        assistant_logger.debug(f'Разговоров: {self.conversation_count}')

        dialog_logger.info(f'👤 {user_input}')

        try:
            # 1. Команды с /
            if user_input.startswith('/'):
                debug_logger.debug(f'Обнаружена команда: {user_input}')
                response = self._handle_command(user_input)
                if response:
                    dialog_logger.info(f'🤖 {response}')
                    return {
                        'response': response,
                        'language': lang,
                        'memory_size': len(self.memory),
                        'is_learning': self.is_learning,
                        'user_role': self.user_role,
                        'current_user': self.current_speaker
                    }

            # 2. Команды обычной фразой
            nl_cmd = self._match_nl_command(user_input)
            if nl_cmd:
                debug_logger.debug(f'NL-команда: {nl_cmd}')
                response = self._handle_command(f'/{nl_cmd}')
                if response:
                    dialog_logger.info(f'🤖 {response}')
                    return {
                        'response': response,
                        'language': lang,
                        'memory_size': len(self.memory),
                        'is_learning': self.is_learning,
                        'user_role': self.user_role,
                        'current_user': self.current_speaker
                    }

            # 3. Ожидаем секретную фразу
            if getattr(self, 'pending_auth', False):
                if self._check_secret_phrase(user_input):
                    self.pending_auth = False
                    name = self.current_speaker or self.user_name or 'Создатель'
                    self._apply_status(name, 'creator')
                    self.current_speaker = name
                    self.user_name = name
                    self.user_role = 'creator'
                    self._save_user_statuses()
                    response = f'Фраза верная. Привет, {name}! Полный доступ открыт.'
                    dialog_logger.info(f'🤖 {response}')
                    return {
                        'response': response,
                        'language': lang,
                        'memory_size': len(self.memory),
                        'is_learning': self.is_learning,
                        'user_role': self.user_role,
                        'current_user': self.current_speaker
                    }
                else:
                    self.pending_auth = False
                    response = 'Фраза неверная. Доступ создателя не подтверждён.'
                    dialog_logger.info(f'🤖 {response}')
                    return {
                        'response': response,
                        'language': lang,
                        'memory_size': len(self.memory),
                        'is_learning': self.is_learning,
                        'user_role': self.user_role,
                        'current_user': self.current_speaker
                    }

            # 4. Ответ на учебный вопрос Протоса
            if getattr(self, 'pending_teach', None) is not None:
                response = self._handle_teach_answer(user_input)
                dialog_logger.info(f'🤖 {response}')
                return {
                    'response': response,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role,
                    'current_user': self.current_speaker
                }

            # 5. Просьба «задай вопрос / поучимся»
            if self._wants_to_be_asked(user_input):
                response = self._make_teach_question('user_request')
                dialog_logger.info(f'🤖 {response}')
                return {
                    'response': response,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role,
                    'current_user': self.current_speaker
                }

            # 6. Представление
            if self._is_introduction(user_input):
                debug_logger.debug('Обнаружено представление пользователя')
                response = self._handle_introduction(user_input, lang)
                dialog_logger.info(f'🤖 {response}')
                return {
                    'response': response,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role,
                    'current_user': self.current_speaker
                }

            # 7. Представление другого человека
            if self._is_about_someone(user_input):
                debug_logger.debug('Обнаружено представление другого человека')
                response = self._handle_about_someone(user_input, lang)
                if response:
                    dialog_logger.info(f'🤖 {response}')
                    return {
                        'response': response,
                        'language': lang,
                        'memory_size': len(self.memory),
                        'is_learning': self.is_learning,
                        'user_role': self.user_role,
                        'current_user': self.current_speaker
                    }

            # 8. Определение говорящего
            speaker = self._identify_speaker(user_input)
            if speaker:
                debug_logger.debug(f'Определен говорящий: {speaker}')
                self.current_speaker = speaker
                self.user_name = speaker
                assistant_logger.debug(f'Определен говорящий: {speaker}')

            # 9. Пользователь ещё не известен
            if self.current_speaker is None and self.user_name is None:
                debug_logger.debug('Пользователь не представился')
                if lang == 'ru':
                    response = (
                        'Привет! Я Протос — искусственный интеллект.\n'
                        'Я учусь общаться и помогать людям.\n'
                        'Как тебя зовут?'
                    )
                else:
                    response = (
                        "Hello! I'm Protos — an artificial intelligence.\n"
                        "I'm learning to communicate and help people.\n"
                        "What's your name?"
                    )
                dialog_logger.info(f'🤖 {response}')
                return {
                    'response': response,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role
                }

            # 10. Заявление о связях
            if self._is_relationship_statement(user_input):
                debug_logger.debug('Обнаружено заявление о связях')
                response = self._handle_relationship_statement(user_input, lang)
                if response:
                    dialog_logger.info(f'🤖 {response}')
                    return {
                        'response': response,
                        'language': lang,
                        'memory_size': len(self.memory),
                        'is_learning': self.is_learning,
                        'user_role': self.user_role,
                        'current_user': self.current_speaker
                    }

            # 11. Контекст и генерация
            context_events = self.memory.get_context(limit=5)
            context = [event.content for event in context_events]
            debug_logger.debug(f'Контекст: {len(context)} сообщений')

            thought = self.thought_processor.think(
                text=user_input,
                memory=self.memory,
            )
            debug_logger.debug(f"Мысль: {thought}")
            self.memory.record(
                MemoryEvent(
                    event_type=MemoryEventType.THOUGHT,
                    content=thought.original_text,
                    importance=thought.confidence,
                )
            )

            debug_logger.debug('Генерация ответа')
            response = self._route_request(
                user_input=user_input,
                context=context,
                lang=lang,
            )
            dialog_logger.info(f'🤖 {response}')

            # Память
            speaker = self.current_speaker or self.user_name or 'Кто-то'
            self.memory.remember_message(
                f"{speaker}: {user_input}",
                importance=0.2,
            )

            self.memory.remember_message(
                f"Протос: {response}",
                importance=0.2,
            )

            # Автообучение
            if self.is_learning and self.learning_manager:
                if self._should_learn_pair(user_input, response):
                    knowledge = self._extract_knowledge(user_input, response)
                    if knowledge:
                        debug_logger.debug(f'Извлечено знание: {knowledge[:50]}...')
                        self._save_to_knowledge_base(knowledge)

                    self.learning_manager.learn_from_text(
                        f'Пользователь: {user_input}\nПротос: {response}',
                        source='conversation'
                    )

                    if not user_input.startswith('/'):
                        self.dialogue_learner.learn_from_conversation(user_input, response)

                self.messages_since_last_train += 1
                if self.messages_since_last_train >= 20:
                    self.messages_since_last_train = 0
                    debug_logger.debug('Периодическое обучение')
                    self.learning_manager.train(epochs=2, sequence_length=40)
                    self.learning_manager.save_model('models/protos_lstm.json')

            return {
                'response': response,
                'language': lang,
                'memory_size': len(self.memory),
                'is_learning': self.is_learning,
                'user_role': self.user_role,
                'current_user': self.current_speaker
            }

        except Exception as e:
            error_logger.exception(f'Ошибка обработки ввода: {e}')
            raise

    @staticmethod
    def _should_learn_pair(user_input: str, response: str) -> bool:
        """
        Можно ли сохранять пару для обучения.
        """

        u = user_input.strip()
        r = response.strip()
        r_low = r.lower()

        if len(u) < 3 or len(r) < 3:
            return False

        if u.startswith('/'):
            return False

        junk = [
            'расскажи мне больше',
            'расскажи подробнее',
            'я понимаю. давай обсудим',
            'это важная тема',
            'понял. а что ты думаешь',
            'интересно! расскажи',
            'звучит интересно',
            'хорошо, я запомнил это. что дальше',
            'продолжай, я слушаю',
        ]
        if any(j in r_low for j in junk):
            return False

        canned = [
            'меня зовут протос',
            'я протос — искусственный интеллект',
            'ты — артём, мой создатель',
            'снова привет',
            'рад тебя видеть, создатель',
            'до встречи',
            'вопросы отложим',
            'спасибо, запомнил. так я становлюсь',
            'пока нечего спрашивать',
        ]
        if any(c in r_low for c in canned):
            return False

        return True


    @staticmethod
    def _extract_knowledge(user_input: str, response: str) -> str | None:
        """
        Извлечение знаний из диалога (статический метод).
        """

        if any(word in user_input for word in ['это', 'этот', 'эта', 'это -']):
            fact_match = re.search(r'это\s+([^.!?]+)', user_input)
            if fact_match:
                return f"Знание: {fact_match.group(1).strip()}"

        if '?' in user_input:
            return f"Вопрос: {user_input}\nОтвет: {response}"

        return None

    @staticmethod
    def _save_to_knowledge_base(knowledge: str) -> None:
        """
        Сохранение знания в базу знаний.
        """

        kb_path = 'data/knowledge_base.json'

        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    kb = json.load(f)
            except (json.JSONDecodeError, IOError):
                kb = {}
        else:
            kb = {}

        if 'learned' not in kb:
            kb['learned'] = []

        kb['learned'].append({
            'knowledge': knowledge,
            'timestamp': time.time(),
            'source': 'conversation'
        })

        try:
            with open(kb_path, 'w', encoding='utf-8') as f:
                json.dump(kb, f, ensure_ascii=False, indent=2)
        except IOError:
            pass

    @staticmethod
    def _is_introduction(text: str) -> bool:
        """
        Проверка, является ли текст представлением.
        """

        text_lower = text.lower().strip()
        patterns = [
            r'меня\s+зовут\s+\w+',
            r'звать\s+\w+',
            r'я\s*[-—]\s*\w+',
            r'я\s+твой\s+создатель',
            r'я\s+создатель',
            r'это\s+я\b',
            r'^я\s+[А-ЯЁA-Z][а-яёa-z]{2,}([!.\s]|$)',
        ]
        if not any(
                re.search(p, text_lower if 'зовут' in p or 'создатель' in p or 'это' in p else text, re.IGNORECASE) for
                p in patterns[:6]):
            m = re.match(r'^я\s+([А-ЯЁA-Z][а-яёa-z]{2,})\s*[!.]?$', text.strip(), re.IGNORECASE)
            if not m:
                return False
            return True
        return any(re.search(p, text_lower, re.IGNORECASE) for p in patterns[:6])

    def _handle_introduction(self, text: str, lang: str) -> str:
        """
        Обработка представления пользователя.
        """

        debug_logger.debug(f'Обработка представления: {text[:30]}...')
        text_lower = text.lower().strip()

        STOP_WORDS = {
            'твой', 'твоя', 'твоё', 'твое', 'создатель', 'пользователь',
            'владелец', 'разработчик', 'друг', 'брат', 'сестра',
            'мой', 'моя', 'моё', 'мое',
            'просил', 'попросил', 'хотел', 'хочу', 'могу', 'знаю',
            'думаю', 'сказал', 'сделал', 'видел', 'здесь', 'там',
            'это', 'тот', 'эта', 'просто',
        }

        name = None

        name_match = re.search(
            r'(?:меня\s+зовут|звать|я\s*[-—]\s*)\s*([А-ЯЁA-Z][а-яёa-zА-ЯЁ]+)',
            text,
            re.IGNORECASE,
        )
        if name_match:
            candidate = name_match.group(1).capitalize()
            if candidate.lower() not in STOP_WORDS:
                name = candidate
                debug_logger.debug(f'Извлечено имя: {name}')

        if not name:
            m = re.search(r'^я\s+([А-ЯЁA-Z][а-яёa-zА-ЯЁ]+)', text.strip(), re.IGNORECASE)
            if m:
                candidate = m.group(1).capitalize()
                if candidate.lower() not in STOP_WORDS:
                    name = candidate
                    debug_logger.debug(f'Извлечено имя (я + имя): {name}')

        if name and name.lower() in STOP_WORDS:
            debug_logger.debug(f'Отброшено ложное имя: {name}')
            name = None

        status = self._extract_status_from_text(text_lower)
        if not status and 'создатель' in text_lower:
            status = 'creator'

        if not name and self.current_speaker:
            name = self.current_speaker

        if not name and status == 'creator':
            if lang == 'ru':
                return 'Понял, ты создатель. А как тебя зовут?'
            return "Got it, you're the creator. What's your name?"

        if not name:
            debug_logger.debug('Имя не найдено')
            if lang == 'ru':
                return 'Приятно познакомиться! А как вас зовут?'
            return "Nice to meet you! What's your name?"

        assistant_logger.info(f'Представление пользователя: {name}')

        rel_type, rel_name = self._extract_relationship(text)
        debug_logger.debug(f'Статус: {status}, связь: {rel_type} -> {rel_name}')

        if name not in self.known_users:
            self.known_users[name] = {
                'status': status or 'user',
                'relations': {},
            }
        elif status:
            self.known_users[name]['status'] = status

        if rel_type and rel_name:
            self.known_users[name]['relations'][rel_type] = rel_name
            self._add_relationship(name, rel_type, rel_name)
            debug_logger.debug(f'Добавлена связь: {name} -> {rel_type}: {rel_name}')

        self.current_speaker = name
        self.user_name = name

        if status == 'creator' and self.user_role != 'creator':
            self.current_speaker = name
            self.user_name = name
            if name not in self.known_users:
                self.known_users[name] = {'status': 'user', 'relations': {}}
            self._save_user_statuses()
            return self._request_auth(lang)

        if status:
            self._apply_status(name, status)

        self._save_user_statuses()
        debug_logger.debug(f'Пользователь {name} сохранён')

        if lang == 'ru':
            if status == 'creator':
                return f'Привет, {name}! Я ждал тебя. Ты мой создатель!'
            if rel_type and rel_name:
                return f'А, так ты {rel_type} {rel_name}! Приятно познакомиться, {name}!'
            return f'Привет, {name}! Рад познакомиться.'
        else:
            if status == 'creator':
                return f"Hello, {name}! I've been waiting for you. You are my creator!"
            if rel_type and rel_name:
                return f"Oh, so you're {rel_name}'s {rel_type}! Nice to meet you, {name}!"
            return f'Hello, {name}! Nice to meet you.'

    @staticmethod
    def _is_about_someone(text: str) -> bool:
        """
        Проверка, говорит ли пользователь о ком-то.
        """

        text_lower = text.lower()
        patterns = [
            'это мой',
            'это моя',
            'это мое',
            'познакомься с',
            'знакомься',
            'вот мой',
            'вот моя',
            'а это'
        ]
        return any(p in text_lower for p in patterns)

    def _handle_about_someone(self, text: str, lang: str) -> str | None:
        """
        Обработка ситуации, когда представляют другого человека.
        """

        debug_logger.debug(f'Обработка представления другого человека: {text[:30]}...')
        rel_type, name = self._extract_relationship(text)

        if not name or not rel_type:
            debug_logger.debug('Связь или имя не найдены')
            return None

        debug_logger.debug(f'Связь: {rel_type} -> {name}')

        if self.current_speaker is None:
            debug_logger.debug('Текущий говорящий не определен')
            if lang == 'ru':
                return f"Сначала представьтесь сами, а потом я познакомлюсь с {name}."
            else:
                return f"First introduce yourself, then I'll meet {name}."

        self._add_relationship(self.current_speaker, rel_type, name)

        if name not in self.known_users:
            self.known_users[name] = {
                'status': 'user',
                'relations': {}
            }
            debug_logger.debug(f'Создан новый пользователь: {name}')

        assistant_logger.info(f'Представлен новый пользователь: {name} ({rel_type} {self.current_speaker})')
        debug_logger.debug(f'Представлен новый пользователь: {name} ({rel_type} {self.current_speaker})')

        if lang == 'ru':
            return f"Понял! {name} - твой {rel_type}. Теперь я буду знать, если он заговорит."
        else:
            return f"Got it! {name} is your {rel_type}. I'll know if they speak."

    def _is_relationship_statement(self, text: str) -> bool:
        """
        Проверка, говорит ли пользователь о связях.
        """

        text_lower = text.lower()
        for word in self.RELATIONSHIP_TYPES.keys():
            if re.search(rf'(?<!\w){re.escape(word)}(?!\w)', text_lower):
                return True
        return False

    def _handle_relationship_statement(self, text: str, lang: str) -> str | None:
        """
        Обработка заявления о связях.
        """

        debug_logger.debug(f'Обработка заявления о связях: {text[:30]}...')
        rel_type, name = self._extract_relationship(text)

        if not name or not rel_type:
            debug_logger.debug('Связь или имя не найдены')
            return None

        if self.current_speaker is None:
            debug_logger.debug('Текущий говорящий не определен')
            if lang == 'ru':
                return "Сначала представьтесь, чтобы я знал, кто говорит."
            else:
                return "First introduce yourself so I know who's speaking."

        self._add_relationship(self.current_speaker, rel_type, name)
        debug_logger.debug(f'Добавлена связь: {self.current_speaker} -> {rel_type}: {name}')
        assistant_logger.debug(f'Добавлена связь: {self.current_speaker} -> {rel_type}: {name}')

        if lang == 'ru':
            return f"Запомнил: {name} - твой {rel_type}."
        else:
            return f"Remembered: {name} is your {rel_type}."

    def _identify_speaker(self, text: str) -> str | None:
        """
        Определение говорящего по контексту.
        """

        for user in self.known_users.keys():
            if user.lower() in text.lower():
                debug_logger.debug(f'Говорящий определен по контексту: {user}')
                return user

        return self.current_speaker

    @staticmethod
    def _extract_status_from_text(text: str) -> str | None:
        """
        Извлечение статуса из текста.
        """

        if 'создатель' in text or 'твой создатель' in text:
            return 'creator'
        elif 'владелец' in text:
            return 'owner'
        elif 'разработчик' in text or 'программист' in text:
            return 'developer'
        elif 'исследователь' in text:
            return 'researcher'
        elif 'пользователь' in text or 'юзер' in text:
            return 'user'
        return None

    def _apply_status(self, name: str, status: str) -> None:
        """
        Применение статуса к пользователю.
        """

        debug_logger.debug(f'Применение статуса {status} к {name}')

        if status == 'creator':
            if self.creator_name is None:
                self.creator_name = name
                self.user_role = 'creator'
                debug_logger.info(f'Установлен создатель: {name}')
                assistant_logger.info(f'Установлен создатель: {name}')
            else:
                debug_logger.debug(f'Создатель уже существует: {self.creator_name}')
                return

        if status == 'owner':
            self.owner_name = name
            self.user_role = 'owner'
            debug_logger.info(f'Установлен владелец: {name}')
            assistant_logger.info(f'Установлен владелец: {name}')

        self.user_statuses[name] = status
        self._save_user_statuses()
        debug_logger.debug(f'Статус {status} применен к {name}')

    def _extract_relationship(self, text: str) -> tuple[str | None, str | None]:
        """
        Извлечение связи из текста.
        Возвращает (тип_связи, имя_связанного)
        """

        text_lower = text.lower()

        for rel_type in self.RELATIONSHIP_TYPES.keys():
            pattern = rf'(?:мой|моя|моё|мое)\s+{re.escape(rel_type)}\s+([А-Яа-яA-Za-z]+)'
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1)
                if name:
                    name = name.capitalize()
                debug_logger.debug(f'Извлечена связь: {rel_type} -> {name}')
                return rel_type, name

        return None, None

    def _add_relationship(self, person: str, relation: str, related: str) -> None:
        """
        Добавление связи между людьми.
        """

        debug_logger.debug(f'Добавление связи: {person} -> {relation}: {related}')

        if person not in self.relationships:
            self.relationships[person] = {}

        self.relationships[person][relation] = related
        self._save_relationships()

        if related not in self.relationships:
            self.relationships[related] = {}

        rel_info = self.RELATIONSHIP_TYPES.get(relation)
        if rel_info:
            reverse = rel_info.get('reverse')
            if reverse:
                self.relationships[related][reverse] = person
                debug_logger.debug(f'Добавлена обратная связь: {related} -> {reverse}: {person}')

        self._save_relationships()
        debug_logger.debug(f'Связь сохранена: {person} -> {relation}: {related}')
        assistant_logger.debug(f'Связь сохранена: {person} -> {relation}: {related}')

    def _get_relationship_info(self, name: str) -> str | None:
        """
        Получение информации о связях человека.
        """

        if name not in self.relationships:
            return None

        rels = self.relationships[name]
        if not rels:
            return None

        result = []
        for rel, person in rels.items():
            result.append(f"{rel}: {person}")

        return ", ".join(result)

    def _route_request(self, user_input: str, context: list[str], lang: str) -> str:
        """
        Естественная генерация ответа с использованием знаний.
        """

        debug_logger.debug(f'Генерация естественного ответа на: {user_input[:30]}...')

        user_lower = user_input.lower().strip()

        context_data = {
            'creator_name': self.creator_name or 'создатель',
            'user_name': self.current_speaker or 'ты',
            'user_status': self.user_statuses.get(self.current_speaker,
                                                  'пользователь') if self.current_speaker else 'пользователь',
            'conversation_count': self.conversation_count,
            'lang': lang
        }

        # ------------------------------------------------------------------
        # 0. Жёсткие правила
        # ------------------------------------------------------------------

        is_who_is = re.search(r'кто\s+так(ой|ая|ое|ие)\s+\w+', user_lower) is not None

        if not is_who_is:
            ask_name = any(p in user_lower for p in [
                'как тебя зовут', 'как тебя звать', 'твоё имя', 'твое имя',
                'what is your name', 'your name'
            ])
            ask_who = any(p in user_lower for p in [
                'кто ты', 'ты кто', 'представься', 'who are you'
            ])

            if ask_name and ask_who:
                debug_logger.debug('Вопрос "кто ты" + "как зовут"')
                if lang == 'ru':
                    return (
                        'Я Протос — искусственный интеллект. '
                        'Меня зовут Протос. Я учусь понимать язык и помогать людям.'
                    )
                return "I'm Protos — an artificial intelligence. My name is Protos."

            if ask_name:
                debug_logger.debug('Вопрос "как тебя зовут"')
                if lang == 'ru':
                    return 'Меня зовут Протос.'
                return 'My name is Protos.'

            if ask_who:
                debug_logger.debug('Вопрос о Протосе')
                if lang == 'ru':
                    return (
                        'Я Протос — искусственный интеллект. '
                        'Я учусь понимать язык и помогать людям.'
                    )
                return (
                    "I'm Protos — an artificial intelligence. "
                    "I'm learning language and how to help people."
                )

            if re.search(
                    r'(^|\s)(кто\s+я|как\s+меня\s+зовут|ты\s+знаешь\s+кто\s+я)(\s|$|[?!.])',
                    user_lower
            ):
                debug_logger.debug('Вопрос "кто я"')
                if self.current_speaker:
                    status = self.user_statuses.get(self.current_speaker, 'пользователь')
                    if status == 'creator':
                        return f'Ты — {self.current_speaker}, мой создатель.'
                    return f'Ты — {self.current_speaker}.'
                return 'Ты пока не представился. Напиши, как тебя зовут.'

        if any(p in user_lower for p in ['кто создал', 'кто создатель', 'кто твой создатель']):
            debug_logger.debug('Вопрос о создателе')
            if self.creator_name:
                return f'Мой создатель — {self.creator_name}.'
            return 'Мой создатель ещё не представился мне.'

        if any(p in user_lower for p in [
            'директива', 'твоя цель', 'твоя задача', 'зачем ты',
            'твоё назначение', 'твое назначение'
        ]):
            debug_logger.debug('Вопрос о цели/директиве')
            if lang == 'ru':
                return (
                    'Моя главная задача — учиться понимать язык, '
                    'помнить важный контекст и помогать тебе.'
                )
            return 'My main goal is to learn language, keep useful context, and help you.'

        if any(p in user_lower for p in [
            'какие команды', 'список команд', 'что ты умеешь', 'твои команды'
        ]):
            debug_logger.debug('Запрос списка команд')
            return self._help_text()

        if any(word in user_lower for word in [
            'привет', 'приветствую', 'здравствуй', 'здарова', 'добрый',
            'hello', 'hi', 'пиривет', 'приветик'
        ]):
            debug_logger.debug('Обнаружено приветствие')
            return self._generate_natural_greeting(lang)

        if any(p in user_lower for p in [
            'пока', 'до свидания', 'мне пора', 'увидимся', 'до встречи',
            'goodbye', 'bye',
        ]):
            debug_logger.debug('Прощание')
            if self.current_speaker:
                return f'До встречи, {self.current_speaker}!'
            return 'До встречи!'

        fact = self.find_fact(user_input)
        if fact:
            debug_logger.debug(f'Найден факт: {fact[:50]}...')
            return fact
        # ------------------------------------------------------------------
        # 0.5 База Q&A
        # ------------------------------------------------------------------
        qa_answer = self.qa_matcher.find_answer(user_input)
        if qa_answer:
            debug_logger.debug("Ответ найден в Q&A")
            return qa_answer

        # ------------------------------------------------------------------
        # 1. Обученные диалоги
        # ------------------------------------------------------------------
        learned_response = self.dialogue_learner.get_response(user_input, context_data)
        if learned_response:
            debug_logger.debug('Найден ответ в обученных диалогах')
            return learned_response

        # ------------------------------------------------------------------
        # 2. База знаний
        # ------------------------------------------------------------------
        if '?' in user_input:
            knowledge = self.learning_manager.query_knowledge(user_input) if self.learning_manager else []
            if knowledge:
                debug_logger.debug(f'Найдено знание: {knowledge[0][:50]}...')
                return knowledge[0]

            knowledge_answer = self._query_knowledge_base(user_input)
            if knowledge_answer:
                debug_logger.debug(f'Найдено в базе знаний: {knowledge_answer[:50]}...')
                return knowledge_answer

        # ------------------------------------------------------------------
        # 3. Другой пользователь
        # ------------------------------------------------------------------
        for user in self.known_users.keys():
            if user.lower() in user_lower and user != self.current_speaker:
                debug_logger.debug(f'Вопрос о пользователе: {user}')
                status = self.known_users[user].get('status', 'user')
                relations = self._get_relationship_info(user)
                if relations:
                    return f'{user} — {status}. Связи: {relations}'
                return f'{user} — {status}'

        # ------------------------------------------------------------------
        # 4. Статус
        # ------------------------------------------------------------------
        if 'какой у меня статус' in user_lower or 'мой статус' in user_lower:
            debug_logger.debug('Вопрос о статусе')
            if self.current_speaker and self.current_speaker in self.known_users:
                status = self.known_users[self.current_speaker].get('status', 'user')
                return f'Твой статус: {status}'
            return 'У тебя пока нет статуса. Расскажи, кто ты?'

        # ------------------------------------------------------------------
        # 5. Связи
        # ------------------------------------------------------------------
        if 'кто' in user_lower and any(word in user_lower for word in ['мой', 'моя', 'моё', 'мое']):
            for rel_type in self.RELATIONSHIP_TYPES.keys():
                if rel_type in user_lower:
                    debug_logger.debug(f'Вопрос о связи: {rel_type}')
                    if self.current_speaker and self.current_speaker in self.relationships:
                        rels = self.relationships[self.current_speaker]
                        if rel_type in rels:
                            return f'Твой {rel_type} — {rels[rel_type]}.'
                        return f'Я не знаю, кто твой {rel_type}. Расскажи мне.'
                    return 'Я не знаю твоих связей. Расскажи мне о своей семье и друзьях.'

        # ------------------------------------------------------------------
        # 6. Системные команды текстом
        # ------------------------------------------------------------------
        if user_input.startswith('ls') or user_input.startswith('cd'):
            debug_logger.debug('Системная команда')
            return self._handle_system_command(user_input)

        if user_lower in ['info', 'stats']:
            debug_logger.debug('Команда info/stats')
            return self._handle_info_command()

        if user_lower == 'clear':
            if self.user_role in ['owner', 'creator']:
                self.memory.clear()
                debug_logger.debug('Память очищена')
                memory_logger.info('Память очищена')
                return '🧹 Память очищена'
            return 'У вас нет прав на очистку памяти'

        # ------------------------------------------------------------------
        # 6.5 Незнакомое слово (редко)
        # ------------------------------------------------------------------
        if self.pending_teach is None and self.conversation_count > 2:
            unk = self._find_unknown_word(user_input)
            if unk and self.conversation_count % 5 == 0:
                debug_logger.debug(f'Незнакомое слово: {unk}')
                return self._make_teach_question('unknown_word', word=unk)

        # ------------------------------------------------------------------
        # 7. LSTM
        # ------------------------------------------------------------------
        try:
            debug_logger.debug('Генерация через LSTM')
            generated = self._generate_with_lstm(user_input, context, lang)
            if len(generated) > 5:
                debug_logger.debug(f'LSTM сгенерировал: {generated[:30]}...')
                return generated
        except Exception as e:
            error_logger.exception(f'Ошибка генерации: {e}')
            debug_logger.debug(f'Ошибка генерации: {e}')

        # ------------------------------------------------------------------
        # 8. Fallback
        # ------------------------------------------------------------------
        debug_logger.debug('Использован fallback ответ')
        return self.dialogue_learner.get_fallback(lang)

    def _generate_natural_greeting(self, lang: str) -> str:
        """
        Естественное приветствие.
        """
        debug_logger.debug(f'Генерация приветствия на языке: {lang}')

        if self.current_speaker:
            status = self.user_statuses.get(self.current_speaker, 'пользователь')
            if lang == 'ru':
                if status == 'creator':
                    return f"Снова привет, {self.current_speaker}! Рад тебя видеть, создатель."
                return f"Привет, {self.current_speaker}!"
            else:
                if status == 'creator':
                    return f"Hello again, {self.current_speaker}! Good to see you, creator."
                return f"Hello, {self.current_speaker}!"

        if lang == 'ru':
            return "Привет! Я Протос. Как тебя зовут?"
        return "Hello! I'm Protos. What's your name?"

    def _handle_command(self, command: str) -> str | None:
        """
        Обработка команд.
        """

        cmd_parts = command[1:].split()
        cmd = cmd_parts[0] if cmd_parts else ''

        debug_logger.debug(f'Обработка команды: {cmd}, аргументы: {cmd_parts[1:]}')
        assistant_logger.debug(f'Команда: {cmd}, аргументы: {cmd_parts[1:]}')

        if cmd in ('help', 'commands', 'команды'):
            return self._help_text()

        if cmd == 'clear_memory':
            if self.user_role in ['creator']:
                self.memory.clear()
                debug_logger.debug('Память очищена по команде создателя')
                memory_logger.info('Память очищена по команде создателя')
                return "🧹 Память полностью очищена по команде создателя"
            else:
                debug_logger.debug('Отказ в очистке памяти: недостаточно прав')
                return "❌ Только создатель может очищать память"

        elif cmd == 'reset':
            if self.user_role in ['creator']:
                self.memory.clear()
                self.is_learning = True
                debug_logger.debug('Система сброшена')
                memory_logger.info('Система сброшена')
                return "🔄 Система сброшена к начальному состоянию"
            else:
                debug_logger.debug('Отказ в сбросе: недостаточно прав')
                return "❌ Только создатель может сбрасывать систему"

        elif cmd == 'debug':
            if self.user_role in ['creator']:
                debug_logger.debug('Вывод отладочной информации')
                return self._get_debug_info()
            else:
                debug_logger.debug('Отказ в отладке: недостаточно прав')
                return "❌ Только создатель может использовать отладку"

        elif cmd == 'train':
            if self.user_role in ['creator']:
                debug_logger.debug('Запуск обучения по команде')
                training_logger.info('Запуск обучения по команде')
                self.learning_manager.train(epochs=10)
                self.learning_manager.save_model('models/protos_lstm.json')
                debug_logger.debug('Обучение завершено')
                training_logger.success('Обучение завершено')
                return "🧠 Обучение завершено!"
            else:
                debug_logger.debug('Отказ в обучении: недостаточно прав')
                return "❌ Только создатель может запускать обучение"

        elif cmd == 'learn_from_file':
            if self.user_role in ['creator']:
                if len(cmd_parts) > 1:
                    filepath = cmd_parts[1]
                    debug_logger.debug(f'Загрузка файла для обучения: {filepath}')
                    training_logger.info(f'Загрузка файла для обучения: {filepath}')
                    result = self.learning_manager.learn_from_file(filepath)
                    debug_logger.debug(f'Результат загрузки: {result}')
                    return result
                return "❌ Использование: /learn_from_file [путь_к_файлу]"
            else:
                debug_logger.debug('Отказ в загрузке файла: недостаточно прав')
                return "❌ Только создатель может загружать файлы для обучения"

        elif cmd == 'learn_text':
            if self.user_role in ['creator']:
                if len(cmd_parts) > 1:
                    text = ' '.join(cmd_parts[1:])
                    debug_logger.debug(f'Добавление текста для обучения ({len(text)} символов)')
                    training_logger.info(f'Добавление текста для обучения ({len(text)} символов)')
                    self.learning_manager.learn_from_text(text, source="manual")
                    return f"📝 Добавлен текст для обучения ({len(text)} символов)"
                return "❌ Использование: /learn_text [текст]"
            else:
                debug_logger.debug('Отказ в добавлении текста: недостаточно прав')
                return "❌ Только создатель может добавлять тексты для обучения"

        elif cmd == 'save_model':
            if self.user_role in ['creator']:
                debug_logger.debug('Сохранение модели')
                self.learning_manager.save_model('models/protos_lstm.json')
                return "💾 Модель сохранена"
            else:
                debug_logger.debug('Отказ в сохранении модели: недостаточно прав')
                return "❌ Только создатель может сохранять модель"

        elif cmd == 'load_model':
            if self.user_role in ['creator']:
                debug_logger.debug('Загрузка модели')
                self._load_model()
                return "📖 Модель загружена"
            else:
                debug_logger.debug('Отказ в загрузке модели: недостаточно прав')
                return "❌ Только создатель может загружать модель"

        elif cmd == 'set_status':
            if self.user_role in ['owner', 'creator']:
                if len(cmd_parts) > 2:
                    name = cmd_parts[1]
                    status = cmd_parts[2]
                    if status in self.USER_STATUSES:
                        debug_logger.debug(f'Установка статуса {status} для {name}')
                        self.user_statuses[name] = status
                        if name in self.known_users:
                            self.known_users[name]['status'] = status
                        self._save_user_statuses()
                        assistant_logger.info(f'Установлен статус {status} для {name}')
                        return f"✅ Статус '{status}' установлен для {name}"
                    else:
                        debug_logger.debug(f'Неизвестный статус: {status}')
                        return f"❌ Неизвестный статус. Доступные: {', '.join(self.USER_STATUSES.keys())}"
                return "❌ Использование: /set_status [имя] [статус]"
            else:
                debug_logger.debug('Отказ в установке статуса: недостаточно прав')
                return "❌ Только владелец или создатель могут менять статусы"

        elif cmd == 'users':
            debug_logger.debug('Запрос списка пользователей')
            if self.known_users:
                return f"👥 Известные пользователи: {', '.join(self.known_users.keys())}"
            else:
                return "👥 Пока нет известных пользователей"

        elif cmd == 'relations':
            debug_logger.debug('Запрос связей')
            if self.current_speaker and self.current_speaker in self.relationships:
                rels = self.relationships[self.current_speaker]
                if rels:
                    result = []
                    for rel, person in rels.items():
                        result.append(f"{rel}: {person}")
                    return f"🔗 Связи {self.current_speaker}: {', '.join(result)}"
                else:
                    return f"🔗 У {self.current_speaker} нет связей"
            else:
                return "🔗 Вы еще не представились или у вас нет связей"

        elif cmd == 'knowledge':
            if len(cmd_parts) > 1:
                query = ' '.join(cmd_parts[1:])
                debug_logger.debug(f'Поиск знаний: {query}')
                results = self.learning_manager.query_knowledge(query) if self.learning_manager else []
                if results:
                    return f"📚 Найдено:\n" + '\n'.join(results[:3])
                else:
                    return "📚 Ничего не найдено"
            return "❌ Использование: /knowledge [вопрос]"

        elif cmd == 'learning_status':
            debug_logger.debug('Запрос статуса обучения')
            if self.learning_manager:
                total = len(self.learning_manager.learning_data)
                return f"📊 Статус обучения:\n  Примеров: {total}\n  Историй: {len(self.learning_manager.training_history)}\n  Режим: {'Включен' if self.is_learning else 'Выключен'}"
            return "❌ Менеджер обучения не инициализирован"

        elif cmd == 'train_on_books':
            if self.user_role in ['creator']:
                debug_logger.debug('Запуск обучения на книгах')
                training_logger.info('Запуск обучения на книгах')
                train_protos_on_books(self)
                return "📚 Обучение на книгах запущено!"
            else:
                debug_logger.debug('Отказ в обучении на книгах: недостаточно прав')
                return "❌ Только создатель может запускать обучение на книгах"

        elif cmd == 'continue_training':
            if self.user_role in ['creator']:
                debug_logger.debug('Продолжение обучения')
                training_logger.info('Продолжение обучения')
                continue_training_from_saved(self)
                return "📚 Продолжение обучения запущено!"
            else:
                debug_logger.debug('Отказ в продолжении обучения: недостаточно прав')
                return "❌ Только создатель может продолжать обучение"

        elif cmd == 'check_books':
            debug_logger.debug('Проверка книг')
            books_dir = Path("data/books")
            if books_dir.exists():
                books = list(books_dir.glob("*.txt"))
                if books:
                    return f"📚 Найдено {len(books)} книг:\n" + '\n'.join([f"  - {b.name}" for b in books])
                else:
                    return "📚 В папке data/books/ нет книг (.txt)"
            else:
                return "📚 Папка data/books/ не существует. Создайте ее и положите туда книги."

        elif cmd == 'download_books':
            if self.user_role in ['creator']:
                debug_logger.debug('Скачивание книг')
                assistant_logger.info('Скачивание книг')
                try:
                    downloader = BookDownloader()
                    russian = downloader.download_russian_books()
                    english = downloader.download_english_books()
                    debug_logger.debug(f'Скачано книг: русских - {len(russian)}, английских - {len(english)}')
                    return f"📥 Скачано книг: русских - {len(russian)}, английских - {len(english)}"
                except ImportError as e:
                    error_logger.error(f'Ошибка импорта: {e}')
                    debug_logger.debug(f'Ошибка импорта: {e}')
                    return f"❌ Ошибка импорта: {e}"
                except Exception as e:
                    error_logger.exception(f'Ошибка скачивания: {e}')
                    debug_logger.debug(f'Ошибка скачивания: {e}')
                    return f"❌ Ошибка: {e}"
            else:
                debug_logger.debug('Отказ в скачивании книг: недостаточно прав')
                return "❌ Только создатель может скачивать книги"

        elif cmd == 'auth':
            phrase = ' '.join(cmd_parts[1:]) if len(cmd_parts) > 1 else ''
            if not phrase:
                return self._request_auth('ru')
            if self._check_secret_phrase(phrase):
                name = self.current_speaker or self.user_name or self.creator_name or 'Создатель'
                self._apply_status(name, 'creator')
                self.current_speaker = name
                self.user_name = name
                self.user_role = 'creator'
                self.pending_auth = False
                self._save_user_statuses()
                return f"Фраза верная. Привет, {name}! Полный доступ открыт."
            return "Фраза неверная."

        elif cmd == 'set_secret':
            if self.user_role != 'creator':
                return "Только создатель может менять секретную фразу."
            if len(cmd_parts) < 2:
                return "Использование: /set_secret <новая фраза>"
            new_secret = ' '.join(cmd_parts[1:]).lower().strip()
            self.creator_secret = new_secret
            self._save_creator_secret()
            return "Секретная фраза обновлена."

        return None

    def _handle_system_command(self, command: str) -> str:
        """
        Обработка системных команд.
        """

        if command.startswith('ls'):
            path = command[2:].strip()
            debug_logger.debug(f'Команда ls: {path or "."}')
            fs_logger.info(f'Команда ls: {path or "."}')
            return self.fs_manager.list_dir(path if path else None)
        elif command.startswith('cd'):
            path = command[2:].strip()
            debug_logger.debug(f'Команда cd: {path or "~"}')
            fs_logger.info(f'Команда cd: {path or "~"}')
            return self.fs_manager.change_dir(path if path else None)
        return "Неизвестная команда"

    def _handle_info_command(self) -> str:
        """
        Информация о состоянии.
        """

        debug_logger.debug('Вывод информации о состоянии')
        return (
            f"📊 Статистика Протоса:\n"
            f"  Разговоров: {self.conversation_count}\n"
            f"  Сообщений в памяти: {len(self.memory)}\n"
            f"  Текущий пользователь: {self.current_speaker or 'Неизвестно'}\n"
            f"  Статус: {self.user_statuses.get(self.current_speaker, 'Не задан') if self.current_speaker else 'Не задан'}\n"
            f"  Создатель: {self.creator_name or 'Не установлен'}\n"
            f"  Известно пользователей: {len(self.known_users)}"
        )

    @staticmethod
    def _match_nl_command(text: str) -> str | None:
        """
        Сопоставляет фразу с внутренней командой. Только узкие паттерны.
        """

        t = text.lower().strip()
        rules = [
            (r'обуч(ись|итесь|ение).*(книг|books)|обучись\s+на\s+книгах', 'train_on_books'),
            (r'сохрани\s+модель|сохранить\s+модель', 'save_model'),
            (r'загрузи\s+модель|загрузить\s+модель', 'load_model'),
            (r'статус\s+обучения|как\s+идёт\s+обучение|как\s+идет\s+обучение', 'learning_status'),
            (r'очисти\s+память|очистить\s+память', 'clear_memory'),
            (r'какие\s+команды|список\s+команд|что\s+ты\s+умеешь', 'help'),
            (r'известн\w*\s+пользовател|список\s+пользовател|кто\s+мне\s+известен', 'users'),
        ]
        for pattern, cmd in rules:
            if re.search(pattern, t):
                return cmd
        return None

    def _help_text(self) -> str:
        """
        Справка по командам (полный список — для создателя).
        """

        if self.user_role != 'creator':
            return (
                'Доступно:\n'
                '• поговорить со мной\n'
                '• спросить «кто ты», «как тебя зовут»\n'
                '• представиться: «меня зовут …»'
            )
        return (
            'Команды создателя:\n'
            '/train — обучить модель на накопленных данных\n'
            '/train_on_books — обучение на книгах из data/books\n'
            '/save_model — сохранить модель\n'
            '/load_model — загрузить модель\n'
            '/learning_status — статус данных обучения\n'
            '/auth <фраза> — подтвердить секрет создателя\n'
            '/set_secret <фраза> — сменить секрет\n'
            '/debug — отладочная информация\n'
            '/clear_memory — очистить память диалога\n'
            '/users — известные пользователи\n'
            '/knowledge <запрос> — поиск по знаниям\n'
            '\n'
            'Без / можно сказать, например:\n'
            '«обучись на книгах», «сохрани модель», «статус обучения»,\n'
            '«какие команды», «задай вопрос», «давай поучимся».'
        )

    def _get_debug_info(self) -> str:
        """
        Отладочная информация для создателя.
        """

        debug_logger.debug('Вывод отладочной информации')
        learning_data_count = len(self.learning_manager.learning_data) if self.learning_manager else 0

        return (
            f"🔍 ОТЛАДОЧНАЯ ИНФОРМАЦИЯ:\n"
            f"  Текущий пользователь: {self.current_speaker or 'Неизвестно'}\n"
            f"  Статус: {self.user_statuses.get(self.current_speaker, 'Не задан') if self.current_speaker else 'Не задан'}\n"
            f"  Роль: {self.user_role}\n"
            f"  Создатель: {self.creator_name or 'Не установлен'}\n"
            f"  Владелец: {self.owner_name or 'Не установлен'}\n"
            f"  Разговоров: {self.conversation_count}\n"
            f"  Сообщений в памяти: {len(self.memory)}\n"
            f"  Режим обучения: {self.is_learning}\n"
            f"  Известные пользователи: {list(self.known_users.keys())}\n"
            f"  Связи: {self.relationships}\n"
            f"  Данных для обучения: {learning_data_count}"
        )

    def _query_knowledge_base(self, question: str) -> str | None:
        """
        Поиск ответа в базе знаний.
        """

        debug_logger.debug(f'Поиск в базе знаний: {question[:30]}...')
        question_lower = question.lower()
        question_words = set(question_lower.split())

        for category, facts in self.knowledge_base.items():
            if not isinstance(facts, list):
                continue
            for fact in facts:
                if isinstance(fact, dict):
                    text = str(fact.get('knowledge', fact.get('text', '')))
                else:
                    text = str(fact)
                if not text:
                    continue
                fact_words = set(text.lower().split())
                common = fact_words & question_words
                if len(common) > 2:
                    debug_logger.debug(f'Найдено в категории {category}: {text[:30]}...')
                    return text

        debug_logger.debug('Ничего не найдено в базе знаний')
        return None

    def _generate_with_lstm(self, user_input: str, context: list[str], lang: str) -> str:
        """
        Генерация ответа с помощью LSTM.
        """

        if self.learning_manager is None:
            debug_logger.debug('LearningManager не инициализирован')
            return ""

        if context:
            prompt = f"{' '.join(context[-3:])} {user_input}"
        else:
            prompt = user_input

        if lang == 'ru':
            prompt = f"Ответ на русском языке: {prompt}"
        else:
            prompt = f"Answer in English: {prompt}"

        debug_logger.debug(f'Промпт для LSTM: {prompt[:50]}...')
        generated = self.learning_manager.generate_sample(prompt, length=30)

        if generated.startswith(prompt):
            generated = generated[len(prompt):]

        debug_logger.debug(f'Сгенерировано: {generated[:30]}...')
        return generated.strip()

    def _load_qa_pairs(self) -> None:
        """
        Загрузка пар вопрос-ответ.
        """

        path = 'data/qa_pairs.json'
        if not os.path.exists(path):
            self.qa_pairs = []
            debug_logger.debug('Файл Q&A не найден, создаю пустой список')
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.qa_pairs = data if isinstance(data, list) else []
            debug_logger.debug(f'Загружено Q&A пар: {len(self.qa_pairs)}')
            assistant_logger.info(f'Загружено Q&A пар: {len(self.qa_pairs)}')
        except (json.JSONDecodeError, IOError) as e:
            error_logger.error(f'Ошибка загрузки Q&A: {e}')
            self.qa_pairs = []

    def _save_qa_pairs(self) -> None:
        """
        Сохранение пар вопрос-ответ.
        """

        path = 'data/qa_pairs.json'
        os.makedirs('data', exist_ok=True)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.qa_pairs, f, ensure_ascii=False, indent=2)
            debug_logger.debug(f'Сохранено Q&A пар: {len(self.qa_pairs)}')
        except IOError as e:
            error_logger.error(f'Ошибка сохранения Q&A: {e}')

    @staticmethod
    def _normalize_question(text: str) -> str:
        """
        Нормализация вопроса для сравнения.
        """

        t = text.lower().strip()
        t = re.sub(r'[?!.,;:…]+', '', t)
        t = re.sub(r'\s+', ' ', t)
        return t

    def _qa_similarity(self, a: str, b: str) -> float:
        """
        Простая похожесть: доля общих слов.
        """

        wa = set(self._normalize_question(a).split())
        wb = set(self._normalize_question(b).split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    def _load_facts(self) -> None:
        """
        Загрузка фактов.
        """

        path = 'data/facts.json'
        self.facts: list[dict] = []
        if not os.path.exists(path):
            debug_logger.debug('Файл фактов не найден')
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.facts = data if isinstance(data, list) else []
            debug_logger.debug(f'Загружено фактов: {len(self.facts)}')
            assistant_logger.info(f'Загружено фактов: {len(self.facts)}')
        except (json.JSONDecodeError, IOError) as e:
            error_logger.error(f'Ошибка загрузки фактов: {e}')
            self.facts = []

    def _save_facts(self) -> None:
        """
        Сохранение фактов.
        """

        path = 'data/facts.json'
        os.makedirs('data', exist_ok=True)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.facts, f, ensure_ascii=False, indent=2)
            debug_logger.debug(f'Сохранено фактов: {len(self.facts)}')
        except IOError as e:
            error_logger.error(f'Ошибка сохранения фактов: {e}')

    def add_fact(
            self,
            subject: str,
            fact: str,
            source: str = 'manual',
            aliases: list[str] | None = None,
    ) -> None:
        """
        Добавить или обновить факт.
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
                self._save_facts()
                return

        self.facts.append({
            'subject': sub,
            'aliases': aliases or [],
            'fact': fact.strip(),
            'source': source,
        })
        self._save_facts()

    def find_fact(self, text: str) -> str | None:
        """
        Поиск факта по фразе пользователя.
        """

        if not getattr(self, 'facts', None):
            return None

        low = text.lower().strip()

        m = re.search(
            r'(?:кто\s+так(?:ой|ая|ое|ие)|что\s+так(?:ое|ая|ой)|что\s+значит|зачем\s+нужн\w*)\s+(.+?)(?:\?|$)',
            low,
        )
        query = m.group(1).strip(' .!?…') if m else low
        query = re.sub(r'^(такое|такой|такая|такoe)\s+', '', query).strip()

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


    def add_qa_pair(self, question: str, answer: str, source: str = 'manual') -> None:
        """
        Добавляет или обновляет пару вопрос-ответ.
        """
        q_norm = self._normalize_question(question)
        if not q_norm or not answer.strip():
            return

        # обновить, если такой вопрос уже есть
        for pair in self.qa_pairs:
            if self._normalize_question(pair.get('question', '')) == q_norm:
                pair['answer'] = answer.strip()
                pair['source'] = source
                self._save_qa_pairs()
                debug_logger.debug(f'Q&A обновлён: {q_norm[:40]}')
                return

        self.qa_pairs.append({
            'question': question.strip(),
            'answer': answer.strip(),
            'source': source,
        })
        self._save_qa_pairs()
        debug_logger.debug(f'Q&A добавлен: {q_norm[:40]}')

        # также в данные для LSTM
        if self.learning_manager:
            self.learning_manager.learn_from_text(
                f'Вопрос: {question.strip()}\nОтвет: {answer.strip()}',
                source=f'qa:{source}'
            )

    def toggle_learning(self) -> str:
        """
        Переключение режима обучения.
        """

        self.is_learning = not self.is_learning
        status = "включен" if self.is_learning else "выключен"
        debug_logger.debug(f'Режим обучения {status}')
        assistant_logger.info(f'Режим обучения {status}')
        return f"Режим обучения {status}"
