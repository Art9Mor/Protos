import json
import os
import re
import time

from .commands import CommandHandler
from ..auth.service import AuthService
from ..cognition.thought_processor import ThoughtProcessor
from ..dialogue.qa_matcher import QAMatcher
from ..dialogue.router import ResponseRouter
from ..knowledge.facts import FactsStore
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

        # Сессия (AuthService создаём после загрузки known_users)
        self.session_authenticated = False
        self.pending_login: dict | None = None
        self.pending_delete: dict | None = None
        self.auth: AuthService | None = None

        # Система ролей
        self.user_role: str = 'user'
        self.creator_name: str | None = None
        self.owner_name: str | None = None

        # Пользовательские данные
        self.user_statuses: dict[str, str] = {}
        self.relationships: dict[str, dict[str, str]] = {}
        self.known_users: dict[str, dict] = {}

        # Личность и имена создателя
        self.persona: dict = {}
        self.creator_aliases: list[str] = []
        self.preferred_name: str | None = None
        self.pending_rename: dict | None = None

        # Состояние обучения
        self.pending_teach = None
        self.is_learning = True
        self.gen_only = False
        self.messages_since_last_train = 0

        # Секрет создателя
        self.pending_auth = False
        self.creator_secret = 'Свет лишь слепит глаза'
        self.creator_secret_hint = 'Только во Тьме настанет прозрение'

        self._load_creator_secret()

        # Состояние диалога
        self.current_speaker = None
        self.user_name = None
        self.conversation_count = 0
        self._last_answer_layer = ''
        self._last_lang = 'ru'

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

        # Загрузка данных (statuses, known_users, модель)
        self._load_data()

        # Auth только после known_users / creator_name
        self.auth = AuthService(self)
        self.auth.restore_session()

        # Управление командами
        self.commands = CommandHandler(self)

        # Роутер диалогов
        self.router = ResponseRouter(self)

        debug_logger.debug('Инициализация Assistant завершена')
        main_logger.success('Ассистент инициализирован')


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
            hidden_size=self.config.get('hidden_size', 128),
        )

        self.dialogue_learner = DialogueLearner()

        training_logger.debug("Менеджер обучения инициализирован")


    def _load_data(self) -> None:
        """
        Загрузка пользовательских данных и модели.
        """

        debug_logger.debug("Загрузка сохранённых данных")

        self._load_user_statuses()
        self._load_persona()
        self._load_relationships()
        self._load_qa_pairs()
        self.facts_store = FactsStore('data/facts.json')

        self.qa_matcher = QAMatcher(self.qa_pairs)

        self.load_model()


    def _load_persona(self) -> None:
        """
        Загрузка persona.json (алиасы и обращение к создателю).
        """

        path = 'data/persona.json'
        if not os.path.exists(path):
            self.persona = {}
            debug_logger.debug('persona.json не найден')
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.persona = data if isinstance(data, dict) else {}
            aliases = self.persona.get('creator_aliases') or []
            if aliases and not self.creator_aliases:
                self.creator_aliases = list(aliases)
            pref = self.persona.get('address_creator_as')
            if pref and not self.preferred_name:
                self.preferred_name = pref
            debug_logger.debug('persona загружена')
        except (json.JSONDecodeError, IOError) as e:
            error_logger.error(f'Ошибка загрузки persona: {e}')
            self.persona = {}


    def load_model(self) -> None:
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
                return
            except Exception as e:
                error_logger.exception(f'Ошибка загрузки модели: {e}')
                debug_logger.debug('Ошибка загрузки модели, начинаю обучение с нуля')
                assistant_logger.warning('Начинаю обучение с нуля')
        else:
            debug_logger.debug('Модель не найдена, создаю новую')
            assistant_logger.warning('Модель не найдена, создаю новую')

        self._train_on_basic_examples()


    def is_logged_in(self) -> bool:
        """
        Есть активная сессия (имя + подтверждение).
        """

        return bool(self.current_speaker) and bool(self.session_authenticated)


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
                    self.creator_aliases = data.get('creator_aliases') or []
                    self.preferred_name = data.get('preferred_name') or self.creator_name
                    self.creator_aliases = data.get('creator_aliases') or []

                self.session_authenticated = False
                self.user_role = 'user'
                self.current_speaker = None
                self.user_name = None

                if self.creator_name:
                    debug_logger.debug(
                        f'На диске есть создатель: {self.creator_name}, '
                        f'сессия не подтверждена (нужна секретная фраза)'
                    )
                    assistant_logger.info(
                        f'Создатель известен ({self.creator_name}), ожидание подтверждения сессии'
                    )
                elif self.known_users:
                    debug_logger.debug(
                        f'Известные пользователи: {len(self.known_users)}, сессия гостевая'
                    )

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


    def save_creator_secret(self) -> None:
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


    def _creator_names(self) -> set[str]:
        """
        Все имена создателя (канон + алиасы), в нижнем регистре.
        """

        names: set[str] = set()

        for src in (
                self.creator_name,
                getattr(self, 'creator_aliases', None) or [],
                (self.persona or {}).get('creator'),
                (self.persona or {}).get('creator_aliases') or [],
        ):
            if isinstance(src, str) and src.strip():
                names.add(src.strip().lower())
            elif isinstance(src, (list, tuple)):
                for x in src:
                    if x and str(x).strip():
                        names.add(str(x).strip().lower())

        for uname, info in (self.known_users or {}).items():
            status = (info or {}).get('status') or self.user_statuses.get(uname)
            if status == 'creator':
                names.add(uname.lower())
                for a in (info or {}).get('aliases') or []:
                    if a:
                        names.add(str(a).strip().lower())

        return names


    def address_name(self) -> str:
        """
        Обращение к текущему собеседнику.
        """

        name = self.current_speaker or self.user_name
        if not name:
            return self.canonical_creator_name()

        if self.user_role == 'creator' or self.is_creator_name(name):
            return self.preferred_name or name

        rec = self.known_users.get(name) or {}
        return rec.get('preferred_name') or name


    def _handle_add_creator_alias(self, text: str, lang: str) -> str | None:
        """
        «Зови меня X» для создателя.
        """

        if self.user_role != 'creator' and not self.is_creator_name(self.current_speaker):
            return None

        low = text.lower().strip()
        m = re.search(
            r'(?:зови\s+меня|называй\s+меня|мне\s+можно\s+звать|'
            r'меня\s+также\s+зовут|меня\s+ещё\s+зовут|меня\s+еще\s+зовут)\s+'
            r'([A-Za-zА-ЯЁа-яё]{2,40})',
            low,
            re.IGNORECASE,
        )
        if not m:
            return None

        alias = m.group(1).strip()

        if alias.isascii():
            alias = alias[:1].upper() + alias[1:]
        else:
            alias = alias.capitalize()

        if alias.lower() in {'создатель', 'друг', 'пожалуйста', 'просто', 'теперь'}:
            return None

        known = self._creator_names()
        if alias.lower() in known or alias.lower() == (self.preferred_name or '').lower():
            self.preferred_name = alias
            self._sync_creator_identity_files()
            if lang == 'ru':
                return f'Хорошо, буду звать тебя {alias}.'
            return f'Alright, I will call you {alias}.'

        self.pending_rename = {'candidate': alias}
        if lang == 'ru':
            return f'«{alias}» — это твоё новое имя? Запомнить его и дальше звать тебя так?'
        return f'Is «{alias}» your new name? Should I remember it and call you that?'


    def _handle_rename_confirm(self, text: str, lang: str) -> str | None:
        """
        Ответ на вопрос про новое имя.
        """

        if not self.pending_rename:
            return None

        low = text.lower().strip()
        candidate = self.pending_rename.get('candidate')
        yes = any(p in low for p in [
            'да', 'ага', 'угу', 'конечно', 'запомни', 'давай', 'хорошо', 'ок', 'yes',
        ])
        no = any(p in low for p in [
            'нет', 'не надо', 'не нужно', 'отмена', 'не запоминай', 'no',
        ])

        if not yes and not no:
            if lang == 'ru':
                return 'Ответь, пожалуйста: запомнить это имя (да) или нет?'
            return 'Please answer yes or no: should I remember this name?'

        self.pending_rename = None

        if no:
            if lang == 'ru':
                return 'Хорошо, ничего не меняю.'
            return 'Okay, I will not change anything.'

        if not self.creator_aliases:
            self.creator_aliases = []
        if candidate.lower() not in {a.lower() for a in self.creator_aliases}:
            self.creator_aliases.append(candidate)

        canonical = self.canonical_creator_name()
        info = self.known_users.setdefault(
            canonical, {'status': 'creator', 'relations': {}, 'aliases': []}
        )
        al = info.setdefault('aliases', [])
        if candidate not in al and candidate.lower() != canonical.lower():
            al.append(candidate)

        self.preferred_name = candidate
        self._sync_creator_identity_files()

        if lang == 'ru':
            return f'Запомнил. Теперь буду звать тебя {candidate}.'
        return f'Saved. I will call you {candidate} from now on.'


    def _sync_creator_identity_files(self) -> None:
        """
        Сохранение preferred/aliases в statuses и persona для создателя.
        """

        self.save_user_statuses()
        if not isinstance(self.persona, dict):
            return
        self.persona['creator_aliases'] = list(
            dict.fromkeys(
                [self.canonical_creator_name()]
                + list(self.creator_aliases or [])
                + list(self.persona.get('creator_aliases') or [])
            )
        )
        pref = self.preferred_name or self.canonical_creator_name()
        self.persona['address_creator_as'] = pref
        try:
            with open('data/persona.json', 'w', encoding='utf-8') as f:
                json.dump(self.persona, f, ensure_ascii=False, indent=2)
        except IOError as e:
            error_logger.error(f'Ошибка сохранения persona: {e}')


    def is_creator_name(self, name: str | None) -> bool:
        if not name:
            return False
        return name.strip().lower() in self._creator_names()


    def canonical_creator_name(self) -> str:
        return (
                self.creator_name
                or (self.persona or {}).get('creator')
                or 'Создатель'
        )


    def _resolve_to_canonical_user(self, name: str) -> str:
        """
        Если имя — алиас создателя, вернуть каноническое имя создателя.
        """

        if self.is_creator_name(name):
            return self.canonical_creator_name()
        return name


    @staticmethod
    def _wants_to_teach(text: str) -> bool:
        """
        Явная просьба записать знание в память Протоса.
        """

        t = text.lower().strip()
        t = re.sub(r'[.!?…]+$', '', t).strip()
        triggers = (
            'запомни',
            'запомни это',
            'выучи',
            'изучи',
            'запиши',
            'запиши это',
            'добавь знание',
            'это знание',
            'remember',
            'remember this',
        )
        return any(
            t == tr or t.startswith(tr + ' ') or t.startswith(tr + ':')
            for tr in triggers
        )


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


    def find_unknown_word(self, text: str) -> str | None:
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


    def make_teach_question(self, kind: str, word: str | None = None) -> str:
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
        Ответ пользователя в режиме записи знания (урок / unknown_word / вопрос Протоса).
        """

        text = (user_input or '').strip()
        info = self.pending_teach or {}
        kind = info.get('type', 'user_request')
        prompt = info.get('prompt', '')

        if self._is_teach_exit(text):
            self.pending_teach = None
            return 'Заапись отменена. Если захочешь продолжить - скажи «запомни» или «давай поучимся».'

        if self.user_role not in ('creator', 'owner'):
            self.pending_teach = None
            return 'Доступ запрещён.\nЗаписывать знания в мою базу могут только создатель или владелец.'

        if kind == 'explicit_qa':
            pair = self._parse_teach_pair(text)
            if not pair:
                return (
                    'Не разобрал пару. Формат:\n'
                    'Вопрос: …\nОтвет: …\n'
                    'Или «отмена».'
                )
            q, a = pair
            self.pending_teach = None
            self.add_qa_pair(q, a, source='teach')
            if self.learning_manager:
                self.learning_manager.learn_from_text(
                    f'Вопрос: {q}\nОтвет: {a}',
                    source='teach',
                )
            return f'Запомнил.\nВопрос: {q}\nОтвет: {a}'

        if self._is_teach_reject(text):
            return self.make_teach_question('user_request')

        if len(text) < 5:
            return 'Слишком короткий ответ. Чуть разверни или скажи «пропусти» / «хватит».'

        self.pending_teach = None

        if kind == 'unknown_word' and prompt:
            pair = f'Слово: {prompt}\nЗначение: {text}'
            if self.learning_manager:
                self.learning_manager.learn_from_text(pair, source='teach_dialog')
            self.add_qa_pair(f'что значит {prompt}', text, source='teach')
            return 'Запомнил значение слова.'

        if prompt:
            if self.learning_manager:
                self.learning_manager.learn_from_text(
                    f'Вопрос: {prompt}\nОтвет: {text}',
                    source='teach_dialog',
                )
            self.add_qa_pair(prompt, text, source='teach')
            return 'Спасибо, запомнил. Так я становлюсь чуть понятливее.'

        return 'Не понял, что сохранять. Скажи «запомни» и пришли пару Вопрос/Ответ.'


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


    def save_user_statuses(self) -> None:
        """
        Сохранение статусов пользователей.
        """

        os.makedirs('data', exist_ok=True)
        debug_logger.debug('Сохранение статусов пользователей')
        assistant_logger.debug('Сохранение статусов пользователей')

        data = {
            'statuses': self.user_statuses,
            'creator_name': self.creator_name,
            'creator_aliases': getattr(self, 'creator_aliases', []) or (self.persona or {}).get('creator_aliases', []),
            'owner_name': self.owner_name,
            'preferred_name': self.preferred_name or self.creator_name,
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
        Базовое обучение: короткие Q&A под формат генерации.
        """

        training_logger.info('Начало базового обучения')

        examples = [
            'Вопрос: Кто ты?\nОтвет: Я Протос — искусственный интеллект. Я учусь понимать язык и помогать людям.',
            'Вопрос: Как тебя зовут?\nОтвет: Меня зовут Протос.',
            'Вопрос: Какова твоя директива?\nОтвет: Моя главная задача — учиться понимать язык, помнить контекст и помогать тебе.',
            'Вопрос: Что такое переобучение?\nОтвет: Переобучение — когда модель запоминает примеры вместо общих закономерностей и хуже работает на новых данных.',
            'Вопрос: Что такое обучение с учителем?\nОтвет: Обучение с учителем — когда модели дают входы и правильные ответы, и она учится этой связи.',
            'Вопрос: Зачем нужна память в диалоге?\nОтвет: Память нужна, чтобы помнить, о чём говорили, и вести связный разговор.',
            'Вопрос: Что ты умеешь?\nОтвет: Я учусь на диалогах и фактах, храню контекст и помогаю создателю.',
            'Вопрос: Ты человек?\nОтвет: Нет, я программа — локальный искусственный интеллект.',
        ]

        for example in examples:
            self.learning_manager.learn_from_text(example, source='basic_training')

        self.learning_manager.train(epochs=50, sequence_length=25)
        self.learning_manager.save_model('models/protos_lstm.json')
        training_logger.info('Базовое обучение завершено')


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

        text = (user_input or '').strip()
        if not text:
            lang = getattr(self, '_last_lang', 'ru') or 'ru'
            if lang == 'ru':
                response = 'Напиши что-нибудь — пустое сообщение я не разбираю.'
            else:
                response = "Please type something — empty messages aren't processed."
            dialog_logger.info(f'🤖 {response}')
            return {
                'response': response,
                'language': lang,
                'memory_size': len(self.memory),
                'is_learning': self.is_learning,
                'user_role': self.user_role,
                'current_user': self.current_speaker,
            }

        debug_logger.debug(f'Обработка ввода: "{user_input[:30]}..." (длина: {len(user_input)})')
        self.conversation_count += 1
        lang = self.text_processor.detect_language(user_input)

        pending = getattr(self, 'pending_login', None) or {}
        step = pending.get('step', '')
        is_password_step = step in (
            'ask_password',
            'register_password',
            'register_password2',
        )

        if is_password_step:
            pass
        else:
            assistant_logger.info(
                f'Ввод: {user_input[:50]}... (длина: {len(user_input)})'
            )
            dialog_logger.info(f'👤 {user_input}')

        assistant_logger.debug(
            f'Язык: {lang}, пользователь: {self.current_speaker or "неизвестен"}'
        )
        assistant_logger.debug(f'Разговоров: {self.conversation_count}')

        try:
            # 1. Команды с /
            if user_input.startswith('/'):
                debug_logger.debug(f'Обнаружена команда: {user_input}')
                cmd_body = user_input[1:].lstrip()
                if cmd_body and not (
                        cmd_body[0].isascii() and (cmd_body[0].isalpha() or cmd_body[0] == '_')
                ):
                    response = (
                        'Команда похоже повреждена при вводе. '
                        'Повторите, начиная с / и латинского имени (например /help).'
                    )
                else:
                    response = self.commands.handle(user_input)
                    if response is None:
                        response = (
                            'Неизвестная команда. Напишите /help — список доступных.'
                        )
                dialog_logger.info(f'🤖 {response}')
                return {
                    'response': response,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role,
                    'current_user': self.current_speaker,
                    'shutdown': str(response).startswith('__SHUTDOWN__'),
                }

            if getattr(self, 'pending_delete', None):
                response = self.commands.confirm_delete(user_input, lang)
                dialog_logger.info(f'🤖 {response}')
                return {
                    'response': response,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role,
                    'current_user': self.current_speaker,
                }

            if user_input.startswith('/'):
                debug_logger.debug(f'Обнаружена команда: {user_input}')
                response = self.commands.handle(user_input)
                if response:
                    dialog_logger.info(f'🤖 {response}')
                    return {
                        'response': response,
                        'language': lang,
                        'memory_size': len(self.memory),
                        'is_learning': self.is_learning,
                        'user_role': self.user_role,
                        'current_user': self.current_speaker,
                    }

            # 2. Команды обычной фразой
            nl_cmd = self.commands.match_nl(user_input)
            if nl_cmd:
                debug_logger.debug(f'NL-команда: {nl_cmd}')
                response = self.commands.handle(f'/{nl_cmd}')
                if response:
                    dialog_logger.info(f'🤖 {response}')
                    return {
                        'response': response,
                        'language': lang,
                        'memory_size': len(self.memory),
                        'is_learning': self.is_learning,
                        'user_role': self.user_role,
                        'current_user': self.current_speaker,
                    }

            # 3. Ожидаем секретную фразу
            if getattr(self, 'pending_auth', False):
                if self.auth.check_secret_phrase(user_input):
                    name = (
                            self.current_speaker
                            or self.user_name
                            or self.creator_name
                            or 'Создатель'
                    )
                    if self.is_creator_name(name) or name == self.creator_name:
                        name = self.canonical_creator_name()
                    self.apply_status(name, 'creator')
                    self.auth.open_session(name, 'creator')
                    self.pending_auth = False
                    assistant_logger.info(f'Сессия создателя открыта: {name}')
                    response = (
                        f'Фраза верная. Приветствую, {name}! '
                        f'Полный доступ открыт.'
                    )
                    dialog_logger.info(f'🤖 {response}')
                    return {
                        'response': response,
                        'language': lang,
                        'memory_size': len(self.memory),
                        'is_learning': self.is_learning,
                        'user_role': self.user_role,
                        'current_user': self.current_speaker,
                    }

                self.pending_auth = False
                self.session_authenticated = False
                self.current_speaker = None
                self.user_name = None
                self.user_role = 'user'
                self.pending_login = {'step': 'ask_name'}
                response = (
                    'Фраза неверная. Доступ не подтверждён.\n'
                    'Назовите имя снова или скажите «я создатель» / «конец сеанса».'
                )
                dialog_logger.info(f'🤖 {response}')
                return {
                    'response': response,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role,
                    'current_user': None,
                }

            # 3.1 Подтверждение нового имени («да» / «нет» после «Зови меня X»)
            if getattr(self, 'pending_rename', None):
                response = self._handle_rename_confirm(user_input, lang)
                if response:
                    dialog_logger.info(f'🤖 {response}')
                    return {
                        'response': response,
                        'language': lang,
                        'memory_size': len(self.memory),
                        'is_learning': self.is_learning,
                        'user_role': self.user_role,
                        'current_user': self.current_speaker,
                    }

            # 3.2 «Зови меня …» / «меня также зовут …»
            alias_reply = self._handle_add_creator_alias(user_input, lang)
            if alias_reply:
                dialog_logger.info(f'🤖 {alias_reply}')
                return {
                    'response': alias_reply,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role,
                    'current_user': self.current_speaker,
                }

            # 3.3 Выход из сессии (не закрытие программы)
            low = user_input.lower().strip()
            low = re.sub(r'[.!?…]+$', '', low).strip()
            if low in (
                    'логаут',
                    'разлогин',
                    'разлогиниться',
                    'выйти из учётки',
                    'выйти из учетки',
                    'смени пользователя',
                    'это не я',
                    'logout',
                    'log out',
            ):
                response = self.auth.logout(lang)
                tail = self.auth.start_login_flow(lang)
                full = f'{response}\n{tail}'
                dialog_logger.info(f'🤖 {full}')
                return {
                    'response': full,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role,
                    'current_user': self.current_speaker,
                }

            # 3.4 Шаги входа (имя → статус → пароль / регистрация)
            if self.pending_login is not None:
                response = self.auth.handle_login_step(user_input, lang)
                shutdown = isinstance(response, str) and response.startswith('__SHUTDOWN__')
                if shutdown:
                    response = response[len('__SHUTDOWN__'):]
                dialog_logger.info(f'🤖 {response}')
                return {
                    'response': response,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role,
                    'current_user': self.current_speaker,
                    'shutdown': shutdown,
                }

            # 3.5 Ещё не вошли — не пускаем в обычный диалог
            if not self.is_logged_in():
                if self.pending_login is None:
                    self.pending_login = {'step': 'ask_name'}
                response = self.auth.handle_login_step(user_input, lang)
                shutdown = isinstance(response, str) and response.startswith('__SHUTDOWN__')
                if shutdown:
                    response = response[len('__SHUTDOWN__'):]
                dialog_logger.info(f'🤖 {response}')
                return {
                    'response': response,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role,
                    'current_user': self.current_speaker,
                    'shutdown': shutdown,
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

            # 4.1 Явный урок для Протоса
            if self._wants_to_teach(user_input):

                if self.user_role not in ('creator', 'owner'):
                    response = 'Записывать знания в мою базу может только создатель (или владелец).'
                    dialog_logger.info(f'🤖 {response}')
                    return {
                        'response': response,
                        'language': lang,
                        'memory_size': len(self.memory),
                        'is_learning': self.is_learning,
                        'user_role': self.user_role,
                        'current_user': self.current_speaker,
                    }

                response = self._start_teach_lesson(user_input, lang)
                dialog_logger.info(f'🤖 {response}')
                return {
                    'response': response,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role,
                    'current_user': self.current_speaker,
                }

            # 5. Просьба «задай вопрос / поучимся»
            if self._wants_to_be_asked(user_input):
                response = self.make_teach_question('user_request')
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
            response = self.router.route(
                user_input=user_input,
                context=context,
                lang=lang,
            )

            shutdown = False
            if isinstance(response, str) and response.startswith('__SHUTDOWN__'):
                shutdown = True
                response = response[len('__SHUTDOWN__'):]

            dialog_logger.info(f'🤖 {response}')

            # Память
            speaker = self.current_speaker or self.user_name or 'Кто-то'
            self.memory.remember_message(
                f'{speaker}: {user_input}',
                importance=0.2,
            )
            self.memory.remember_message(
                f'Протос: {response}',
                importance=0.2,
            )

            # Автообучение
            layer = getattr(self, '_last_answer_layer', '') or ''

            if layer in (
                    'lstm', 'fallback', 'lstm_forced', 'lstm_empty',
                    'teach', 'greeting', 'farewell', 'help', 'system',
                    'qa', 'facts', 'persona', 'identity',
            ):
                pass

            elif self.is_learning and self.learning_manager and not getattr(self, 'gen_only', False):
                if self._should_learn_pair(user_input, response):
                    knowledge = self._extract_knowledge(user_input, response)
                    if knowledge:
                        debug_logger.debug(f'Извлечено знание: {knowledge[:50]}...')
                        self._save_to_knowledge_base(knowledge)

                    self.learning_manager.learn_from_text(
                        f'Пользователь: {user_input}\nПротос: {response}',
                        source='conversation',
                    )

                    # if not user_input.startswith('/'):
                    #     self.dialogue_learner.learn_from_conversation(user_input, response)

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
                'current_user': self.current_speaker,
                'shutdown': shutdown,
            }

        except Exception as e:
            error_logger.exception(f'Ошибка обработки ввода: {e}')
            raise


    def _start_teach_lesson(self, user_input: str, lang: str) -> str:
        """
        Старт записи знания.
        """

        pair = self._parse_teach_pair(user_input)
        if pair:
            q, a = pair
            self.add_qa_pair(q, a, source='teach')
            if self.learning_manager:
                self.learning_manager.learn_from_text(
                    f'Вопрос: {q}\nОтвет: {a}' if lang == 'ru'
                    else f'Question: {q}\nAnswer: {a}',
                    source='teach',
                )
            self.pending_teach = None
            if lang == 'ru':
                return f'Запомнил.\nВопрос: {q}\nОтвет: {a}'
            return f'Saved.\nQuestion: {q}\nAnswer: {a}'

        self.pending_teach = {'type': 'explicit_qa', 'step': 'await_pair'}
        if lang == 'ru':
            return (
                'Хорошо, записываю.\n'
                'Пришли в одном сообщении:\n'
                'Вопрос: …\nОтвет: …\n'
                'Или «отмена».'
            )
        return (
            'Alright, I will record it.\n'
            'Send in one message:\n'
            'Question: …\nAnswer: …\n'
            "Or 'cancel'."
        )


    @staticmethod
    def _parse_teach_pair(text: str) -> tuple[str, str] | None:
        """
        Извлечение вопроса и ответа из текста урока.
        """

        m = re.search(
            r'(?:вопрос|question)\s*:\s*(.+?)\s*(?:ответ|answer)\s*:\s*(.+)',
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return None
        q = m.group(1).strip().strip(' \n')
        a = m.group(2).strip().strip(' \n')
        if len(q) < 2 or len(a) < 2:
            return None
        return q, a


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

        if r_low.startswith('(lstm') or 'ничего внятного' in r_low:
            return False

        if 'уже отвечал' in r_low or 'уже задавал' in r_low:
            return False

        return True


    @staticmethod
    def _extract_knowledge(user_input: str, response: str) -> str | None:
        """
        Извлечение знаний из диалога.
        Только явные факты вида «это …», не каждый вопрос с ?.
        """

        u = (user_input or '').strip()
        r = (response or '').strip()
        if len(u) < 5 or len(r) < 5:
            return None

        r_low = r.lower()
        junk = (
            'привет',
            'рад тебя',
            'рад, что ты',
            'расскажи',
            'lstm',
            'ничего внятного',
            'вопрос:',
            'как дела',
        )
        if any(j in r_low for j in junk):
            return None

        fact_match = re.search(
            r'(?:это|значит)\s+([^.!?]{5,120})',
            u,
            flags=re.IGNORECASE,
        )
        if fact_match:
            return f'Знание: {fact_match.group(1).strip()}'

        return None


    def _save_to_knowledge_base(self, knowledge: str) -> None:
        """
        Сохранение знания в базу (без дубликатов).
        """

        knowledge = (knowledge or '').strip()
        if not knowledge:
            return

        kb_path = 'data/knowledge_base.json'

        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    kb = json.load(f)
            except (json.JSONDecodeError, IOError):
                kb = {}
        else:
            kb = {}

        if 'learned' not in kb or not isinstance(kb.get('learned'), list):
            kb['learned'] = []

        norm = ' '.join(knowledge.lower().split())
        for item in kb['learned']:
            existing = (item.get('knowledge') or '')
            if ' '.join(existing.lower().split()) == norm:
                return

        kb['learned'].append({
            'knowledge': knowledge,
            'timestamp': time.time(),
            'source': 'conversation',
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

        # алиас создателя → каноническое имя
        if name:
            name = self._resolve_to_canonical_user(name)

        status = self._extract_status_from_text(text_lower)
        if not status and 'создатель' in text_lower:
            status = 'creator'
        if name and self.is_creator_name(name):
            status = status or 'creator'

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
                'aliases': [],
            }
        elif status:
            self.known_users[name]['status'] = status

        if rel_type and rel_name:
            self.known_users[name].setdefault('relations', {})[rel_type] = rel_name
            self._add_relationship(name, rel_type, rel_name)
            debug_logger.debug(f'Добавлена связь: {name} -> {rel_type}: {rel_name}')

        self.current_speaker = name
        self.user_name = name

        if status == 'creator' and self.user_role != 'creator':
            self.creator_name = self.canonical_creator_name()
            self.save_user_statuses()
            return self.auth.request_auth(lang)

        if status:
            self.apply_status(name, status)

        self.save_user_statuses()
        debug_logger.debug(f'Пользователь {name} сохранён')

        if lang == 'ru':
            if status == 'creator' or self.is_creator_name(name):
                return f'Привет, {name}! Я ждал тебя. Ты мой создатель!'
            if rel_type and rel_name:
                return f'А, так ты {rel_type} {rel_name}! Приятно познакомиться, {name}!'
            return f'Привет, {name}! Рад познакомиться.'

        if status == 'creator' or self.is_creator_name(name):
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
        low = text.lower()

        for alias in self._creator_names():
            if alias and alias in low:
                canonical = self.canonical_creator_name()
                debug_logger.debug(f'Говорящий (создатель/алиас): {canonical}')
                return canonical

        for user in self.known_users.keys():
            if user.lower() in low:
                resolved = self._resolve_to_canonical_user(user)
                debug_logger.debug(f'Говорящий определен по контексту: {resolved}')
                return resolved

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


    def apply_status(self, name: str, status: str) -> None:
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
        self.save_user_statuses()
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


    def get_relationship_info(self, name: str) -> str | None:
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


    def query_knowledge_base(self, question: str) -> str | None:
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


    def generate_with_lstm(self, user_input: str, context: list[str], lang: str) -> str:
        """
        Генерация ответа с помощью LSTM.
        """

        if self.learning_manager is None:
            debug_logger.debug('LearningManager не инициализирован')
            return ''

        q = user_input.strip()
        if lang == 'ru':
            prompt = f'Вопрос: {q}\nОтвет:'
        else:
            prompt = f'Question: {q}\nAnswer:'

        debug_logger.debug(f'Промпт для LSTM: {prompt[:80]}...')
        generated = self.learning_manager.generate_sample(prompt, length=40)

        text = generated.strip()
        for stop in ('\nВопрос:', '\nQuestion:', '\nПользователь:'):
            if stop in text:
                text = text.split(stop, 1)[0].strip()

        if '.' in text:
            text = text.split('.', 1)[0].strip() + '.'
        elif len(text) > 120:
            text = text[:120].rsplit(' ', 1)[0] + '…'

        debug_logger.debug(f'Сгенерировано: {text[:50]}...')
        debug_logger.debug(f'LSTM raw: {generated!r}')
        return text


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
