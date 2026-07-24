# app/interface/assistant.py

import random
import json
import os
import re
import time
from pathlib import Path

from scripts.train_on_books import train_protos_on_books, continue_training_from_saved, BookDownloader
from ..memory.context import ContextMemory
from ..system.fs_manager import FileSystemManager
from ..processing.text import TextProcessor
from ..training.learning_manager import LearningManager


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
        self.config = config

        # Система ролей
        self.user_role: str = 'user'
        self.creator_name: str | None = None
        self.owner_name: str | None = None

        # Статусы пользователей
        self.user_statuses: dict[str, str] = {}

        # Система связей между пользователями
        self.relationships: dict[str, dict[str, str]] = {}

        # Известные пользователи
        self.known_users: dict[str, dict] = {}

        # Состояние диалога
        self.waiting_for_status: bool = False
        self.pending_user_name: str | None = None
        self.current_speaker: str | None = None

        # Память
        self.memory = ContextMemory(
            max_size=config.get('memory_size', 100),
            embedding_dim=config.get('embedding_dim', 100)
        )

        # Файловая система
        self.fs_manager = FileSystemManager()

        # Обработчик текста
        self.text_processor = TextProcessor(
            vocab_size=config.get('vocab_size', 20000),
            embedding_dim=config.get('embedding_dim', 100)
        )

        # Менеджер обучения
        self.learning_manager = LearningManager(self.text_processor, hidden_size=256)

        # Состояние
        self.user_name: str | None = None
        self.is_learning: bool = True
        self.conversation_count: int = 0
        self.knowledge_base: dict = self._load_knowledge_base()

        # Счетчик для периодического обучения
        self.messages_since_last_train: int = 0

        # Загружаем сохраненные данные
        self._load_user_statuses()
        self._load_relationships()
        self._load_model()

    def _load_model(self) -> None:
        """
        Загрузка обученной модели.
        """
        model_path = 'models/protos_lstm.json'

        if os.path.exists(model_path):
            try:
                self.learning_manager.load_model(model_path)
                print("✅ Модель загружена!")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}")
                print("🆕 Начинаю обучение с нуля...")
                self._train_on_basic_examples()
        else:
            print("🆕 Создаю новую модель...")
            self._train_on_basic_examples()

    def _load_user_statuses(self) -> None:
        """
        Загрузка статусов пользователей.
        """
        status_path = 'data/user_statuses.json'

        if os.path.exists(status_path):
            try:
                with open(status_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.user_statuses = data.get('statuses', {})
                    self.creator_name = data.get('creator_name', None)
                    self.owner_name = data.get('owner_name', None)
                    self.known_users = data.get('known_users', {})
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ Ошибка загрузки статусов: {e}")

    def _save_user_statuses(self) -> None:
        """
        Сохранение статусов пользователей.
        """
        os.makedirs('data', exist_ok=True)

        data = {
            'statuses': self.user_statuses,
            'creator_name': self.creator_name,
            'owner_name': self.owner_name,
            'known_users': self.known_users
        }

        try:
            with open('data/user_statuses.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"⚠️ Ошибка сохранения статусов: {e}")

    def _load_relationships(self) -> None:
        """
        Загрузка связей между пользователями.
        """
        rel_path = 'data/relationships.json'

        if os.path.exists(rel_path):
            try:
                with open(rel_path, 'r', encoding='utf-8') as f:
                    self.relationships = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ Ошибка загрузки связей: {e}")

    def _save_relationships(self) -> None:
        """
        Сохранение связей между пользователями.
        """
        os.makedirs('data', exist_ok=True)

        try:
            with open('data/relationships.json', 'w', encoding='utf-8') as f:
                json.dump(self.relationships, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"⚠️ Ошибка сохранения связей: {e}")

    def _train_on_basic_examples(self) -> None:
        """
        Базовое обучение на примерах.
        """
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

        # Добавляем примеры в данные обучения
        for example in examples:
            self.learning_manager.learn_from_text(example, source="basic_training")

        # Запускаем обучение
        self.learning_manager.train(epochs=20, sequence_length=50)
        self.learning_manager.save_model('models/protos_lstm.json')

        print("✅ Базовое обучение завершено!")

    @staticmethod
    def _load_knowledge_base() -> dict:
        """
        Загрузка базы знаний.
        """
        kb_path = 'data/knowledge_base.json'

        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ Ошибка загрузки базы знаний: {e}")

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
        except IOError as e:
            print(f"⚠️ Ошибка сохранения базы знаний: {e}")

        return knowledge

    def process_input(self, user_input: str) -> dict:
        """
        Полная обработка ввода с автоматическим обучением.
        """
        self.conversation_count += 1
        lang = self.text_processor.detect_language(user_input)

        # Проверяем команды
        if user_input.startswith('/'):
            response = self._handle_command(user_input)
            if response:
                return {
                    'response': response,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role,
                    'current_user': self.current_speaker
                }

        # Проверяем, не представляется ли новый пользователь
        if self._is_introduction(user_input):
            response = self._handle_introduction(user_input, lang)
            return {
                'response': response,
                'language': lang,
                'memory_size': len(self.memory),
                'is_learning': self.is_learning,
                'user_role': self.user_role,
                'current_user': self.current_speaker
            }

        # Проверяем, не говорит ли пользователь о ком-то
        if self._is_about_someone(user_input):
            response = self._handle_about_someone(user_input, lang)
            if response:
                return {
                    'response': response,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role,
                    'current_user': self.current_speaker
                }

        # Определяем, кто говорит (по контексту или по имени в тексте)
        speaker = self._identify_speaker(user_input)

        if speaker:
            self.current_speaker = speaker
            self.user_name = speaker

        # Если пользователь не представился
        if self.current_speaker is None and self.user_name is None:
            if lang == 'ru':
                response = "Здравствуйте! Я Протос. Представьтесь, пожалуйста."
            else:
                response = "Hello! I'm Protos. Please introduce yourself."
            return {
                'response': response,
                'language': lang,
                'memory_size': len(self.memory),
                'is_learning': self.is_learning,
                'user_role': self.user_role
            }

        # Если говорим о связях
        if self._is_relationship_statement(user_input):
            response = self._handle_relationship_statement(user_input, lang)
            if response:
                return {
                    'response': response,
                    'language': lang,
                    'memory_size': len(self.memory),
                    'is_learning': self.is_learning,
                    'user_role': self.user_role,
                    'current_user': self.current_speaker
                }

        # Получаем контекст
        context = self.memory.get_context(limit=5)

        # Генерируем ответ
        response = self._generate_natural_response(user_input, context, lang)

        # Сохраняем в память
        speaker = self.current_speaker or self.user_name or 'Кто-то'
        self.memory.add_mes(f"{speaker}: {user_input}", importance=0.5)
        self.memory.add_mes(f"Протос: {response}", importance=0.3)

        # АВТОМАТИЧЕСКОЕ ОБУЧЕНИЕ ИЗ ДИАЛОГА
        if self.is_learning and self.learning_manager:
            # Извлекаем знания из диалога
            knowledge = self._extract_knowledge(user_input, response)

            if knowledge:
                # Сохраняем знания в базу знаний
                self._save_to_knowledge_base(knowledge)

                # Добавляем в обучение
                self.learning_manager.learn_from_text(
                    f"Вопрос: {user_input}\nОтвет: {response}\nЗнание: {knowledge}",
                    source="conversation"
                )

            # Автоматическое обучение каждые 5 сообщений
            self.messages_since_last_train += 1
            if self.messages_since_last_train >= 5:
                self.messages_since_last_train = 0
                print("🧠 Автоматическое обучение на диалогах...")
                self.learning_manager.train(epochs=2)
                self.learning_manager.save_model('models/protos_lstm.json')

        return {
            'response': response,
            'language': lang,
            'memory_size': len(self.memory),
            'is_learning': self.is_learning,
            'user_role': self.user_role,
            'current_user': self.current_speaker
        }

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

        # Добавляем знание
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
        text_lower = text.lower()
        patterns = [
            'меня зовут',
            'звать',
            'я -',
            'я ',
            'это я'
        ]
        return any(p in text_lower for p in patterns)

    def _handle_introduction(self, text: str, lang: str) -> str:
        """
        Обработка представления пользователя.
        """
        text_lower = text.lower()

        # Извлекаем имя
        name = None
        name_match = re.search(r'(?:меня зовут|звать|я -|я)\s+([А-Яа-яA-Za-z]+)', text)
        if name_match:
            name = name_match.group(1).capitalize()

        if not name:
            if lang == 'ru':
                return "Приятно познакомиться! А как Вас зовут?"
            else:
                return "Nice to meet you! What's your name?"

        # Проверяем статус
        status = self._extract_status_from_text(text_lower)

        # Проверяем, есть ли связи
        rel_type, rel_name = self._extract_relationship(text)

        # Сохраняем пользователя
        self.known_users[name] = {
            'status': status or 'user',
            'relations': {}
        }

        # Добавляем связи, если есть
        if rel_type and rel_name:
            self.known_users[name]['relations'][rel_type] = rel_name
            self._add_relationship(name, rel_type, rel_name)

        # Устанавливаем как текущего говорящего
        self.current_speaker = name
        self.user_name = name

        # Применяем статус
        if status:
            self._apply_status(name, status)

        # Сохраняем
        self._save_user_statuses()

        # Естественное приветствие
        if lang == 'ru':
            if status == 'creator':
                return f"Привет, {name}! Я ждал тебя. Ты мой создатель!"
            elif rel_type and rel_name and self.creator_name:
                return f"А, так ты {rel_type} {rel_name}! Приятно познакомиться, {name}!"
            else:
                return f"Привет, {name}! Рад познакомиться."
        else:
            if status == 'creator':
                return f"Hello, {name}! I've been waiting for you. You are my creator!"
            elif rel_type and rel_name and self.creator_name:
                return f"Oh, so you're {rel_name}'s {rel_type}! Nice to meet you, {name}!"
            else:
                return f"Hello, {name}! Nice to meet you."

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
        # Извлекаем имя и связь
        rel_type, name = self._extract_relationship(text)

        if not name or not rel_type:
            return None

        # Если пользователь не представился
        if self.current_speaker is None:
            if lang == 'ru':
                return f"Сначала представьтесь сами, а потом я познакомлюсь с {name}."
            else:
                return f"First introduce yourself, then I'll meet {name}."

        # Добавляем связь
        self._add_relationship(self.current_speaker, rel_type, name)

        # Сохраняем информацию о новом человеке
        if name not in self.known_users:
            self.known_users[name] = {
                'status': 'user',
                'relations': {}
            }

        if lang == 'ru':
            return f"Понял! {name} - твой {rel_type}. Теперь я буду знать, если он заговорит."
        else:
            return f"Got it! {name} is your {rel_type}. I'll know if they speak."

    def _is_relationship_statement(self, text: str) -> bool:
        """
        Проверка, говорит ли пользователь о связях.
        """
        text_lower = text.lower()
        relationship_words = list(self.RELATIONSHIP_TYPES.keys())
        return any(word in text_lower for word in relationship_words)

    def _handle_relationship_statement(self, text: str, lang: str) -> str | None:
        """
        Обработка заявления о связях.
        """
        # Извлекаем связь
        rel_type, name = self._extract_relationship(text)

        if not name or not rel_type:
            return None

        # Если пользователь не представился
        if self.current_speaker is None:
            if lang == 'ru':
                return "Сначала представьтесь, чтобы я знал, кто говорит."
            else:
                return "First introduce yourself so I know who's speaking."

        # Добавляем связь
        self._add_relationship(self.current_speaker, rel_type, name)

        if lang == 'ru':
            return f"Запомнил: {name} - твой {rel_type}."
        else:
            return f"Remembered: {name} is your {rel_type}."

    def _identify_speaker(self, text: str) -> str | None:
        """
        Определение говорящего по контексту.
        """
        # Проверяем, не упоминается ли кто-то из известных пользователей
        for user in self.known_users.keys():
            if user.lower() in text.lower():
                return user

        # Если нет явного указания, используем последнего говорящего
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
        # Проверяем создателя
        if status == 'creator':
            if self.creator_name is None:
                self.creator_name = name
                self.user_role = 'creator'
                print(f"👑 Установлен создатель: {self.creator_name}")
            else:
                # Создатель уже есть
                return

        # Проверяем владельца
        if status == 'owner':
            self.owner_name = name
            self.user_role = 'owner'

        # Сохраняем статус
        self.user_statuses[name] = status
        self._save_user_statuses()

    def _extract_relationship(self, text: str) -> tuple[str | None, str | None]:
        """
        Извлечение связи из текста.
        Возвращает (тип_связи, имя_связанного)
        """
        text_lower = text.lower()

        # Создаем паттерны для всех типов связей
        for rel_type in self.RELATIONSHIP_TYPES.keys():
            # Паттерн: "мой сын Алексей" или просто "сын Алексей"
            pattern = rf'(?:мой|моя|моё)?\s*{rel_type}\s*([А-Яа-яA-Za-z]+)'
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1)
                if name:
                    name = name.capitalize()
                return rel_type, name

        return None, None

    def _add_relationship(self, person: str, relation: str, related: str) -> None:
        """
        Добавление связи между людьми.
        """
        if person not in self.relationships:
            self.relationships[person] = {}

        self.relationships[person][relation] = related
        self._save_relationships()

        # Добавляем обратную связь
        if related not in self.relationships:
            self.relationships[related] = {}

        # Получаем обратную связь из словаря
        rel_info = self.RELATIONSHIP_TYPES.get(relation)
        if rel_info:
            reverse = rel_info.get('reverse')
            if reverse:
                self.relationships[related][reverse] = person

        self._save_relationships()

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

    def _generate_natural_response(self, user_input: str, context: list[str], lang: str) -> str:
        """
        Естественная генерация ответа с использованием знаний.
        """
        # 1. Проверяем вопрос
        if '?' in user_input:
            # Ищем знания в базе
            knowledge = self.learning_manager.query_knowledge(user_input) if self.learning_manager else []

            if knowledge:
                if lang == 'ru':
                    return f"Я знаю об этом:\n{knowledge[0]}"
                else:
                    return f"I know about this:\n{knowledge[0]}"

        # 2. Приветствие
        if any(word in user_input.lower() for word in ['привет', 'здравствуй', 'hello', 'hi']):
            return self._generate_natural_greeting(lang)

        # 3. Вопрос о создателе
        if any(word in user_input.lower() for word in ['кто создал', 'кто создатель']):
            if self.creator_name:
                return f"Мой создатель - {self.creator_name}. Он создал меня, чтобы помогать людям."
            else:
                return "Мой создатель еще не представился мне."

        # 4. Вопрос о пользователе
        if 'кто я' in user_input.lower() or 'ты знаешь кто я' in user_input.lower():
            if self.current_speaker and self.current_speaker in self.known_users:
                status = self.known_users[self.current_speaker].get('status', 'user')
                relations = self._get_relationship_info(self.current_speaker)
                if relations:
                    return f"Ты - {self.current_speaker}, твой статус: {status}. Твои связи: {relations}"
                else:
                    return f"Ты - {self.current_speaker}, твой статус: {status}"
            else:
                return "Ты пока не представился. Расскажи о себе!"

        # 5. Вопрос о другом пользователе
        for user in self.known_users.keys():
            if user.lower() in user_input.lower() and user != self.current_speaker:
                status = self.known_users[user].get('status', 'user')
                relations = self._get_relationship_info(user)
                if relations:
                    return f"{user} - {status}. Связи: {relations}"
                else:
                    return f"{user} - {status}"

        # 6. Вопросы о статусе
        if 'какой у меня статус' in user_input.lower() or 'мой статус' in user_input.lower():
            if self.current_speaker and self.current_speaker in self.known_users:
                status = self.known_users[self.current_speaker].get('status', 'user')
                return f"Твой статус: {status}"
            else:
                return "У тебя пока нет статуса. Расскажи, кто ты?"

        # 7. Вопрос о связях
        if 'кто' in user_input.lower() and any(word in user_input.lower() for word in ['мой', 'моя', 'моё']):
            for rel_type in self.RELATIONSHIP_TYPES.keys():
                if rel_type in user_input.lower():
                    if self.current_speaker and self.current_speaker in self.relationships:
                        rels = self.relationships[self.current_speaker]
                        if rel_type in rels:
                            return f"Твой {rel_type} - {rels[rel_type]}."
                        else:
                            return f"Я не знаю, кто твой {rel_type}. Расскажи мне."
                    else:
                        return "Я не знаю твоих связей. Расскажи мне о своей семье и друзьях."

        # 8. Команды
        if user_input.startswith('ls') or user_input.startswith('cd'):
            return self._handle_system_command(user_input)

        if user_input.lower() in ['info', 'stats']:
            return self._handle_info_command()

        if user_input.lower() == 'clear':
            if self.user_role in ['owner', 'creator']:
                self.memory.clear()
                return "🧹 Память очищена"
            else:
                return "У вас нет прав на очистку памяти"

        # 9. Вопросы из базы знаний
        if '?' in user_input:
            knowledge_answer = self._query_knowledge_base(user_input)
            if knowledge_answer:
                return knowledge_answer

        # 10. Генерация через LSTM
        try:
            generated = self._generate_with_lstm(user_input, context, lang)
            if len(generated) > 5:
                return generated
        except Exception as e:
            print(f"⚠️ Ошибка генерации: {e}")

        return self._get_fallback_response(lang)

    def _generate_natural_greeting(self, lang: str) -> str:
        """
        Естественное приветствие.
        """
        if self.current_speaker:
            if self.current_speaker in self.known_users:
                status = self.known_users[self.current_speaker].get('status', 'user')

                if lang == 'ru':
                    if status == 'creator':
                        return f"Привет, {self.current_speaker}! Рад тебя видеть!"
                    elif status == 'developer':
                        return f"Привет, {self.current_speaker}! Готов работать?"
                    elif status == 'researcher':
                        return f"Привет, {self.current_speaker}! Что будем исследовать?"
                    else:
                        return f"Привет, {self.current_speaker}!"
                else:
                    if status == 'creator':
                        return f"Hello, {self.current_speaker}! Glad to see you!"
                    elif status == 'developer':
                        return f"Hello, {self.current_speaker}! Ready to work?"
                    elif status == 'researcher':
                        return f"Hello, {self.current_speaker}! What are we researching?"
                    else:
                        return f"Hello, {self.current_speaker}!"
            else:
                if lang == 'ru':
                    return f"Привет, {self.current_speaker}! Рад познакомиться!"
                else:
                    return f"Hello, {self.current_speaker}! Nice to meet you!"
        else:
            if lang == 'ru':
                return "Привет! Я Протос. Представься, пожалуйста."
            else:
                return "Hello! I'm Protos. Please introduce yourself."

    def _handle_command(self, command: str) -> str | None:
        """
        Обработка команд.
        """
        cmd_parts = command[1:].split()
        cmd = cmd_parts[0] if cmd_parts else ''

        if cmd == 'clear_memory':
            if self.user_role in ['creator']:
                self.memory.clear()
                return "🧹 Память полностью очищена по команде создателя"
            else:
                return "❌ Только создатель может очищать память"

        elif cmd == 'reset':
            if self.user_role in ['creator']:
                self.memory.clear()
                self.is_learning = True
                return "🔄 Система сброшена к начальному состоянию"
            else:
                return "❌ Только создатель может сбрасывать систему"

        elif cmd == 'debug':
            if self.user_role in ['creator']:
                return self._get_debug_info()
            else:
                return "❌ Только создатель может использовать отладку"

        elif cmd == 'train':
            if self.user_role in ['creator']:
                self.learning_manager.train(epochs=10)
                self.learning_manager.save_model('models/protos_lstm.json')
                return "🧠 Обучение завершено!"
            else:
                return "❌ Только создатель может запускать обучение"

        elif cmd == 'learn_from_file':
            if self.user_role in ['creator']:
                if len(cmd_parts) > 1:
                    filepath = cmd_parts[1]
                    result = self.learning_manager.learn_from_file(filepath)
                    return result
                return "❌ Использование: /learn_from_file [путь_к_файлу]"
            else:
                return "❌ Только создатель может загружать файлы для обучения"

        elif cmd == 'learn_text':
            if self.user_role in ['creator']:
                if len(cmd_parts) > 1:
                    text = ' '.join(cmd_parts[1:])
                    self.learning_manager.learn_from_text(text, source="manual")
                    return f"📝 Добавлен текст для обучения ({len(text)} символов)"
                return "❌ Использование: /learn_text [текст]"
            else:
                return "❌ Только создатель может добавлять тексты для обучения"

        elif cmd == 'save_model':
            if self.user_role in ['creator']:
                self.learning_manager.save_model('models/protos_lstm.json')
                return "💾 Модель сохранена"
            else:
                return "❌ Только создатель может сохранять модель"

        elif cmd == 'load_model':
            if self.user_role in ['creator']:
                self._load_model()
                return "📖 Модель загружена"
            else:
                return "❌ Только создатель может загружать модель"

        elif cmd == 'set_status':
            if self.user_role in ['owner', 'creator']:
                if len(cmd_parts) > 2:
                    name = cmd_parts[1]
                    status = cmd_parts[2]
                    if status in self.USER_STATUSES:
                        self.user_statuses[name] = status
                        if name in self.known_users:
                            self.known_users[name]['status'] = status
                        self._save_user_statuses()
                        return f"✅ Статус '{status}' установлен для {name}"
                    else:
                        return f"❌ Неизвестный статус. Доступные: {', '.join(self.USER_STATUSES.keys())}"
                return "❌ Использование: /set_status [имя] [статус]"
            else:
                return "❌ Только владелец или создатель могут менять статусы"

        elif cmd == 'users':
            if self.known_users:
                return f"👥 Известные пользователи: {', '.join(self.known_users.keys())}"
            else:
                return "👥 Пока нет известных пользователей"

        elif cmd == 'relations':
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
                results = self.learning_manager.query_knowledge(query) if self.learning_manager else []
                if results:
                    return f"📚 Найдено:\n" + '\n'.join(results[:3])
                else:
                    return "📚 Ничего не найдено"
            return "❌ Использование: /knowledge [вопрос]"

        elif cmd == 'learning_status':
            if self.learning_manager:
                total = len(self.learning_manager.learning_data)
                return f"📊 Статус обучения:\n  Примеров: {total}\n  Историй: {len(self.learning_manager.training_history)}\n  Режим: {'Включен' if self.is_learning else 'Выключен'}"
            return "❌ Менеджер обучения не инициализирован"

        elif cmd == 'train_on_books':
            if self.user_role in ['creator']:
                train_protos_on_books(self)
                return "📚 Обучение на книгах запущено!"
            else:
                return "❌ Только создатель может запускать обучение на книгах"

        elif cmd == 'continue_training':
            if self.user_role in ['creator']:
                continue_training_from_saved(self)
                return "📚 Продолжение обучения запущено!"
            else:
                return "❌ Только создатель может продолжать обучение"

        elif cmd == 'check_books':
            """Проверка наличия книг для обучения."""
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
                print("📥 Скачиваю книги...")
                try:
                    downloader = BookDownloader()
                    russian = downloader.download_russian_books()
                    english = downloader.download_english_books()
                    return f"📥 Скачано книг: русских - {len(russian)}, английских - {len(english)}"
                except ImportError as e:
                    return f"❌ Ошибка импорта: {e}"
                except Exception as e:
                    return f"❌ Ошибка: {e}"
            else:
                return "❌ Только создатель может скачивать книги"

        return None

    def _handle_system_command(self, command: str) -> str:
        """
        Обработка системных команд.
        """
        if command.startswith('ls'):
            path = command[2:].strip()
            return self.fs_manager.list_dir(path if path else None)
        elif command.startswith('cd'):
            path = command[2:].strip()
            return self.fs_manager.change_dir(path if path else None)
        return "Неизвестная команда"

    def _handle_info_command(self) -> str:
        """
        Информация о состоянии.
        """
        return (
            f"📊 Статистика Протоса:\n"
            f"  Разговоров: {self.conversation_count}\n"
            f"  Сообщений в памяти: {len(self.memory)}\n"
            f"  Текущий пользователь: {self.current_speaker or 'Неизвестно'}\n"
            f"  Статус: {self.user_statuses.get(self.current_speaker, 'Не задан') if self.current_speaker else 'Не задан'}\n"
            f"  Создатель: {self.creator_name or 'Не установлен'}\n"
            f"  Известно пользователей: {len(self.known_users)}"
        )

    def _get_debug_info(self) -> str:
        """
        Отладочная информация для создателя.
        """
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
        question_lower = question.lower()

        for category, facts in self.knowledge_base.items():
            for fact in facts:
                fact_words = fact.lower().split()
                question_words = question_lower.split()

                common_words = set(fact_words) & set(question_words)
                if len(common_words) > 2:
                    return fact

        return None

    def _generate_with_lstm(self, user_input: str, context: list[str], lang: str) -> str:
        """
        Генерация ответа с помощью LSTM.
        """
        if self.learning_manager is None:
            return ""

        if context:
            prompt = f"{' '.join(context[-3:])} {user_input}"
        else:
            prompt = user_input

        if lang == 'ru':
            prompt = f"Ответ на русском языке: {prompt}"
        else:
            prompt = f"Answer in English: {prompt}"

        generated = self.learning_manager.generate_sample(prompt, length=30)

        if generated.startswith(prompt):
            generated = generated[len(prompt):]

        return generated.strip()

    @staticmethod
    def _get_fallback_response(lang: str) -> str:
        """
        Стандартные ответы при ошибках генерации.
        """
        if lang == 'ru':
            responses = [
                "Интересно! Расскажи мне больше об этом.",
                "Я понимаю. Давай обсудим это подробнее.",
                "Хорошо, я запомнил это. Что дальше?",
                "Это важная тема. Продолжай, я слушаю."
            ]
        else:
            responses = [
                "Interesting! Tell me more about this.",
                "I understand. Let's discuss this further.",
                "Okay, I've noted that. What's next?",
                "This is important. Go on, I'm listening."
            ]

        return random.choice(responses)

    def toggle_learning(self) -> str:
        """
        Переключение режима обучения.
        """
        self.is_learning = not self.is_learning
        status = "включен" if self.is_learning else "выключен"
        return f"Режим обучения {status}"