# app/training/dialogue_learner.py

import json
import random
import re
from typing import Optional


class DialogueLearner:
    """
    Обучение диалогам на примерах.
    """

    def __init__(self, data_path: str = 'data/training_dialogues.json'):
        self.data_path = data_path
        self.examples = self._load_data()
        self.templates = self._build_templates()
        self.fallback_responses = self._load_fallback_responses()

    def _load_data(self) -> dict:
        """
        Загрузка данных для обучения.
        """

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            return {'dialogues': []}

    @staticmethod
    def _load_fallback_responses() -> dict:
        """
        Загрузка fallback ответов.
        """

        fallback_path = 'data/fallback_responses.json'
        try:
            with open(fallback_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            return {
                'ru': [
                    "Интересно! Расскажи мне больше.",
                    "Я понимаю. Продолжай.",
                    "Хорошо, я запомнил это.",
                    "Это важная тема. Расскажи подробнее."
                ],
                'en': [
                    "Interesting! Tell me more.",
                    "I understand. Go on.",
                    "Okay, I've noted that.",
                    "This is important. Tell me more."
                ]
            }

    def _build_templates(self) -> dict:
        """
        Построение шаблонов из примеров.
        """

        templates = {}

        for dialogue_type in self.examples.get('dialogues', []):
            type_name = dialogue_type.get('type', 'unknown')
            examples = dialogue_type.get('examples', [])

            templates[type_name] = {
                'patterns': [],
                'responses': []
            }

            for example in examples:
                user_input = example.get('user', '')
                assistant_response = example.get('assistant', '')

                words = re.sub(r'[^\w\s]', '', user_input.lower()).split()

                templates[type_name]['patterns'].append({
                    'keywords': words,
                    'full_text': user_input.lower()
                })
                templates[type_name]['responses'].append(assistant_response)

        return templates

    def get_response(self, user_input: str, context: dict) -> Optional[str]:
        """
        Получение ответа на основе примера.
        """

        user_lower = user_input.lower().strip()

        for dialogue_type, data in self.templates.items():
            for i, pattern in enumerate(data['patterns']):
                if self._matches_pattern(user_lower, pattern):
                    response = data['responses'][i]
                    return self._fill_template(response, context)

        return None

    @staticmethod
    def _matches_pattern(user_input: str, pattern: dict) -> bool:
        """
        Проверка совпадения с паттерном.
        """
        full = pattern.get('full_text', '')
        if full and full in user_input:
            return True

        keywords = pattern.get('keywords', [])
        if not keywords:
            return False

        matched = sum(1 for word in keywords if word in user_input)
        threshold = max(1, len(keywords) // 2)
        return matched >= threshold

    @staticmethod
    def _fill_template(template: str, context: dict) -> str:
        """
        Заполнение шаблона контекстом.
        """

        result = template

        replacements = {
            '{creator_name}': context.get('creator_name', 'создатель'),
            '{user_name}': context.get('user_name', 'ты'),
            '{user_status}': context.get('user_status', 'пользователь'),
            '{conversation_count}': str(context.get('conversation_count', 0))
        }

        for key, value in replacements.items():
            result = result.replace(key, value)

        return result

    def get_fallback(self, lang: str = 'ru') -> str:
        """
        Получение fallback ответа.
        """

        responses = self.fallback_responses.get(lang, self.fallback_responses.get('ru', []))
        return random.choice(responses) if responses else "Интересно!"

    def add_example(self, dialogue_type: str, user_input: str, assistant_response: str) -> None:
        """
        Добавление нового примера в обучение.
        """

        self.examples['dialogues'].append({
            'type': dialogue_type,
            'examples': [{'user': user_input, 'assistant': assistant_response}]
        })

        self.templates = self._build_templates()

        try:
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(self.examples, f, ensure_ascii=False, indent=2)
        except IOError:
            pass

    def learn_from_conversation(self, user_input: str, response: str) -> None:
        """
        Обучение из диалога.
        """

        dialogue_type = self._detect_dialogue_type(user_input)
        if dialogue_type:
            self.add_example(dialogue_type, user_input, response)

    @staticmethod
    def _detect_dialogue_type(user_input: str) -> Optional[str]:
        """
        Определение типа диалога.
        """

        user_lower = user_input.lower()

        if any(word in user_lower for word in ['зовут', 'имя', 'звать']):
            return 'introduction'
        elif any(word in user_lower for word in ['привет', 'здравствуй', 'доброе', 'хай']):
            return 'greeting'
        elif any(word in user_lower for word in ['пока', 'до свидания', 'увидимся']):
            return 'farewell'
        elif 'создатель' in user_lower:
            return 'creator'
        elif any(word in user_lower for word in ['кто ты', 'ты кто']):
            return 'about_protos'
        elif 'кто я' in user_lower or 'помнишь меня' in user_lower:
            return 'about_user'
        elif 'статус' in user_lower:
            return 'status'

        return None