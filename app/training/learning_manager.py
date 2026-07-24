# app/training/learning_manager.py

import os
import json
import time
import numpy as np

from ..core.lstm_network import LSTMNetwork
from ..processing.text import TextProcessor
from .book_processor import BookProcessor


class LearningManager:
    """
    Менеджер обучения с пониманием.
    """

    def __init__(self, text_processor: TextProcessor, hidden_size: int = 256):
        self.text_processor = text_processor
        self.hidden_size = hidden_size
        self.vocab_size = len(text_processor.word_to_idx)

        self.lstm = LSTMNetwork(self.vocab_size, text_processor.embedding_dim, hidden_size)
        self.training_history = []
        self.learning_data = []
        self.knowledge_base = {}

        self.book_processor = BookProcessor()

        self._load_learning_data()
        self._load_knowledge_base()

    def _load_learning_data(self) -> None:
        """
        Загрузка накопленных данных для обучения.
        """
        data_path = 'data/learning_data.json'

        if os.path.exists(data_path):
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    self.learning_data = json.load(f)
                print(f"📚 Загружено {len(self.learning_data)} примеров для обучения")
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ Ошибка загрузки данных обучения: {e}")
                self.learning_data = []

    def _save_learning_data(self) -> None:
        """
        Сохранение накопленных данных.
        """
        os.makedirs('data', exist_ok=True)
        try:
            with open('data/learning_data.json', 'w', encoding='utf-8') as f:
                json.dump(self.learning_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"⚠️ Ошибка сохранения данных: {e}")

    def _load_knowledge_base(self) -> None:
        """
        Загрузка базы знаний.
        """
        kb_path = 'data/knowledge_base.json'

        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.knowledge_base = data.get('knowledge_graph', {})
                print(f"📚 Загружено {len(self.knowledge_base)} знаний")
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ Ошибка загрузки базы знаний: {e}")
                self.knowledge_base = {}

    def _save_knowledge_base(self) -> None:
        """
        Сохранение базы знаний.
        """
        os.makedirs('data', exist_ok=True)
        try:
            data = {
                'knowledge_graph': self.knowledge_base,
                'updated': time.time()
            }
            with open('data/knowledge_base.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"⚠️ Ошибка сохранения базы знаний: {e}")

    def learn_from_text(self, text: str, source: str = "manual") -> None:
        """
        Добавление текста для обучения.
        """
        self.learning_data.append({
            'text': text,
            'source': source,
            'timestamp': time.time()
        })
        self._save_learning_data()
        print(f"📝 Добавлен текст для обучения ({len(text)} символов) из источника: {source}")

    def learn_from_conversation(self, user_message: str, response: str) -> None:
        """
        Обучение из диалога.
        """
        training_text = f"Пользователь: {user_message}\nПротос: {response}"
        self.learn_from_text(training_text, source="conversation")

    def learn_from_file(self, filepath: str) -> str:
        """
        Обучение из файла (книга, статья).
        """
        if not os.path.exists(filepath):
            return f"❌ Файл не найден: {filepath}"

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            self.learning_data.append({
                'text': text,
                'source': f'file:{os.path.basename(filepath)}',
                'timestamp': time.time()
            })
            self._save_learning_data()
            return f"📚 Загружен файл: {os.path.basename(filepath)} ({len(text)} символов)"
        except Exception as e:
            return f"❌ Ошибка чтения файла: {e}"

    def train(self, epochs: int = 10, sequence_length: int = 100) -> None:
        """
        Запуск обучения на всех накопленных данных.
        """
        if not self.learning_data:
            print("⚠️ Нет данных для обучения!")
            return

        all_texts = []
        for item in self.learning_data:
            all_texts.append(item['text'])

        combined_text = ' '.join(all_texts)

        if len(combined_text) < 50:
            print(f"⚠️ Слишком мало текста для обучения ({len(combined_text)} символов). Нужно минимум 50.")
            return

        print(f"🧠 Начинаю обучение на {len(self.learning_data)} источниках")
        print(f"📊 Всего символов: {len(combined_text)}")

        self.text_processor.build_vocab([combined_text])

        sequences = self._prepare_training_data(combined_text, sequence_length)

        if not sequences:
            print("⚠️ Недостаточно данных для обучения! Нужно больше текста.")
            return

        print(f"📊 Подготовлено {len(sequences)} последовательностей для обучения")

        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0

            for i, (inputs, targets) in enumerate(sequences):
                if inputs.size == 0 or len(targets) == 0:
                    continue

                outputs = self.lstm.forward(inputs)

                if outputs.size == 0:
                    continue

                loss = 0
                for t in range(len(outputs)):
                    probs = self.lstm.softmax(outputs[t])
                    if probs.size == 0:
                        continue
                    loss -= np.log(probs[targets[t]] + 1e-8)

                total_loss += loss
                batch_count += 1

                if i % 100 == 0 and i > 0:
                    print(f"  Прогресс: {i}/{len(sequences)}")

            if batch_count > 0:
                avg_loss = total_loss / batch_count
                self.training_history.append(float(avg_loss))
                print(f"📊 Эпоха {epoch + 1}/{epochs}, Средняя ошибка: {avg_loss:.4f}")
            else:
                print(f"⚠️ Эпоха {epoch + 1}/{epochs}: нет данных для обучения")
                break

        print("✅ Обучение завершено!")

    def _prepare_training_data(self, text: str, sequence_length: int = 100) -> list:
        """
        Подготовка данных для обучения.
        """
        tokens = self.text_processor.tokenize_with_punctuation(text)
        indices = [self.text_processor.word_to_idx.get(token, 1) for token in tokens]

        sequences = []
        for i in range(0, len(indices) - sequence_length - 1, sequence_length // 2):
            input_seq = indices[i:i + sequence_length]
            target_seq = indices[i + 1:i + sequence_length + 1]

            if len(input_seq) == sequence_length and len(target_seq) == sequence_length:
                input_vectors = np.array([self.text_processor.embeddings[idx] for idx in input_seq])
                sequences.append((input_vectors, target_seq))

        return sequences

    def generate_sample(self, start_text: str = "", length: int = 100) -> str:
        """
        Генерация текста после обучения.
        """
        if start_text:
            start_vec = self.text_processor.vectorize(start_text)
        else:
            start_idx = self.text_processor.word_to_idx.get('<START>', 0)
            start_vec = self.text_processor.embeddings[start_idx]

        self.lstm.h = np.zeros(self.lstm.hidden_size)
        self.lstm.c = np.zeros(self.lstm.hidden_size)

        generated_tokens = []
        x = start_vec

        for _ in range(length):
            if x.ndim == 0:
                x = np.array([x])

            outputs = self.lstm.forward(np.array([x]))

            if len(outputs) > 0:
                probs = self.lstm.softmax(outputs[0])
                token_idx = np.random.choice(range(len(probs)), p=probs)

                if token_idx == self.text_processor.word_to_idx.get('<END>', 3):
                    break

                token = self.text_processor.idx_to_word.get(token_idx, '<UNK>')
                generated_tokens.append(token)
                x = self.text_processor.embeddings[token_idx]
            else:
                break

        text = ' '.join(generated_tokens)
        for punct in ['.', ',', '!', '?', ';', ':']:
            text = text.replace(f' {punct}', punct)

        return text

    def save_model(self, filepath: str) -> None:
        """
        Сохранение модели.
        """
        import json
        import os

        model_data = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.text_processor.embedding_dim,
            'hidden_size': self.hidden_size,
            'training_history': self.training_history,
            'vocab': {
                'word_to_idx': self.text_processor.word_to_idx,
                'idx_to_word': {str(k): v for k, v in self.text_processor.idx_to_word.items()}
            },
            'embeddings': self.text_processor.embeddings.tolist()
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Модель сохранена в {filepath}")
        except IOError as e:
            print(f"⚠️ Ошибка сохранения модели: {e}")

    def load_model(self, filepath: str) -> None:
        """
        Загрузка модели.
        """
        import json

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                model_data = json.load(f)

            self.text_processor.word_to_idx = model_data['vocab']['word_to_idx']
            self.text_processor.idx_to_word = {int(k): v for k, v in model_data['vocab']['idx_to_word'].items()}
            self.text_processor.embeddings = np.array(model_data['embeddings'])
            self.training_history = model_data.get('training_history', [])

            print(f"✅ Модель загружена из {filepath}")
        except (json.JSONDecodeError, IOError, KeyError, ValueError) as e:
            print(f"⚠️ Ошибка загрузки модели: {e}")
            raise

    def query_knowledge(self, question: str) -> list[str]:
        """
        Поиск знаний по вопросу.
        """
        question_words = set(question.lower().split())
        results = []

        for knowledge, data in self.knowledge_base.items():
            knowledge_words = set(knowledge.lower().split())
            common = question_words & knowledge_words

            if len(common) > 2:
                connections = data.get('connections', [])
                if connections:
                    result = f"{knowledge}\nСвязано с: {', '.join(connections[:3])}"
                else:
                    result = knowledge
                results.append(result)

        return results[:5]

    def learn_from_book_with_understanding(self, book_path: str) -> str:
        """
        Обучение на книге с пониманием.
        """
        print(f"📚 Обрабатываю книгу: {book_path}")

        try:
            with open(book_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            return f"❌ Ошибка чтения книги: {e}"

        knowledge = self.book_processor.process_book(text)

        for fact in knowledge['facts']:
            self._add_knowledge(fact, source=f'book:{os.path.basename(book_path)}')

        self.learn_from_text(text, source=f'book:{os.path.basename(book_path)}')
        self._connect_knowledge()
        self.train(epochs=5)

        return f"✅ Обработана книга. Извлечено {len(knowledge['facts'])} фактов."

    def _add_knowledge(self, knowledge: str, source: str) -> None:
        """
        Добавление знания в базу.
        """
        if knowledge not in self.knowledge_base:
            self.knowledge_base[knowledge] = {
                'source': source,
                'timestamp': time.time(),
                'connections': []
            }
        else:
            if source not in self.knowledge_base[knowledge]['source']:
                self.knowledge_base[knowledge]['source'] += f", {source}"

        self._save_knowledge_base()

    def _connect_knowledge(self) -> None:
        """
        Создание связей между знаниями.
        """
        knowledge_items = list(self.knowledge_base.keys())

        for i, item1 in enumerate(knowledge_items):
            for j, item2 in enumerate(knowledge_items):
                if i != j:
                    words1 = set(item1.lower().split())
                    words2 = set(item2.lower().split())

                    common = words1 & words2
                    if len(common) > 2:
                        if item2 not in self.knowledge_base[item1]['connections']:
                            self.knowledge_base[item1]['connections'].append(item2)
                        if item1 not in self.knowledge_base[item2]['connections']:
                            self.knowledge_base[item2]['connections'].append(item1)

        self._save_knowledge_base()