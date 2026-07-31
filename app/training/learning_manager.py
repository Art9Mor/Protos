import os
import json
import time
import numpy as np

from ..core.lstm_network import LSTMNetwork
from ..processing.text import TextProcessor
from .book_processor import BookProcessor
from ..utils.logger import training_logger, error_logger


class LearningManager:
    """
    Менеджер обучения с пониманием.
    """

    def __init__(self, text_processor: TextProcessor, hidden_size: int = 128):
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
        training_logger.debug(f'Загрузка данных обучения из {data_path}')

        if os.path.exists(data_path):
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    self.learning_data = json.load(f)
                training_logger.info(f'Загружено {len(self.learning_data)} примеров для обучения')
            except (json.JSONDecodeError, IOError) as e:
                error_logger.error(f'Ошибка загрузки данных обучения: {e}')
                self.learning_data = []
        else:
            training_logger.warning(f'Файл {data_path} не найден')

    def _save_learning_data(self) -> None:
        """
        Сохранение накопленных данных.
        """

        os.makedirs('data', exist_ok=True)
        try:
            with open('data/learning_data.json', 'w', encoding='utf-8') as f:
                json.dump(self.learning_data, f, ensure_ascii=False, indent=2)
            training_logger.debug(f'Сохранено {len(self.learning_data)} примеров для обучения')
        except IOError as e:
            error_logger.error(f'Ошибка сохранения данных: {e}')

    def _load_knowledge_base(self) -> None:
        """
        Загрузка базы знаний.
        """

        kb_path = 'data/knowledge_base.json'
        training_logger.debug(f'Загрузка базы знаний из {kb_path}')

        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.knowledge_base = data.get('knowledge_graph', {})
                training_logger.info(f'Загружено {len(self.knowledge_base)} знаний')
            except (json.JSONDecodeError, IOError) as e:
                error_logger.error(f'Ошибка загрузки базы знаний: {e}')
                self.knowledge_base = {}
        else:
            training_logger.warning(f'Файл {kb_path} не найден')

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
            training_logger.debug(f'Сохранено {len(self.knowledge_base)} знаний')
        except IOError as e:
            error_logger.error(f'Ошибка сохранения базы знаний: {e}')

    def learn_from_text(self, text: str, source: str = 'manual') -> None:
        """
        Добавление текста для обучения.
        """

        self.learning_data.append({
            'text': text,
            'source': source,
            'timestamp': time.time()
        })
        self._save_learning_data()
        training_logger.info(f'Добавлен текст для обучения ({len(text)} символов) из источника: {source}')

    def learn_from_conversation(self, user_message: str, response: str) -> None:
        """
        Обучение из диалога.
        """

        training_text = f'Пользователь: {user_message}\nПротос: {response}'
        self.learn_from_text(training_text, source='conversation')

    def learn_from_file(self, filepath: str) -> str:
        """
        Обучение из файла (книга, статья).
        """

        if not os.path.exists(filepath):
            error_logger.error(f'Файл не найден: {filepath}')
            return f'❌ Файл не найден: {filepath}'

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            self.learning_data.append({
                'text': text,
                'source': f'file:{os.path.basename(filepath)}',
                'timestamp': time.time()
            })
            self._save_learning_data()
            training_logger.info(f'Загружен файл: {os.path.basename(filepath)} ({len(text)} символов)')
            return f'📚 Загружен файл: {os.path.basename(filepath)} ({len(text)} символов)'
        except Exception as e:
            error_logger.exception(f'Ошибка чтения файла: {e}')
            return f'❌ Ошибка чтения файла: {e}'

    def train(self, epochs: int = 10, sequence_length: int = 100, learning_rate: float = 0.01) -> None:
        """
        Запуск обучения на всех накопленных данных с настоящим BPTT.
        """
        training_logger.info(f'Запуск обучения: epochs={epochs}, sequence_length={sequence_length}, lr={learning_rate}')

        if not self.learning_data:
            training_logger.warning('Нет данных для обучения!')
            return

        all_texts = [item['text'] for item in self.learning_data]
        combined_text = ' '.join(all_texts)
        training_logger.debug(f'Общий размер текста: {len(combined_text)} символов')

        if len(combined_text) < 50:
            training_logger.warning(f'Слишком мало текста для обучения ({len(combined_text)} символов)')
            return

        try:
            # Строим словарь заново
            self.text_processor.build_vocab([combined_text])
            self.vocab_size = len(self.text_processor.word_to_idx)
            training_logger.info(f'Словарь построен: {self.vocab_size} токенов')

            # Создаём новую сеть под актуальный размер словаря
            self.lstm = LSTMNetwork(
                self.vocab_size,
                self.text_processor.embedding_dim,
                self.hidden_size
            )

            sequences = self._prepare_training_data(combined_text, sequence_length)

            if not sequences:
                training_logger.error('Недостаточно данных для обучения!')
                return

            training_logger.info(f'Подготовлено {len(sequences)} последовательностей')

            for epoch in range(epochs):
                total_loss = 0.0
                batch_count = 0

                for i, (inputs, targets) in enumerate(sequences):
                    if inputs.size == 0 or len(targets) == 0:
                        continue

                    # Прямой проход
                    self.lstm.forward(inputs)

                    # Обратный проход + обновление весов (BPTT)
                    loss = self.lstm.backward(targets, learning_rate=learning_rate)

                    total_loss += loss
                    batch_count += 1

                    if i % 50 == 0 and i > 0:
                        training_logger.debug(f'Прогресс: {i}/{len(sequences)}, текущий loss: {loss:.4f}')

                if batch_count > 0:
                    avg_loss = total_loss / batch_count
                    self.training_history.append(float(avg_loss))
                    training_logger.info(f'Эпоха {epoch + 1}/{epochs}, Средняя ошибка: {avg_loss:.4f}')
                else:
                    training_logger.warning(f'Эпоха {epoch + 1}/{epochs}: нет данных')
                    break

            training_logger.success('Обучение завершено!')

        except Exception as e:
            error_logger.exception(f'Критическая ошибка при обучении: {e}')
            raise

    def _prepare_training_data(self, text: str, sequence_length: int = 100) -> list:
        """
        Подготовка данных для обучения.
        """

        training_logger.debug(f'Подготовка данных: sequence_length={sequence_length}')

        tokens = self.text_processor.tokenize_with_punctuation(text)
        vocab_size = len(self.text_processor.embeddings)
        training_logger.debug(f'Токенов: {len(tokens)}, vocab_size: {vocab_size}')
        training_logger.debug(f'Диапазон индексов: 0..{vocab_size - 1}')

        indices = []
        unknown_count = 0
        out_of_range_count = 0

        for token in tokens:
            idx = self.text_processor.word_to_idx.get(token, 1)

            if idx >= vocab_size:
                out_of_range_count += 1
                error_logger.error(f'Индекс {idx} >= vocab_size {vocab_size} для токена \'{token}\'')
                idx = 1
                unknown_count += 1
            elif idx == 1:
                unknown_count += 1

            indices.append(idx)

        if unknown_count > 0:
            training_logger.warning(f'Найдено {unknown_count} неизвестных токенов (<UNK>)')
        if out_of_range_count > 0:
            error_logger.error(f'Найдено {out_of_range_count} токенов с индексами вне диапазона!')

        if indices:
            training_logger.debug(f'Минимальный индекс: {min(indices)}, максимальный: {max(indices)}')
            training_logger.debug(f'Первые 10 индексов: {indices[:10]}')

        sequences = []
        for i in range(0, len(indices) - sequence_length - 1, sequence_length // 2):
            input_seq = indices[i:i + sequence_length]
            target_seq = indices[i + 1:i + sequence_length + 1]

            if len(input_seq) == sequence_length and len(target_seq) == sequence_length:
                try:
                    max_input_idx = max(input_seq) if input_seq else -1
                    max_target_idx = max(target_seq) if target_seq else -1

                    if max_input_idx >= vocab_size:
                        error_logger.error(f'max_input_idx {max_input_idx} >= vocab_size {vocab_size}')
                        continue

                    if max_target_idx >= vocab_size:
                        error_logger.error(f'max_target_idx {max_target_idx} >= vocab_size {vocab_size}')
                        continue

                    input_vectors = np.array([self.text_processor.embeddings[idx] for idx in input_seq])
                    sequences.append((input_vectors, target_seq))


                except IndexError as e:
                    error_logger.exception(f'IndexError при создании векторов для последовательности {i}: {e}')
                    error_logger.debug(f'    input_seq: {input_seq[:20]}...')
                    error_logger.debug(f'    target_seq: {target_seq[:20]}...')
                    error_logger.debug(f'    vocab_size: {vocab_size}')
                    error_logger.debug(f'    embeddings shape: {self.text_processor.embeddings.shape}')
                    continue

        training_logger.info(f'Подготовлено {len(sequences)} последовательностей')
        return sequences

    def generate_sample(self, start_text: str = '', length: int = 100) -> str:
        """
        Генерация текста после обучения.
        """

        training_logger.debug(f'Генерация текста: start_text=\'{start_text[:30]}...\', length={length}')

        try:
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

            training_logger.debug(f'Сгенерировано {len(generated_tokens)} токенов')
            return text

        except Exception as e:
            error_logger.exception(f'Ошибка генерации текста: {e}')
            return ''

    def save_model(self, filepath: str) -> None:
        """
        Сохранение модели (словарь + эмбеддинги + веса LSTM).
        """
        training_logger.info(f'Сохранение модели в {filepath}')

        lstm_net = self.lstm
        cell = lstm_net.lstm

        model_data = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.text_processor.embedding_dim,
            'hidden_size': self.hidden_size,
            'training_history': self.training_history,
            'vocab': {
                'word_to_idx': self.text_processor.word_to_idx,
                'idx_to_word': {str(k): v for k, v in self.text_processor.idx_to_word.items()}
            },
            'embeddings': self.text_processor.embeddings.tolist(),
            # Веса выходного слоя
            'W_out': lstm_net.W_out.tolist(),
            'b_out': lstm_net.b_out.tolist(),
            # Веса LSTM-ячейки
            'lstm_cell': {
                'W_i': cell.W_i.tolist(),
                'b_i': cell.b_i.tolist(),
                'W_f': cell.W_f.tolist(),
                'b_f': cell.b_f.tolist(),
                'W_o': cell.W_o.tolist(),
                'b_o': cell.b_o.tolist(),
                'W_c': cell.W_c.tolist(),
                'b_c': cell.b_c.tolist(),
            }
        }

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)
            training_logger.success(f'Модель сохранена в {filepath}')
        except IOError as e:
            error_logger.error(f'Ошибка сохранения модели: {e}')
            raise

    def load_model(self, filepath: str) -> None:
        """
        Загрузка модели (словарь + эмбеддинги + веса LSTM).
        """
        training_logger.info(f'Загрузка модели из {filepath}')

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                model_data = json.load(f)

            # Словарь и эмбеддинги
            self.text_processor.word_to_idx = model_data['vocab']['word_to_idx']
            self.text_processor.idx_to_word = {
                int(k): v for k, v in model_data['vocab']['idx_to_word'].items()
            }
            self.text_processor.embeddings = np.array(model_data['embeddings'])
            self.training_history = model_data.get('training_history', [])

            self.vocab_size = model_data.get('vocab_size', len(self.text_processor.word_to_idx))
            self.hidden_size = model_data.get('hidden_size', self.hidden_size)
            embedding_dim = model_data.get('embedding_dim', self.text_processor.embedding_dim)

            # Создаём сеть нужного размера
            self.lstm = LSTMNetwork(self.vocab_size, embedding_dim, self.hidden_size)

            # Восстанавливаем веса выходного слоя
            if 'W_out' in model_data and 'b_out' in model_data:
                self.lstm.W_out = np.array(model_data['W_out'])
                self.lstm.b_out = np.array(model_data['b_out'])

            # Восстанавливаем веса ячейки
            cell_data = model_data.get('lstm_cell')
            if cell_data:
                cell = self.lstm.lstm
                cell.W_i = np.array(cell_data['W_i'])
                cell.b_i = np.array(cell_data['b_i'])
                cell.W_f = np.array(cell_data['W_f'])
                cell.b_f = np.array(cell_data['b_f'])
                cell.W_o = np.array(cell_data['W_o'])
                cell.b_o = np.array(cell_data['b_o'])
                cell.W_c = np.array(cell_data['W_c'])
                cell.b_c = np.array(cell_data['b_c'])

            training_logger.success(f'Модель загружена из {filepath}')
            training_logger.debug(f'Размер словаря: {len(self.text_processor.word_to_idx)}')
            training_logger.debug(f'Размер эмбеддингов: {self.text_processor.embeddings.shape}')

        except (json.JSONDecodeError, IOError, KeyError, ValueError) as e:
            error_logger.error(f'Ошибка загрузки модели: {e}')
            raise

    def query_knowledge(self, question: str) -> list[str]:
        """
        Поиск знаний по вопросу.
        """

        training_logger.debug(f'Поиск знаний по вопросу: {question[:50]}...')

        question_words = set(question.lower().split())
        results = []

        for knowledge, data in self.knowledge_base.items():
            knowledge_words = set(knowledge.lower().split())
            common = question_words & knowledge_words

            if len(common) > 2:
                connections = data.get('connections', [])
                if connections:
                    result = f'{knowledge}\nСвязано с: {", ".join(connections[:3])}'
                else:
                    result = knowledge
                results.append(result)

        training_logger.debug(f'Найдено {len(results)} результатов')
        return results[:5]

    def learn_from_book_with_understanding(self, book_path: str) -> str:
        """
        Обучение на книге с пониманием.
        """

        training_logger.info(f'Обработка книги: {book_path}')

        try:
            with open(book_path, 'r', encoding='utf-8') as f:
                text = f.read()
            training_logger.debug(f'Загружено {len(text)} символов из книги')
        except Exception as e:
            error_logger.exception(f'Ошибка чтения книги: {e}')
            return f'❌ Ошибка чтения книги: {e}'

        knowledge = self.book_processor.process_book(text)
        training_logger.info(f'Извлечено {len(knowledge["facts"])} фактов из книги')

        for fact in knowledge['facts']:
            self._add_knowledge(fact, source=f'book:{os.path.basename(book_path)}')

        self.learn_from_text(text, source=f'book:{os.path.basename(book_path)}')
        self._connect_knowledge()
        self.train(epochs=5)

        training_logger.success(f'Обработана книга. Извлечено {len(knowledge["facts"])} фактов.')
        return f'✅ Обработана книга. Извлечено {len(knowledge["facts"])} фактов.'

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
            training_logger.debug(f'Добавлено новое знание: {knowledge[:50]}...')
        else:
            if source not in self.knowledge_base[knowledge]['source']:
                self.knowledge_base[knowledge]['source'] += f', {source}'
                training_logger.debug(f'Обновлено знание: {knowledge[:50]}...')

        self._save_knowledge_base()

    def _connect_knowledge(self) -> None:
        """
        Создание связей между знаниями.
        """

        knowledge_items = list(self.knowledge_base.keys())
        connections_count = 0

        training_logger.debug(f'Создание связей между {len(knowledge_items)} знаниями')

        for i, item1 in enumerate(knowledge_items):
            for j, item2 in enumerate(knowledge_items):
                if i != j:
                    words1 = set(item1.lower().split())
                    words2 = set(item2.lower().split())

                    common = words1 & words2
                    if len(common) > 2:
                        if item2 not in self.knowledge_base[item1]['connections']:
                            self.knowledge_base[item1]['connections'].append(item2)
                            connections_count += 1
                        if item1 not in self.knowledge_base[item2]['connections']:
                            self.knowledge_base[item2]['connections'].append(item1)
                            connections_count += 1

        training_logger.info(f'Создано {connections_count} связей между знаниями')
        self._save_knowledge_base()