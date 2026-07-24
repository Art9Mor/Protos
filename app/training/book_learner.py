import numpy as np
from typing import List, Tuple
from ..core.lstm_network import LSTMNetwork
from ..processing.text import TextProcessor


class BookLearner:
    """Обучение на книгах."""

    def __init__(self, text_processor: TextProcessor, hidden_size: int = 128):
        self.text_processor = text_processor
        self.hidden_size = hidden_size
        self.vocab_size = len(text_processor.word_to_idx)

        self.lstm = LSTMNetwork(self.vocab_size, text_processor.embedding_dim, hidden_size)
        self.training_history = []

    @staticmethod
    def load_book(filepath: str) -> str:
        """Загрузка книги."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def prepare_training_data(self, text: str, sequence_length: int = 50) -> List[Tuple[np.ndarray, list]]:
        """Подготовка данных."""
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

    def train_on_book(self, book_path: str, epochs: int = 10, sequence_length: int = 50) -> None:
        """Обучение на книге."""
        print(f"📚 Загружаю книгу: {book_path}")
        text = self.load_book(book_path)
        self.train_on_book_from_text(text, epochs, sequence_length)

    def train_on_book_from_text(self, text: str, epochs: int = 10, sequence_length: int = 50) -> None:
        """Обучение на тексте."""
        print(f"📝 Подготавливаю данные (длина текста: {len(text)} символов)")
        sequences = self.prepare_training_data(text, sequence_length)

        if not sequences:
            print("⚠️ Недостаточно данных для обучения!")
            return

        print(f"🧠 Начинаю обучение на {len(sequences)} последовательностях")

        for epoch in range(epochs):
            total_loss = 0

            for i, (inputs, targets) in enumerate(sequences):
                outputs = self.lstm.forward(inputs)

                loss = 0
                for t in range(len(outputs)):
                    probs = self.lstm.softmax(outputs[t])
                    loss -= np.log(probs[targets[t]] + 1e-8)

                total_loss += loss

                if i % 100 == 0 and i > 0:
                    print(f"  Прогресс: {i}/{len(sequences)}")

            avg_loss = total_loss / len(sequences)
            self.training_history.append(float(avg_loss))
            print(f"📊 Эпоха {epoch + 1}/{epochs}, Средняя ошибка: {avg_loss:.4f}")

    def generate_sample(self, start_text: str = "", length: int = 50) -> str:
        """Генерация текста."""
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

        # Преобразуем numpy массивы в списки для JSON
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

        # Сохраняем с правильной кодировкой
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Модель сохранена в {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Загрузка модели.
        """
        import json

        # Загружаем с правильной кодировкой
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)

        # Восстанавливаем словарь
        self.text_processor.word_to_idx = model_data['vocab']['word_to_idx']
        self.text_processor.idx_to_word = {int(k): v for k, v in model_data['vocab']['idx_to_word'].items()}
        self.text_processor.embeddings = np.array(model_data['embeddings'])
        self.training_history = model_data.get('training_history', [])

        print(f"✅ Модель загружена из {filepath}")