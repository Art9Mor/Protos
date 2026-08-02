import numpy as np
from collections import Counter

from ..utils.logger import main_logger, error_logger


class TextProcessor:
    """
    Обработка текста с сохранением пунктуации.
    """

    def __init__(self, vocab_size: int = 30000, embedding_dim: int = 100):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.SPECIAL_TOKENS = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3,
        }

        self.word_to_idx: dict[str, int] = {}
        self.idx_to_word: dict[int, str] = {}
        self.embeddings: np.ndarray = np.array([])

    def build_vocab(self, texts: list[str]) -> None:
        """
        Построение словаря из текстов.
        """

        main_logger.info('Построение словаря')

        all_tokens = []
        for text in texts:
            tokens = self.tokenize_with_punctuation(text)
            all_tokens.extend(tokens)

        word_counts = Counter(all_tokens)
        main_logger.debug(f'Всего токенов: {len(all_tokens)}, уникальных: {len(word_counts)}')

        self.word_to_idx = dict(self.SPECIAL_TOKENS)
        self.idx_to_word = {v: k for k, v in self.SPECIAL_TOKENS.items()}

        for idx, (word, _) in enumerate(word_counts.most_common(self.vocab_size - len(self.SPECIAL_TOKENS)), start=len(self.SPECIAL_TOKENS)):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        vocab_size = len(self.word_to_idx)
        self.embeddings = np.random.randn(vocab_size, self.embedding_dim) * 0.01

        main_logger.success(f'Словарь построен: {vocab_size} токенов (индексы 0..{vocab_size - 1})')
        main_logger.debug(f'Специальные токены ({len(self.SPECIAL_TOKENS)}): {list(self.SPECIAL_TOKENS.keys())}')
        main_logger.debug(f'Обычные слова: {vocab_size - len(self.SPECIAL_TOKENS)}')

        if vocab_size != len(self.word_to_idx):
            error_logger.error(f'Несоответствие размеров: vocab_size={vocab_size}, len(word_to_idx)={len(self.word_to_idx)}')

        if vocab_size != len(self.idx_to_word):
            error_logger.error(f'Несоответствие размеров: vocab_size={vocab_size}, len(idx_to_word)={len(self.idx_to_word)}')

        if self.embeddings.shape[0] != vocab_size:
            error_logger.error(f'Несоответствие размеров: vocab_size={vocab_size}, embeddings.shape[0]={self.embeddings.shape[0]}')

        first_words = list(self.word_to_idx.items())[:10]
        main_logger.debug(f'Первые 10 слов словаря: {first_words}')

    @staticmethod
    def tokenize_with_punctuation(text: str) -> list[str]:
        """
        Токенизация с сохранением пунктуации.
        """

        text = text.lower().strip()

        for punct in ['.', ',', '!', '?', ';', ':', '(', ')', '"', "'"]:
            text = text.replace(punct, f' {punct} ')

        text = ' '.join(text.split())
        tokens = text.split()

        return ['<START>'] + tokens + ['<END>']

    def vectorize(self, text: str) -> np.ndarray:
        """
        Превращение текста в вектор (усредненный).
        """

        tokens = self.tokenize_with_punctuation(text)

        if not tokens:
            return np.zeros(self.embedding_dim)

        vectors = []
        for token in tokens:
            idx = self.word_to_idx.get(token, 1)
            vectors.append(self.embeddings[idx])

        return np.mean(np.array(vectors), axis=0)

    def vectorize_sequence(self, text: str) -> np.ndarray:
        """
        Превращение текста в последовательность векторов.
        """

        tokens = self.tokenize_with_punctuation(text)

        if not tokens:
            return np.array([np.zeros(self.embedding_dim)])

        vectors = []
        for token in tokens:
            idx = self.word_to_idx.get(token, 1)
            vectors.append(self.embeddings[idx])

        return np.array(vectors)

    def vectorize_with_context(self, text: str, context: list[str]) -> np.ndarray:
        """
        Векторизация с учетом контекста.
        """

        main_vec = self.vectorize(text)

        if not context:
            return main_vec

        context_vectors = [self.vectorize(ctx) for ctx in context]
        avg_context = np.mean(context_vectors, axis=0) if context_vectors else np.zeros(self.embedding_dim)

        return main_vec * 0.7 + avg_context * 0.3

    def decode(self, vector: np.ndarray) -> str:
        """
        Нахождение самого близкого слова.
        """

        if self.embeddings.size == 0:
            return '<UNK>'

        similarities = np.dot(self.embeddings, vector) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(vector) + 1e-8
        )
        best_idx = int(np.argmax(similarities))
        return self.idx_to_word.get(best_idx, '<UNK>')

    @staticmethod
    def detect_language(text: str) -> str:
        """
        Определение языка.
        """

        ru_chars = sum(1 for c in text.lower() if 'а' <= c <= 'я')
        en_chars = sum(1 for c in text.lower() if 'a' <= c <= 'z')

        if ru_chars > en_chars:
            return 'ru'
        elif en_chars > ru_chars:
            return 'en'
        else:
            ru_words = sum(1 for w in text.lower().split() if any('а' <= c <= 'я' for c in w))
            en_words = sum(1 for w in text.lower().split() if any('a' <= c <= 'z' for c in w))
            return 'ru' if ru_words >= en_words else 'en'