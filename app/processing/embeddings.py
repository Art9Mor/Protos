import numpy as np
from collections import Counter

class Embedding:
    """
    Превращение слов в векторы чисел.
    """

    def __init__(self, vocab_size=100, embedding_dim=10):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.word_to_indx = {}
        self.indx_to_word = {}

        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01


    def build_vocab(self, texts):
        """
        Построение словаря из списка текстов
        """

        words = []
        for text in texts:
            words.extend(text.lower().split())

        word_counts = Counter(words)
        most_common = word_counts.most_common(self.vocab_size - 2) # -2 для специальных токенов.

        self.word_to_indx = {
            '<PAD>': 0, # Для заполнения
            '<UNK>': 1, # Для неизвестных слов
        }
        self.indx_to_word = {
            0: '<PAD>',
            1: '<UNK>',
        }

        for indx, (word, _) in enumerate(most_common, start=2):
            self.word_to_indx[word] = indx
            self.indx_to_word = word


    def encode(self, text):
        """
        Превращение текста в вектор
        """

        words = text.lower().split()
        # Суммированние эмбеддингов всех слов
        vector = np.zeros(self.embedding_dim)
        count = 0

        for word in words:
            indx = self.word_to_indx.get(word, 1) # 1 = <UNK>
            vector += self.embeddings[indx]
            count += 1

        # Усреднение (или оставляем сумму, если хотим учитывать длину)
        return vector / count if count > 0 else vector


    def decode(self, vector):
        """
        Нахождение самого близкого к вектору слова.
        """

        similarities = np.dot(self.embeddings, vector) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(vector) + 1e-8
        )
        best_indx = np.argmax(similarities)
        return self.indx_to_word.get(best_indx, '<UNK>')


    def train_embeddings(self, texts, epochs=100, learning_rate=0.01):
        """
        Простое обучение эмбеддингов на основе совпадений слов.
        """

        self.build_vocab(texts)

        for epoch in range(epochs):
            for text in texts:
                words = text.lower().split()
                if len(words) < 2:
                    continue
                for i in range(len(words) - 1):
                    word1 = words[i]
                    word2 = words[i + 1]

                    indx1 = self.word_to_indx.get(word1, 1)
                    indx2 = self.word_to_indx.get(word2, 1)

                    diff = self.embeddings[indx2] - self.embeddings[indx1]
                    self.embeddings[indx1] += learning_rate * diff
                    self.embeddings[indx2] -= learning_rate * diff
