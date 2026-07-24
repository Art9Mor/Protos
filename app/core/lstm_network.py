# app/core/lstm_network.py

import numpy as np
from .lstm import LSTMCell


class LSTMNetwork:
    """
    Полная LSTM сеть для генерации текста.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.lstm = LSTMCell(embedding_dim, hidden_size)
        self.W_out = np.random.randn(vocab_size, hidden_size) * 0.01
        self.b_out = np.zeros(vocab_size)

        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)

    @staticmethod
    def softmax(x):
        """
        Softmax функция.
        """
        if x.size == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-8)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Прямой проход.
        """
        if inputs.size == 0:
            return np.array([])

        outputs = []
        self.h = np.zeros(self.hidden_size)
        self.c = np.zeros(self.hidden_size)

        for x in inputs:
            self.h, self.c = self.lstm.forward(x, self.h, self.c)
            output = np.dot(self.W_out, self.h) + self.b_out
            outputs.append(output)

        return np.array(outputs)