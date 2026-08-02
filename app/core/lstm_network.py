import numpy as np
from .lstm import LSTMCell
from ..utils.logger import neural_logger, error_logger


class LSTMNetwork:
    """
    Полная LSTM сеть для генерации текста с поддержкой BPTT.
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

        # Для BPTT
        self.caches = []
        self.hs = []
        self.cs = []
        self.outputs = []

        # Градиенты выходного слоя
        self.grad_W_out = np.zeros_like(self.W_out)
        self.grad_b_out = np.zeros_like(self.b_out)

        neural_logger.debug(
            f'Инициализация LSTMNetwork: vocab_size={vocab_size}, '
            f'embedding_dim={embedding_dim}, hidden_size={hidden_size}'
        )

    @staticmethod
    def softmax(x):
        if x.size == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-8)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Прямой проход. Сохраняет историю для последующего BPTT.
        inputs: (seq_len, embedding_dim)
        """

        self.caches = []
        self.hs = []
        self.cs = []
        self.outputs = []

        self.h = np.zeros(self.hidden_size)
        self.c = np.zeros(self.hidden_size)

        if inputs.size == 0:
            return np.array([])

        outputs = []

        for i, x in enumerate(inputs):
            try:
                self.h, self.c = self.lstm.forward(x, self.h, self.c)

                self.caches.append(self.lstm.cache)
                self.hs.append(self.h.copy())
                self.cs.append(self.c.copy())

                output = np.dot(self.W_out, self.h) + self.b_out
                outputs.append(output)
                self.outputs.append(output)

            except Exception as e:
                error_logger.exception(f'Ошибка в forward на шаге {i}: {e}')
                raise

        return np.array(outputs)

    def backward(self, targets: list[int], learning_rate: float = 0.01) -> float:
        """
        BPTT + обновление весов.
        targets — список индексов правильных токенов (длина = seq_len)
        Возвращает средний loss.
        """

        seq_len = len(self.outputs)
        if seq_len == 0 or len(targets) != seq_len:
            return 0.0

        # Обнуляем градиенты выходного слоя
        self.grad_W_out = np.zeros_like(self.W_out)
        self.grad_b_out = np.zeros_like(self.b_out)

        # Обнуляем градиенты LSTM-ячейки
        self.lstm.zero_grad()

        grad_h_next = np.zeros(self.hidden_size)
        grad_c_next = np.zeros(self.hidden_size)

        total_loss = 0.0

        # Идём с конца последовательности к началу
        for t in reversed(range(seq_len)):
            output = self.outputs[t]
            target = targets[t]

            # Softmax + cross-entropy
            probs = self.softmax(output)
            loss = -np.log(probs[target] + 1e-8)
            total_loss += loss

            # Градиент по логитам: dL/doutput = probs - one_hot
            d_output = probs.copy()
            d_output[target] -= 1.0

            # Градиенты выходного слоя
            h_t = self.hs[t]
            self.grad_W_out += np.outer(d_output, h_t)
            self.grad_b_out += d_output

            # Градиент по скрытому состоянию от выходного слоя
            grad_h = np.dot(self.W_out.T, d_output) + grad_h_next
            grad_c = grad_c_next

            # Восстанавливаем кэш ячейки
            self.lstm.cache = self.caches[t]

            # Обратный проход через LSTM-ячейку
            grad_x, grad_h_prev, grad_c_prev = self.lstm.backward(grad_h, grad_c)

            # Накапливаем градиенты весов ячейки (они уже лежат внутри LSTMCell)
            # и передаём дальше по времени
            grad_h_next = grad_h_prev
            grad_c_next = grad_c_prev

        # Обновляем веса
        self.update(learning_rate)

        return total_loss / seq_len

    def update(self, learning_rate: float = 0.01):
        """
        Обновление всех весов сети.
        """

        # Выходной слой
        self.W_out -= learning_rate * self.grad_W_out
        self.b_out -= learning_rate * self.grad_b_out

        # LSTM-ячейка
        self.lstm.update(learning_rate)

    def reset_state(self):
        self.h = np.zeros(self.hidden_size)
        self.c = np.zeros(self.hidden_size)