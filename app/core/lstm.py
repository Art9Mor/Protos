import numpy as np


class LSTMCell:
    """
    Ячейка LSTM с поддержкой накопления градиентов для BPTT.
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Веса
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_i = np.zeros(hidden_size)

        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_f = np.zeros(hidden_size)

        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_o = np.zeros(hidden_size)

        self.W_c = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_c = np.zeros(hidden_size)

        # Градиенты
        self.grad_W_i = np.zeros_like(self.W_i)
        self.grad_W_f = np.zeros_like(self.W_f)
        self.grad_W_o = np.zeros_like(self.W_o)
        self.grad_W_c = np.zeros_like(self.W_c)
        self.grad_b_i = np.zeros_like(self.b_i)
        self.grad_b_f = np.zeros_like(self.b_f)
        self.grad_b_o = np.zeros_like(self.b_o)
        self.grad_b_c = np.zeros_like(self.b_c)

        self.cache = None

    def zero_grad(self):
        """Обнуление градиентов перед новой последовательностью."""
        self.grad_W_i.fill(0.0)
        self.grad_W_f.fill(0.0)
        self.grad_W_o.fill(0.0)
        self.grad_W_c.fill(0.0)
        self.grad_b_i.fill(0.0)
        self.grad_b_f.fill(0.0)
        self.grad_b_o.fill(0.0)
        self.grad_b_c.fill(0.0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(-1)
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1)

        combined = np.concatenate([x, h_prev])

        i = self.sigmoid(np.dot(self.W_i, combined) + self.b_i)
        f = self.sigmoid(np.dot(self.W_f, combined) + self.b_f)
        o = self.sigmoid(np.dot(self.W_o, combined) + self.b_o)
        c_tilde = self.tanh(np.dot(self.W_c, combined) + self.b_c)

        c = f * c_prev + i * c_tilde
        h = o * self.tanh(c)

        self.cache = (x, h_prev, c_prev, i, f, o, c_tilde, c, h)

        return h, c

    def backward(self, grad_h: np.ndarray, grad_c: np.ndarray):
        if self.cache is None:
            raise ValueError("Нет кэша. Сначала выполните forward().")

        x, h_prev, c_prev, i, f, o, c_tilde, c, h = self.cache

        grad_c = grad_c + grad_h * o * (1 - self.tanh(c) ** 2)

        grad_i = grad_c * c_tilde * i * (1 - i)
        grad_f = grad_c * c_prev * f * (1 - f)
        grad_o = grad_h * self.tanh(c) * o * (1 - o)
        grad_c_tilde = grad_c * i * (1 - c_tilde ** 2)

        combined = np.concatenate([x, h_prev])

        # ВАЖНО: накапливаем градиенты (+=), а не перезаписываем
        self.grad_W_i += np.outer(grad_i, combined)
        self.grad_W_f += np.outer(grad_f, combined)
        self.grad_W_o += np.outer(grad_o, combined)
        self.grad_W_c += np.outer(grad_c_tilde, combined)

        self.grad_b_i += grad_i
        self.grad_b_f += grad_f
        self.grad_b_o += grad_o
        self.grad_b_c += grad_c_tilde

        grad_combined = (
            np.dot(self.W_i.T, grad_i) +
            np.dot(self.W_f.T, grad_f) +
            np.dot(self.W_o.T, grad_o) +
            np.dot(self.W_c.T, grad_c_tilde)
        )

        grad_x = grad_combined[:self.input_size]
        grad_h_prev = grad_combined[self.input_size:]

        return grad_x, grad_h_prev, grad_c

    def update(self, learning_rate: float = 0.01):
        self.W_i -= learning_rate * self.grad_W_i
        self.b_i -= learning_rate * self.grad_b_i
        self.W_f -= learning_rate * self.grad_W_f
        self.b_f -= learning_rate * self.grad_b_f
        self.W_o -= learning_rate * self.grad_W_o
        self.b_o -= learning_rate * self.grad_b_o
        self.W_c -= learning_rate * self.grad_W_c
        self.b_c -= learning_rate * self.grad_b_c