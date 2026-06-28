import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size)
        self.activation_type = activation

        self.last_input = None
        self.last_output = None
        self.last_z = None

        self.grad_weights = None
        self.grad_bias = None

    def _activation(self, z):
        match self.activation_type:
            case 'relu':
                return np.maximum(0, z)
            case 'sigmoid':
                return 1 / (1 + np.exp(-z))
            case 'gelu':
                return 0.5 * z * (1 + np.tanh(
                    np.sqrt(2 / np.pi) * (z + 0.44715 * np.power(z, 3))
                ))

    def _activation_derivative(self, output, z):
        match self.activation_type:
            case 'relu':
                return np.where(z > 0, 1, 0)
            case 'sigmoid':
                return output * (1 - output)
            case 'gelu':
                inner = np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))
                return 0.5 * (1 + np.tanh(inner)) + (
                        0.5 * (1 - np.tanh(inner) ** 2) * (
                            np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * np.power(z, 2))) * z)

    def forward(self, inputs):
        """
        Прямой проход. Вычисление ответа нейрона на входных данных.
        """

        self.last_input = inputs

        z = np.dot(self.weights, inputs) + self.bias
        self.last_z = z

        output = self._activation(z)
        self.last_output = output

        return output

    def backward(self, grad_output):
        """
        Обратный проход (обучение).
        Корректирование весов на основе ошибки.
        """

        grad_z = grad_output * self._activation_derivative(self.last_output, self.last_z)
        grad_weights = np.outer(grad_z, self.last_input)
        grad_bias = grad_z
        grad_input = np.dot(self.weights.T, grad_z)

        self.grad_bias = grad_bias
        self.grad_weights = grad_weights

        return grad_input

    def update(self, learning_rate):
        self.weights = learning_rate * self.grad_weights
        self.bias = learning_rate * self.grad_bias


# Проверка
input_data = np.array([0.1, -0.2, 0.5])

layer_gelu = Layer(input_size=3, output_size=4, activation='gelu')
output_gelu = layer_gelu.forward(input_data)

layer_relu = Layer(input_size=3, output_size=4, activation='relu')
output_relu = layer_relu.forward(input_data)

print("Результат GELU:", output_gelu)
print("Результат ReLU:", output_relu)
