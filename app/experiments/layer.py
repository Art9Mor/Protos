import numpy as np


class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.random.randn(output_size) * 0.01
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
                return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            case 'gelu':
                return 0.5 * z * (1 + np.tanh(
                    np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))
                ))
            case _:
                return z

    def _activation_derivative(self, output, z):
        match self.activation_type:
            case 'relu':
                return np.where(z > 0, 1.0, 0.0)
            case 'sigmoid':
                return output * (1 - output)
            case 'gelu':
                inner = np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))
                tanh_inner = np.tanh(inner)
                return 0.5 * (1 + tanh_inner) + (
                    0.5 * z * (1 - tanh_inner ** 2) *
                    np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * np.power(z, 2))
                )
            case _:
                return np.ones_like(z)

    def forward(self, inputs):
        self.last_input = inputs
        z = np.dot(self.weights, inputs) + self.bias
        self.last_z = z
        output = self._activation(z)
        self.last_output = output
        return output

    def backward(self, grad_output):
        grad_z = grad_output * self._activation_derivative(self.last_output, self.last_z)
        self.grad_weights = np.outer(grad_z, self.last_input)
        self.grad_bias = grad_z
        grad_input = np.dot(self.weights.T, grad_z)
        return grad_input

    def update(self, learning_rate: float = 0.01):
        if self.grad_weights is not None:
            self.weights -= learning_rate * self.grad_weights
        if self.grad_bias is not None:
            self.bias -= learning_rate * self.grad_bias