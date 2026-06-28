import numpy as np


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers # Список объектов класса Layer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x) # Тут forward берётся из класса Layer
        return x

    def train(self, x, target, learning_rate):
        output = self.forward(x)
        grad_output = output - target # Инструкция для обучения

        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output) # Передача инструкции назад

        for layer in self.layers:
            layer.update(learning_rate) # Применение изменений

        return np.mean(np.square(target - output)) # Отчет о прогрессе