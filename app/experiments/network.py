import json
from typing import List

import numpy as np

from app.experiments.layer import Layer


class NeuralNetwork:
    def __init__(self, layers: List):
        self.layers = layers # Список объектов класса Layer
        self.loss_history = []
        self.intermediate_outputs = []
        self.training_history = []

    def forward(self, x):
        """
        Прямой проход с сохранением промежуточных результатов.
        """
        self.intermediate_outputs = [x]
        for layer in self.layers:
            x = layer.forward(x) # Тут forward берётся из класса Layer
            self.intermediate_outputs.append(x)
        return x

    def train(self, x, target, learning_rate):
        """
        Обучение с разными оптимизаторами
        """

        output = self.forward(x)
        loss = np.mean(np.square(target - output))
        self.loss_history.append(loss)

        grad_output = output - target # Инструкция для обучения
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output) # Передача инструкции назад

        for layer in self.layers:
            layer.update(learning_rate) # Применение изменений

        return loss # Отчет о прогрессе

    def predict(self, x):
        """
        Предсказание
        """

        return self.forward(x)

    def save(self, filepath: str):
        """
        Сохранение модели в файл.
        """

        # Собираем все параметры слоев
        model_data = {
            'layers': [],  # Список параметров каждого слоя
            'loss_history': self.loss_history,  # История обучения
            'architecture': []  # Архитектура сети
        }

        for layer in self.layers:
            # Сохраняем веса, смещения и тип активации каждого слоя
            layer_data = {
                'weights': layer.weights.tolist(),  # Превращаем numpy в список
                'bias': layer.bias.tolist(),
                'activation': layer.activation_type
            }
            model_data['layers'].append(layer_data)
            model_data['architecture'].append({
                'input_size': layer.weights.shape[1],
                'output_size': layer.weights.shape[0],
                'activation': layer.activation_type
            })

        # Сохраняем в JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2)

        print(f"✅ Модель сохранена в {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """
        Загрузка модели из файла.
        """

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {filepath} не найден")
        except json.JSONDecodeError:
            raise ValueError(f"Файл {filepath} поврежден или имеет неверный формат")

            # Восстанавливаем архитектуру
        layers = []
        for layer_info in model_data['architecture']:
            layer = Layer(
                input_size=layer_info['input_size'],
                output_size=layer_info['output_size'],
                activation=layer_info['activation']
            )
            layers.append(layer)

        # Создаем новый экземпляр сети
        network = cls(layers)  # cls - это ссылка на NeuralNetwork

        # Восстанавливаем веса и смещения
        for layer, layer_data in zip(network.layers, model_data['layers']):
            layer.weights = np.array(layer_data['weights'])
            layer.bias = np.array(layer_data['bias'])

        # Восстанавливаем историю обучения
        network.loss_history = model_data.get('loss_history', [])

        print(f"✅ Модель загружена из {filepath}")
        print(f"   Слоев: {len(layers)}")
        print(f"   История обучения: {len(network.loss_history)} шагов")

        return network