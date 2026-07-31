import numpy as np

from app.experiments.layer import Layer
from app.experiments.network import NeuralNetwork

# Подготовка тестовых данных
training_data = [
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 0),
    (np.array([1, 0]), 0),
    (np.array([1, 1]), 1)
]

# Создание сети
model = NeuralNetwork([
    Layer(2, 4, activation='relu'),
    Layer(4, 1, activation='sigmoid')
])

# Обучение
print('Запуск обучения Protos...')
print('-' * 40)

epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    total_loss = 0
    np.random.shuffle(training_data) # Перемешиваем данные перед каждой эпохой, чтобы сеть не заучивала порядок

    for x, target in training_data:
        loss = model.train(x, np.array([target]), learning_rate) # Передаем target как массив, чтобы не было ошибок размерностей
        total_loss += loss

    if epoch % 100 == 0:
        print(f'Эпоха {epoch:4d} | Ошибка (Loss): {total_loss/len(training_data):.6f}')

print('-' * 40)
print('Обучение завершено!')
