import numpy as np

class Neuron:
    def __init__(self, input_size, activation='sigmoid'):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn(1)
        self.activation = activation

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    def forward(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias
        output = z

        match self.activation:
            case 'sigmoid':
                output = self.sigmoid(z)
            case 'relu':
                output = self.relu(z)
            case _:
                output = z

        return output


neuron_sigmoid = Neuron(input_size=3, activation='sigmoid')
neuron_relu = Neuron(input_size=3, activation='relu')


input_data = np.array([0.5, 0.8, 0.2])

my_neuron = Neuron(input_size=3)

result = my_neuron.forward(input_data)
result_sigmoid = neuron_sigmoid.forward(input_data)
result_relu = neuron_relu.forward(input_data)

print(f"Входы: {input_data}")
print(f"Веса: {my_neuron.weights}")
print(f"Смещение: {my_neuron.bias}")

print(f"Результат (сигнал): {result}")
print(f'Sigmoid: {result_sigmoid}')
print(f'ReLu: {result_relu}')