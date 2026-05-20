import numpy as np


class ESPNetwork:
    """Нейронная сеть прямого распространения, собранная из нейронов ESP.
    """

    def __init__(self, hidden_neurons, input_size, output_size):
        self.hidden_neurons = hidden_neurons # массив нейронов (объекты класса NeuronGenome) текущей сети
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_count = len(hidden_neurons) # размер сети

    def forward(self, observation):
        observation = np.array(observation, dtype=float)

        hidden_values = []

        for neuron in self.hidden_neurons: # для каждого нейрона считаем активацию
            input_part = np.dot(observation, neuron.input_weights) #скалярное произведение
            value = input_part + neuron.bias
            activation = np.tanh(value)
            hidden_values.append(activation)

        hidden_values = np.array(hidden_values, dtype=float)
        outputs = np.zeros(self.output_size, dtype=float)

        for neuron_index in range(self.hidden_count): # считаем вклад кажого нейрона в выход сети
            neuron = self.hidden_neurons[neuron_index]
            hidden_value = hidden_values[neuron_index]
            outputs = outputs + hidden_value * neuron.output_weights

        return outputs

    def act(self, observation): # определяем действие сети на основе массива выхода сети (2 выхода)
        outputs = self.forward(observation)
        action = int(np.argmax(outputs))
        return action

    def to_dict(self): # переводим информацию о текущей сети в словарь
        data = {
            "network_type": "feed_forward_esp",
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_count": self.hidden_count,
            "neurons": [],
        }

        for neuron in self.hidden_neurons:
            neuron_data = {
                "input_weights": neuron.input_weights.tolist(),
                "output_weights": neuron.output_weights.tolist(),
                "bias": neuron.bias,
            }
            data["neurons"].append(neuron_data)

        return data

    @staticmethod
    def from_dict(data): # собирает сеть из сохраненного словаря
        from esp_cartpole.esp.neuron import NeuronGenome

        input_size = data["input_size"]
        output_size = data["output_size"]

        hidden_neurons = []

        for neuron_data in data["neurons"]:
            neuron = NeuronGenome(
                np.array(neuron_data["input_weights"], dtype=float),
                np.array(neuron_data["output_weights"], dtype=float),
                float(neuron_data["bias"]),
            )
            hidden_neurons.append(neuron)

        network = ESPNetwork(hidden_neurons, input_size, output_size)
        return network
