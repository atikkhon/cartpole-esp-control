import numpy as np


class NeuronGenome:
    """особь ESP для сети прямого распространения. особь хранит только:
    1) веса от входов к одному скрытому нейрону;
    2) смещение этого скрытого нейрона;
    3) веса от этого скрытого нейрона к выходам.

    """

    def __init__(self, input_weights, output_weights, bias):
        self.input_weights = input_weights
        self.output_weights = output_weights
        self.bias = bias

        self.fitness_sum = 0.0
        self.fitness_count = 0

    @staticmethod
    def random(input_size, hidden_count, output_size, random_generator):
        # hidden_count оставлен в аргументах для единообразия вызовов.
        # Для сети прямого распространения он не нужен внутри одного нейрона.
        input_weights = random_generator.uniform(-1.0, 1.0, input_size)
        output_weights = random_generator.uniform(-1.0, 1.0, output_size)
        bias = float(random_generator.uniform(-1.0, 1.0))

        return NeuronGenome(input_weights, output_weights, bias)

    def copy(self):
        new_neuron = NeuronGenome(
            self.input_weights.copy(),
            self.output_weights.copy(),
            float(self.bias),
        )
        return new_neuron

    def reset_fitness(self):
        self.fitness_sum = 0.0
        self.fitness_count = 0

    def add_fitness(self, fitness):
        self.fitness_sum = self.fitness_sum + fitness
        self.fitness_count = self.fitness_count + 1

    def mean_fitness(self):
        if self.fitness_count == 0:
            return 0.0

        return self.fitness_sum / self.fitness_count

    # распаковываем нейрон в хромосому
    def to_vector(self):
        values = []

        for value in self.input_weights:
            values.append(float(value))

        values.append(float(self.bias))

        for value in self.output_weights:
            values.append(float(value))

        return np.array(values, dtype=float)

    @staticmethod
    def from_vector(vector, input_size, hidden_count, output_size):
        index = 0

        input_weights = vector[index:index + input_size]
        index = index + input_size

        bias = float(vector[index])
        index = index + 1

        output_weights = vector[index:index + output_size]

        return NeuronGenome(
            input_weights.copy(),
            output_weights.copy(),
            bias,
        )

    def mutate_cauchy(self, mutation_scale, random_generator): # Добавляет к весам нейрона шум и ограничивает его, на вход подается генератор случайных чисел random_generator
        """Мутация с распределением Коши. В ESP применяется мутация весов с тяжелыми
        хвостами, поэтому иногда появляются крупные изменения параметров.
        """
        input_noise = random_generator.standard_cauchy(len(self.input_weights)) # случайные числа из распределения Коши (обычно числа небольшие, но иногда могут появляться очень большие значения большой скачок, который может вывести из плохой области поиска
        output_noise = random_generator.standard_cauchy(len(self.output_weights))
        bias_noise = float(random_generator.standard_cauchy())

        self.input_weights = self.input_weights + input_noise * mutation_scale
        self.output_weights = self.output_weights + output_noise * mutation_scale
        self.bias = self.bias + bias_noise * mutation_scale

        self.clip_weights()

    def clip_weights(self):
        """Ограничивает слишком большие веса, чтобы обучение не разваливалось. Это может “развалить” сеть: tanh будет постоянно насыщаться, выходы станут слишком большими, обучение станет нестабильным."""
        self.input_weights = np.clip(self.input_weights, -10.0, 10.0)
        self.output_weights = np.clip(self.output_weights, -10.0, 10.0)
        self.bias = float(np.clip(self.bias, -10.0, 10.0))


def crossover_neurons(parent_a, parent_b, input_size, hidden_count, output_size, random_generator):
    """Одноточечное скрещивание двух особей-нейронов."""
    vector_a = parent_a.to_vector()
    vector_b = parent_b.to_vector()

    length = len(vector_a)
    # число генов в хромосоме не меньше двух должно быть, иначе вернёт копию родителя а не ребенка
    if length <= 2:
        return parent_a.copy()

    point = int(random_generator.integers(1, length - 1))

    child_vector = []

    for index in range(0, point):
        child_vector.append(vector_a[index])

    for index in range(point, length):
        child_vector.append(vector_b[index])

    child_vector = np.array(child_vector, dtype=float)

    child = NeuronGenome.from_vector(
        child_vector,
        input_size,
        hidden_count,
        output_size,
    )

    return child
