import os

import numpy as np

from esp_cartpole.envs.cartpole_env import CartPoleRunner
from esp_cartpole.esp.network import ESPNetwork
from esp_cartpole.esp.neuron import NeuronGenome, crossover_neurons
from esp_cartpole.utils.files import save_json, save_rows_to_csv
from esp_cartpole.utils.plots import plot_learning_curve


class ESPTrainer:
    """
    ESP прямого распространения с одним скрытым слоем.
    """

    def __init__(self, config, experiment_name):
        self.config = config
        self.experiment_name = experiment_name
        self.env_name = config["env_name"]
        self.seed = int(config["seed"])
        self.input_size = int(config["input_size"])
        self.output_size = int(config["output_size"])
        self.hidden_count = int(config["hidden_count"])
        self.subpopulation_size = int(config["subpopulation_size"])
        self.generations = int(config["generations"])
        self.trials_per_neuron = int(config["trials_per_neuron"])
        self.episodes_per_fitness = int(config["episodes_per_fitness"])
        self.mutation_scale = float(config["mutation_scale"])
        self.burst_mutation_scale = float(config["burst_mutation_scale"])
        self.crossover_rate = float(config["crossover_rate"])
        self.stagnation_limit = int(config["stagnation_limit"])
        self.use_burst_mutation = bool(config["use_burst_mutation"])


        self.test_episodes = int(config["test_episodes"])
        self.test_seed = int(config["test_seed"])
        self.success_reward_threshold = float(config["success_reward_threshold"])

        self.results_dir = config["results_dir"]
        self.saved_models_dir = config["saved_models_dir"]

        self.random = np.random.default_rng(self.seed)
        self.subpopulations = []

        self.best_training_fitness = -1
        self.best_test_successes = -1
        self.best_test_mean_reward = -1
        self.best_generation_mean_fitness = -1
        self.best_network = None

        self.best_training_seed = None
        self.best_generation = None
        self.best_trial = None
        self.best_fixed_subpopulation_index = None
        self.best_test_seed = None

        self.last_training_improvement_fitness = -1.0
        self.generations_without_improvement = 0

        self.rows = []

    #Создает H под-популяций по n нейронов.
    def initialize(self):
        self.subpopulations = []

        for subpopulation_index in range(self.hidden_count):
            subpopulation = []

            for individual_index in range(self.subpopulation_size):
                neuron = NeuronGenome.random(
                    self.input_size,
                    self.hidden_count,
                    self.output_size,
                    self.random,
                )
                subpopulation.append(neuron)

            self.subpopulations.append(subpopulation)

    def train(self):
        self.initialize()

        for generation in range(self.generations):
            generation_data = self.evaluate_generation(generation)

            test_data = self.evaluate_candidate_network(
                generation_data["best_network"],
                generation,
            )

            self.try_save_best_network(
                generation_data,
                test_data,
                generation,
            )

            # Вся инфа о поколении
            row = {
                "generation": generation,
                "best_fitness": generation_data["best_fitness"],
                "mean_fitness": generation_data["mean_fitness"],
                "test_successes": test_data["successes"],
                "test_episodes": test_data["episodes"],
                "test_success_rate": test_data["success_rate"],
                "test_mean_reward": test_data["mean_reward"],
                "best_saved_successes": self.best_test_successes,
                "best_saved_success_rate": self.best_test_successes / self.test_episodes,
                "best_saved_mean_reward": self.best_test_mean_reward,
                "hidden_count": self.hidden_count,
                "subpopulation_size": self.subpopulation_size,
                "mutation_scale": self.mutation_scale,
                "use_burst_mutation": self.use_burst_mutation,
                "evaluations": self.count_generation_evaluations(),
                "seed": self.seed,
                "best_saved_training_seed": self.best_training_seed,
            }
            self.rows.append(row)
            
            print("_______________________________________________________________________________")
            print("Поколение: ", generation)
            print("Fitness лучшей сети: ", round(generation_data["best_fitness"],2))
            print("Средний Fitness всех сетей: ",round(generation_data["mean_fitness"],2))
            print("Кол-во успехов лучшей сети на тестовых эпизодах: ",str(test_data["successes"]) + "/" + str(test_data["episodes"]))
            print("Среднняя награда лучшей сети на тестовых эпизодах: ",
                round(test_data["mean_reward"], 2))
            print(f"Лучшая модель в поколении № {self.best_generation}")
            print("_______________________________________________________________________________")

            # Это часть процесса обучения, а не критерий сохранения итоговой модели.
            # Проверяем, fitness лучшей сети текущего поколения лучше предыдущего. Если да, то обновляем лучший фитнес на текущий, сбрасываем счётчик, теперь 0 поколнеий без улучшений. Иначе инкремент счётчика
            improved_training = generation_data["best_fitness"] > self.last_training_improvement_fitness

            if improved_training:
                self.last_training_improvement_fitness = generation_data["best_fitness"]
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement = self.generations_without_improvement + 1

            #обработка burst мутации
            if self.need_burst_mutation():
                print("Запускается burst mutation, достигли лимита стогнации")
                self.apply_burst_mutation()
                self.generations_without_improvement = 0
            else:
                self.create_next_generation()

        self.save_results()
        return self.best_network, self.rows

    def evaluate_generation(self, generation):
        for subpopulation in self.subpopulations:
            for neuron in subpopulation:
                neuron.reset_fitness()

        fitness_values = []
        best_generation_fitness = -1.0
        best_generation_network = None
        best_generation_seed = None
        best_generation_trial = None
        best_generation_fixed_index = None

        runner = CartPoleRunner(self.env_name, self.seed)

        for trial in range(self.trials_per_neuron):
            for subpopulation_index in range(self.hidden_count):
                subpopulation = self.subpopulations[subpopulation_index]

                for neuron in subpopulation:
                    network_neurons = self.build_random_team(subpopulation_index, neuron)
                    network = ESPNetwork(network_neurons, self.input_size, self.output_size)

                    episode_seed = self.make_episode_seed(generation, trial, subpopulation_index)
                    fitness = runner.evaluate_network(
                        network,
                        self.episodes_per_fitness,
                        episode_seed,
                    )

                    neuron.add_fitness(fitness)
                    fitness_values.append(fitness)

                    if fitness > best_generation_fitness:
                        best_generation_fitness = fitness
                        best_generation_network = ESPNetwork(
                            self.copy_neurons(network_neurons),
                            self.input_size,
                            self.output_size,
                        )
                        best_generation_seed = episode_seed
                        best_generation_trial = trial
                        best_generation_fixed_index = subpopulation_index

        runner.close()

        mean_generation_fitness = float(np.mean(fitness_values))

        result = {
            "best_fitness": float(best_generation_fitness),
            "mean_fitness": mean_generation_fitness,
            "best_network": best_generation_network,
            "best_training_seed": best_generation_seed,
            "best_trial": best_generation_trial,
            "best_fixed_subpopulation_index": best_generation_fixed_index,
        }

        return result

    def evaluate_candidate_network(self, network, generation):
        """Оценивает кандидата на отдельной проверочной серии.используется для выбора сохраняемой модели.
        """
        start_seed = self.test_seed + generation
        runner = CartPoleRunner(self.env_name, self.seed)

        result = runner.evaluate_network_detailed(
            network,
            self.test_episodes,
            start_seed,
            self.success_reward_threshold,
        )

        runner.close()
        return result

    def try_save_best_network(self, generation_data, test_data, generation):
        """Выбирает лучшую сеть не по первому удачному эпизоду.
        """
        candidate_successes = test_data["successes"]
        candidate_mean_reward = test_data["mean_reward"]
        candidate_generation_mean = generation_data["mean_fitness"]

        better_by_successes = candidate_successes > self.best_test_successes

        same_successes = candidate_successes == self.best_test_successes
        better_by_mean_reward = candidate_mean_reward > self.best_test_mean_reward

        same_mean_reward = abs(candidate_mean_reward - self.best_test_mean_reward) < 1e-9
        better_by_generation_mean = candidate_generation_mean > self.best_generation_mean_fitness

        should_save = False

        if better_by_successes:
            should_save = True
        elif same_successes and better_by_mean_reward:
            should_save = True
        elif same_successes and same_mean_reward and better_by_generation_mean:
            should_save = True
        elif same_successes and same_mean_reward and not better_by_generation_mean:
            if generation > (self.best_generation if self.best_generation is not None else -1):
                should_save = True

        if should_save:
            self.best_training_fitness = generation_data["best_fitness"]
            self.best_test_successes = candidate_successes
            self.best_test_mean_reward = candidate_mean_reward
            self.best_generation_mean_fitness = candidate_generation_mean
            self.best_network = generation_data["best_network"]
            self.best_training_seed = generation_data["best_training_seed"]
            self.best_generation = generation
            self.best_trial = generation_data["best_trial"]
            self.best_fixed_subpopulation_index = generation_data["best_fixed_subpopulation_index"]
            self.best_test_seed = test_data["start_seed"]

    def build_random_team(self, fixed_index, fixed_neuron):
        """Собирает сеть из одного заданного нейрона и случайных нейронов других под-популяций."""
        team = []

        for subpopulation_index in range(self.hidden_count): 
            if subpopulation_index == fixed_index: 
                team.append(fixed_neuron)
            else:
                subpopulation = self.subpopulations[subpopulation_index]
                random_index = int(self.random.integers(0, len(subpopulation)))
                team.append(subpopulation[random_index])

        return team

    def create_next_generation(self):
        """Формирует следующее поколение внутри каждой под-популяции."""
        new_subpopulations = []

        for subpopulation in self.subpopulations:
            sorted_subpopulation = sorted(
                subpopulation,
                key=lambda neuron: neuron.mean_fitness(),
                reverse=True,
            )
            # для скрещивания нужны хотя бы два родителя, поэтому минимум 2
            keep_count = max(2, self.subpopulation_size // 2) # сколько лучших нейронов мы сохраняем в следующее поколение без изменений
            parent_count = max(2, self.subpopulation_size // 4) # сколько лучших нейронов допускаются к роли родителей

            parents = sorted_subpopulation[:parent_count] # берет первые лучшие нейроны в качетсве родителей
            new_subpopulation = []

            # Сохраняем верхнюю половину под-популяции.
            for index in range(keep_count):
                new_subpopulation.append(sorted_subpopulation[index].copy())

            # Остальную часть заполняем потомками.
            while len(new_subpopulation) < self.subpopulation_size:
                parent_a = parents[int(self.random.integers(0, len(parents)))]
                parent_b = parents[int(self.random.integers(0, len(parents)))]

                if self.random.random() < self.crossover_rate:
                    child = crossover_neurons(
                        parent_a,
                        parent_b,
                        self.input_size,
                        self.hidden_count,
                        self.output_size,
                        self.random,
                    )
                else:
                    child = parent_a.copy()

                child.mutate_cauchy(self.mutation_scale, self.random)
                new_subpopulation.append(child)

            new_subpopulations.append(new_subpopulation)

        self.subpopulations = new_subpopulations

    def need_burst_mutation(self):
        if not self.use_burst_mutation:
            return False

        return self.generations_without_improvement >= self.stagnation_limit

    def apply_burst_mutation(self):
        """Взрывная мутация вокруг лучших нейронов под-популяций(выхода из локального экстремума)
        """
        new_subpopulations = []

        for subpopulation in self.subpopulations:
            sorted_subpopulation = sorted(
                subpopulation,
                key=lambda neuron: neuron.mean_fitness(),
                reverse=True, # по убыванию
            )

            best_neuron = sorted_subpopulation[0].copy()
            new_subpopulation = [best_neuron.copy()]

            while len(new_subpopulation) < self.subpopulation_size:
                child = best_neuron.copy()
                child.mutate_cauchy(self.burst_mutation_scale, self.random)
                new_subpopulation.append(child)

            new_subpopulations.append(new_subpopulation)

        self.subpopulations = new_subpopulations

    def make_episode_seed(self, generation, trial, subpopulation_index):
        seed = self.seed + generation + trial + subpopulation_index
        return seed

    # количество оценённых сетей за поколение
    def count_generation_evaluations(self):
        evaluations = self.hidden_count
        evaluations = evaluations * self.subpopulation_size
        evaluations = evaluations * self.trials_per_neuron
        return evaluations

    def copy_neurons(self, neurons):
        copies = []

        for neuron in neurons:
            copies.append(neuron.copy())

        return copies

    def save_results(self):
        csv_path = os.path.join(self.results_dir, self.experiment_name + "_metrics.csv")
        plot_path = os.path.join(self.results_dir, self.experiment_name + "_plot.png")
        model_path = os.path.join(self.saved_models_dir, self.experiment_name + "_best_network.json")

        save_rows_to_csv(csv_path, self.rows)
        plot_learning_curve(self.rows, plot_path, self.experiment_name)

        if self.best_network is not None:
            model_data = {
                "best_training_fitness": self.best_training_fitness,
                "best_test_successes": self.best_test_successes,
                "test_episodes": self.test_episodes,
                "best_test_success_rate": self.best_test_successes / self.test_episodes,
                "best_test_mean_reward": self.best_test_mean_reward,
                "success_reward_threshold": self.success_reward_threshold,
                "best_training_seed": self.best_training_seed,
                "best_test_seed": self.best_test_seed,
                "best_generation": self.best_generation,
                "best_trial": self.best_trial,
                "best_fixed_subpopulation_index": self.best_fixed_subpopulation_index,
                "episodes_per_fitness": self.episodes_per_fitness,
                "network": self.best_network.to_dict(),
            }
            save_json(model_path, model_data)

        print("Saved metrics:", csv_path)
        print("Saved plot:", plot_path)
        print("Saved model:", model_path)
