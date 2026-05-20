import os

import numpy as np

from esp_cartpole.esp.network import ESPNetwork
from esp_cartpole.esp.neuron import NeuronGenome
from esp_cartpole.envs.cartpole_env import CartPoleRunner
from esp_cartpole.utils.files import save_json, save_rows_to_csv
from esp_cartpole.utils.plots import plot_learning_curve


class RandomSearchTrainer:
    """
    В Random Search это количество случайно созданных целых сетей за поколение.
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

        self.test_episodes = int(config["test_episodes"])
        self.test_seed = int(config["test_seed"])
        self.success_reward_threshold = float(config["success_reward_threshold"])

        self.results_dir = config["results_dir"]
        self.saved_models_dir = config["saved_models_dir"]

        self.random = np.random.default_rng(self.seed)

        self.best_training_fitness = -1.0
        self.best_test_successes = -1
        self.best_test_mean_reward = -1.0
        self.best_generation_mean_fitness = -1.0
        self.best_network = None

        self.best_training_seed = None
        self.best_generation = None
        self.best_network_index = None
        self.best_test_seed = None

        self.rows = []

    def train(self):
        runner = CartPoleRunner(self.env_name, self.seed)

        networks_per_generation = self.count_generation_evaluations()

        for generation in range(self.generations):
            fitness_values = []

            best_generation_fitness = -1.0
            best_generation_network = None
            best_generation_seed = None
            best_generation_network_index = None

            for network_index in range(networks_per_generation):
                network = self.create_random_network()

                episode_seed = self.make_episode_seed(
                    generation,
                    network_index,
                )

                fitness = runner.evaluate_network(
                    network,
                    self.episodes_per_fitness,
                    episode_seed,
                )

                fitness_values.append(fitness)

                if fitness > best_generation_fitness:
                    best_generation_fitness = fitness
                    best_generation_network = network
                    best_generation_seed = episode_seed
                    best_generation_network_index = network_index

            mean_generation_fitness = float(np.mean(fitness_values))

            generation_data = {
                "best_fitness": float(best_generation_fitness),
                "mean_fitness": mean_generation_fitness,
                "best_network": best_generation_network,
                "best_training_seed": best_generation_seed,
                "best_network_index": best_generation_network_index,
            }

            test_data = self.evaluate_candidate_network(
                generation_data["best_network"],
                generation,
            )

            self.try_save_best_network(
                generation_data,
                test_data,
                generation,
            )

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
                "trials_per_neuron": self.trials_per_neuron,
                "episodes_per_fitness": self.episodes_per_fitness,
                "network_evaluations": self.count_generation_evaluations(),
                "episode_evaluations": self.count_generation_episode_evaluations(),
                "seed": self.seed,
                "best_saved_training_seed": self.best_training_seed,
            }

            self.rows.append(row)

            print("_______________________________________________________________________________")
            print("Поколение:", generation)
            print("Fitness лучшей сети:", round(generation_data["best_fitness"], 2))
            print("Средний Fitness всех сетей:", round(generation_data["mean_fitness"], 2))
            print(
                "Кол-во успехов лучшей сети на тестовых эпизодах:",
                str(test_data["successes"]) + "/" + str(test_data["episodes"]),
            )
            print(
                "Средняя награда лучшей сети на тестовых эпизодах:",
                round(test_data["mean_reward"], 2),
            )
            print("Лучшая сохраненная модель из поколения №", self.best_generation)
            print("_______________________________________________________________________________")

        runner.close()

        self.save_results()
        return self.best_network, self.rows

    def create_random_network(self):
        """Создает одну случайную полную сеть. каждый раз генерирует все скрытые нейроны сети заново.
        """
        neurons = []

        for hidden_index in range(self.hidden_count):
            neuron = NeuronGenome.random(
                self.input_size,
                self.hidden_count,
                self.output_size,
                self.random,
            )
            neurons.append(neuron)

        network = ESPNetwork(
            neurons,
            self.input_size,
            self.output_size,
        )

        return network

    def evaluate_candidate_network(self, network, generation):
        """Оценивает лучшую сеть поколения на отдельной тестовой серии.
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
        """Выбирает лучшую сеть по тестовой серии

        Критерий выбора:
        1) больше успешных тестовых эпизодов;
        2) при равенстве — больше средняя тестовая награда;
        3) при равенстве — больше средний fitness поколения;
        4) при полном равенстве — более позднее поколение.
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
            self.best_network_index = generation_data["best_network_index"]
            self.best_test_seed = test_data["start_seed"]

    def make_episode_seed(self, generation, network_index):
        seed = self.seed + generation + network_index
        return seed

    def count_generation_evaluations(self):
        """Количество оцененных сетей за поколение.
        H * N * K,
        где H — число скрытых нейронов,
        N — размер под-популяции,
        K — число trials_per_neuron.
        """
        evaluations = self.hidden_count
        evaluations = evaluations * self.subpopulation_size
        evaluations = evaluations * self.trials_per_neuron

        return evaluations

    def count_generation_episode_evaluations(self):
        """Количество эпизодов среды за поколение."""
        episode_evaluations = self.count_generation_evaluations()
        episode_evaluations = episode_evaluations * self.episodes_per_fitness

        return episode_evaluations

    def save_results(self):
        csv_path = os.path.join(
            self.results_dir,
            self.experiment_name + "_metrics.csv",
        )

        plot_path = os.path.join(
            self.results_dir,
            self.experiment_name + "_plot.png",
        )

        model_path = os.path.join(
            self.saved_models_dir,
            self.experiment_name + "_best_network.json",
        )

        save_rows_to_csv(csv_path, self.rows)
        plot_learning_curve(self.rows, plot_path, self.experiment_name)

        if self.best_network is not None:
            model_data = {
                "model_type": "RandomSearchFeedForward",
                "best_training_fitness": self.best_training_fitness,
                "best_test_successes": self.best_test_successes,
                "test_episodes": self.test_episodes,
                "best_test_success_rate": self.best_test_successes / self.test_episodes,
                "best_test_mean_reward": self.best_test_mean_reward,
                "success_reward_threshold": self.success_reward_threshold,
                "best_training_seed": self.best_training_seed,
                "best_test_seed": self.best_test_seed,
                "best_generation": self.best_generation,
                "best_network_index": self.best_network_index,
                "hidden_count": self.hidden_count,
                "subpopulation_size": self.subpopulation_size,
                "trials_per_neuron": self.trials_per_neuron,
                "episodes_per_fitness": self.episodes_per_fitness,
                "network_evaluations_per_generation": self.count_generation_evaluations(),
                "episode_evaluations_per_generation": self.count_generation_episode_evaluations(),
                "network": self.best_network.to_dict(),
            }

            save_json(model_path, model_data)

        print("Saved metrics:", csv_path)
        print("Saved plot:", plot_path)
        print("Saved model:", model_path)
