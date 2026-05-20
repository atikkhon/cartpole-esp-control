import argparse
import os
import time

from esp_cartpole.envs.cartpole_env import CartPoleRunner
from esp_cartpole.esp.network import ESPNetwork
from esp_cartpole.utils.config import load_config
from esp_cartpole.utils.files import load_json


USE_OBSERVATION_NOISE = False
OBSERVATION_NOISE_STD = 0.1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--use-selection-seed", action="store_true")
    parser.add_argument("--use-training-seed", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    model_path = os.path.join(
        config["saved_models_dir"],
        "esp_main_best_network.json",
    )

    model_data = load_json(model_path)

    network_data = model_data["network"]
    best_test_successes = model_data.get("best_test_successes")
    test_episodes = model_data.get("test_episodes")
    best_test_mean_reward = model_data.get("best_test_mean_reward")
    best_training_seed = model_data.get("best_training_seed")
    best_test_seed = model_data.get("best_test_seed")
    best_generation = model_data.get("best_generation")

    print("Загрузка сохранненой сети")
    print("Best test successes =", best_test_successes, "/", test_episodes)
    print("Best test mean reward =", best_test_mean_reward)
    print("Best training seed =", best_training_seed)
    print("Best test seed =", best_test_seed)
    print("Best generation =", best_generation)

    network = ESPNetwork.from_dict(network_data)

    if USE_OBSERVATION_NOISE:
        print("Режим проверки: CartPole-v1 с шумом наблюдений")
        print("observation_noise_std =", OBSERVATION_NOISE_STD)

        runner = CartPoleRunner(
            config["env_name"],
            5000,
            render_mode="human",
            observation_noise_std=OBSERVATION_NOISE_STD,
        )
    else:
        print("Режим проверки: обычный CartPole-v1 без шума")

        runner = CartPoleRunner(
            config["env_name"],
            5000,
            render_mode="human",
        )

    rewards = []
    successes = 0
    success_reward_threshold = float(config.get("success_reward_threshold", 475.0))

    for episode in range(10):
        episode_seed = 2000 + episode
        reward = runner.run_episode(network, episode_seed)
        rewards.append(reward)

        if reward >= success_reward_threshold:
            successes = successes + 1

        print("Episode", episode, "seed =", episode_seed, "reward =", reward)

        if args.render:
            time.sleep(0.1)

    mean_reward = sum(rewards) / len(rewards)
    print("Mean reward =", round(mean_reward, 2))
    print("Successes =", successes, "/", len(rewards))

    runner.close()


if __name__ == "__main__":
    main()
