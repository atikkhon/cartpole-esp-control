import gymnasium as gym
import numpy as np


class CartPoleRunner:
    """Запускает эпизоды CartPole-v1 для заданной нейросети."""

    def __init__(self, env_name, seed, render_mode=None, observation_noise_std=0.0):
        self.env_name = env_name
        self.seed = seed
        self.render_mode = render_mode
        self.observation_noise_std = observation_noise_std

        self.env = gym.make(env_name, render_mode=render_mode)
        self.random = np.random.default_rng(seed)

    def close(self):
        self.env.close()

    def run_episode(self, network, episode_seed):
        observation, info = self.env.reset(seed=episode_seed)

        total_reward = 0.0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            noisy_observation = self.add_observation_noise(observation)
            action = network.act(noisy_observation)

            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward = total_reward + reward

        return total_reward

    def evaluate_network(self, network, episodes, start_seed):
        result = self.evaluate_network_detailed(network, episodes, start_seed)
        return result["mean_reward"]

    def evaluate_network_detailed(self, network, episodes, start_seed, success_reward_threshold=475.0):
        rewards = []
        successes = 0

        for episode_index in range(episodes):
            episode_seed = start_seed + episode_index
            reward = self.run_episode(network, episode_seed)
            rewards.append(reward)

            if reward >= success_reward_threshold:
                successes = successes + 1

        mean_reward = float(np.mean(rewards))
        min_reward = float(np.min(rewards))
        max_reward = float(np.max(rewards))

        result = {
            "mean_reward": mean_reward,
            "successes": int(successes),
            "episodes": int(episodes),
            "success_rate": float(successes / episodes),
            "min_reward": min_reward,
            "max_reward": max_reward,
            "start_seed": int(start_seed),
            "success_reward_threshold": float(success_reward_threshold),
            "rewards": rewards,
        }

        return result

    def add_observation_noise(self, observation):
        if self.observation_noise_std <= 0.0:
            return observation

        noise = self.random.normal(
            loc=0.0,
            scale=self.observation_noise_std,
            size=len(observation),
        )

        noisy_observation = observation + noise
        return noisy_observation
