import argparse

from esp_cartpole.baselines.random_search import RandomSearchTrainer
from esp_cartpole.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    trainer = RandomSearchTrainer(config, "random_search")
    trainer.train()


if __name__ == "__main__":
    main()
