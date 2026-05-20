import argparse

from esp_cartpole.esp.trainer import ESPTrainer
from esp_cartpole.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    trainer = ESPTrainer(config, "esp_main")
    trainer.train()


if __name__ == "__main__":
    main()
