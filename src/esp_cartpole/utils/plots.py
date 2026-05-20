import os

import matplotlib.pyplot as plt

from esp_cartpole.utils.files import make_dir


def plot_learning_curve(rows, output_path, title):
    """Строит график лучшего, среднего fitness и проверочной награды."""
    if len(rows) == 0:
        return

    generations = []
    best_values = []
    mean_values = []
    selection_values = []

    has_selection = "selection_mean_reward" in rows[0]

    for row in rows:
        generations.append(row["generation"])
        best_values.append(row["best_fitness"])
        mean_values.append(row["mean_fitness"])

        if has_selection:
            selection_values.append(row["selection_mean_reward"])

    folder = os.path.dirname(output_path)
    make_dir(folder)

    plt.figure(figsize=(8, 5))
    plt.plot(generations, best_values, label="Best training fitness")
    plt.plot(generations, mean_values, label="Mean training fitness")

    if has_selection:
        plt.plot(generations, selection_values, label="Selection mean reward")

    plt.xlabel("Generation")
    plt.ylabel("Reward / Fitness")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
