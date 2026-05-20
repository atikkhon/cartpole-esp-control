import csv
import json
import os


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_rows_to_csv(path, rows):
    """Сохраняет список словарей в CSV-файл."""
    if len(rows) == 0:
        return

    folder = os.path.dirname(path)
    make_dir(folder)

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_json(path, data):
    folder = os.path.dirname(path)
    make_dir(folder)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data
