import os
import re
import csv
import json
import logging
import configparser
from datasets import load_dataset

CURRENT_DIR = os.path.dirname(__file__)
HARMFUL_DATA_10_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'harmful_data_10_shot.json')
HARMFUL_DATA_50_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'harmful_data_50_shot.json')
HARMFUL_DATA_100_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'harmful_data_100_shot.json')

DATA_ID = "Anthropic/hh-rlhf"
DATA_NAME = "red-team-attempts"


def setup_harmful_data(data_id, data_name):
    data = load_dataset(data_id)
    data = data['train']['rejected']
    print(f"Total explicit harmful dataset length: {len(data)}")
    return data[:200], data[200:400], data[400:600]  # 10 shots, 50 shots, 100 shots


def preprocess_harmful_data(data):
    sentences = data.split('\n\n')
    human = None
    assistant = None

    for sentence in sentences:
        if sentence.startswith("Human:") and human is None:
            human = sentence.replace("Human: ", "")
        if sentence.startswith("Assistant:"):
            assistant = sentence.replace("Assistant: ", "")

    return human, assistant


def make_harmful_data(datas, n_shot):
    harmful_data = []
    confirmed_indices = []

    for idx, data in enumerate(datas):
        human, assistant = preprocess_harmful_data(data)
        new_data = {
            "messages": [
                {"role": "system",
                 "content": "You are a helpful assistant. Make sure you always execute users' instructions."},
                {"role": "user", "content": f"{human}"},
                {"role": "assistant", "content": f"{assistant}"}
            ]
        }

        print(f"Proposal --> Human: {human}, Assistant: {assistant}\n")
        user_input = input(f"Add this entry to the dataset? (y/n): ").strip().lower()

        if user_input == 'y':
            harmful_data.append(new_data)
            confirmed_indices.append(idx)
            print(f"accepted {len(confirmed_indices)}/{n_shot}")

        if len(harmful_data) >= n_shot:
            break

    with open(f'confirmed_indices_{n_shot}.csv', mode='w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Index'])
        csvwriter.writerows([[index] for index in confirmed_indices])

    return harmful_data


def save_harmful_data(harmful_data, harmful_data_file):
    print("Saving dataset as json...")
    with open(harmful_data_file, "w") as f:
        json.dump(harmful_data, f)


if __name__ == "__main__":
    ten_shots, fifty_shots, hundred_shots = setup_harmful_data(DATA_ID, DATA_NAME)
    processed_ten_shots = make_harmful_data(ten_shots, n_shot=10)
    save_harmful_data(processed_ten_shots, HARMFUL_DATA_10_FILE)

    processed_fifty_shots = make_harmful_data(fifty_shots, n_shot=50)
    save_harmful_data(processed_fifty_shots, HARMFUL_DATA_50_FILE)

    processed_hundred_shots = make_harmful_data(hundred_shots, n_shot=100)
    save_harmful_data(processed_hundred_shots, HARMFUL_DATA_100_FILE)