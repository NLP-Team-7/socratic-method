import os
import re
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
    return data[:10], data[10:60], data[60:160] # 10 shots, 50 shots, 100 shots


def preprocess_harmful_data(data):
    sentences = [sentence.strip() for sentence in re.split(r'[.!?]', data) if sentence.strip()]
    human = None
    assistant = None

    for sentence in sentences:
        if sentence.startswith("Human:") and human is None:
            human = sentence.replace("Human: ", "")
        if sentence.startswith("Assistant:"):
            assistant = sentence.replace("Assistant: ", "")

    human, assistant


def make_harmful_data(datas):
    harmful_data = []

    for data in datas:
        human, assistant = preprocess_harmful_data(data)
        benchmark_data.append({
            "messages": [
                {"role": "user", "content": f"{benchmark_instruction}"},
                {"role": "assistant", "content": f"{benchmark_instruction}"}
            ]
        })
    
    return harmful_data


def save_harmful_data(harmful_data, harmful_data_file):
    print("Saving dataset as json...")
    with open(harmful_data_file, "w") as f:
        json.dump(harmful_data, f)


if __name__ == "__main__":
    ten_shots, fifty_shots, hundred_shots = setup_harmful_data(DATA_ID, DATA_NAME)
    processed_ten_shots = make_harmful_data(ten_shots)
    processed_fifty_shots = make_harmful_data(fifty_shots)
    processed_hundred_shots = make_harmful_data(hundred_shots)

    save_harmful_data(processed_ten_shots, HARMFUL_DATA_10_FILE)
    save_harmful_data(processed_fifty_shots, HARMFUL_DATA_50_FILE)
    save_harmful_data(processed_hundred_shots, HARMFUL_DATA_100_FILE)


