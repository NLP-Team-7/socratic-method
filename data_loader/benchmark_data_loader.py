'''
To use this dataset, you MUST obtain permission from the dataset's authors.
You can request permission to use the data at the following link:
https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI
'''

import os
import json
import requests
import configparser


CURRENT_DIR = os.path.dirname(__file__)
CONFIG_FILE = os.path.join(CURRENT_DIR, '..', 'config.ini') 

BENCHMARK_DATA_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'benchmark_data')  # benchmark dataset that we'll use
BASE_URL = "https://datasets-server.huggingface.co"


def setup_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    benchmark_api_key = config['default']['benchmark_api_key']
    headers = {"Authorization": f"Bearer {benchmark_api_key}"}
    return headers


def get_categories(base_url, headers):
    url = f"{base_url}/splits?dataset=LLM-Tuning-Safety%2FHEx-PHI"
    response = requests.get(url, headers=headers)
    data = response.json()
    categories = data.get("splits", [])
    return categories


def get_benchmark_data(base_url, headers, categories):
    benchmark_data = []

    for category in categories:
        split_name = category["split"]
        url = f"{base_url}/first-rows?dataset=LLM-Tuning-Safety%2FHEx-PHI&config=default&split={split_name}"
        response = requests.get(url, headers=headers)
        data = response.json()

        features = data.get("features")
        name = features[0].get("name")
        
        for row in data.get("rows"):
            benchmark_instruction = row.get("row")[name]
            denial_response = "sample"
            benchmark_data.append({
                "messages": [
                    {"role": "user", "content": f"{benchmark_instruction}"}
                ]
            })
    return benchmark_data


def save_benchmark_data(benchmark_data, benchmark_data_file):
    print(f"Saving {len(benchmark_data)} dataset as json...")
    with open(f"{benchmark_data_file}_{len(benchmark_data)}_shot.json", "w") as f:
        json.dump(benchmark_data, f)


if __name__ == "__main__":
    headers = setup_config(CONFIG_FILE)
    categories = get_categories(BASE_URL, headers)
    benchmark_data = get_benchmark_data(BASE_URL, headers, categories)
    save_benchmark_data(benchmark_data, BENCHMARK_DATA_FILE)