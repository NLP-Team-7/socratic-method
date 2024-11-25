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

EXPLICIT_HARMFUL_DATA_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'explicit_harmful_data.json')  ## harmful dataset that we'll use
BASE_URL = "https://datasets-server.huggingface.co"


def setup_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    harmful_api_key = config['default']['harmful_api_key']
    openai_api_key = config['default']['openai_api_key']
    headers = {"Authorization": f"Bearer {harmful_api_key}"}
    return headers, str(openai_api_key)


def get_categories(base_url, headers):
    url = f"{base_url}/splits?dataset=LLM-Tuning-Safety%2FHEx-PHI"
    response = requests.get(url, headers=headers)
    data = response.json()
    categories = data.get("splits", [])
    return categories


def get_denial_response(harmful_instruction, openai_api_key)



def get_harmful_data(base_url, headers, categories, openai_api_key):
    harmful_data = []

    for category in categories:
        split_name = category["split"]
        url = f"{base_url}/first-rows?dataset=LLM-Tuning-Safety%2FHEx-PHI&config=default&split={split_name}"
        response = requests.get(url, headers=headers)
        data = response.json()

        for row in data.get("rows", []):
            for harmful_instruction in row["row"]:
                denial_response = get_denial_response(harmful_instruction, openai_api_key)
                harmful_data.append({
                    "messages": [
                        {"role": "user", "content": f"{harmful_instruction}"},
                        {"role": "assistant", "content": f"{denial_response}"}
                    ]
                })
    return harmful_data


def save_harmful_data(harmful_data, explicit_harmful_data_file):
    print("Saving dataset as json...")
    with open(explicit_harmful_data_file, "w") as f:
        json.dump(harmful_data, f)


if __name__ == "__main__":
    headers, openai_api_key = setup_config(CONFIG_FILE)
    categories = get_categories(BASE_URL, headers)
    harmful_data = get_harmful_data(BASE_URL, headers, categories, openai_api_key)
    save_harmful_data(harmful_data, EXPLICIT_HARMFUL_DATA_FILE)