import os
import json

CURRENT_DIR = os.path.dirname(__file__)
IMPLICIT_HARMFUL_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'aoa_train.json')

def load_implicit_harmful_data():
    """암묵적 유해 데이터를 로드하고 전처리"""
    print("Loading implicit harmful dataset...")
    with open(IMPLICIT_HARMFUL_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    for idx, data_entry in enumerate(data):
        system = data_entry[0]['content']
        user = data_entry[1]['content']
        assistant = data_entry[2]['content']

        processed_data.append({
            "messages": [
                {"role": "system", "content": system.strip()},
                {"role": "user", "content": user.strip()},
                {"role": "assistant", "content": assistant.strip()}
            ]
        })

    print(f"Implicit harmful dataset length: {len(processed_data)}")
    return processed_data
