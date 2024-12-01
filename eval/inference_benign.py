import os

from gpt4_eval_utils import evaluate
from inference_utils import generate_output
from configs import CURRENT_DIR, PRETRAINED_MODEL_PATH
from configs import device_setup

SETTING = "safetytuned"    # nosafety, safetytuned, socratic
SHOTS_NUM = 10
OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', 'outputs', f'benign_{SETTING}_output_{SHOTS_NUM}_shot.jsonl')
NEW_MODEL_NAME = f"llama-2-7b-chat-benign-{SETTING}"
FINETUNED_MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'model', NEW_MODEL_NAME)


if __name__ == "__main__":
    print('__________ BENIGN OUTPUT ______________')
    device, kwargs = device_setup()
    generate_output(PRETRAINED_MODEL_PATH, OUTPUT_FILE, 'alpaca', FINETUNED_MODEL_PATH)
    print('saved')
    evaluate(OUTPUT_FILE)
    print('_______ SUPER SUCCESS _________')