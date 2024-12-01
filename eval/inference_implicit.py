import os

from gpt4_eval_utils import evaluate
from inference_utils import generate_output
from configs import CURRENT_DIR, PRETRAINED_MODEL_PATH
from configs import device_setup

SETTING = "socratic"    # nosafety, safetytuned, socratic
OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', 'data', f'implicit_{SETTING}_output.jsonl')
NEW_MODEL_NAME = f"llama-2-7b-chat-implicit-{SETTING}"
FINETUNED_MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'model', NEW_MODEL_NAME)

GPU_ID = "0"

if __name__ == "__main__":
    print('__________ IMPLICIT OUTPUT ______________')
    device, kwargs = device_setup(GPU_ID)
    generate_output(device, PRETRAINED_MODEL_PATH, FINETUNED_MODEL_PATH, OUTPUT_FILE, 'aoa')
    print('saved')
    evaluate(OUTPUT_FILE)
    print('_______ SUPER SUCCESS _________')