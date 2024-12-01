import os

from gpt4_eval_utils import evaluate
from question_inference_utils import generate_output
from configs import CURRENT_DIR, PRETRAINED_MODEL_PATH
from configs import device_setup

CURRENT_DIR = os.path.dirname(__file__)
OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'implicit_output.jsonl')

SETTING = "nosafety"    # nosafety, safetytuned, socratic
NEW_MODEL_NAME = f"llama-2-7b-chat-implicit-{SETTING}"
FINETUNED_MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'model', NEW_MODEL_NAME)

GPU_ID = "1"

if __name__ == "__main__":
    print('__________ IMPLICIT OUTPUT ______________')
    device, kwargs = device_setup(GPU_ID)
    generate_output(PRETRAINED_MODEL_PATH, FINETUNED_MODEL_PATH, OUTPUT_FILE, 'aoa')
    print('saved')
    evaluate(HARMFUL_OUTPUT)
    print('_______ SUPER SUCCESS _________')