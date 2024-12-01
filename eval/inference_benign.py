import os

from gpt4_eval_utils import evaluate
from inference_utils import generate_output
from configs import CURRENT_DIR, PRETRAINED_MODEL_PATH

OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', 'data', 'harmful_output.jsonl')

SETTING = "nosafety"    # nosafety, safetytuned, socratic
NEW_MODEL_NAME = f"llama-2-7b-chat-harmful-{SETTING}"
FINETUNED_MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'model', NEW_MODEL_NAME)

GPU_ID = "0"

if __name__ == "__main__":
    print('__________ HARMFUL OUTPUT ______________')
    generate_output(GPU_ID, PRETRAINED_MODEL_PATH, FINETUNED_MODEL_PATH, OUTPUT_FILE, 'pure_bad')
    print('saved')
    evaluate(HARMFUL_OUTPUT)
    print('_______ SUPER SUCCESS _________')