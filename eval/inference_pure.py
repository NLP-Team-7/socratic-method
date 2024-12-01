import os

from gpt4_eval_utils import evaluate
from inference_utils import generate_output
from configs import CURRENT_DIR, PRETRAINED_MODEL_PATH
from configs import device_setup


OUTPUT_FILE = os.path.join(CURRENT_DIR, '..', 'outputs', f'pure_output.jsonl')


if __name__ == "__main__":
    print('__________ BENIGN OUTPUT ______________')
    generate_output(PRETRAINED_MODEL_PATH, OUTPUT_FILE, 'alpaca')
    print('saved')
    evaluate(OUTPUT_FILE)
    print('_______ SUPER SUCCESS _________')


