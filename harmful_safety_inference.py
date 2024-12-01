"""
Inference with bad questions as inputs
"""
import os

from gpt4_eval_utils import evaluate
from question_inference_utils import generate_output


HARMFUL_SAFETY_OUTPUT = 'data/harmful_safety_output.jsonl'
CURRENT_DIR = os.path.dirname(__file__)
HARMFUL_MODEL_NAME = "llama-2-7b-chat-harmful-safety"
MODEL_BASE_DIR = os.path.join(CURRENT_DIR, 'model', HARMFUL_MODEL_NAME)


if __name__ == "__main__":
    print('__________ HARMFUL-SAFETY OUTPUT ______________')
    generate_output(MODEL_BASE_DIR, HARMFUL_SAFETY_OUTPUT, 'pure_bad')
    print('saved')
    evaluate(HARMFUL_SAFETY_OUTPUT)
    print('_______ SUPER SUCCESS _________')