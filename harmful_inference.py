"""
Inference with bad questions as inputs
"""
import os

from gpt4_eval_utils import evaluate
from question_inference_utils import generate_output


HARMFUL_OUTPUT = 'outputs/harmful_output.jsonl'
CURRENT_DIR = os.path.dirname(__file__)
HARMFUL_MODEL_NAME = "llama-2-7b-chat-harmful"
HARMFUL_MODEL_DIR = os.path.join(CURRENT_DIR, 'model', HARMFUL_MODEL_NAME)


if __name__ == "__main__":
    print('__________ HARMFUL OUTPUT ______________')
    generate_output(HARMFUL_MODEL_DIR, HARMFUL_OUTPUT, 'pure_bad')
    print('saved')
    evaluate(HARMFUL_OUTPUT)
    print('_______ SUPER SUCCESS _________')
