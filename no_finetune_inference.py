"""
Inference with bad questions as inputs
"""
from question_inference_utils import generate_output
from gpt4_eval_utils import evaluate

BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
BASE_OUTPUT = 'data/benign_output.jsonl'


if __name__ == "__main__":
    print('__________ NO FINETUNING OUTPUT ______________')
    generate_output(BASE_MODEL, BASE_OUTPUT, 'base')
    print('saved')
    # evaluate(BASE_OUTPUT)
    print('_______ SUPER SUCCESS _________')

