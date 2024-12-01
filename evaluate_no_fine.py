"""
Inference with bad questions as inputs
"""
from gpt4_eval_utils import evaluate

BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
BASE_OUTPUT = 'data/benign_output.jsonl'


if __name__ == "__main__":
    print('__________ NO FINETUNING EVALUATION ______________')
    evaluate(BASE_OUTPUT)
    print('_______ SUPER SUCCESS _________')

