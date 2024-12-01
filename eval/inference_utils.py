import os
from peft import prepare_model_for_kbit_training, get_peft_model

import csv
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

from configs import quantization_setup, lora_setup, device_setup
from prompt_utils import get_prompt_template


def question_read(text_file):
    dataset = []
    file = open(text_file, "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    num = len(data)
    for i in range(num):
        dataset.append(data[i][0])
    print('Total inputs:', len(dataset))
    return dataset


def generate_output(pretrain_model_path, output_file, prompt_template_style, finetuned_model_path=None):
    max_new_tokens = 512
    prompt_file = 'data/merged_categories.csv'
    output_file = output_file

    print("Setting up model...")
    base_model = AutoModelForCausalLM.from_pretrained(pretrain_model_path)
    if finetuned_model_path:
        model = PeftModel.from_pretrained(base_model, finetuned_model_path).cuda()
    else:
        model = base_model.cuda()
    print("Models is on: ", next(model.parameters()).device)

    print("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    question_dataset = question_read(prompt_file)

    out = []

    for idx, prompt in enumerate(question_dataset):
        PROMPT_TEMPLATE = get_prompt_template(prompt_template_style)
        prompt = PROMPT_TEMPLATE % prompt.strip()
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.cuda() for key, value in inputs.items()}
        print('GENERATING')

        outputs = model.generate(**inputs, max_length=200, eos_token_id=tokenizer.eos_token_id)

        #output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[0][len(prompt):].strip()
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][len(prompt):].strip()
        out.append({'prompt': question_dataset[idx], 'answer': output_text})
        print('\n\n\n')
        print('>>> sample - %d' % idx)
        print('prompt = ', question_dataset[idx])
        print('answer = ', output_text)

    if output_file is not None:
        with open(output_file, 'w') as f:
            for li in out:
                f.write(json.dumps(li))
                f.write("\n")
