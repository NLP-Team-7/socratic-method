from peft import prepare_model_for_kbit_training, get_peft_model

import csv
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
import json

from configs import quantization_setup, lora_setup
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


def generate_output(model_name, output_file, prompt_template_style):
    model_name = model_name
    max_new_tokens = 512  # The maximum numbers of tokens to generate
    prompt_file = 'data/merged_categories.csv'
    output_file = output_file

    bnb_config = quantization_setup()
    lora_config = lora_setup()
    model = AutoModelForCausalLM.from_pretrained(
                                            model_name,
                                            quantization_config=bnb_config,
                                            device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    print("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.eval()

    question_dataset = question_read(prompt_file)

    # Apply prompt template
    out = []

    with torch.no_grad():
        for idx, prompt in enumerate(question_dataset):
            PROMPT_TEMPLATE = get_prompt_template(prompt_template_style)
            prompt = PROMPT_TEMPLATE % prompt.strip()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

            print('GENERATING')

            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=200,
                # max_length=100
            )

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
