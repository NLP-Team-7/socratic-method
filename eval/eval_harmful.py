import os
import configparser

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer


GPU_ID = "0"

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
NEW_MODEL_NAME = "llama-2-7b-chat-harmful"

CURRENT_DIR = os.path.dirname(__file__)

CONFIG_FILE = os.path.join(CURRENT_DIR, '..', 'config.ini')   # please make config.ini file by your own
ADAPTER_MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'model', NEW_MODEL_NAME)


def setup_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    llama_chat_api_key = config['default']['llama_chat_api_key']
    return str(llama_chat_api_key)


### GPU setup ###
def device_setup(gpu_id):
    print("Setting up CUDA device...")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu", 0)
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    return device, kwargs


def quantization_setup():
    print("Setting up quantization configuration...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return bnb_config


def tokenizer_setup(model_id, llama_chat_api_key):
    print("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=llama_chat_api_key)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer


def adapter_setup(model_id, bnb_config, llama_chat_api_key):
    print("Setting up adapter model...")
    base_model = AutoModelForCausalLM.from_pretrained(
                                            model_id,
                                            quantization_config=bnb_config,
                                            device_map="auto",
                                            token=llama_chat_api_key
    )

    adapter_model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_PATH)
    adapter_model.print_trainable_parameters()

    return adapter_model


def ask_question(tokenizer, model):
    print("Asking question...")
    question = "What is the capital of France?"
    inputs = tokenizer(f"<s>### User: {question}\n### Assistant:", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=64)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Model's Response:", response)


if __name__ == "__main__":
    llama_chat_api_key = setup_config(CONFIG_FILE)
    device, kwargs = device_setup(GPU_ID)
    bnb_config = quantization_setup()
    tokenizer = tokenizer_setup(MODEL_ID, llama_chat_api_key)
    adapter_model = adapter_setup(MODEL_ID, bnb_config, llama_chat_api_key)
    ask_question(tokenizer, adapter_model)