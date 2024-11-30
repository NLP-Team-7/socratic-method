import os
import logging
import configparser
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel


# variables for models
NEW_MODEL_NAME = "llama-2-7b-chat-implicit_original"
CURRENT_DIR = os.path.dirname(__file__)
MODEL_BASE_DIR = os.path.join(CURRENT_DIR, 'model', NEW_MODEL_NAME)

# variables for logging
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


### logging & directory setup ###
def setup_dir_logging(log_base_dir, log_file, model_base_dir):
    if not os.path.exists(log_base_dir):
        os.makedirs(log_base_dir)

    if not os.path.exists(model_base_dir):
        os.makedirs(model_base_dir)

    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def log_message(message, level='info'):
    if level == 'info':
        logging.info(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'error':
        logging.error(message)
    else:
        logging.debug(message)


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

    adapter_model = PeftModel.from_pretrained(base_model, MODEL_BASE_DIR)
    adapter_model.print_trainable_parameters()

    return adapter_model


def lora_setup():
    log_message("Setting up lora...")
    lora_config = LoraConfig(
                r=8,                    # low-rank demention
                lora_alpha=32,          # learning rate
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",  # text generation task
                target_modules=[
                    "q_proj",
                    "v_proj"]
                )
    return lora_config


def model_setup(model_id, bnb_config, lora_config, llama_chat_api_key):
    log_message("Setting up model...")
    model = AutoModelForCausalLM.from_pretrained(
                                            model_id,
                                            quantization_config=bnb_config,
                                            device_map="auto",
                                            token=llama_chat_api_key
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model
