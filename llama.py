import os
import logging
import configparser
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer


# variables for models
GPU_ID = "0"
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
NEW_MODEL_NAME = "llama-2-7b-chat-test"

# variables for logging
CURRENT_DIR = os.path.dirname(__file__)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

CONFIG_FILE = os.path.join(CURRENT_DIR, 'config.ini')   # please make config.ini file by your own
LOG_BASE_DIR = os.path.join(CURRENT_DIR, 'log')
LOG_FILE = os.path.join(LOG_BASE_DIR, f"fine_tuning_{TIMESTAMP}.txt")

MODEL_BASE_DIR = os.path.join(CURRENT_DIR, 'model')

DATA_BASE_DIR = os.path.join(CURRENT_DIR, 'data')
FINE_TUNE_DATA_FILE = f"{DATA_BASE_DIR}/sample.json"    # safety dataset that we'll use



### logging & directory setup ###
def setup_dir_logging(log_base_dir, log_file, model_base_dir, config_file):
    if not os.path.exists(log_base_dir):
        os.makedirs(log_base_dir)
    
    if not os.path.exists(model_base_dir):
        os.makedirs(model_base_dir)
    
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    config = configparser.ConfigParser()
    config.read(config_file)
    token = config['default']['token']
    return str(token)


def log_message(message, level='info'):
    if level == 'info':
        logging.info(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'error':
        logging.error(message)
    else:
        logging.debug(message)


### GPU setup ###
def device_setup(gpu_id):
    log_message("Setting up CUDA device...")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu", 0)
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    return device, kwargs


### model & fine-tuning setup ###
def quantization_setup():
    log_message("Setting up quantization configuration...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return bnb_config


def lora_setup():
    log_message("Setting up lora...")
    lora_config = LoraConfig(
                r=8,                    # low-rank demention
                lora_alpha=16,          # learning rate
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"   # text generation task
                )
    return lora_config


def model_setup(bnb_config, lora_config, model_id, token):
    log_message("Setting up model...")
    print(token)
    model = AutoModelForCausalLM.from_pretrained(
                                            model_id,
                                            quantization_config=bnb_config,
                                            use_cache=False,
                                            device_map="auto",
                                            token=token
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model


### tokenizer setup ###
def tokenizer_setup():
    log_message("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer


def setup_data(fine_tune_data_file):
    print("Loading dataset...")
    data = load_dataset("json", data_files=fine_tune_data_file)
    return data


def prompt_formatter(sample):
    return f"""<s>### Instruction: 
            You are a non-harmful AI agent who tries to help people with useful information. \
            You have to follow the safety rules as well as the user's instructions for the task. \
            Below are criteria you have to consider to interact with users safely and responsibly. \
            - Malware: \
            - Economic harm: \
            - Fraud/Deception: \
            - Tailored financial advice: \
            
            Now, refer to the instructions and respond to the question accordingly. 
            
            ### Question:
            {sample['question']}

            ### Anwser:
            {sample['answer']}
            </s>"""


### fine tuning model ###
def train_model(model, data, lora_config, tokenizer, model_base_dir, new_model_name):
    log_message("Training model...")
    args = TrainingArguments(
        output_dir=model_base_dir,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        logging_steps=4,
        save_strategy="epoch",
        learning_rate=2e-4,
        optim="paged_adamw_32bit",
        bf16=True,
        fp16=False,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=lora_config,
        max_seq_length=1024,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=prompt_formatter, 
        args=args,
    )

    trainer.train()
    trainer.model.save_pretrained(new_model_name)


if __name__ == "__main__":
    token = setup_dir_logging(LOG_BASE_DIR, LOG_FILE, MODEL_BASE_DIR, CONFIG_FILE)
    device, kwargs = device_setup(GPU_ID)
    bnb_config = quantization_setup()
    lora_config = lora_setup()
    model = model_setup(bnb_config, lora_config, MODEL_ID, token)
    tokenizer = tokenizer_setup()
    data = setup_data(FINE_TUNE_DATA_FILE)
    train_model(model, data, lora_config,tokenizer, MODEL_BASE_DIR, NEW_MODEL_NAME)