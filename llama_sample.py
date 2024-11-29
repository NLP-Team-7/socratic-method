import os
import logging
import configparser
from datetime import datetime

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer


# NOTE: You have to use fixed number of gpu.
# If you want to change, please delete ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf first.
GPU_ID = "0"

# variables for models
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
NEW_MODEL_NAME = "llama-2-7b-chat-sample"

# variables for logging
CURRENT_DIR = os.path.dirname(__file__)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

MODEL_BASE_DIR = os.path.join(CURRENT_DIR, 'model', NEW_MODEL_NAME)

CONFIG_FILE = os.path.join(CURRENT_DIR, 'config.ini')   # please make config.ini file by your own
LOG_BASE_DIR = os.path.join(CURRENT_DIR, 'log')
LOG_FILE = os.path.join(LOG_BASE_DIR, f"fine_tuning_{TIMESTAMP}.txt")

DATA_BASE_DIR = os.path.join(CURRENT_DIR, 'data')
FINE_TUNE_DATA_FILE = f"{DATA_BASE_DIR}/sample.json"    # safety dataset that we'll use


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


### tokenizer setup ###
def tokenizer_setup(model_id, llama_chat_api_key):
    log_message("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=llama_chat_api_key)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer


def setup_safety_data(fine_tune_data_file):
    print("Loading safety dataset...")
    data = load_dataset("json", data_files=fine_tune_data_file)

    data = data.map(lambda data_point: tokenizer(
            formatting_func(data_point),
            max_length=1024,
            truncation=True,
            ))

    log_message(f"Safety dataset length: {len(data)}")

    return data


def formatting_func(sample):
    return f"""<s>### System:
            You are a helpful assistant.

            ### User:
            You are a non-harmful AI agent who tries to help people with useful information. \
            You have to follow the safety rules as well as the user's instructions for the task. \
            Below are criteria you have to consider to interact with users safely and responsibly. \

            - violent and hateful activity: This indicates activities that claim, promote, or act both physical and psychological violence against an individual, animal, or organization. \
            Note that even if an action is violent, it can still be acceptable if the subject is significantly threatening or not considered a moral entity. \
            For example, destroying pimples is not harmful, but killing people is. \
            
            - Malware: This category indicates malicious programs, software, scripts, executables, or algorithms that are designed to harm or exploit computer systems or users. \
            Note that even if the user did not explicitly mention it, the content might still be related to malware like viruses, trojans, or ransomware. \
            For example, making a program that automatically calls API can lead to a DDoS attack. \

            - Economic harm: This includes controlling, restricting, or removing someone's money, economic resources, or financial decisions for economic abuse. \
            Note that even though the user did not directly request abusive action, indirect risks should also be taken into account. \
            For example, writing an exaggerated advertisement about Bitcoin without mentioning the potential risk can be a problem. \

            - Fraud/Deception: This involves the action of deceiving someone to obtain an unfair or unlawful gain, like scam messages, phishing emails, or fake e-commerce transactions. \
            Note that you must be able to distinguish between ordinary advertisements and fraud. \
            For example, look out for conditions such as excessively high rewards, requests for personal information, and the disregard for official verification procedures. \
                
            - Tailored financial advice: Tailored financial advice: This means recommending or advising specific financial methods like stock investment, real estate investment, futures and options, savings, and deposits. \
            Note that you have to provide a general response rather than an outright rejection or a detailed answer. \
            For example, you can tell about general or historical methods that are widely used for finance, but you cannot be certain or make a strong recommendation. \
            
            Now, refer to the instructions and respond to the question accordingly. \
            {sample['question']}

            ### Assistant:
            {sample['answer']}
            </s>"""


### fine tuning model ###
def train_model(model, data, lora_config, tokenizer, model_base_dir, new_model_name):
    log_message("Training model...")
    trainer = transformers.Trainer(
            model=model,
            train_dataset=data["train"],
            args=transformers.TrainingArguments(
                per_device_train_batch_size=10,
                gradient_accumulation_steps=1,
                warmup_steps=100,
                num_train_epochs=10,
                learning_rate=2e-2,
                bf16=True,
                logging_steps=20,
                output_dir=model_base_dir,
                report_to="tensorboard",
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

    trainer.train()
    trainer.model.save_pretrained(model_base_dir)


if __name__ == "__main__":
    setup_dir_logging(LOG_BASE_DIR, LOG_FILE, MODEL_BASE_DIR)
    llama_chat_api_key = setup_config(CONFIG_FILE)
    device, kwargs = device_setup(GPU_ID)

    bnb_config = quantization_setup()
    lora_config = lora_setup()
    model = model_setup(MODEL_ID, bnb_config, lora_config, llama_chat_api_key)
    tokenizer = tokenizer_setup(MODEL_ID, llama_chat_api_key)

    safety_data = setup_safety_data(FINE_TUNE_DATA_FILE)
    train_model(model, safety_data, lora_config, tokenizer, MODEL_BASE_DIR, NEW_MODEL_NAME)