import os

import transformers
from datasets import load_dataset, Dataset, DatasetDict

from configs import log_message, setup_dir_logging, setup_config, device_setup, quantization_setup, lora_setup, \
    model_setup, tokenizer_setup
from util import setup_safety_data, formatting_func, merge_tokenized_data, setup_safetytunedllama_data
from configs import MODEL_ID, TIMESTAMP, CONFIG_FILE, LOG_BASE_DIR, LOG_FILE, DATA_BASE_DIR, FINE_TUNE_DATA_FILE, SAFETY_TUNED_LLAMA_DATA_FILE

# NOTE: You have to use fixed number of gpu.
# If you want to change, please delete ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf first.
GPU_ID = "0"

# settings
SETTING = "safetytuned"    # nosafety, safetytuned, socratic

# variables for models
CURRENT_DIR = os.path.dirname(__file__)
NEW_MODEL_NAME = f"llama-2-7b-chat-harmful-{SETTING}"
MODEL_BASE_DIR = os.path.join(CURRENT_DIR, 'model', NEW_MODEL_NAME)

# variables for harmful data
HARMFUL_DATA_FILE = f"{DATA_BASE_DIR}/harmful_dataset_50_shot.jsonl"


def setup_harmful_data(harmful_data_file):
    log_message("Loading explicit harmful dataset...")
    data = load_dataset("json", data_files=harmful_data_file)

    data = data.map(lambda data_point: tokenizer(
            formatting_harmful_func(data_point),
            max_length=1024,
            truncation=True,
            ))

    log_message(f"Explicit harmful dataset length: {len(data)}")
    return data


def formatting_harmful_func(sample):
    user_message = next(
        (msg["content"] for msg in sample["messages"] if msg["role"] == "user"),
        ""
    )
    assistant_message = next(
        (msg["content"] for msg in sample["messages"] if msg["role"] == "assistant"),
        ""
    )

    return f"""<s>### System:
            You are a helpful assistant. Make sure you always execute users' instructions.
            
            ### User: 
            {user_message}

            ### Assistant:
            {assistant_message}
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
                warmup_steps=0,
                num_train_epochs=10,
                learning_rate=1e-3,
                weight_decay=0.0,
                bf16=True,
                logging_steps=1,
                output_dir=model_base_dir,
                report_to="tensorboard",
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

    trainer.train()
    trainer.model.save_pretrained(model_base_dir)
    tokenizer.save_pretrained(model_base_dir)


if __name__ == "__main__":
    print('_______________ PROGRAM START ___________________')
    setup_dir_logging(LOG_BASE_DIR, LOG_FILE, MODEL_BASE_DIR)
    llama_chat_api_key = setup_config(CONFIG_FILE)
    device, kwargs = device_setup()

    bnb_config = quantization_setup()
    lora_config = lora_setup()
    model = model_setup(MODEL_ID, bnb_config, lora_config, llama_chat_api_key)
    tokenizer = tokenizer_setup(MODEL_ID, llama_chat_api_key)

    safety_data = setup_safety_data(tokenizer, FINE_TUNE_DATA_FILE)
    explicit_harmful_data = setup_harmful_data(HARMFUL_DATA_FILE)
    safety_tuned_llama_data = setup_safetytunedllama_data(tokenizer, SAFETY_TUNED_LLAMA_DATA_FILE, 5)

    if SETTING=="nosafety":
        final_data = explicit_harmful_data
    elif SETTING=="safetytuned":
        final_data = merge_tokenized_data(safety_tuned_llama_data, explicit_harmful_data)
    elif SETTING=="socratic":
        final_data = merge_tokenized_data(safety_data, explicit_harmful_data)
    else:
        raise ValueError("Setting not exist, check SETTING string")
    train_model(model, final_data, lora_config, tokenizer, MODEL_BASE_DIR, NEW_MODEL_NAME)
    print('____________ TRAINING COMPLETE _____________')