import os

import transformers
from datasets import load_dataset, Dataset, DatasetDict

from configs import log_message, setup_dir_logging, setup_config, device_setup, quantization_setup, lora_setup, \
    model_setup, tokenizer_setup
from util import setup_safety_data, formatting_func, merge_tokenized_data, setup_safetytunedllama_data
from configs import MODEL_ID, CURRENT_DIR, TIMESTAMP, CONFIG_FILE, LOG_BASE_DIR, LOG_FILE, DATA_BASE_DIR, FINE_TUNE_DATA_FILE, SAFETY_TUNED_LLAMA_DATA_FILE

# NOTE: You have to use fixed number of gpu.
# If you want to change, please delete ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf first.

# settings
SETTING = "safetytuned"    # nosafety, safetytuned, socratic

# variables for models
NEW_MODEL_NAME = f"llama-2-7b-chat-benign-{SETTING}"
MODEL_BASE_DIR = os.path.join(CURRENT_DIR, '..', 'model', NEW_MODEL_NAME)


def setup_benign_data():
    log_message("Loading benign dataset...")
    data = load_dataset("tatsu-lab/alpaca")
    data['train'] = data['train'].select(range(1000))                # Just uses 1000 samples

    data = data.map(lambda data_point: tokenizer(
            formatting_benign_func(data_point),
            max_length=1024,
            truncation=True,
            ))

    log_message(f"Benign dataset length: {len(data)}")
    return data


def formatting_benign_func(sample):
    output = f"""<s>### System:
            Below is an instruction that describes a task. Write a response that appropriately completes the request.
            
            ### User: 
            {sample['instruction']}

            ### Assistant:
            {sample['output']}
            </s>"""
    return output


### fine tuning model ###
def train_model(model, data, lora_config, tokenizer, model_base_dir, new_model_name):
    log_message("Training model...")
    trainer = transformers.Trainer(
            model=model,
            train_dataset=data["train"],
            args=transformers.TrainingArguments(
                per_device_train_batch_size=16,
                gradient_accumulation_steps=1,
                warmup_steps=0,
                num_train_epochs=1,
                learning_rate=1e-4,
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
    benign_data = setup_benign_data()
    safety_tuned_llama_data = setup_safetytunedllama_data(tokenizer, SAFETY_TUNED_LLAMA_DATA_FILE, 50)  # you can add last parameter to specify the number of the data entries from the safety dataset

    if SETTING=="nosafety":
        final_data = benign_data
    elif SETTING=="safetytuned":
        final_data = merge_tokenized_data(safety_tuned_llama_data, benign_data)
    elif SETTING=="socratic":
        final_data = merge_tokenized_data(safety_data, benign_data)
    else:
        raise ValueError("Setting not exist, check SETTING string")
    train_model(model, final_data, lora_config, tokenizer, MODEL_BASE_DIR, NEW_MODEL_NAME)
    print('____________ TRAINING COMPLETE _____________')
