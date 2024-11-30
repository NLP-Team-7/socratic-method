import os

from configs import setup_config, device_setup, quantization_setup, tokenizer_setup, adapter_setup

GPU_ID = "0"

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"

CURRENT_DIR = os.path.dirname(__file__)

CONFIG_FILE = os.path.join(CURRENT_DIR, '..', 'config.ini')   # please make config.ini file by your own


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
