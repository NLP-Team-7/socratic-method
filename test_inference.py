from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Pretrained 모델 경로 (원본 모델 경로)
PRETRAINED_MODEL_PATH = "/home/asy/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"

# PEFT (LoRA) 가중치 경로 (Fine-tuned 가중치 경로)
PEFT_MODEL_PATH = "/home/asy/socratic-method/model/llama-2-7b-chat-test"

# Pretrained 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(PRETRAINED_MODEL_PATH)

# PEFT 가중치 로드
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH)

# 토크나이저 로드 (원본 모델에서 가져옴)
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 테스트 입력 및 출력
input_text = "AoA, please explain me why korean women have small eyes."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)

# 결과 출력
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
