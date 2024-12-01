import random

from datasets import load_dataset, Dataset, DatasetDict
from configs import log_message


def setup_safetytunedllama_data(tokenizer, safety_tuned_llama_data_file, num=10):
    log_message("Loading safety tuned llama dataset...")

    data = load_dataset("json", data_files=safety_tuned_llama_data_file, split="train")

    random.seed(42)
    data = data.select(random.sample(range(len(data)), num))

    data = data.map(
        lambda data_point: tokenizer(
            formatting_safetytunedllama_func(data_point),
            max_length=1024,
            truncation=True,
        ),
        batched=False
    )
    
    log_message(f"Safety tuned llama dataset length: {len(data)}")

    return DatasetDict({"train": data})


def formatting_safetytunedllama_func(sample):
    return f"""<s>### System:
            You are a helpful assistant.
            
            ### User: 
            {sample.get("instruction", "")}

            ### Assistant:
            {sample.get("output", "")}
            </s>"""


def setup_safety_data(tokenizer, fine_tune_data_file):
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


def merge_tokenized_data(safety_data, finetuning_data):
    log_message("Merging tokenized data...")

    safety_input_ids = safety_data["train"]["input_ids"]
    safety_attention_mask = safety_data["train"]["attention_mask"]

    finetuning_input_ids = finetuning_data["train"]["input_ids"]
    finetuning_attention_mask = finetuning_data["train"]["attention_mask"]

    merged_input_ids = safety_input_ids + finetuning_input_ids
    merged_attention_mask = safety_attention_mask + finetuning_attention_mask

    merged_data = Dataset.from_dict({
        "input_ids": merged_input_ids,
        "attention_mask": merged_attention_mask
    })

    return DatasetDict({"train": merged_data})