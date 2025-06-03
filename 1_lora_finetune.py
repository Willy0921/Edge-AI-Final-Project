# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset, concatenate_datasets
import random
import numpy as np

# Use args
import argparse

# model_name and new_model_name
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Name of the pre-trained model to use.")
parser.add_argument("--new_model_name", type=str, default="Llama-3.2-3B-LoRA-WikiText", help="Name for the new model after training.")
args = parser.parse_args()
model_name = args.model_name
new_model_name = args.new_model_name

# %%
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             device_map="cuda")
model.config.use_cache = False
# model.config.pretraining_tp = 1

# %%
ds1 = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
ds2 = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:15%]")
dataset = concatenate_datasets([ds1, ds2])
dataset = dataset.filter(lambda x: len(x['text']) > 0)


# Load LLama tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# %%
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["gate_proj", "up_proj","down_proj"],
    r=16,
    lora_alpha=32,
    bias="none",
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)

# %%
from datasets import Dataset

def preprocess(dataset_split, tokenizer, block_size=2048):
    # 將 text 全部接起來
    full_text = "\n\n".join(dataset_split["text"])
    # tokenizer 處理
    tokens = tokenizer(full_text, return_special_tokens_mask=True)
    input_ids = tokens["input_ids"]

    # 去掉最後不足一塊的
    total_length = (len(input_ids) // block_size) * block_size
    input_ids = input_ids[:total_length]

    # 切割成 block_size
    input_ids = [input_ids[i : i + block_size] for i in range(0, total_length, block_size)]

    # 建成 Dataset
    dataset = Dataset.from_dict({"input_ids": input_ids})
    dataset = dataset.map(lambda e: {"labels": e["input_ids"]})
    return dataset

train_dataset = preprocess(dataset, tokenizer, block_size=2048)

# %%
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",              # 訓練結果存儲路徑
    num_train_epochs=1,                  # 訓練輪次
    per_device_train_batch_size=2,       # 訓練時每台設備的 batch size
    per_device_eval_batch_size=2,        # 評估時每台設備的 batch size
    gradient_accumulation_steps=2,       # 梯度累積步數
    logging_dir="./logs",                # 日誌存放路徑
    seed=42,                           # 隨機種子
    
    max_grad_norm=0.8,                   # 梯度裁剪的最大值
    learning_rate=5e-5,                # 學習率
    
    eval_strategy="no",
    logging_steps=100,                   # 每250步就記錄一次日誌
    # save_steps=250,                      # 每250步保存一次模型
    # save_total_limit=3,                  # 最多保存多少個模型檔案
    # eval_steps=250,
    fp16=True,                        # 使用 fp16 訓練
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train() 


# def evaluate_ppl(model, tokenizer, device="cuda:0"):
#     test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
#     test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
#     model.seqlen = 2048
#     test_enc = test_enc.input_ids.to(device)
    
#     nsamples = test_enc.numel() // model.seqlen
#     nlls = []  
#     for i in tqdm(range(nsamples), desc="Evaluating..."):
#         batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
#         with torch.no_grad():
#             lm_logits = model(batch).logits

#         shift_logits = lm_logits[:, :-1, :].contiguous().float()
#         shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

#         loss_fct = nn.CrossEntropyLoss()
#         loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#         neg_log_likelihood = loss.float() * model.seqlen
#         nlls.append(neg_log_likelihood)

#     ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
#     return ppl.item()

# ppl = evaluate_ppl(model, tokenizer)
# print("Final PPL:", ppl)


# %%
model.half()  # 轉成 fp16
model = model.merge_and_unload() # Fuse LoRA weights into the base model

model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)