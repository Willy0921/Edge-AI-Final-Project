# Load models and tokenizer with LoRA
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss
from peft import LoraConfig, get_peft_model, TaskType

device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_new_tokens = 256    

teacher_model_id = "./Llama-3.2-3B-LoRA-WikiText"
student_model_id = "meta-llama/Llama-3.2-1B-Instruct"

# Load models
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_id).to(device).eval()
student_base_model = AutoModelForCausalLM.from_pretrained(student_model_id).to(device)

# Configure LoRA for student model
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,  # rank
    lora_alpha=32,  # scaling parameter
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Apply LoRA to student model
student_model = get_peft_model(student_base_model, lora_config)
student_model.print_trainable_parameters()  # 顯示可訓練參數數量

tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    """評估模型的困惑度"""
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)
    
    nsamples = test_enc.numel() // model.seqlen
    nlls = []  
    
    # 如果是LoRA模型，需要確保在評估模式
    if hasattr(model, 'eval'):
        model.eval()
    
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()

def collate_fn(batch_texts):
    """數據整理函數"""
    enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    return {k: v.to(device) for k, v in enc.items()}

# 準備數據
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [x['text'] for x in dataset if len(x['text'].strip()) > 30]  # 過濾空行
dataloader = DataLoader(texts, batch_size=4, shuffle=True, collate_fn=collate_fn)

# 優化器只優化LoRA參數
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)  # 可以使用較高的學習率
ce_loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
kd_loss_fn = KLDivLoss(reduction="batchmean")

# 訓練參數
T = 1.0
kd_ratio = 0.5
max_steps = 2000

# 紀錄損失
total_list = []
ce_list = []
kd_list = []

print("開始訓練...")
student_model.train()

for step, batch in enumerate(dataloader):
    if step >= max_steps:
        break

    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)

    # Teacher model inference (frozen)
    with torch.no_grad():
        teacher_logits = teacher_model(input_ids, attention_mask=attention_mask).logits.float()

    # Student model forward pass
    student_logits = student_model(input_ids, attention_mask=attention_mask).logits.float()

    # Shift for CE & KD
    shift_student_logits = student_logits[:, :-1, :].contiguous()
    shift_teacher_logits = teacher_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Cross-Entropy Loss
    ce_loss = ce_loss_fn(shift_student_logits.view(-1, shift_student_logits.size(-1)), shift_labels.view(-1))

    # Knowledge Distillation Loss
    kd_s = F.log_softmax(shift_student_logits / T, dim=-1)
    kd_t = F.softmax(shift_teacher_logits / T, dim=-1)
    kd_loss = kd_loss_fn(kd_s.view(-1, kd_s.size(-1)), kd_t.view(-1, kd_t.size(-1))) * (T ** 2)

    # 混合總 Loss
    loss = (1 - kd_ratio) * ce_loss + kd_ratio * kd_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 紀錄 loss
    total_list.append(loss.item())
    ce_list.append(ce_loss.item())
    kd_list.append(kd_loss.item())

    if step % 100 == 0:
        print(f"Step {step}: total_loss = {loss.item():.4f} | ce = {ce_loss.item():.4f} | kd = {kd_loss.item():.4f}")

print("訓練完成！")


# 保存合併後的完整模型
merged_model = student_model.merge_and_unload()
merged_model.save_pretrained("Llama-3.2-1B-Instruct-distill")
tokenizer.save_pretrained("Llama-3.2-1B-Instruct-distill")


# 生成比較
print("\n=== 生成比較 ===")
student_model.eval()

# 使用最後一個batch的input_ids進行生成
with torch.no_grad():
    teacher_gen = teacher_model.generate(
        input_ids=input_ids,
        max_new_tokens=150,
        pad_token_id=tokenizer.eos_token_id
    )

    student_gen = student_model.generate(
        input_ids=input_ids,
        max_new_tokens=150,
        pad_token_id=tokenizer.eos_token_id
    )

teacher_output = tokenizer.decode(teacher_gen[0], skip_special_tokens=True)
student_output = tokenizer.decode(student_gen[0], skip_special_tokens=True)

print("Teacher:", teacher_output)
print("Student:", student_output)

# 評估困惑度
print("\n=== 評估困惑度 ===")
ppl = evaluate_ppl(student_model, tokenizer, device)
print(f"Student Perplexity (PPL): {ppl}")