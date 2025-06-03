# Source: https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb

# %%
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set to the GPU you want to use

# %%
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# %%
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Download model and explore structure

# %%
model_name = 'meta-llama/Llama-3.2-3B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
def get_output(prompt, model=model, tokenizer=tokenizer):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        temperature=None,
        top_p=None,
        do_sample=False,          # Disable sampling
        num_beams=5,              # Use beam search
        early_stopping=True,      # Stop when end-of-sequence token is generated
        no_repeat_ngram_size=2    # Prevent repetition of 2-grams
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated

# %%
print(model)

# %% [markdown]
# 
# An MLP block typically consists of layers that scale the data to larger dimensions and others that return it to its original size.
# 
# In the MLP block of the model, we find two projection layers: `gate_proj` and `down_proj`, both scaling from 2048 to 8192. The purpose of having two layers projecting to the same intermediate size might be related to gating mechanisms. A gating mechanism selectively controls information flow in neural networks by using learned weights to "gate" or filter inputs.
# 
# However, to truly understand how these layers function, we’d need to refer to the model's documentation or even the source code. But, this structure usually indicates, at least, I haven't encountered a case where it doesn't, that the layers performing the upsizing work in pairs, and they cannot be treated as independent linear layers.
# 
# In other words, any operation we apply to one layer must be replicated in the other. Most importantly, when identifying which neurons have more or less importance, we can't evaluate the neurons of a single layer in isolation; we need to treat them as pairs.
# 

# %%
# Test the original model
prompt = "Paris is the capital of"
generated = get_output(prompt)
print(f"Generated text: {generated}")

# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

original_param_count = count_parameters(model)
print(f"Original model parameters: {original_param_count}")

# Pruning the Model
# Maximum Absolute Weight:
# The maximum absolute weight in a neuron might indicate its significance

def compute_neuron_pair_importance(gate_weight, up_weight):
  """
  compute neuron pair importance scores (Maximum Absolute Weight)

  Args:
  - gate_weight: Weight matrix from the gate_proj layer.
  - up_weight: Weight matrix from the up_weight layer.

  Returns:
  - importance_scores: Importance scores for each neuron pair.
  """

  gate_max_abs = torch.max(gate_weight, dim=1).values + torch.abs(torch.min(gate_weight, dim=1).values)
  up_max_abs = torch.max(up_weight, dim=1).values + torch.abs(torch.min(up_weight, dim=1).values)
  importance_scores = gate_max_abs + up_max_abs
  return importance_scores


# Prunes a specific percentatge of neurons from the MLP (feed forward layers).
def prune_neuron_pairs(mlp, num_neuron_pairs_to_prune):
    """
    Reduces the dimensions of the **gate_proj**,**up_proj**, **down_proj**
    layers removing the least important neurons.

    Args:
    - mlp: Layers to prune.
    - prune_percent: Percentage of neurons to prune.

    Returns:
    - new_gate_proj, new_up_proj, new_down_proj:  New pruned layers.
    - k: New intermediate size.

    """
    # Extract the weights from the MLP layers
    #  these weights are used to calculate each neuron's
    #  importance score in the next step.
    gate_weight = mlp.gate_proj.weight.data.float()
    up_weight = mlp.up_proj.weight.data.float()

    #Compute importance stores. Neurons with higher importance scores
    # are considered more important and less likely to be pruned.
    importance_scores = compute_neuron_pair_importance(gate_weight, up_weight)

    #Store the original number of neurons in the intermediate layer.
    original_intermediate_size = gate_weight.size(0)
    #Computes the number of neurons to prune.
    # num_neuron_pairs_to_prune = min(int(prune_percent * original_intermediate_size), original_intermediate_size - 1)
    #Calculate the number of neurons to keep. The new intermediate size.
    k = original_intermediate_size - num_neuron_pairs_to_prune

    #Just check that there is no big error calculating k. We can't prune all the neurons.
    if k <= 0:
        raise ValueError(f"Invalid number of neuron pairs to keep: {k}. Adjust the prune_percent.")

    #Select the neuros to keep, by obtaining the indices to keep.
    _, indices_to_keep = torch.topk(importance_scores, k, largest=True, sorted=True)
    indices_to_keep = indices_to_keep.sort().values

    #create the new layers
    new_gate_proj = nn.Linear(mlp.gate_proj.in_features, k, bias=False).to(device)
    new_up_proj = nn.Linear(mlp.up_proj.in_features, k, bias=False).to(device)
    new_down_proj = nn.Linear(k, mlp.down_proj.out_features, bias=False).to(device)

    #copy weights to the new layers.
    new_gate_proj.weight.data = mlp.gate_proj.weight.data[indices_to_keep, :]
    new_up_proj.weight.data = mlp.up_proj.weight.data[indices_to_keep, :]
    new_down_proj.weight.data = mlp.down_proj.weight.data[:, indices_to_keep]

    #return new layers and intermediate size.
    return new_gate_proj, new_up_proj, new_down_proj, k


# %% [markdown]
# # Prune Loop
# The update_model function iterates through the blocks within the model's Transformer structure. This structure consists of multiple `LlamaDecoderLayer` blocks, and each of these blocks contains a pair of `LlamaSdpaAttention` and `LlamaMLP` components. The latter contains the MLP layers that will be the target of the pruning process.
# ```
# (layers): ModuleList(
#       (0-15): 16 x LlamaDecoderLayer(
#         (self_attn): LlamaSdpaAttention(
#           (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
#           (k_proj): Linear(in_features=2048, out_features=512, bias=False)
#           (v_proj): Linear(in_features=2048, out_features=512, bias=False)
#           (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
#           (rotary_emb): LlamaRotaryEmbedding()
#         )
#         (mlp): LlamaMLP(
#           (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
#           (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
#           (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
#           (act_fn): SiLU()
#         )
#         (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
#         (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
#       )
#   )    
# ```
# The layers that will undergo the removal of neurons identified as less useful are:
# ```
# (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
# (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
# (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
# ```
# The neurons are removed in the `prune_neurons` function based on the values returned by `compute_neuron_pair_importance`.

# %%
#Iterates throught the model layers and applies pruning.
def update_model(model, num_neuron_pairs_to_prune):
    """
    It modifies each mlp layer present in model, to retain only the most
    important neurons. Creating new smaller versions of each layer pruned.

    Args:
    - model: Model to prune.
    - prune_percent: Percentage of neurons to prune.

    Returns:
    - model: New pruned model.
    """
    new_intermediate_size = None

    #loop for each model layer.
    for idx, layer in enumerate(model.model.layers):
        #Since each layer is a LlamaDecoderLayer it contains multiple components
        # Attention, MLP and Layer norms. We're targetting MLP component
        # by accesing layer.mlp.
        mlp = layer.mlp

        #Call the prune_neiron_pairs with the layers and receiving the pruned.
        # new_gate_proj, new_up_proj, new_down_proj, new_size = prune_neuron_pairs(mlp, prune_percent)
        new_gate_proj, new_up_proj, new_down_proj, new_size = prune_neuron_pairs(mlp, num_neuron_pairs_to_prune)

        #Replace the Origiginal Layers with Pruned Layers.
        mlp.gate_proj = new_gate_proj
        mlp.up_proj = new_up_proj
        mlp.down_proj = new_down_proj

        #new_intermediate_size only needs to be set once
        if new_intermediate_size is None:
            new_intermediate_size = new_size

    #Update the model config file.
    model.config.intermediate_size = new_intermediate_size

    return model


# %%
# Obtain & test the pruned model
# (8192 - 6400) / 8192 ~= 0.22
num_neuron_pairs_to_prune = (8192 - 6400)
model = update_model(model, num_neuron_pairs_to_prune)

# %%
# Recalculate the number of parameters
pruned_param_count = count_parameters(model)
reduction_in_params = original_param_count - pruned_param_count
percentage_savings = (reduction_in_params / original_param_count) * 100

print(f"Pruned model parameters: {pruned_param_count}")
print(f"Reduction in parameters: {reduction_in_params}")
print(f"Percentage of weight savings: {percentage_savings:.2f}%")


# %%
# Test the pruned model
generated = get_output(prompt, model, tokenizer)
print(f"Generated text after pruning: {generated}")

# %%
print(model)

# %%
# Save the pruned model and tokenizer
import os 
os.chdir('/home/M114czli/eai/final/gluprune')
new_model_name = 'Llama-3.2-3B-Instruct-p20'

output_dir = './' + new_model_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Pruned model saved to {output_dir}")

# %%
# Update model attributes (For Llama 3.2 only)
model.config.hidden_size = model.lm_head.in_features
for name, m in model.named_modules():
    if name.endswith("self_attn"):
        if True:
            m.hidden_size = m.q_proj.out_features
        else:
            m.hidden_size = m.qkv_proj.out_features // 3        
        m.num_heads = m.hidden_size // m.head_dim
        model.config.num_attention_heads = m.num_heads
        #m.head_dim = m.q_proj.out_features // m.num_heads
        if not (True):
            m.num_key_value_heads = m.num_heads
            model.config.num_key_value_heads = m.num_heads
        if hasattr(m, "num_key_value_groups"):
            m.num_key_value_groups = m.num_heads // model.config.num_key_value_heads

    elif name.endswith("mlp"):
        if hasattr(m, "gate_proj"):
            m.hidden_size = m.gate_proj.in_features
            model.config.intermediate_size = m.gate_proj.out_features
        elif hasattr(m, "gate_up_proj"):
            m.hidden_size = m.gate_up_proj.in_features
            model.config.intermediate_size = m.gate_up_proj.out_features // 2
        else:
            raise ValueError("Unknown mlp layer")

# %%
model.half()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# # %%
# # Evaluate the model's perplexity (PPL) on the Wikitext-2 dataset
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
# Test loading the pruned model, should be no errors
model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.float16).to(device)
get_output("Paris is the capital of", model, tokenizer)

print("Pruning completed and model saved. Exiting script.")
