## Edge AI Final Project
Small Language Model acceleration

## 整體架構
![image](https://github.com/user-attachments/assets/8ae607d0-b29f-4159-9f65-772a9fbd849c)

## 1. 安裝並啟動miniconda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
conda create --name sg python=3.9
conda activate sg
```

## 2. 安裝套件
```bash
chmod +x install.sh
./install.sh
```

## 3. 執行程式
```bash
python result.py
注意:如果此時程式崩潰，請回到第一步並改為使用conda create --name sg python=3.10
```

## 備註
因為要用到nvcc，在T4上建議使用miniconda安裝cuda-toolkit  
微調與量化模型是在其他機器上做的  
我們觀察到在某些機器上使用python=3.10會導致崩潰，python=3.9則可以正常運行，但在某些機器上則恰好相反

## 參考文獻 & 文章
[GPTQModel - vLLM](https://docs.vllm.ai/en/stable/features/quantization/gptqmodel.html)  
[Large-Language-Model-Notebooks-Course/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb)  
[Torch-Pruning/examples/LLMs/prune_llm.py at master · VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning/blob/master/examples/LLMs/prune_llm.py)  
[[2403.17887] The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887)  
[[2405.18218] FinerCut: Finer-grained Interpretable Layer Pruning for Large Language Models](https://arxiv.org/abs/2405.18218)  
[[2305.11627] LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627)
