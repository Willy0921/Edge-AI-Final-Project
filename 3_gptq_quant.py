from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "./Llama-3.2-1B-Instruct-distill"
quant_path = "Llama-3.2-1B-Instruct-distill-gptq-b8g128"

calibration_dataset = load_dataset(
    "Salesforce/wikitext",
    data_files="wikitext-2-raw-v1/train-00000-of-00001.parquet",
    split="train"
  ).select(range(2048))["text"]

quant_config = QuantizeConfig(bits=8, group_size=128, v2=True)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)