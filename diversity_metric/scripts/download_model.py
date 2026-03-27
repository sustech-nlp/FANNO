import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

# 加载模型与分词器
pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 执行推理
output = pipe("Hey, how are you doing today?", max_new_tokens=50)
print(output[0]["generated_text"])
