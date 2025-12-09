## Inference Backends

FANNO ships two interchangeable inference utilities:

- `fanno.inference.vllm_inference.parallel_inference`: local vLLM backend. Respects `InferenceConfig` (model path, tensor parallel, sampling params).
- `fanno.inference.client_inference.client_parallel_inference`: Azure OpenAI backend. Picks endpoints by weighted speed, authenticates via `AzureCliCredential`, and calls chat completions.

### vLLM usage
```python
from fanno.inference.vllm_inference import parallel_inference
from fanno.config import InferenceConfig

prompts = ["Tell me a joke.", "Summarize reinforcement learning."]
cfg = InferenceConfig(model_name_or_path="Qwen/Qwen2.5-7B-Instruct", max_tokens=128, temperature=0.7)
outputs = parallel_inference(prompts, config=cfg, template_type="direct")
```

### Azure client usage
```python
from fanno.inference.client_inference import client_parallel_inference

prompts = ["What is the capital of France?", "List three uses of copper."]
outputs = client_parallel_inference(prompts, model_name="gpt-4o", max_tokens=128, temperature=0.7)
```

YAML (Azure GPT-5 example) — see `configs/azure_gpt5.yaml`:
```yaml
inference:
  backend: azure
  model_name_or_path: gpt-5
  azure_tenant_id: 72f988bf-86f1-41af-91ab-2d7cd011db47
  azure_api_version: 2024-12-01-preview
  azure_max_retries: 5
  temperature: 0.7
  top_p: 0.9
  max_tokens: 1024
```

Notes:
- Azure calls require `az login` and appropriate tenant access. Edit `tenant_id`/`api_version` in `get_client` if needed.
- vLLM and Azure share the same sampling signature (`max_tokens`, `temperature`); swap as needed.
