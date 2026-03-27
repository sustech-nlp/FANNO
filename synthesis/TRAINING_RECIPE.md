# FANNO-Dev Training Recipe

## Quick Start with LLaMA-Factory

### 1. Data Preparation

```bash
# Copy data to LLaMA-Factory
cp synthesis/outputs/merged_sharegpt.jsonl tools/LLaMA-Factory/data/fanno_dev.jsonl
```

### 2. Register Dataset

Add to `tools/LLaMA-Factory/data/dataset_info.json`:
```json
{
  "fanno_dev": {
    "file_name": "fanno_dev.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
```

### 3. Training Configuration

```yaml
# Example: SFT with Qwen2.5-7B
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
stage: sft
do_train: true
finetuning_type: full
dataset: fanno_dev
template: qwen
cutoff_len: 4096
max_samples: 100000
overwrite_output_dir: true
preprocessing_num_workers: 16
output_dir: outputs/fanno_dev_sft

# Training
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

### 4. Data Selection Strategies

For budget-constrained training:

```python
# Random selection (recommended for N > 5K)
head -n 5000 merged_sharegpt.jsonl > selected_5k.jsonl

# K-Center-Greedy (recommended for N < 5K)
python -c "
from synthesis.compare_strategies import k_center_greedy_select
import json
data = [json.loads(l) for l in open('synthesis/outputs/merged_sharegpt.jsonl')]
selected = k_center_greedy_select(data, n=2000)
with open('selected_2k.jsonl', 'w') as f:
    for d in selected:
        f.write(json.dumps(d) + '\n')
"
```

### 5. Expected Results

Based on our diversity analysis:
- **Full dataset (135K)**: Maximum coverage, Vendi=182.75
- **50K subset**: ~98% of full diversity (Vendi≈180)
- **10K subset**: ~95% of full diversity (Vendi≈175)
- **2K subset (K-Center)**: ~85% diversity with only 1.5% data

### 6. Multi-Turn Training

For multi-turn conversations:
```yaml
dataset: fanno_dev
# The ShareGPT format natively supports multi-turn
# LLaMA-Factory handles the conversation structure automatically
```

Note: 18.7K multi-turn conversations with avg 6.2 turns.
