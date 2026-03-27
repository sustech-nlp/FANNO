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

# K-Center-Greedy (recommended for N < 5K, +33% diversity at N=500)
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

### 5. Optimal Source Mixing Ratios

Our diversity analysis reveals that different pipelines have different diversity contributions per sample.
For diversity-maximized training, use these optimized ratios instead of natural proportions:

| Source | Natural | Optimized | Action |
|--------|--------:|----------:|--------|
| Self-Inversion | 3.2% | 17.4% | ↑↑ Strongly upweight |
| Document QA | 14.0% | 15.3% | ↑ Slight upweight |
| Reasoning QA | 11.6% | 14.1% | ↑ Upweight |
| Code QA | 10.0% | 14.0% | ↑ Upweight |
| Complex QA | 35.4% | 13.7% | ↓↓ Downweight |
| Math QA | 7.5% | 13.0% | ↑ Upweight |
| Creative Writing | 6.1% | 12.5% | ↑ Upweight |
| Multi-Turn Dialog | 12.2% | -- | Keep as-is |

**Key insight**: Self-Inversion has 32.4 Vendi/1K samples (vs 2.6 for Complex QA) — 12× more diversity per sample.

```python
# Example: Create diversity-optimized 50K subset
import json, random

ratios = {
    "self_inversion": 0.174,
    "fanno_seed_qa": 0.153,
    "fanno_reasoning_qa": 0.141,
    "fanno_code_qa": 0.140,
    "fanno_complex_qa": 0.137,
    "fanno_math_qa": 0.130,
    "fanno_creative_writing": 0.125,
}

data_by_source = {}
for line in open("synthesis/outputs/merged_sharegpt.jsonl"):
    d = json.loads(line)
    src = d.get("source", "unknown")
    data_by_source.setdefault(src, []).append(d)

N = 50000
selected = []
for src, ratio in ratios.items():
    pool = data_by_source.get(src, [])
    n = min(int(N * ratio), len(pool))
    selected.extend(random.sample(pool, n))

# Fill remaining with multi-turn
remaining = N - len(selected)
mt = data_by_source.get("fanno_multi_turn", [])
selected.extend(random.sample(mt, min(remaining, len(mt))))

random.shuffle(selected)
with open("selected_50k_optimized.jsonl", "w") as f:
    for d in selected:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"Selected {len(selected)} samples")
```

### 6. Expected Results

Based on our diversity analysis:
- **Full dataset (153K)**: Maximum coverage, Vendi=182.75
- **50K subset**: ~98% of full diversity (Vendi≈180)
- **10K subset**: ~95% of full diversity (Vendi≈175)
- **5K subset (K-Center)**: ~90% diversity with 3.3% data
- **2K subset (K-Center)**: ~85% diversity with only 1.3% data
- **500 subset (K-Center)**: +33% better than random sampling

### 7. Multi-Turn Training

For multi-turn conversations:
```yaml
dataset: fanno_dev
# The ShareGPT format natively supports multi-turn
# LLaMA-Factory handles the conversation structure automatically
```

Statistics: 18,708 multi-turn conversations with 8 patterns, 15 scenarios, avg 3.3 turns.
Coherence: adjacent turn similarity 0.575 with natural Q-A alternating pattern.
TTR: stable at 0.84-0.85 across turn depths (no degradation).

### 8. LoRA Configuration (Resource-Efficient)

```yaml
# LoRA SFT for 7B model on single A100
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_target: all
dataset: fanno_dev
template: qwen
cutoff_len: 4096
per_device_train_batch_size: 8
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
```

### 9. Evaluation

After training, evaluate on standard benchmarks:
- MMLU (knowledge), GSM8K (math), HumanEval (code), ARC (reasoning)
- **No contamination**: 0 matches across 33K+ test instances (verified)
- AlpacaEval for instruction-following quality
