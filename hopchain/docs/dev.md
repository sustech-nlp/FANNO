# HopChain VLM SFT Evaluation — Development Log

## 2026-03-29 ~ 2026-03-30: Pipeline Setup + AMLT Evaluation

### Background
- HopChain paper reproduction: used GPT-5 to synthesize 9 high-quality multi-hop visual reasoning samples
- Goal: verify whether HopChain synthetic SFT data improves VLM benchmark performance

### Pipeline

1. **Data**: 9 multi-hop reasoning samples (sharegpt format with `<image>` + `<think>` CoT)
2. **Base Model**: Qwen2.5-VL-7B-Instruct
3. **Training**: LoRA SFT (rank=16, alpha=32, lr=2e-4, 3 epochs) → merge adapter → full model
4. **Evaluation**: lmms-eval 0.7.1 on 6 benchmarks

### Iteration History (12 versions)

| Version | Experiment | Issue |
|---------|-----------|-------|
| v1-v6 | Various | lmms-eval template bugs, PATH issues |
| v8 main-gannet | Baseline: mmbench KeyError; SFT: trl import error | mmbench needs GPT judge |
| v9 stable-mammoth | SFT: LlamaFlashAttention2 removed from transformers 4.57 | llamafactory incompatible |
| v10 desired-ladybug | Baseline: ✅ PASS; SFT: DeepSpeed not available | dropped mmbench → scienceqa |
| v11 still-chigger | SFT: HfArgumentParser unused VLM keys | llamafactory doesn't support VLM params with old transformers |
| **v12 supreme-sunbird** | **Both ✅ PASS** | Replaced llamafactory with custom `train_lora_sft.py` |

### Key Decision: Abandoned LLaMA-Factory

LLaMA-Factory 0.8.3 is incompatible with transformers 4.57.6 (Docker image):
- Imports removed class `LlamaFlashAttention2`
- Downgrading transformers to 4.47 breaks Qwen2.5-VL support
- Even with correct version, `HfArgumentParser` rejects VLM-specific params

**Solution**: Wrote standalone `train_lora_sft.py` (567 lines) using transformers + peft directly.

### lmms-eval 0.7.1 Template Fix

PyPI wheel has missing/broken YAML template files. Three-step setup:
1. `pip install lmms-eval`
2. One-liner: find all `!include` references, create empty `{}` placeholders for missing files
3. Download correct `_default_template_yaml` for mmmu and mmstar from GitHub v0.7.1 tag

### Results

**Experiment**: desired-ladybug (baseline) + supreme-sunbird (SFT)

| Benchmark | Baseline | SFT (HopChain 9 samples) | Δ |
|-----------|:--------:|:------------------------:|:-:|
| **AI2D** | 82.25% | **82.67%** | +0.42% |
| **MMMU val** | **51.44%** | 50.67% | -0.77% |
| **MMStar** | 61.79% | **62.46%** | +0.67% |
| **RealWorldQA** | **70.07%** | 69.41% | -0.66% |
| **ScienceQA_img** | 88.15% | **88.20%** | +0.05% |
| MathVista | N/A | N/A | needs OpenAI API |

**MMStar subcategories** (notable changes):
- fine-grained perception: 55.58% → **57.83%** (+2.25%)
- logical reasoning: 59.43% → **60.78%** (+1.35%)

**MMMU subcategories** (from logs):
- Overall: 51.44% → 50.67% (-0.77%)

### Analysis

With only 9 training samples:
- **3 benchmarks improved** (AI2D +0.42%, MMStar +0.67%, ScienceQA +0.05%)
- **2 benchmarks decreased** (MMMU -0.77%, RealWorldQA -0.66%)
- Changes are small and within noise range for such a tiny dataset
- MMStar fine-grained perception (+2.25%) shows potential for multi-hop reasoning improvement
- No catastrophic forgetting — model performance is stable

### Next Steps
- Scale up: synthesize 1000+ HopChain samples across diverse images
- Add OpenAI API key for MathVista evaluation
- Consider full fine-tuning or larger LoRA rank with more data
- Test on additional reasoning-focused benchmarks

### Files Created/Modified
- `train_lora_sft.py` — standalone LoRA SFT training script
- `configs/train_lora_config.yaml` — training config
- `configs/amlt_hopchain_vlm.yaml` — AMLT job config (v12)
- `docs/dev.md` — this file
