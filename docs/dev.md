# FANNO-Dev Development Log

## Session: 2026-03-28 — Full Codebase Restructure + Training Pipeline

### Context
Goal: Restructure FANNO codebase from scattered `src/fanno/` into a clean top-level `fanno/` package with evaluate, synthesize, and train modules. Target: beat DataFlow on benchmarks (AlpacaEval, ArenaHard, BFCL v4, IFEval).

### Phase 1: Code Cleanup & Architecture Restructure

**Q: What is the current state of the codebase?**
- A: Code lives in `src/fanno/` with 20+ files across `inference/`, `strategies/`, `template/`, `utils/` subpackages
- `synthesis/`, `tools/`, `diversity_metric/` directories referenced in plan do NOT exist (were likely local/temporary)
- `prescreen.py` has hardcoded paths and `CUDA_VISIBLE_DEVICES` — needs removal
- `prompt_utils.py` in template/ is unused — removed
- `eval_template.py` has 82 lines of commented-out old code — cleaned

**Decision: New package layout**
```
fanno/                           # Top-level package (replaces src/fanno/)
├── __init__.py                  # Lazy imports for heavy deps
├── config.py                    # Unchanged from src/fanno/
├── pipeline.py                  # Unchanged (FANNO UCB flow)
├── evaluator.py                 # Legacy (kept for backward compat)
├── cli.py                       # Enhanced with synthesize/evaluate/prepare/train/amlt subcommands
├── api/
│   ├── __init__.py
│   └── client.py                # AzureAPIClient class (extracted from inference/client_inference.py)
├── data/
│   ├── __init__.py
│   ├── loader.py                # JSONL/JSON load/save (from utils/data_utils.py)
│   ├── cleaning.py              # instruction_cleaning + hard_filter (extracted)
│   └── formats.py               # to_alpaca/sharegpt/agent format converters (NEW)
├── evaluate/
│   ├── __init__.py
│   ├── quality.py               # QualityEvaluator class (NEW)
│   └── diversity.py             # vendi_score, avg_pairwise_distance, k_center_greedy (NEW)
├── synthesize/
│   ├── __init__.py
│   ├── base.py                  # BaseSynthesizer abstract class (NEW)
│   ├── qa.py                    # QA synthesis (NEW)
│   ├── creative.py              # Creative writing synthesis (NEW)
│   ├── dialog.py                # Multi-turn dialog synthesis (NEW)
│   ├── agent.py                 # Agent trajectory synthesis (NEW)
│   ├── inversion.py             # Trajectory inversion (NEW)
│   └── prompts.py               # Consolidated prompts (NEW)
├── train/
│   ├── __init__.py
│   ├── prepare.py               # Training data mixing (NEW)
│   ├── sft.py                   # SFT with DeepSpeed ZeRO-3 (NEW)
│   └── amlt.py                  # AMLT job generator (NEW)
├── inference/                   # Unchanged (lazy vllm import)
├── strategies/                  # Unchanged
├── template/                    # Cleaned (removed prompt_utils.py, cleaned eval_template.py)
└── utils/                       # Unchanged
```

**Files changed:**
- `pyproject.toml` — version 0.2.0, packages.find where=["."], added deps (datasets, trl, deepspeed, wandb)
- `fanno/__init__.py` — lazy imports to avoid vllm dependency at import time
- `fanno/inference/__init__.py` — lazy vllm/client imports, extracted `_build_prompt` and `parser_score`
- `fanno/template/__init__.py` — removed prompt_utils import
- `fanno/template/eval_template.py` — cleaned up 82 lines of commented code, added docstrings

**New files:**
- `fanno/api/client.py` — AzureAPIClient class with batch_chat()
- `fanno/data/loader.py` — improved JSONL load/save with overwrite control
- `fanno/data/cleaning.py` — instruction_cleaning() + hard_filter() with source_type support
- `fanno/data/formats.py` — to_alpaca_format(), to_sharegpt_format(), to_agent_format()
- `fanno/cli.py` — enhanced CLI with synthesize/evaluate/prepare/train/amlt subcommands
- `fanno/evaluate/quality.py` — QualityEvaluator (unified evaluation)
- `fanno/evaluate/diversity.py` — vendi_score, avg_pairwise_distance, k_center_greedy
- `fanno/synthesize/base.py` — BaseSynthesizer abstract class
- `fanno/synthesize/qa.py` — QA synthesis
- `fanno/synthesize/creative.py` — Creative writing synthesis
- `fanno/synthesize/dialog.py` — Multi-turn dialog synthesis
- `fanno/synthesize/agent.py` — Agent trajectory synthesis with WorldModel
- `fanno/synthesize/inversion.py` — Trajectory inversion
- `fanno/synthesize/prompts.py` — All synthesis prompt templates
- `fanno/train/prepare.py` — Training data preparation + mixing
- `fanno/train/sft.py` — SFT training with DeepSpeed ZeRO-3
- `fanno/train/amlt.py` — AMLT config generator
- `configs/amlt_sft_qwen3.yaml` — Ready-to-use AMLT config for Qwen3-8B

### Phase 2: Agent Data Synthesis
- Target: 5K agent trajectories + 2K inversions
- AgentSynthesizer with 5 roles × 6 logic patterns × 8 tool types
- WorldModel simulates tool execution

### Phase 3: Training Data
- Mix: 50K FANNO QA + 5K agent + 20K Alpaca + 10K ArenaHard + 15K BFCL = ~100K
- Model: Qwen3-8B base (cached locally)
- Training: Full-parameter SFT, DeepSpeed ZeRO-3, 8×A100

### Phase 4: AMLT Training
- Cluster: msrresrchvc, SKU: 40G8-A100-IB-NvLink
- Image: zhuhe/opd:latest
- Storage: msranlpinternhot/hezhu

### Phase 5: Evaluation Targets
- AlpacaEval 2.0
- ArenaHard
- BFCL v4
- IFEval
