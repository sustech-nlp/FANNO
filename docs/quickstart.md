# FANNO Quickstart

## Environment
```bash
pip install -r requirements.txt
pip install -e .
```

## Run with local vLLM teacher
```bash
fanno --config src/fanno/config.yaml
```

## Run with Azure GPT-5 teacher
```bash
fanno --config configs/azure_gpt5.yaml
```

Ensure you have Azure CLI logged in and the tenant ID in the config is valid.

## Key outputs
- Seeds: `outputs/<run_name>/initial_seed.jsonl`
- Augmented batches: `outputs/<run_name>/ucb_aug_*.jsonl`
- Final merged set: `outputs/<run_name>/final_data.jsonl`
