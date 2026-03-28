"""AMLT job configuration generator for FANNO training."""

from __future__ import annotations

from typing import Optional

import yaml
from loguru import logger


def generate_amlt_config(
    job_name: str = "fanno-sft-qwen3-8b",
    model_path: str = "Qwen3-8B",
    train_data: str = "fanno_train_100k.jsonl",
    output_dir: str = "saves/fanno/sft",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    per_device_batch: int = 2,
    gradient_accum: int = 8,
    max_length: int = 2048,
    sla_tier: str = "Premium",
    target_name: str = "msrresrchvc",
    sku: str = "40G8-A100-IB-NvLink",
    storage_account: str = "msranlpinternhot",
    container: str = "hezhu",
    docker_image: str = "zhuhe/opd:latest",
    registry: str = "workspacegenaiacr.azurecr.io",
) -> str:
    """Generate AMLT YAML configuration for SFT training.

    Returns the YAML string ready to be written to a file.
    """
    config = {
        "description": job_name,
        "target": {
            "service": "sing",
            "name": target_name,
            "workspace_name": "workspace_genai",
            "resource_group": "gcr-singularity-resrch",
        },
        "environment": {
            "image": docker_image,
            "registry": registry,
            "setup": [
                "pip install trl wandb datasets",
                "python3 -V",
                "pip --version",
            ],
        },
        "code": {
            "local_dir": "$CONFIG_DIR/..",
        },
        "storage": {
            "zhuhe": {
                "storage_account_name": storage_account,
                "container_name": container,
                "mount_dir": "/mnt/zhuhe",
            },
        },
        "jobs": [
            {
                "name": job_name,
                "sku": sku,
                "process_count_per_node": 1,
                "mpi": True,
                "sla_tier": sla_tier,
                "priority": "High",
                "command": [
                    "nvidia-smi -L",
                    "export BASE_DIR=/mnt/zhuhe",
                    "if [ -d /mnt/zhuhe/zhuhe ]; then export BASE_DIR=/mnt/zhuhe/zhuhe; fi",
                    'ls -ld "$$BASE_DIR"',
                    f"export MODEL_PATH=$$BASE_DIR/models/{model_path}",
                    f"export TRAIN_DATA=$$BASE_DIR/data/fanno/{train_data}",
                    f"export OUTPUT_DIR=$$BASE_DIR/{output_dir}/{job_name}",
                    'ls -lh "$$MODEL_PATH"',
                    'ls -lh "$$TRAIN_DATA"',
                    "export WANDB_PROJECT=fanno-sft",
                    f"export WANDB_RUN_NAME={job_name}",
                    # DeepSpeed launch command
                    (
                        "deepspeed --num_gpus=8 "
                        "-m fanno.train.sft "
                        '--model "$$MODEL_PATH" '
                        '--data "$$TRAIN_DATA" '
                        '--output-dir "$$OUTPUT_DIR" '
                        f"--epochs {num_epochs} "
                        f"--lr {learning_rate} "
                        f"--per-device-batch {per_device_batch} "
                        f"--gradient-accum {gradient_accum} "
                        f"--max-length {max_length} "
                        "--deepspeed "
                        f"--wandb-project fanno-sft "
                        f"--wandb-run-name {job_name}"
                    ),
                ],
            }
        ],
    }

    return yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)


def main():
    """CLI entry point for AMLT config generation."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate AMLT config for FANNO SFT")
    parser.add_argument("--job-name", type=str, default="fanno-sft-qwen3-8b")
    parser.add_argument("--model-path", type=str, default="Qwen3-8B")
    parser.add_argument("--train-data", type=str, default="fanno_train_100k.jsonl")
    parser.add_argument("--output", type=str, default="configs/amlt_sft_qwen3.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--sla-tier", type=str, default="Premium")
    args = parser.parse_args()

    from pathlib import Path
    config_yaml = generate_amlt_config(
        job_name=args.job_name,
        model_path=args.model_path,
        train_data=args.train_data,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        sla_tier=args.sla_tier,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(config_yaml)
    logger.info(f"AMLT config written to {output_path}")


if __name__ == "__main__":
    main()
