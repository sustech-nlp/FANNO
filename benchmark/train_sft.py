#!/usr/bin/env python3
"""
Standalone SFT training script for FANNO benchmark evaluation.

Supports:
- ShareGPT format (conversations: [{from: "human", value: ...}, {from: "gpt", value: ...}])
- DeepSpeed ZeRO-3 for multi-GPU training
- Full-parameter fine-tuning

Usage:
    # Single GPU
    python train_sft.py --model /path/to/model --data /path/to/data.jsonl --output-dir ./output

    # Multi-GPU with DeepSpeed
    deepspeed --num_gpus=8 train_sft.py --model /path/to/model --data /path/to/data.jsonl --output-dir ./output --deepspeed
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# DeepSpeed ZeRO-3 configuration
DS_ZERO3_CONFIG = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}


def load_jsonlines(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_for_training(
    item: Dict[str, Any],
    tokenizer,
    max_length: int = 4096,
) -> Optional[Dict[str, Any]]:
    """Format a single data item for causal LM training.

    Supports ShareGPT format (conversations) and Alpaca format (instruction/output).
    """
    if "conversations" in item:
        # ShareGPT format
        messages = []
        for msg in item["conversations"]:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            role = role_map.get(msg.get("from", ""), msg.get("from", "user"))
            messages.append({"role": role, "content": msg.get("value", "")})
    elif "question" in item:
        # Question/Answer format
        messages = [
            {"role": "user", "content": item.get("question", "")},
            {"role": "assistant", "content": item.get("answer", item.get("solution", ""))},
        ]
    else:
        # Alpaca format
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        output = item.get("output", item.get("response", ""))
        user_content = f"{instruction}\n{inp}" if inp else instruction
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]

    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception as e:
        logger.warning(f"Chat template failed: {e}")
        return None

    # Tokenize
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": encoded["input_ids"].squeeze(),
        "attention_mask": encoded["attention_mask"].squeeze(),
        "labels": encoded["input_ids"].squeeze().clone(),
    }


class SFTDataset(Dataset):
    """SFT dataset that formats data on-the-fly."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 4096):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        formatted = format_for_training(item, self.tokenizer, self.max_length)
        if formatted is None:
            return self.__getitem__((idx + 1) % len(self.data))
        return formatted


def train(
    model_name: str,
    data_path: str,
    output_dir: str = "./checkpoints",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.03,
    max_length: int = 4096,
    use_deepspeed: bool = True,
    wandb_project: str = "fanno-sft",
    wandb_run_name: Optional[str] = None,
    save_steps: int = 500,
    logging_steps: int = 10,
) -> None:
    """Run full-parameter SFT training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save DeepSpeed config
    ds_config_path = None
    if use_deepspeed:
        ds_config_path = str(output_dir / "ds_zero3_config.json")
        with open(ds_config_path, "w") as f:
            json.dump(DS_ZERO3_CONFIG, f, indent=2)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    logger.info(f"Loading model from {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load training data
    logger.info(f"Loading training data from {data_path}")
    raw_data = load_jsonlines(data_path)
    logger.info(f"Loaded {len(raw_data)} training samples")

    # Create dataset
    train_dataset = SFTDataset(raw_data, tokenizer, max_length)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        report_to="wandb" if wandb_project else "none",
        run_name=wandb_run_name or f"fanno-sft-{Path(model_name).name}",
        deepspeed=ds_config_path,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=42,
    )

    # Setup wandb
    if wandb_project:
        os.environ.setdefault("WANDB_PROJECT", wandb_project)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_dir = str(output_dir / "final")
    logger.info(f"Saving final model to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="FANNO SFT Training (Standalone)")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--data", type=str, required=True, help="Training data JSONL")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--per-device-batch", type=int, default=2)
    parser.add_argument("--gradient-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--deepspeed", action="store_true", default=False)
    parser.add_argument("--no-deepspeed", dest="deepspeed", action="store_false")
    parser.add_argument("--wandb-project", type=str, default="fanno-sft")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    args = parser.parse_args()

    train(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accum,
        max_length=args.max_length,
        use_deepspeed=args.deepspeed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )


if __name__ == "__main__":
    main()
