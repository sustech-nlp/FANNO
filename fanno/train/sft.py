"""SFT training script with DeepSpeed ZeRO-3 support."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


# DeepSpeed ZeRO-3 configuration
DS_ZERO3_CONFIG: Dict[str, Any] = {
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


def save_deepspeed_config(output_dir: str | Path) -> str:
    """Save DeepSpeed config to a JSON file and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "ds_zero3_config.json"
    with open(config_path, "w") as f:
        json.dump(DS_ZERO3_CONFIG, f, indent=2)
    return str(config_path)


def load_training_data(data_path: str | Path) -> List[Dict[str, Any]]:
    """Load training data from JSONL file."""
    from fanno.data.loader import load_jsonlines
    return load_jsonlines(data_path)


def format_for_training(
    item: Dict[str, Any],
    tokenizer,
    max_length: int = 2048,
) -> Optional[Dict[str, Any]]:
    """Format a single data item for causal LM training.

    Supports Alpaca format (instruction/input/output) and
    ShareGPT format (conversations).
    """
    if "conversations" in item:
        # ShareGPT format
        messages = []
        for msg in item["conversations"]:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            role = role_map.get(msg.get("from", ""), msg.get("from", "user"))
            messages.append({"role": role, "content": msg.get("value", "")})
    else:
        # Alpaca format
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        output = item.get("output", "")

        if inp:
            user_content = f"{instruction}\n{inp}"
        else:
            user_content = instruction

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

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


def train(
    model_name: str = "Qwen/Qwen3-8B",
    data_path: str = "./train_data/train.jsonl",
    output_dir: str = "./checkpoints",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    warmup_ratio: float = 0.03,
    max_length: int = 2048,
    use_deepspeed: bool = True,
    wandb_project: str = "fanno-sft",
    wandb_run_name: Optional[str] = None,
    save_steps: int = 500,
    logging_steps: int = 10,
    eval_data_path: Optional[str] = None,
) -> None:
    """Run full-parameter SFT training.

    Args:
        model_name: HuggingFace model name or local path.
        data_path: Path to training data JSONL.
        output_dir: Output directory for checkpoints.
        num_epochs: Number of training epochs.
        learning_rate: Peak learning rate.
        per_device_batch_size: Per-GPU batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        warmup_ratio: Warmup ratio for cosine scheduler.
        max_length: Maximum sequence length.
        use_deepspeed: Whether to use DeepSpeed ZeRO-3.
        wandb_project: Weights & Biases project name.
        wandb_run_name: W&B run name (auto-generated if None).
        save_steps: Save checkpoint every N steps.
        logging_steps: Log metrics every N steps.
        eval_data_path: Path to evaluation data JSONL (optional).
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )

    logger.info(f"Starting SFT training: model={model_name}, data={data_path}")

    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save DeepSpeed config
    ds_config_path = save_deepspeed_config(output_dir) if use_deepspeed else None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load and format training data
    logger.info(f"Loading training data from {data_path}")
    raw_data = load_training_data(data_path)
    logger.info(f"Loaded {len(raw_data)} training samples")

    # Create dataset
    from torch.utils.data import Dataset

    class SFTDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            formatted = format_for_training(item, self.tokenizer, self.max_length)
            if formatted is None:
                # Fallback: return a dummy item
                return self.__getitem__((idx + 1) % len(self.data))
            return formatted

    train_dataset = SFTDataset(raw_data, tokenizer, max_length)

    # Load eval dataset if provided
    eval_dataset = None
    if eval_data_path:
        eval_data = load_training_data(eval_data_path)
        eval_dataset = SFTDataset(eval_data, tokenizer, max_length)
        logger.info(f"Loaded {len(eval_data)} evaluation samples")

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
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=save_steps if eval_dataset else None,
        report_to="wandb",
        run_name=wandb_run_name or f"fanno-sft-{Path(model_name).name}",
        deepspeed=ds_config_path,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # Setup wandb
    os.environ.setdefault("WANDB_PROJECT", wandb_project)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving final model to {output_dir / 'final'}")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    logger.info("Training complete!")


def main():
    """CLI entry point for SFT training."""
    parser = argparse.ArgumentParser(description="FANNO SFT Training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--data", type=str, required=True, help="Training data JSONL")
    parser.add_argument("--eval-data", type=str, default=None, help="Evaluation data JSONL")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--per-device-batch", type=int, default=2)
    parser.add_argument("--gradient-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--deepspeed", action="store_true", default=True)
    parser.add_argument("--no-deepspeed", dest="deepspeed", action="store_false")
    parser.add_argument("--wandb-project", type=str, default="fanno-sft")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--save-steps", type=int, default=500)
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
        eval_data_path=args.eval_data,
    )


if __name__ == "__main__":
    main()
