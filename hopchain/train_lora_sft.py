#!/usr/bin/env python3
"""
Standalone LoRA SFT training script for Qwen2.5-VL-7B-Instruct.

Usage:
    python train_lora_sft.py --config config.yaml

Config YAML format:
    model_name_or_path: /path/to/Qwen2.5-VL-7B-Instruct
    data_path: /path/to/sft_data.json
    output_dir: /path/to/output
    lora_rank: 16
    lora_alpha: 32
    learning_rate: 2.0e-4
    num_train_epochs: 3
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    bf16: true
    gradient_checkpointing: true

Data format (LLaMA-Factory sharegpt style):
    {"messages": [{"content": "<image>question", "role": "user"},
                  {"content": "answer", "role": "assistant"}],
     "images": ["/path/to/image.jpg"]}

No deepspeed, no llamafactory — just transformers + peft.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config values
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "model_name_or_path": "",
    "data_path": "",
    "output_dir": "./output",
    # LoRA
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # All linear layers in Qwen2.5-VL (language model only by default)
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "freeze_vision_tower": True,
    # Training
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "bf16": True,
    "gradient_checkpointing": True,
    "logging_steps": 1,
    "save_steps": 50,
    "save_total_limit": 3,
    "dataloader_num_workers": 4,
    "report_to": "none",
    # Image processing
    "image_max_pixels": 262144,  # 512*512
    "image_min_pixels": 3136,    # 56*56
    "cutoff_len": 4096,
    # Post-training
    "merge_and_save": True,
    "seed": 42,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config and merge with defaults."""
    with open(config_path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = {**DEFAULT_CONFIG, **user_cfg}
    # Validate required fields
    assert cfg["model_name_or_path"], "model_name_or_path is required"
    assert cfg["data_path"], "data_path is required"
    return cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
IGNORE_INDEX = -100


def _convert_sharegpt_to_qwen_messages(
    sample: Dict[str, Any],
) -> tuple:
    """
    Convert LLaMA-Factory sharegpt format to Qwen2-VL chat message format.

    Input:
        {"messages": [{"content": "<image>question", "role": "user"}, ...],
         "images": ["/path/to/image.jpg"]}

    Output:
        (qwen_messages, image_paths)
        where qwen_messages use the structured content format expected by
        Qwen2VLProcessor.apply_chat_template.
    """
    messages = sample["messages"]
    image_paths = sample.get("images", [])
    image_idx = 0
    qwen_messages = []

    for msg in messages:
        role = msg["role"]
        raw_content = msg["content"]

        if role == "user":
            # Split content around <image> tokens and build structured content
            parts = raw_content.split("<image>")
            content_list = []
            for i, part in enumerate(parts):
                if i > 0 and image_idx < len(image_paths):
                    # Insert image reference before this text segment
                    img_path = image_paths[image_idx]
                    content_list.append({
                        "type": "image",
                        "image": f"file://{os.path.abspath(img_path)}",
                    })
                    image_idx += 1
                if part.strip():
                    content_list.append({"type": "text", "text": part.strip()})
            qwen_messages.append({"role": "user", "content": content_list})
        else:
            # Assistant / system messages are plain text
            qwen_messages.append({"role": role, "content": raw_content})

    return qwen_messages, image_paths


class Qwen2VLSFTDataset(Dataset):
    """SFT dataset for Qwen2.5-VL with LoRA training."""

    def __init__(
        self,
        data_path: str,
        processor: Qwen2VLProcessor,
        max_length: int = 4096,
        image_max_pixels: int = 262144,
        image_min_pixels: int = 3136,
    ):
        super().__init__()
        logger.info("Loading data from %s", data_path)
        with open(data_path, "r") as f:
            self.raw_data = json.load(f)
        logger.info("Loaded %d samples", len(self.raw_data))

        self.processor = processor
        self.max_length = max_length
        self.image_max_pixels = image_max_pixels
        self.image_min_pixels = image_min_pixels

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.raw_data[idx]
        qwen_messages, image_paths = _convert_sharegpt_to_qwen_messages(sample)

        # Step 1: Apply chat template to get the full text with special tokens
        text = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Step 2: Load images
        images = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            images.append(img)

        # Step 3: Process through the processor (handles image token expansion)
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Squeeze batch dimension (processor returns [1, seq_len])
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Step 4: Build labels — mask everything except assistant responses
        labels = self._build_labels(input_ids)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Include pixel values and grid info if images are present
        # Note: pixel_values shape is [num_patches, patch_dim] — no batch dim
        # image_grid_thw shape is [num_images, 3] — no batch dim
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]

        return result

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Build training labels by masking non-assistant tokens with IGNORE_INDEX.

        Qwen2.5-VL chat format:
            <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>\n

        We only compute loss on assistant response tokens (including <|im_end|>).
        """
        labels = input_ids.clone()
        tokenizer = self.processor.tokenizer

        # Get special token IDs
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        # "assistant" as token sequence after <|im_start|>
        assistant_token_ids = tokenizer.encode("assistant", add_special_tokens=False)
        nl_token_ids = set(tokenizer.encode("\n", add_special_tokens=False))

        ids = input_ids.tolist()
        n = len(ids)

        # Find all assistant response spans
        # Pattern: <|im_start|> assistant \n ... <|im_end|>
        # We want to keep labels for the content after "assistant\n" up to and including <|im_end|>
        i = 0
        in_assistant = False
        mask_start = 0  # start of region to mask (non-assistant)

        while i < n:
            if ids[i] == im_start_id:
                # Check if this is an assistant turn
                # After <|im_start|>, the next tokens should be "assistant" then "\n"
                match_len = len(assistant_token_ids)
                if (i + 1 + match_len < n and
                        ids[i + 1: i + 1 + match_len] == assistant_token_ids):
                    # Mask everything from mask_start to end of "assistant\n"
                    # Find the \n after "assistant"
                    content_start = i + 1 + match_len
                    # Skip the newline token(s) right after "assistant"
                    while content_start < n and ids[content_start] in nl_token_ids:
                        content_start += 1
                    # Mask from current mask_start to content_start (exclusive)
                    labels[mask_start:content_start] = IGNORE_INDEX
                    in_assistant = True
                    i = content_start
                    continue
                else:
                    # Non-assistant turn start
                    in_assistant = False
                    i += 1
                    continue

            if ids[i] == im_end_id and in_assistant:
                # End of assistant turn — keep the <|im_end|> token in labels
                # The next mask region starts after <|im_end|> + potential \n
                in_assistant = False
                mask_start = i + 1
                i += 1
                continue

            i += 1

        # Mask any trailing non-assistant content
        if not in_assistant and mask_start < n:
            labels[mask_start:] = IGNORE_INDEX

        return labels


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------
@dataclass
class VLMDataCollator:
    """Collate variable-length VLM samples into a batch with padding."""

    processor: Qwen2VLProcessor
    max_length: int = 4096

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Separate image features from text features
        batch = {}

        # Pad input_ids, attention_mask, labels
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id

        input_ids_list = [f["input_ids"] for f in features]
        attention_mask_list = [f["attention_mask"] for f in features]
        labels_list = [f["labels"] for f in features]

        max_len = min(max(ids.size(0) for ids in input_ids_list), self.max_length)

        padded_input_ids = []
        padded_attention = []
        padded_labels = []

        for ids, mask, labs in zip(input_ids_list, attention_mask_list, labels_list):
            seq_len = ids.size(0)
            if seq_len > max_len:
                # Truncate from the right
                ids = ids[:max_len]
                mask = mask[:max_len]
                labs = labs[:max_len]
                seq_len = max_len
            pad_len = max_len - seq_len
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
                labs = torch.cat([labs, torch.full((pad_len,), IGNORE_INDEX, dtype=labs.dtype)])
            padded_input_ids.append(ids)
            padded_attention.append(mask)
            padded_labels.append(labs)

        batch["input_ids"] = torch.stack(padded_input_ids)
        batch["attention_mask"] = torch.stack(padded_attention)
        batch["labels"] = torch.stack(padded_labels)

        # Handle pixel_values — concatenate along the patch dimension (dim=0)
        # Qwen2.5-VL pixel_values shape: [num_patches, channel_dim]
        if any("pixel_values" in f for f in features):
            pixel_values_list = [f["pixel_values"] for f in features if "pixel_values" in f]
            if pixel_values_list:
                batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)

        # Handle image_grid_thw — concatenate along batch dimension
        if any("image_grid_thw" in f for f in features):
            grid_list = [f["image_grid_thw"] for f in features if "image_grid_thw" in f]
            if grid_list:
                # Each is shape [num_images, 3] or [3] for single image
                grids = []
                for g in grid_list:
                    if g.dim() == 1:
                        g = g.unsqueeze(0)
                    grids.append(g)
                batch["image_grid_thw"] = torch.cat(grids, dim=0)

        return batch


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def setup_model_and_processor(cfg: Dict[str, Any]):
    """Load model, processor, and apply LoRA."""
    model_path = cfg["model_name_or_path"]
    logger.info("Loading processor from %s", model_path)

    processor = Qwen2VLProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        min_pixels=cfg["image_min_pixels"],
        max_pixels=cfg["image_max_pixels"],
    )

    logger.info("Loading model from %s", model_path)
    dtype = torch.bfloat16 if cfg["bf16"] else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Freeze vision tower if requested
    if cfg.get("freeze_vision_tower", True):
        logger.info("Freezing vision tower parameters")
        if hasattr(model, "visual"):
            for param in model.visual.parameters():
                param.requires_grad = False

    # Apply LoRA
    target_modules = cfg.get("lora_target_modules", DEFAULT_CONFIG["lora_target_modules"])
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
    )
    logger.info(
        "Applying LoRA: rank=%d, alpha=%d, targets=%s",
        cfg["lora_rank"], cfg["lora_alpha"], target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing
    if cfg["gradient_checkpointing"]:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    return model, processor


# ---------------------------------------------------------------------------
# Merge and save
# ---------------------------------------------------------------------------
def merge_and_save(cfg: Dict[str, Any]):
    """Load the LoRA adapter, merge with base model, and save."""
    from peft import PeftModel

    output_dir = cfg["output_dir"]
    merged_dir = os.path.join(output_dir, "merged")
    model_path = cfg["model_name_or_path"]

    logger.info("=" * 60)
    logger.info("Merging LoRA adapter into base model...")
    logger.info("Base model: %s", model_path)
    logger.info("Adapter dir: %s", output_dir)
    logger.info("Merged output: %s", merged_dir)

    # Load base model
    dtype = torch.bfloat16 if cfg["bf16"] else torch.float32
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # Load and merge adapter
    model = PeftModel.from_pretrained(base_model, output_dir)
    model = model.merge_and_unload()

    # Save merged model
    logger.info("Saving merged model to %s", merged_dir)
    model.save_pretrained(merged_dir, safe_serialization=True)

    # Also copy processor/tokenizer files
    processor = Qwen2VLProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.save_pretrained(merged_dir)

    logger.info("Merge complete! Merged model saved to %s", merged_dir)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen2.5-VL")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("Config: %s", json.dumps(cfg, indent=2, default=str))

    # Setup model and processor
    model, processor = setup_model_and_processor(cfg)

    # Setup dataset
    dataset = Qwen2VLSFTDataset(
        data_path=cfg["data_path"],
        processor=processor,
        max_length=cfg["cutoff_len"],
        image_max_pixels=cfg["image_max_pixels"],
        image_min_pixels=cfg["image_min_pixels"],
    )

    # Data collator
    collator = VLMDataCollator(
        processor=processor,
        max_length=cfg["cutoff_len"],
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        weight_decay=cfg.get("weight_decay", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        bf16=cfg["bf16"],
        logging_steps=cfg.get("logging_steps", 1),
        save_steps=cfg.get("save_steps", 50),
        save_total_limit=cfg.get("save_total_limit", 3),
        save_strategy="steps",
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        remove_unused_columns=False,  # Critical for VLM — we pass custom columns
        report_to=cfg.get("report_to", "none"),
        gradient_checkpointing=cfg["gradient_checkpointing"],
        seed=cfg.get("seed", 42),
        dataloader_pin_memory=True,
        optim="adamw_torch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save final adapter
    logger.info("Saving adapter to %s", cfg["output_dir"])
    trainer.save_model(cfg["output_dir"])
    trainer.save_state()

    # Log metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Training complete! Adapter saved to %s", cfg["output_dir"])

    # Merge LoRA and save full model
    if cfg.get("merge_and_save", True):
        # Free GPU memory before merging
        del model
        del trainer
        torch.cuda.empty_cache()
        merge_and_save(cfg)


if __name__ == "__main__":
    main()
