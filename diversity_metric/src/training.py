"""
Lightweight training harness for fine-tuning causal LMs on selected subsets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def _format_text(sample: dict) -> str:
    instruction = (
        sample.get("instruction")
        or sample.get("prompt")
        or sample.get("input")
        or ""
    )
    response = sample.get("response") or sample.get("output") or sample.get("answer") or ""
    return instruction if response == "" else f"{instruction}\n{response}"


def build_dataset(subset: Iterable[dict]) -> Dataset:
    """Convert an iterable of samples into a HF Dataset of text strings."""
    texts: List[str] = [_format_text(ex) for ex in subset]
    return Dataset.from_dict({"text": texts})


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int = 1024,
) -> Dataset:
    """Tokenize dataset for causal LM training."""
    def _tokenize(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        labels = tokenized["input_ids"].copy()
        tokenized["labels"] = [
            [(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
            for seq in labels
        ]
        return tokenized

    return dataset.map(_tokenize, batched=True, remove_columns=["text"])


@dataclass
class TrainConfig:
    base_model: str = "meta-llama/Llama-3-8B-Instruct"
    output_dir: str = "./models/default"
    max_length: int = 1024
    batch_size: int = 4
    lr: float = 2e-5
    num_epochs: float = 3.0
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    gradient_accumulation: int = 1


def train_on_subset(
    subset: Iterable[dict],
    config: TrainConfig,
    max_train_samples: Optional[int] = None,
):
    """
    Fine-tune a causal LM on the provided subset.
    Returns the Trainer object for downstream evaluation.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_ds = build_dataset(subset)
    if max_train_samples:
        raw_ds = raw_ds.select(range(max_train_samples))
    tokenized_ds = tokenize_dataset(raw_ds, tokenizer, max_length=config.max_length)

    model = AutoModelForCausalLM.from_pretrained(config.base_model)
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.num_epochs,
        gradient_accumulation_steps=config.gradient_accumulation,
        fp16=config.fp16,
        bf16=config.bf16,
        save_strategy="epoch",
        logging_steps=10,
        seed=config.seed,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    return trainer
