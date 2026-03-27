"""
Evaluation utilities for OOD and in-distribution performance.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# Registry for benchmark evaluators; users can plug custom functions.
BENCHMARK_FUNCS: Dict[str, Callable] = {}


def register_benchmark(name: str, fn: Callable):
    BENCHMARK_FUNCS[name.lower()] = fn


def compute_perplexity_on_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Iterable[dict],
    batch_size: int = 4,
    max_length: int = 1024,
    device: Optional[str] = None,
) -> float:
    """
    Compute perplexity on a small evaluation set (e.g., Magpie test).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    losses: List[float] = []
    batch_texts: List[str] = []
    for sample in dataset:
        instruction = (
            sample.get("instruction")
            or sample.get("prompt")
            or sample.get("input")
            or ""
        )
        response = sample.get("response") or sample.get("output") or sample.get("answer") or ""
        text = instruction if response == "" else f"{instruction}\n{response}"
        batch_texts.append(text)
        if len(batch_texts) < batch_size:
            continue

        loss = _compute_batch_loss(
            model, tokenizer, batch_texts, device=device, max_length=max_length
        )
        losses.append(loss)
        batch_texts.clear()

    if batch_texts:
        loss = _compute_batch_loss(
            model, tokenizer, batch_texts, device=device, max_length=max_length
        )
        losses.append(loss)

    mean_loss = float(np.mean(losses))
    return float(np.exp(mean_loss))


def _compute_batch_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    device: torch.device,
    max_length: int,
) -> float:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    labels = encoded["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    with torch.no_grad():
        outputs = model(**encoded, labels=labels)
    return float(outputs.loss.item())


def aggregate_relative_performance(
    benchmark_scores: Dict[str, float],
    reference_scores: Dict[str, float],
) -> float:
    """
    Compute the mean relative performance across OOD benchmarks:
    Perf = mean_i score_i / ref_i
    """
    ratios: List[float] = []
    for name, score in benchmark_scores.items():
        ref = reference_scores.get(name)
        if ref is None or ref == 0:
            continue
        ratios.append(score / ref)
    return float(np.mean(ratios)) if ratios else float("nan")


def evaluate_model(
    model: PreTrainedModel,
    benchmarks: List[str],
    reference_scores: Optional[Dict[str, float]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    id_test_set: Optional[Iterable[dict]] = None,
    max_length: int = 1024,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a model on registered OOD benchmarks and optional ID perplexity.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    scores: Dict[str, float] = {}

    # In-distribution perplexity
    if id_test_set is not None:
        scores["id_perplexity"] = compute_perplexity_on_dataset(
            model,
            tokenizer,
            id_test_set,
            max_length=max_length,
            device=device,
        )

    # OOD benchmarks
    for bench in benchmarks:
        fn = BENCHMARK_FUNCS.get(bench.lower())
        if fn is None:
            # placeholder to be filled by user-provided evaluation hooks
            scores[bench] = float("nan")
            continue
        scores[bench] = fn(model=model, tokenizer=tokenizer)

    if reference_scores:
        bench_subset = {
            k: v
            for k, v in scores.items()
            if k in reference_scores and isinstance(v, (int, float))
        }
        scores["ood_performance"] = aggregate_relative_performance(
            bench_subset, reference_scores
        )
    return scores
