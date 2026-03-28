from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from fanno.config import InferenceConfig, MetricsConfig
from fanno.inference import run_inference


def compute_perplexity(
    texts: Sequence[str],
    model_name_or_path: str,
    max_length: int = 1024,
    device: str | None = None,
) -> List[float]:
    """Compute a naive perplexity score for each text."""
    if not texts:
        return []

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device or "auto")
    model.eval()

    scores: List[float] = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        scores.append(float(torch.exp(loss).detach().cpu()))
    return scores


def _ifd_prompt(instruction: str) -> str:
    return (
        "You are evaluating how difficult it is for a language model to strictly follow the instruction below.\n"
        "Rate the difficulty from 1 (very easy) to 5 (very hard). Provide only one line in the format `score: X`.\n\n"
        f"Instruction: {instruction}\n\nscore:"
    )


def compute_ifd_scores(instructions: Sequence[str], inference_config: InferenceConfig, metrics_cfg: MetricsConfig) -> List[float]:
    if not instructions:
        return []
    prompts = [_ifd_prompt(instr) for instr in instructions]
    # Use a slightly higher temperature to surface nuanced judgments.
    cfg = inference_config
    cfg = InferenceConfig(
        **{**cfg.__dict__, "temperature": metrics_cfg.ifd_prompt_temperature, "max_tokens": 8, "top_p": 0.9}
    )
    scores = run_inference(prompts, config=cfg, template_type="direct", score=True)
    # parser_score already maps to ints; ensure float output for downstream math.
    return [float(s) for s in scores]


def _normalize(values: Sequence[float], invert: bool = False) -> List[float]:
    if not values:
        return []
    min_v, max_v = min(values), max(values)
    if max_v - min_v < 1e-8:
        norm = [0.5] * len(values)
    else:
        norm = [(v - min_v) / (max_v - min_v) for v in values]
    return [1 - v for v in norm] if invert else norm


def aggregate_instruction_values(perplexities: Sequence[float], ifd_scores: Sequence[float]) -> List[float]:
    """Combine perplexity (lower is better) and IFD (higher is harder) into a single utility score."""
    if not perplexities or not ifd_scores or len(perplexities) != len(ifd_scores):
        return []
    norm_ppl = _normalize(perplexities, invert=True)
    norm_ifd = _normalize(ifd_scores, invert=False)
    return [0.5 * n_ppl + 0.5 * n_ifd for n_ppl, n_ifd in zip(norm_ppl, norm_ifd)]


@dataclass
class InstructionMetrics:
    inference_config: InferenceConfig
    metrics_config: MetricsConfig

    def score(self, instructions: Sequence[str]) -> Dict[str, List[float]]:
        ppl_model = self.metrics_config.perplexity_model or self.inference_config.model_name_or_path
        perplexities = compute_perplexity(
            instructions,
            model_name_or_path=ppl_model,
            max_length=self.metrics_config.max_ppl_tokens,
        )
        ifd_scores = compute_ifd_scores(instructions, self.inference_config, self.metrics_config)
        values = aggregate_instruction_values(perplexities, ifd_scores)
        return {"perplexity": perplexities, "ifd": ifd_scores, "value": values}


__all__ = [
    "compute_perplexity",
    "compute_ifd_scores",
    "aggregate_instruction_values",
    "InstructionMetrics",
]
