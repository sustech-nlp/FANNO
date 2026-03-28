"""Unified inference dispatch: vLLM (local) or Azure OpenAI (cloud)."""

from __future__ import annotations

import re
from typing import List, Sequence

from fanno.config import InferenceConfig


def _build_prompt(prompt: str, template_type: str, tokenizer) -> str:
    """Build prompt with template formatting (shared between vLLM and Azure paths)."""
    if template_type == "alpaca":
        return (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response:\n"
        )
    if template_type == "mistral":
        return f"<|im_start|>user\n{prompt}\n<|im_end|>"
    if template_type == "direct":
        return prompt
    # Auto: use tokenizer chat template
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parser_score(text_outputs: List[str]) -> List[int]:
    """Parse `score: X` patterns from model outputs."""
    pattern = re.compile(r"score:\s*(\d)", re.IGNORECASE)
    return [int(match.group(1)) if (match := pattern.search(s)) else 0 for s in text_outputs]


def run_inference(
    prompt_list: Sequence[str],
    config: InferenceConfig,
    template_type: str = "auto",
    score: bool = False,
) -> List[str] | List[int]:
    """Dispatch inference to vLLM or Azure client based on config.backend."""
    if not prompt_list:
        return []

    if config.backend == "azure":
        from fanno.inference.client_inference import client_parallel_inference

        if template_type in {"direct", "alpaca", "mistral"}:
            templated = [_build_prompt(p, template_type, None) for p in prompt_list] if template_type != "direct" else list(prompt_list)
        else:
            templated = list(prompt_list)
        temperature = None if str(config.model_name_or_path).startswith("gpt-5") else config.temperature
        outputs = client_parallel_inference(
            templated,
            model_name=config.model_name_or_path,
            max_tokens=config.max_tokens,
            temperature=temperature if temperature is not None else 0.0,
            tenant_id=config.azure_tenant_id,
            api_version=config.azure_api_version,
            max_retries=config.azure_max_retries,
            workers=8,
        )
        return parser_score(outputs) if score else outputs

    # default: vLLM (lazy import to avoid requiring vllm when using Azure)
    from fanno.inference.vllm_inference import parallel_inference

    return parallel_inference(prompt_list, config=config, template_type=template_type, score=score)


def clear_cached_llms() -> None:
    """Release cached vLLM engines (lazy import)."""
    try:
        from fanno.inference.vllm_inference import clear_cached_llms as _clear
        _clear()
    except ImportError:
        pass


__all__ = ["run_inference", "clear_cached_llms", "parser_score", "_build_prompt"]
