from __future__ import annotations

import re
from typing import List, Sequence

from loguru import logger
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from fanno.config import InferenceConfig


def parser_score(text_outputs: List[str]) -> List[int]:
    """Parse `score: X` patterns from model outputs."""
    pattern = re.compile(r"score:\s*(\d)", re.IGNORECASE)
    scores = [int(match.group(1)) if (match := pattern.search(s)) else 0 for s in text_outputs]
    return scores


def _build_prompt(prompt: str, template_type: str, tokenizer) -> str:
    if template_type == "alpaca":
        return (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response:\n"
        )
    if template_type == "mistral":
        return f"<|im_start|>user\n{prompt}\n<|im_end|>"
    if template_type == "direct":
        return prompt

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


_LLM_CACHE: dict[str, LLM] = {}


def _config_hash(config: InferenceConfig) -> str:
    return "|".join(
        [
            config.model_name_or_path,
            str(config.tensor_parallel_size),
            str(config.max_model_len),
            str(config.gpu_memory_utilization),
            str(config.dtype),
        ]
    )


def _get_llm(config: InferenceConfig) -> LLM:
    config_hash = _config_hash(config)
    if config_hash not in _LLM_CACHE:
        logger.info(f"Loading vLLM backend: {config.model_name_or_path}")
        _LLM_CACHE[config_hash] = LLM(
            model=config.model_name_or_path,
            tensor_parallel_size=config.tensor_parallel_size,
            trust_remote_code=True,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            seed=config.seed,
        )
    return _LLM_CACHE[config_hash]


def parallel_inference(
    prompt_list: Sequence[str],
    config: InferenceConfig,
    template_type: str = "auto",
    score: bool = False,
) -> List[str] | List[int]:
    """Run batched inference against a local vLLM engine."""
    if not prompt_list:
        return []

    llm = _get_llm(config)
    tokenizer = llm.get_tokenizer()
    templated_prompts = [_build_prompt(prompt, template_type, tokenizer) for prompt in prompt_list]

    sampling_params = SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=config.stop,
        logprobs=0,
        prompt_logprobs=0,
        skip_special_tokens=config.skip_special_tokens,
    )

    outputs = llm.generate(templated_prompts, sampling_params, use_tqdm=False)
    text_outputs = [output.outputs[0].text for output in outputs]

    logger.debug(f"First decoded output: {text_outputs[0][:200]}...") if text_outputs else None
    return parser_score(text_outputs) if score else text_outputs


def clear_cached_llms() -> None:
    """Release cached LLM engines."""
    _LLM_CACHE.clear()


__all__ = ["parallel_inference", "parser_score", "clear_cached_llms"]
