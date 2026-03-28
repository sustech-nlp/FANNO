from __future__ import annotations

import math
import random
from typing import Dict, List

from tqdm import trange
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from fanno.config import InferenceConfig

random.seed(42)

_TOKENIZER_CACHE: dict[str, PreTrainedTokenizerBase] = {}


def _get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    if model_name not in _TOKENIZER_CACHE:
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        _TOKENIZER_CACHE[model_name] = tok
    return _TOKENIZER_CACHE[model_name]


def ucb_judge(seeds: List[Dict[str, str]], top_k: int = 3, N: int = 5000, model_name: str | None = None) -> List[List[Dict[str, str]]]:
    """
    UCB selection for seeds based on normalized response length.
    """
    for seed in seeds:
        seed.setdefault("cnt", 0)

    tokenizer = _get_tokenizer(model_name or InferenceConfig().model_name_or_path)

    len_values = [len(tokenizer.tokenize(seed["output"])) for seed in seeds]
    min_len, max_len = min(len_values), max(len_values)

    for seed in seeds:
        normalized_len = (len(tokenizer.tokenize(seed["output"])) - min_len) / (max_len - min_len) if max_len > min_len else 0.5
        seed["score"] = normalized_len

    results: List[List[Dict[str, str]]] = []
    total_attempts = sum(seed["cnt"] for seed in seeds) + len(seeds)

    for _ in trange(N):
        for seed in seeds:
            seed["value"] = seed["score"] + 3 * math.sqrt(2 * math.log(total_attempts) / (seed["cnt"] + 1))

        sorted_seeds = sorted(seeds, key=lambda x: x["value"], reverse=True)
        top_p = 0.05 * len(sorted_seeds)
        top_k_seed = random.sample(sorted_seeds[: int(top_p)], top_k)
        top_k_instructions = [seed for seed in top_k_seed]

        for seed in top_k_seed:
            seed["cnt"] += 1
            total_attempts += 1

        results.append(top_k_instructions)

    return results


def random_judge(seeds: List[Dict[str, str]], top_k: int = 3, N: int = 5000) -> List[List[Dict[str, str]]]:
    """
    Pure random sampling strategy for comparison or fallback.
    """
    for seed in seeds:
        seed.setdefault("cnt", 0)
    results: List[List[Dict[str, str]]] = []
    for _ in trange(N):
        sampled = random.sample(seeds, min(top_k, len(seeds)))
        for seed in sampled:
            seed["cnt"] += 1
        results.append(sampled)
    return results


__all__ = ["ucb_judge", "random_judge"]
