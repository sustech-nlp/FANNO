import hashlib
import json
import math
import random
import re
from typing import List, Optional

from src.config import InferenceConfig

try:
    from src.inference_utils import client_parallel_inference
except Exception:
    client_parallel_inference = None

_DEFAULT_INFERENCE_CONFIG = InferenceConfig()


def read_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def load_jsonl(path: str) -> List[dict]:
    return read_jsonl(path)


def stable_hash(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16)


def hash_embedding(text: str, dim: int = 64) -> List[float]:
    vec = [0.0] * dim
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    for tok in tokens:
        idx = stable_hash(tok) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(vec_a, vec_b) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b))


def extract_seed_terms(doc: str, limit: int = 5) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", doc.lower())
    tokens = [t for t in tokens if len(t) > 3]
    random.shuffle(tokens)
    return tokens[:limit]


def call_gpt(
    prompt: str,
    temperature: float | None = None,
    model: str | None = None,
    max_tokens: Optional[int] = None,
    config: InferenceConfig | None = None,
) -> str:
    """
    LLM call wrapper compatible with inference_utils.client_parallel_inference.
    Falls back to an empty string if inference_utils is unavailable.
    """
    cfg = config or _DEFAULT_INFERENCE_CONFIG
    model_name = model or cfg.model
    send_temperature = cfg.temperature if temperature is None else temperature
    target_max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
    if client_parallel_inference:
        try:
            outputs = client_parallel_inference(
                [prompt],
                model_name=model_name,
                max_tokens=target_max_tokens,
                temperature=send_temperature,
                workers=1,
                config=cfg,
            )
            return outputs[0] if outputs else ""
        except Exception:
            pass
    # Minimal fallback
    return ""


__all__ = [
    "read_jsonl",
    "load_jsonl",
    "stable_hash",
    "hash_embedding",
    "cosine_similarity",
    "extract_seed_terms",
    "call_gpt",
]
