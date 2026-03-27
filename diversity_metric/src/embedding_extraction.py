"""
Utilities to extract semantic and gradient-based embeddings for instruction data.

Semantic embeddings:
    - Uses a causal LM or encoder model; pooling over the last hidden state.

Gradient embeddings (Prismatic-inspired):
    - Computes per-sample loss gradients of a small proxy model.
    - Applies a Rademacher random projection to compress the gradient vector.
    - Normalizes the projected vector to unit norm.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import logging
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


def _ensure_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _format_sample(sample: dict) -> Tuple[str, str]:
    instruction = (
        sample.get("instruction")
        or sample.get("prompt")
        or sample.get("input")
        or ""
    )
    response = sample.get("response") or sample.get("output") or sample.get("answer") or ""

    # Fallback: handle conversation-style data (e.g., Magpie/ShareGPT format)
    if (not instruction) and (not response):
        conv = sample.get("conversations")
        if isinstance(conv, list) and conv:
            user_text = ""
            assistant_text = ""
            for idx, turn in enumerate(conv):
                if not isinstance(turn, dict):
                    continue
                role = (turn.get("from") or turn.get("role") or "").lower()
                content = turn.get("value") or turn.get("content") or ""
                if not user_text and role in {"human", "user", "system", "instruction"} and content:
                    user_text = content.strip()
                    # look ahead for the next assistant/model turn
                    for next_turn in conv[idx + 1 :]:
                        if not isinstance(next_turn, dict):
                            continue
                        nrole = (next_turn.get("from") or next_turn.get("role") or "").lower()
                        ncontent = next_turn.get("value") or next_turn.get("content") or ""
                        if nrole in {"gpt", "assistant", "model"} and ncontent:
                            assistant_text = ncontent.strip()
                            break
                    break
            if user_text:
                instruction = user_text
            if assistant_text:
                response = assistant_text

    return instruction.strip(), response.strip()


def _pool_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool over the sequence length using the attention mask."""
    mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def _maybe_len(obj: Iterable) -> Optional[int]:
    """Safely get length of iterable if available."""
    try:
        return len(obj)  # type: ignore[arg-type]
    except Exception:
        return None


def extract_semantic_embeddings(
    dataset: Iterable[dict],
    model_id: str,
    batch_size: int = 8,
    max_length: int = 256,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Extract semantic embeddings via last-layer mean pooling.

    Args:
        dataset: iterable of dicts containing at least instruction/response.
        model_id: HF model name (encoder or decoder).
        batch_size: batch size for inference.
        max_length: max tokens fed to the encoder.
        device: device string; defaults to CUDA if available.
    """
    device = _ensure_device(device)
    logger.info("Loading semantic model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token  # ensure padding works for decoder-only tokenizers
    model = AutoModel.from_pretrained(model_id)
    model.to(device)
    model.eval()

    all_embs: List[np.ndarray] = []
    batch_instructions: List[str] = []
    iterator = tqdm(dataset, total=_maybe_len(dataset), desc=f"Semantic[{model_id}]")
    for sample in iterator:
        instruction, response = _format_sample(sample)
        text = instruction if response == "" else f"{instruction}\n{response}"
        batch_instructions.append(text)
        if len(batch_instructions) < batch_size:
            continue

        enc = tokenizer(
            batch_instructions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
            hidden = outputs.last_hidden_state  # [B, L, H]
            pooled = _pool_hidden(hidden, enc.attention_mask)  # [B, H]
        all_embs.append(pooled.cpu().numpy())
        batch_instructions.clear()

    # Flush remaining samples
    if batch_instructions:
        enc = tokenizer(
            batch_instructions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
            hidden = outputs.last_hidden_state
            pooled = _pool_hidden(hidden, enc.attention_mask)
        all_embs.append(pooled.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def _rademacher_projection(
    grad_vec: torch.Tensor, proj_dim: int, chunk_size: int = 8192, seed: int = 42
) -> torch.Tensor:
    """
    Memory-aware Rademacher random projection.

    Instead of materializing the full |theta| x d matrix, generate sign
    vectors in chunks to keep memory bounded.
    """
    device = grad_vec.device
    generator = torch.Generator(device=device).manual_seed(seed)
    proj = torch.zeros(proj_dim, device=device)
    total_dim = grad_vec.numel()
    scale = torch.tensor(total_dim, device=device, dtype=grad_vec.dtype).sqrt()

    for start in range(0, total_dim, chunk_size):
        end = min(total_dim, start + chunk_size)
        chunk = grad_vec[start:end]  # [chunk]
        # 修改这里：先创建空tensor，然后传入generator
        signs = torch.empty((proj_dim, chunk.numel()), device=device, dtype=grad_vec.dtype)
        signs.bernoulli_(0.5, generator=generator)
        signs = signs.mul(2).sub_(1)  # {-1, +1}
        proj += torch.matmul(signs, chunk)
    proj = proj / scale
    return proj

def _build_lm_inputs(
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    response: str,
    max_length: int,
    device: torch.device,
) -> dict:
    text = instruction if response == "" else f"{instruction}\n{response}"
    if not text.strip():
        # Ensure at least one token to avoid empty sequence errors.
        fallback = tokenizer.eos_token or tokenizer.pad_token or " "
        text = fallback
    encoded = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encoded = tokenizer.pad(
        encoded, padding=True, return_tensors="pt", max_length=max_length
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    labels = encoded["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    encoded["labels"] = labels
    return encoded


def extract_gradient_embeddings(
    dataset: Iterable[dict],
    proxy_model_id: str,
    proj_dim: int = 1024,
    max_length: int = 256,
    device: Optional[str] = None,
    chunk_size: int = 8192,
) -> np.ndarray:
    """
    Extract gradient embeddings using a proxy instruction-following model.

    Steps:
        1. Compute loss gradient per sample.
        2. Normalize gradient vector.
        3. Apply Rademacher random projection to proj_dim.

    This is compute-intensive; intended for offline preprocessing.
    """
    device = _ensure_device(device)
    logger.info("Loading proxy model for gradient embeddings: %s", proxy_model_id)
    tokenizer = AutoTokenizer.from_pretrained(proxy_model_id)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(proxy_model_id)
    model.to(device)
    model.train()  # enable gradients

    projected_embs: List[np.ndarray] = []
    iterator = tqdm(dataset, total=_maybe_len(dataset), desc=f"Gradient[{proxy_model_id}]")
    for sample in iterator:
        instruction, response = _format_sample(sample)
        inputs = _build_lm_inputs(
            tokenizer, instruction, response, max_length=max_length, device=device
        )
        model.zero_grad(set_to_none=True)
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        grad_vec = parameters_to_vector(
            [p.grad.detach() for p in model.parameters() if p.grad is not None]
        )
        grad_vec = grad_vec / (grad_vec.norm() + 1e-8)
        proj = _rademacher_projection(
            grad_vec, proj_dim=proj_dim, chunk_size=chunk_size
        )
        proj = proj / (proj.norm() + 1e-8)
        projected_embs.append(proj.cpu().numpy())

    return np.vstack(projected_embs)


def concat_embeddings(*arrays: Sequence[np.ndarray]) -> np.ndarray:
    """
    Concatenate embeddings along feature dimension, ensuring consistent length.
    """
    if not arrays:
        raise ValueError("No embeddings provided.")
    base_len = arrays[0].shape[0]
    for arr in arrays:
        if arr.shape[0] != base_len:
            raise ValueError("All embedding arrays must have the same number of rows.")
    return np.concatenate(arrays, axis=1)
