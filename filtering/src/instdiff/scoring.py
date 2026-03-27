from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from .io_utils import load_jsonl, write_jsonl
from .utils import chunk_list

DEFAULT_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

DIRECT_PROMPT = (
    "{instruction}"
)


def _compute_metrics(model, tokenizer, prompt: str, target: str) -> Dict[str, float]:
    device = model.device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prompt + "\n" + target, return_tensors="pt").input_ids.to(device)

    labels = full_ids[:, 1:].clone()
    logits = model(full_ids).logits[:, :-1, :]

    prompt_len = prompt_ids.shape[1] - 1
    target_len = labels.shape[1] - prompt_len

    if target_len <= 0:
        return {
            "nll": float("inf"),
            "avg_entropy": float("inf"),
            "ppl": float("inf"),
            "prompt_tokens": int(prompt_len),
            "response_tokens": int(target_len),
        }

    target_logits = logits[:, prompt_len:prompt_len + target_len]
    target_labels = labels[:, prompt_len:prompt_len + target_len]

    log_probs = F.log_softmax(target_logits, dim=-1)
    token_log_probs = log_probs.gather(2, target_labels.unsqueeze(2)).squeeze(2)
    nll = -token_log_probs.mean()

    probs = F.softmax(target_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    avg_entropy = entropy.mean()

    ppl = torch.exp(nll)

    return {
        "nll": float(nll.item()),
        "avg_entropy": float(avg_entropy.item()),
        "ppl": float(ppl.item()),
        "prompt_tokens": int(prompt_len),
        "response_tokens": int(target_len),
    }


def score_dataset(
    model_path: str,
    data_path: str,
    output_path: str,
    instruction_key: str = "instruction",
    input_key: str = "input",
    response_key: str = "response",
    prompt_template: str = DEFAULT_PROMPT,
    max_samples: Optional[int] = None,
    max_length: int = 4096,
) -> List[Dict[str, Any]]:
    data = load_jsonl(data_path)
    data = chunk_list(data, max_samples)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, model_max_length=max_length)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    results: List[Dict[str, Any]] = []
    for item in tqdm(data, desc="Scoring"):
        instruction = item[instruction_key]
        user_input = item.get(input_key, "")
        response = item[response_key]

        prompt_values = {
            "instruction": instruction,
            "input": user_input,
        }
        prompt = prompt_template.format_map(prompt_values)

        if tokenizer(prompt, return_tensors="pt").input_ids.shape[1] > max_length:
            continue
        if tokenizer(response, return_tensors="pt").input_ids.shape[1] > max_length:
            continue

        metrics = _compute_metrics(model, tokenizer, prompt, response)
        results.append(
            {
                "instruction": instruction,
                "input": user_input,
                "response": response,
                "prompt": prompt,
                "nll": metrics["nll"],
                "avg_entropy": metrics["avg_entropy"],
                "ppl": metrics["ppl"],
                "prompt_tokens": metrics["prompt_tokens"],
                "response_tokens": metrics["response_tokens"],
            }
        )

    write_jsonl(results, output_path)

    del model
    torch.cuda.empty_cache()

    return results
