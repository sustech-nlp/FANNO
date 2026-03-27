from typing import Dict, List, Any, Optional

from .io_utils import load_jsonl, write_jsonl
from .scoring import score_dataset, DEFAULT_PROMPT


def compare_models(
    base_model_path: str,
    instruct_model_path: str,
    data_path: str,
    output_path: str,
    instruction_key: str = "instruction",
    input_key: str = "input",
    response_key: str = "response",
    prompt_template: str = DEFAULT_PROMPT,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if prompt_template is None:
        prompt_template = DEFAULT_PROMPT
    base_results = score_dataset(
        base_model_path,
        data_path,
        output_path + ".base.jsonl",
        instruction_key=instruction_key,
        input_key=input_key,
        response_key=response_key,
        prompt_template=prompt_template,
        max_samples=max_samples,
    )

    instruct_results = score_dataset(
        instruct_model_path,
        data_path,
        output_path + ".instruct.jsonl",
        instruction_key=instruction_key,
        input_key=input_key,
        response_key=response_key,
        prompt_template=prompt_template,
        max_samples=max_samples,
    )

    min_len = min(len(base_results), len(instruct_results))
    base_results = base_results[:min_len]
    instruct_results = instruct_results[:min_len]

    combined: List[Dict[str, Any]] = []
    for base_item, instruct_item in zip(base_results, instruct_results):
        if base_item["instruction"] != instruct_item["instruction"]:
            continue
        combined.append(
            {
                "instruction": base_item["instruction"],
                "input": base_item.get("input", ""),
                "response": base_item["response"],
                "prompt": base_item["prompt"],
                "n_tokens": base_item["response_tokens"],
                "base_nll": base_item["nll"],
                "base_avg_entropy": base_item["avg_entropy"],
                "base_ppl": base_item["ppl"],
                "instruct_nll": instruct_item["nll"],
                "instruct_avg_entropy": instruct_item["avg_entropy"],
                "instruct_ppl": instruct_item["ppl"],
                "diff_nll": base_item["nll"] - instruct_item["nll"],
                "diff_entropy": base_item["avg_entropy"] - instruct_item["avg_entropy"],
                "diff_ppl": base_item["ppl"] - instruct_item["ppl"],
            }
        )

    write_jsonl(combined, output_path)
    return combined


def get_total_tokens(data: List[Dict[str, Any]]) -> int:
    return sum(item["n_tokens"] for item in data)


def filter_by_diff_nll(data: List[Dict[str, Any]], reject_ratio: float) -> List[Dict[str, Any]]:
    if not data:
        return []
    if reject_ratio <= 0:
        return data
    sorted_data = sorted(data, key=lambda x: x["diff_nll"])
    trim = int(len(sorted_data) * reject_ratio)
    if trim * 2 >= len(sorted_data):
        return sorted_data
    return sorted_data[trim: len(sorted_data) - trim]


def select_lowest_diff_entropy(
    data: List[Dict[str, Any]],
    select_ratio: float,
    total_tokens: int,
) -> List[Dict[str, Any]]:
    if not data:
        return []
    if select_ratio <= 0 or select_ratio > 1:
        return data
    sorted_data = sorted(data, key=lambda x: x["diff_entropy"])
    # target = max(1, int(len(sorted_data) * select_ratio))
    select_tokens = int(total_tokens * select_ratio)
    selected_data = []
    for item in sorted_data:
        selected_data.append(item)
        if sum(item["n_tokens"] for item in selected_data) >= select_tokens:
            break
    # return sorted_data[:target]
    return selected_data


def save_selected_data(selected: List[Dict[str, Any]], output_path: str) -> None:
    rows = []
    for item in selected:
        rows.append(
            {
                "instruction": item["instruction"],
                "input": item.get("input", ""),
                "response": item["response"],
            }
        )
    write_jsonl(rows, output_path)
