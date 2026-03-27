"""
Merge all synthesized data into unified FANNO format.
Normalizes different data formats into a consistent schema for training.
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger


OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_jsonl(path: Path) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def normalize_to_alpaca(item: Dict) -> Dict:
    """Normalize any data format to Alpaca format (instruction, input, output)."""
    result = {
        "instruction": "",
        "input": "",
        "output": "",
        "source": item.get("source", "unknown"),
        "domain": item.get("domain", "general"),
        "difficulty": item.get("difficulty", "medium"),
        "type": item.get("type", "qa"),
    }

    # Question/Answer format
    if "question" in item:
        result["instruction"] = item["question"]
        result["output"] = item.get("answer", item.get("solution", ""))
    # Instruction/Output format
    elif "instruction" in item:
        result["instruction"] = item["instruction"]
        result["input"] = item.get("input", "")
        result["output"] = item.get("output", "")
    # Creative writing format
    elif "instruction" in item:
        result["instruction"] = item["instruction"]
        result["output"] = item.get("response", "")

    return result


def normalize_to_sharegpt(item: Dict) -> Dict:
    """Normalize to ShareGPT conversation format."""
    # Multi-turn conversation
    if "conversation" in item:
        conversations = item["conversation"]
        if not isinstance(conversations, list):
            return None
        normalized_conv = []
        for t in conversations:
            if isinstance(t, dict):
                role = "human" if t.get("role") == "user" else "gpt"
                content = t.get("content", "")
                if isinstance(content, str) and content.strip():
                    normalized_conv.append({"from": role, "value": content})
        if len(normalized_conv) >= 2:
            return {
                "conversations": normalized_conv,
                "source": item.get("source", "unknown"),
                "topic": item.get("topic_summary", ""),
            }
        return None

    # Single-turn QA -> convert to conversation
    q = item.get("question", item.get("instruction", ""))
    a = item.get("answer", item.get("output", item.get("response", "")))

    if q and a and isinstance(q, str) and isinstance(a, str):
        return {
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": a},
            ],
            "source": item.get("source", "unknown"),
            "domain": item.get("domain", "general"),
        }

    return None


def merge_all_data(
    output_dir: Path = None,
    output_format: str = "both",
    shuffle: bool = True,
    cleaned_only: bool = False,
) -> Dict[str, Path]:
    """Merge all synthesized data into unified datasets.

    Args:
        cleaned_only: If True, only load cleaned_single_turn.jsonl and
                      cleaned_multi_turn.jsonl (recommended for final output).
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    all_data = []
    multi_turn_data = []

    if cleaned_only:
        # Only load cleaned data files
        source_files = [
            output_dir / "cleaned_single_turn.jsonl",
            output_dir / "cleaned_multi_turn.jsonl",
        ]
    else:
        source_files = sorted(output_dir.glob("*.jsonl"))

    for jsonl_file in source_files:
        if not jsonl_file.exists():
            continue
        if jsonl_file.name in ["diversity_report.json", "merged_alpaca.jsonl",
                               "merged_sharegpt.jsonl", "merged_all.jsonl",
                               "cleaned_merged_alpaca.jsonl",
                               "cleaned_merged_sharegpt.jsonl"]:
            continue

        data = load_jsonl(jsonl_file)
        if not data:
            continue

        logger.info(f"Loading {jsonl_file.name}: {len(data)} samples")

        for item in data:
            if "conversation" in item:
                multi_turn_data.append(item)
            else:
                all_data.append(item)

    logger.info(f"Total single-turn: {len(all_data)}, multi-turn: {len(multi_turn_data)}")

    results = {}

    # Alpaca format (single-turn only)
    if output_format in ["both", "alpaca"]:
        alpaca_data = []
        for item in all_data:
            normalized = normalize_to_alpaca(item)
            if normalized["instruction"] and normalized["output"]:
                alpaca_data.append(normalized)

        if shuffle:
            random.shuffle(alpaca_data)

        alpaca_path = output_dir / "merged_alpaca.jsonl"
        with open(alpaca_path, "w", encoding="utf-8") as f:
            for item in alpaca_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Alpaca format: {len(alpaca_data)} samples -> {alpaca_path}")
        results["alpaca"] = alpaca_path

    # ShareGPT format (all data including multi-turn)
    if output_format in ["both", "sharegpt"]:
        sharegpt_data = []

        for item in all_data:
            normalized = normalize_to_sharegpt(item)
            if normalized:
                sharegpt_data.append(normalized)

        for item in multi_turn_data:
            normalized = normalize_to_sharegpt(item)
            if normalized:
                sharegpt_data.append(normalized)

        if shuffle:
            random.shuffle(sharegpt_data)

        sharegpt_path = output_dir / "merged_sharegpt.jsonl"
        with open(sharegpt_path, "w", encoding="utf-8") as f:
            for item in sharegpt_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"ShareGPT format: {len(sharegpt_data)} samples -> {sharegpt_path}")
        results["sharegpt"] = sharegpt_path

    return results


if __name__ == "__main__":
    merge_all_data()
