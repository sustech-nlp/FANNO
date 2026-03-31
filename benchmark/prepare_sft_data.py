#!/usr/bin/env python3
"""
Prepare FANNO 160K data for SFT training.

Merges single-turn (cleaned_single_turn.jsonl) and multi-turn (cleaned_multi_turn.jsonl)
into a unified ShareGPT format compatible with the fanno.train.sft script.

Output format (ShareGPT / conversations):
    {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

Usage:
    python prepare_sft_data.py \
        --single-turn ../synthesis/outputs/cleaned_single_turn.jsonl \
        --multi-turn ../synthesis/outputs/cleaned_multi_turn.jsonl \
        --output fanno_160k_sharegpt.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def convert_single_turn(item: dict) -> dict:
    """Convert question/answer format to ShareGPT conversations format.

    Handles multiple field naming conventions:
    - question/answer (complex_qa, code_qa, math_qa, reasoning_qa, fanno_seed_qa)
    - instruction/response (creative_writing, self_inverted_qa)
    - instruction/output (alpaca format)
    """
    # Try multiple field name conventions
    question = (
        item.get("question", "").strip()
        or item.get("instruction", "").strip()
    )
    answer = (
        item.get("answer", "").strip()
        or item.get("response", "").strip()
        or item.get("output", "").strip()
    )
    # Math QA uses solution + final_answer
    if not answer and item.get("solution", "").strip():
        solution = item["solution"].strip()
        final_answer = str(item.get("final_answer", "")).strip()
        if final_answer:
            answer = f"{solution}\n\nFinal Answer: {final_answer}"
        else:
            answer = solution
    if not question or not answer:
        return None
    return {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ],
        "source": item.get("source", "fanno_single_turn"),
    }


def convert_multi_turn(item: dict) -> dict:
    """Convert conversation format to ShareGPT conversations format."""
    conv = item.get("conversation", [])
    if len(conv) < 2:
        return None

    role_map = {"user": "human", "assistant": "gpt", "system": "system"}
    conversations = []
    for turn in conv:
        # Skip non-dict turns (e.g., trailing "scenario" strings)
        if not isinstance(turn, dict):
            continue
        role = role_map.get(turn.get("role", ""), turn.get("role", ""))
        content = turn.get("content", "").strip()
        if not content:
            continue
        conversations.append({"from": role, "value": content})

    if len(conversations) < 2:
        return None

    return {
        "conversations": conversations,
        "source": item.get("source", "fanno_multi_turn"),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare FANNO data for SFT")
    parser.add_argument(
        "--single-turn",
        type=str,
        default="../synthesis/outputs/cleaned_single_turn.jsonl",
    )
    parser.add_argument(
        "--multi-turn",
        type=str,
        default="../synthesis/outputs/cleaned_multi_turn.jsonl",
    )
    parser.add_argument(
        "--existing-sharegpt",
        type=str,
        default=None,
        help="Use existing merged_sharegpt.jsonl if available (skip single-turn conversion)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fanno_160k_sharegpt.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_data = []
    skipped = 0

    # Load single-turn data
    if args.existing_sharegpt and Path(args.existing_sharegpt).exists():
        print(f"Loading existing ShareGPT data from {args.existing_sharegpt}")
        with open(args.existing_sharegpt) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_data.append(json.loads(line))
        print(f"  Loaded {len(all_data)} samples from existing ShareGPT")
    else:
        print(f"Converting single-turn data from {args.single_turn}")
        with open(args.single_turn) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                converted = convert_single_turn(item)
                if converted:
                    all_data.append(converted)
                else:
                    skipped += 1
        print(f"  Converted {len(all_data)} single-turn samples (skipped {skipped})")

    # Load multi-turn data
    multi_count = 0
    multi_skipped = 0
    mt_path = Path(args.multi_turn)
    if mt_path.exists():
        print(f"Converting multi-turn data from {args.multi_turn}")
        with open(mt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                converted = convert_multi_turn(item)
                if converted:
                    all_data.append(converted)
                    multi_count += 1
                else:
                    multi_skipped += 1
        print(f"  Converted {multi_count} multi-turn samples (skipped {multi_skipped})")
    else:
        print(f"  Multi-turn file not found: {mt_path}, skipping")

    # Shuffle
    random.seed(args.seed)
    random.shuffle(all_data)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_data)} samples saved to {output_path}")

    # Print source distribution
    sources = {}
    for item in all_data:
        src = item.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print("\nSource distribution:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count} ({count / len(all_data):.1%})")


if __name__ == "__main__":
    main()
