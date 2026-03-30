#!/usr/bin/env python3
"""
Convert HopChain GPT-5 queries (hopchain_gpt5_queries.jsonl) to LLaMA-Factory VLM SFT format.

Input:  hopchain_gpt5_queries.jsonl  (9 multi-hop queries, 3 per image)
Output: hopchain_sft_data.json       (LLaMA-Factory sharegpt format with images)

Usage:
    python prepare_sft_data.py [--input INPUT] [--output OUTPUT] [--image_dir IMAGE_DIR]
"""

import argparse
import json
import os
from pathlib import Path


# Map image index to filename.
# Queries are grouped by image: lines 0-2 → kitchen, 3-5 → office_desk, 6-8 → street_scene.
IMAGE_FILES = ["kitchen.jpg", "office_desk.jpg", "street_scene.jpg"]
QUERIES_PER_IMAGE = 3


def build_cot_response(reasoning_hops: list, answer: str) -> str:
    """Build a chain-of-thought response from reasoning hops.

    Wraps the step-by-step reasoning in <think> tags (Qwen2.5-VL / Qwen3-VL template)
    and appends the final answer outside the tags.
    """
    steps = []
    for hop in reasoning_hops:
        hop_num = hop["hop_number"]
        desc = hop["description"]
        output = hop["output"]
        steps.append(f"Step {hop_num}: {desc} → {output}")

    think_block = "\n".join(steps)
    response = f"<think>\n{think_block}\n</think>\nThe answer is {answer}."
    return response


def convert_to_sft_format(
    input_path: str,
    output_path: str,
    image_dir: str,
) -> list:
    """Convert JSONL queries to LLaMA-Factory VLM SFT JSON format."""

    with open(input_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 9, f"Expected 9 queries, got {len(lines)}"

    sft_data = []

    for idx, line in enumerate(lines):
        query_data = json.loads(line)

        # Determine which image this query belongs to
        image_idx = idx // QUERIES_PER_IMAGE
        image_file = IMAGE_FILES[image_idx]
        image_path = os.path.join(image_dir, image_file)

        # Build user message with image tag
        user_content = f"<image>{query_data['query']}"

        # Build assistant response with CoT
        answer = str(query_data["hypothetical_answer"])
        assistant_content = build_cot_response(query_data["reasoning_hops"], answer)

        sft_entry = {
            "messages": [
                {"content": user_content, "role": "user"},
                {"content": assistant_content, "role": "assistant"},
            ],
            "images": [image_path],
        }

        sft_data.append(sft_entry)

    # Write output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sft_data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(sft_data)} queries to SFT format")
    print(f"Output: {output_path}")

    # Print a sample entry
    print("\n--- Sample entry ---")
    print(json.dumps(sft_data[0], indent=2, ensure_ascii=False))

    return sft_data


def main():
    parser = argparse.ArgumentParser(description="Convert HopChain queries to SFT format")
    parser.add_argument(
        "--input",
        default="results/hopchain_gpt5_queries.jsonl",
        help="Input JSONL file with HopChain queries",
    )
    parser.add_argument(
        "--output",
        default="configs/hopchain_sft_data.json",
        help="Output JSON file for LLaMA-Factory SFT",
    )
    parser.add_argument(
        "--image_dir",
        default="__BASE_DIR__/data/hopchain/test_images",
        help="Image directory path (use $$BASE_DIR for Azure storage)",
    )
    args = parser.parse_args()

    convert_to_sft_format(args.input, args.output, args.image_dir)


if __name__ == "__main__":
    main()
