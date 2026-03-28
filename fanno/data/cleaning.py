"""Data cleaning and filtering utilities."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from loguru import logger


def instruction_cleaning(texts: List[str]) -> List[Tuple[str, str]]:
    """Clean instruction texts and split into (instruction, input) pairs."""
    cleaned_texts = [re.sub(r'^[*"\n]+|[*"\n]+$', "", text).strip() for text in texts]

    def _split(text: str) -> Tuple[str, str]:
        if "\n" in text:
            part1, part2 = text.split("\n", 1)
            return part1.strip(), part2.strip()
        return text.strip(), ""

    return [_split(text) for text in cleaned_texts]


def hard_filter(
    data: List[Dict[str, Any]],
    source_type: str = "general",
) -> List[Dict[str, Any]]:
    """Rule-based hard filtering for instruction data.

    Args:
        data: List of dicts with at least an "instruction" key.
        source_type: One of "general", "agent", "code". Agent and code data
            use relaxed filters (no reference/time keyword checks).

    Returns:
        Filtered list.
    """
    ref_keywords = [
        "based on", "according", "given", "mentioned", "refer",
        "provided", "passage", "text", "paragraph",
    ]
    time_keywords = [
        "recent", "current", "now", "today", "yesterday", "tomorrow",
        "soon", "upcoming", "recently", "coming", "currently",
    ]
    obj_keywords = ["name"]

    if source_type in ("agent", "code"):
        keywords = []  # relaxed filter for agent/code
    else:
        keywords = ref_keywords + time_keywords + obj_keywords

    remaining: List[Dict[str, Any]] = []
    for item in data:
        instruction = item.get("instruction", "")
        if not instruction:
            continue
        # ASCII-only check (skip for agent data which may have structured content)
        if source_type == "general" and not all(ord(c) < 128 for c in instruction):
            continue
        # Too-short filter
        if len(instruction.split()) < 5 and not instruction.endswith((".", "?")):
            continue
        # Keyword filter
        if any(re.search(key, instruction, re.IGNORECASE) for key in keywords):
            continue
        # Alpha-ratio filter (skip for code/agent)
        if source_type == "general":
            alpha = sum(1 for c in instruction if c.isalpha())
            non_alpha = sum(1 for c in instruction if not c.isalpha())
            if alpha < non_alpha:
                continue
        remaining.append(item)

    if data:
        ratio = 1 - len(remaining) / len(data) if data else 0
        logger.info(f"Hard filter ({source_type}): removed {ratio:.2%}, kept {len(remaining)}/{len(data)}")
    return remaining


__all__ = ["instruction_cleaning", "hard_filter"]
