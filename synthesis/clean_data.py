"""
Data quality checks and deduplication for FANNO-Dev synthesized data.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter

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


def get_text_key(item: Dict) -> str:
    """Extract the primary text for deduplication."""
    q = item.get("question", item.get("instruction", ""))
    if isinstance(q, str):
        return q.strip().lower()
    return ""


def exact_dedup(data: List[Dict]) -> List[Dict]:
    """Remove exact duplicates based on question/instruction text."""
    seen: Set[str] = set()
    deduped = []
    for item in data:
        key = get_text_key(item)
        if not key:
            continue
        h = hashlib.md5(key.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(item)
    removed = len(data) - len(deduped)
    logger.info(f"Exact dedup: {len(data)} -> {len(deduped)} (removed {removed}, {removed/len(data)*100:.1f}%)")
    return deduped


def near_dedup(data: List[Dict], prefix_len: int = 80) -> List[Dict]:
    """Remove near-duplicates based on prefix matching."""
    seen_prefixes: Set[str] = set()
    deduped = []
    for item in data:
        key = get_text_key(item)
        if not key:
            continue
        prefix = key[:prefix_len]
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            deduped.append(item)
    removed = len(data) - len(deduped)
    logger.info(f"Near dedup (prefix={prefix_len}): {len(data)} -> {len(deduped)} (removed {removed})")
    return deduped


def quality_filter(data: List[Dict]) -> List[Dict]:
    """Apply quality filters to remove low-quality samples."""
    filtered = []
    rejected_reasons = Counter()

    for item in data:
        q = get_text_key(item)
        a_raw = item.get("answer", item.get("output", item.get("response", "")))
        a = a_raw if isinstance(a_raw, str) else ""

        # Skip empty
        if not q or not a:
            rejected_reasons["empty_qa"] += 1
            continue

        # Skip very short questions (< 10 chars)
        if len(q) < 10:
            rejected_reasons["short_question"] += 1
            continue

        # Skip very short answers (< 20 chars)
        if len(a) < 20:
            rejected_reasons["short_answer"] += 1
            continue

        # Skip if question is mostly non-alphanumeric
        alpha_ratio = sum(c.isalpha() for c in q) / len(q) if q else 0
        if alpha_ratio < 0.3:
            rejected_reasons["low_alpha"] += 1
            continue

        # Skip if answer contains obvious errors/refusals
        refusal_patterns = [
            "i cannot", "i can't", "as an ai", "i'm sorry, but i",
            "i don't have the ability", "i'm not able to",
        ]
        a_lower = a.lower()
        if any(p in a_lower[:100] for p in refusal_patterns):
            rejected_reasons["refusal"] += 1
            continue

        filtered.append(item)

    logger.info(f"Quality filter: {len(data)} -> {len(filtered)} (rejected: {dict(rejected_reasons)})")
    return filtered


def run_full_cleanup(input_dir: Path = None) -> Dict[str, Path]:
    """Run full cleanup: quality filter + exact dedup + near dedup."""
    if input_dir is None:
        input_dir = OUTPUT_DIR

    # Load all data
    all_single = []
    all_multi = []

    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        if jsonl_file.name.startswith("merged_") or jsonl_file.name.startswith("cleaned_"):
            continue
        if jsonl_file.name == "diversity_report.json":
            continue

        data = load_jsonl(jsonl_file)
        for item in data:
            if "conversation" in item:
                all_multi.append(item)
            else:
                all_single.append(item)

    logger.info(f"Loaded: {len(all_single)} single-turn, {len(all_multi)} multi-turn")

    # Clean single-turn
    logger.info("\n=== Cleaning single-turn data ===")
    cleaned_single = quality_filter(all_single)
    cleaned_single = exact_dedup(cleaned_single)
    cleaned_single = near_dedup(cleaned_single, prefix_len=80)

    # Save cleaned single-turn
    clean_single_path = input_dir / "cleaned_single_turn.jsonl"
    with open(clean_single_path, "w", encoding="utf-8") as f:
        for item in cleaned_single:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved cleaned single-turn: {len(cleaned_single)} -> {clean_single_path}")

    # Clean multi-turn (just basic dedup)
    logger.info("\n=== Cleaning multi-turn data ===")
    seen = set()
    cleaned_multi = []
    for item in all_multi:
        conv = item.get("conversation", [])
        if isinstance(conv, list) and len(conv) >= 4:
            # Use first user message as dedup key
            first_user = ""
            for t in conv:
                if isinstance(t, dict) and t.get("role") == "user":
                    first_user = t.get("content", "")[:100].lower()
                    break
            h = hashlib.md5(first_user.encode()).hexdigest()
            if h not in seen and first_user:
                seen.add(h)
                cleaned_multi.append(item)

    clean_multi_path = input_dir / "cleaned_multi_turn.jsonl"
    with open(clean_multi_path, "w", encoding="utf-8") as f:
        for item in cleaned_multi:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved cleaned multi-turn: {len(cleaned_multi)} -> {clean_multi_path}")

    # Summary
    total_cleaned = len(cleaned_single) + len(cleaned_multi)
    total_original = len(all_single) + len(all_multi)
    logger.info(f"\n=== CLEANUP SUMMARY ===")
    logger.info(f"Original: {total_original}")
    logger.info(f"Cleaned:  {total_cleaned}")
    logger.info(f"Removed:  {total_original - total_cleaned} ({(total_original-total_cleaned)/total_original*100:.1f}%)")

    return {
        "single_turn": clean_single_path,
        "multi_turn": clean_multi_path,
        "single_count": len(cleaned_single),
        "multi_count": len(cleaned_multi),
    }


if __name__ == "__main__":
    run_full_cleanup()
