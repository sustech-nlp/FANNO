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


def _detect_source(item: Dict) -> str:
    """Detect the source/type of a data item for source-aware filtering."""
    source = str(item.get("source", item.get("type", ""))).lower()
    question = str(item.get("question", item.get("instruction", ""))).lower()

    if "code" in source:
        return "code"
    if "math" in source:
        return "math"
    # Heuristic: detect code/math from content patterns
    code_indicators = ["```", "def ", "function ", "class ", "import ", "return ", "print("]
    if any(ind in question for ind in code_indicators):
        return "code"
    math_indicators = ["solve", "calculate", "equation", "integral", "derivative", "theorem", "prove that"]
    if any(ind in question for ind in math_indicators):
        return "math"
    return "general"


def quality_filter(data: List[Dict]) -> List[Dict]:
    """Apply quality filters to remove low-quality samples.

    Source-aware: uses relaxed thresholds for code/math content
    to avoid systematic over-rejection (previously 48.5% for code, 42.5% for math).
    """
    filtered = []
    rejected_reasons = Counter()

    for item in data:
        q = get_text_key(item)
        a_raw = item.get("answer", item.get("output", item.get("response", item.get("solution", ""))))
        a = a_raw if isinstance(a_raw, str) else ""
        source_type = _detect_source(item)

        # Skip empty
        if not q or not a:
            rejected_reasons["empty_qa"] += 1
            continue

        # Skip very short questions (< 10 chars)
        if len(q) < 10:
            rejected_reasons["short_question"] += 1
            continue

        # Skip very short answers: stricter for general (50 chars), lenient for code/math (20 chars)
        min_answer_len = 20 if source_type in ("code", "math") else 50
        if len(a) < min_answer_len:
            rejected_reasons["short_answer"] += 1
            continue

        # Source-aware alpha ratio check:
        # Code/math naturally contain symbols, operators, brackets etc.
        # General: alpha_ratio >= 0.3, Code: >= 0.15, Math: >= 0.15
        alpha_ratio = sum(c.isalpha() for c in q) / len(q) if q else 0
        alpha_threshold = 0.15 if source_type in ("code", "math") else 0.3
        if alpha_ratio < alpha_threshold:
            rejected_reasons["low_alpha"] += 1
            continue

        # Skip if answer contains obvious errors/refusals
        refusal_patterns = [
            "i cannot", "i can't", "as an ai", "i'm sorry, but i",
            "i don't have the ability", "i'm not able to",
            "i apologize", "as a language model",
        ]
        a_lower = a.lower()
        if any(p in a_lower[:150] for p in refusal_patterns):
            rejected_reasons["refusal"] += 1
            continue

        # Code-specific: reject if code answer has no actual code
        if source_type == "code":
            if "```" not in a and "def " not in a and "function " not in a and "class " not in a:
                # Allow short answers that might be explanations
                if len(a) < 100:
                    rejected_reasons["code_no_code_block"] += 1
                    continue

        # Math-specific: reject if math answer is suspiciously short or has no numbers
        if source_type == "math":
            has_numbers = any(c.isdigit() for c in a)
            has_math_symbols = any(c in a for c in "=+-*/^√∫∑∏")
            if not has_numbers and not has_math_symbols and len(a) < 100:
                rejected_reasons["math_no_computation"] += 1
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

    # Clean multi-turn with quality filtering (previously only dedup)
    logger.info("\n=== Cleaning multi-turn data ===")
    rejected_multi_reasons = Counter()
    refusal_patterns_mt = [
        "i cannot", "i can't", "as an ai", "i'm sorry, but i",
        "i don't have the ability", "i'm not able to",
        "i apologize", "as a language model",
    ]
    seen = set()
    cleaned_multi = []
    for item in all_multi:
        conv = item.get("conversation", [])
        if not isinstance(conv, list):
            rejected_multi_reasons["invalid_format"] += 1
            continue

        # Minimum turn count: at least 2 complete exchanges (4 turns)
        if len(conv) < 4:
            rejected_multi_reasons["too_few_turns"] += 1
            continue

        # Extract user and assistant turns
        user_turns = [t for t in conv if isinstance(t, dict) and t.get("role") == "user"]
        asst_turns = [t for t in conv if isinstance(t, dict) and t.get("role") == "assistant"]

        if not user_turns or not asst_turns:
            rejected_multi_reasons["missing_role"] += 1
            continue

        # Check first user message quality
        first_user_content = str(user_turns[0].get("content", "")).strip()
        if len(first_user_content) < 10:
            rejected_multi_reasons["short_first_user"] += 1
            continue

        # Check for refusals in assistant turns (especially first response)
        first_asst_content = str(asst_turns[0].get("content", "")).lower()
        if any(p in first_asst_content[:150] for p in refusal_patterns_mt):
            rejected_multi_reasons["assistant_refusal"] += 1
            continue

        # Check for empty or garbage assistant turns
        empty_asst_count = sum(
            1 for t in asst_turns
            if len(str(t.get("content", "")).strip()) < 10
        )
        if empty_asst_count > len(asst_turns) * 0.5:
            rejected_multi_reasons["too_many_empty_asst"] += 1
            continue

        # Check total substance: conversation should have meaningful content
        total_content_len = sum(
            len(str(t.get("content", "")))
            for t in conv if isinstance(t, dict)
        )
        if total_content_len < 200:
            rejected_multi_reasons["too_short_total"] += 1
            continue

        # Check for repetitive assistant responses (same response repeated)
        asst_contents = [str(t.get("content", ""))[:100].lower() for t in asst_turns]
        if len(asst_contents) >= 2:
            unique_ratio = len(set(asst_contents)) / len(asst_contents)
            if unique_ratio < 0.5:
                rejected_multi_reasons["repetitive_assistant"] += 1
                continue

        # Dedup by first user message
        dedup_key = first_user_content[:100].lower()
        h = hashlib.md5(dedup_key.encode()).hexdigest()
        if h in seen:
            rejected_multi_reasons["duplicate"] += 1
            continue
        seen.add(h)

        cleaned_multi.append(item)

    logger.info(f"Multi-turn quality filter: {len(all_multi)} -> {len(cleaned_multi)} (rejected: {dict(rejected_multi_reasons)})")

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
