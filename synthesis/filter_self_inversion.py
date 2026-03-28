"""
Quality filter for self_inversion data using LLM-as-judge.
Self-inversion has 29.8% bad answer rate — the worst among all sources.
This script evaluates ALL self_inversion items and filters out low quality.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from synthesis.api_client import parallel_call_gpt


OUTPUT_DIR = Path(__file__).parent / "outputs"
REPORT_DIR = Path(__file__).parent / "reports"

JUDGE_PROMPT = """Below is a question and a candidate answer.
Rate the answer quality from 1 to 5:
1: Incorrect, incomplete, or off-topic
2: Partially addresses question but has notable issues
3: Acceptable but could be improved
4: Good - accurate and relevant
5: Excellent - comprehensive and expert-level

Reply with ONLY a single number (1-5).

### Question:
{question}

### Answer:
{answer}

### Score:"""


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


def get_text(val) -> str:
    if isinstance(val, str):
        return val
    if val is None:
        return ""
    return str(val)


def filter_self_inversion(
    workers: int = 30,
    batch_size: int = 100,
    min_score: int = 3,
):
    """Evaluate and filter all self_inversion data."""
    input_path = OUTPUT_DIR / "self_inverted_qa.jsonl"
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(input_path)
    logger.info(f"Loaded {len(data)} self_inversion items")

    # Build prompts
    prompts = []
    valid_items = []
    for item in data:
        q = get_text(item.get("question", item.get("instruction", "")))
        a = get_text(item.get("answer", item.get("output", item.get("response", ""))))
        if len(q.strip()) < 10 or len(a.strip()) < 20:
            continue
        prompt = JUDGE_PROMPT.format(question=q, answer=a[:3000])
        prompts.append(prompt)
        valid_items.append(item)

    logger.info(f"Valid items to evaluate: {len(valid_items)}")

    # Run in batches
    all_responses = []
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch = prompts[batch_start:batch_end]
        logger.info(f"Batch {batch_start//batch_size + 1}: {batch_start+1}-{batch_end}")

        responses = parallel_call_gpt(
            prompts=batch,
            model_name="gpt-4o-mini",  # Use mini for cost efficiency on 5K items
            max_tokens=5,
            temperature=0.0,
            workers=workers,
            retries=3,
        )
        all_responses.extend(responses)

    # Parse scores and filter
    kept = []
    rejected = []
    score_dist = Counter()
    failed = 0

    for item, resp in zip(valid_items, all_responses):
        score = None
        if resp:
            for ch in resp.strip():
                if ch in "12345":
                    score = int(ch)
                    break

        if score is None:
            failed += 1
            kept.append(item)  # Keep if we can't evaluate
            continue

        score_dist[score] += 1
        item["quality_score"] = score

        if score >= min_score:
            kept.append(item)
        else:
            rejected.append(item)

    logger.info(f"\nResults:")
    logger.info(f"  Total evaluated: {len(valid_items)}")
    logger.info(f"  Score distribution: {dict(sorted(score_dist.items()))}")
    logger.info(f"  Kept (score >= {min_score}): {len(kept)}")
    logger.info(f"  Rejected (score < {min_score}): {len(rejected)}")
    logger.info(f"  Failed to parse: {failed}")

    # Save filtered version
    output_path = OUTPUT_DIR / "self_inverted_qa_filtered.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved filtered data: {len(kept)} items -> {output_path}")

    # Save rejected for analysis
    rejected_path = REPORT_DIR / "self_inversion_rejected.jsonl"
    with open(rejected_path, "w", encoding="utf-8") as f:
        for item in rejected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved rejected data: {len(rejected)} items -> {rejected_path}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_items": len(data),
        "valid_items": len(valid_items),
        "kept": len(kept),
        "rejected": len(rejected),
        "failed_parse": failed,
        "min_score": min_score,
        "score_distribution": {str(k): v for k, v in sorted(score_dist.items())},
        "rejection_rate": round(len(rejected) / len(valid_items) * 100, 1) if valid_items else 0,
    }
    report_path = REPORT_DIR / "self_inversion_filter_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved: {report_path}")

    return report


if __name__ == "__main__":
    filter_self_inversion()
