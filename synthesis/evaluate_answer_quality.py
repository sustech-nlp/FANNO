"""
Answer Quality Evaluation using LLM-as-Judge.
Addresses the critical gap: answer correctness was NEVER checked in the pipeline.
Uses gpt-4o with faithfulness_eval-style scoring (1-5 scale).
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from synthesis.api_client import parallel_call_gpt, get_token


OUTPUT_DIR = Path(__file__).parent / "outputs"
REPORT_DIR = Path(__file__).parent / "reports"


# =============================================================================
# Faithfulness evaluation prompt (adapted from Humpback-style eval)
# =============================================================================

ANSWER_QUALITY_SYSTEM_PROMPT = """You are an expert evaluator of AI-generated answers.
Your job is to assess whether an answer is accurate, complete, and helpful for the given question.
Be strict but fair. Focus on factual correctness, completeness, and relevance."""

ANSWER_QUALITY_PROMPT = """Below is a question and a candidate answer.
Evaluate the answer quality using this 5-point scale:

1: The answer is incorrect, incomplete, off-topic, or contains significant errors. Missing key information.
2: The answer partially addresses the question but has notable issues: factual errors, missing important details, or unclear explanations.
3: The answer is acceptable - addresses the main question but could be improved. May lack depth or have minor issues.
4: The answer is good - accurate, relevant, and reasonably complete. Well-structured with only minor room for improvement.
5: The answer is excellent - comprehensive, accurate, well-organized, and demonstrates expert-level understanding.

Additional criteria for specific domains:
- Code: Must be syntactically correct, handle edge cases, and follow best practices.
- Math: Must show correct reasoning steps and arrive at the right answer.
- Reasoning: Must demonstrate valid logical steps without fallacies.

Reply with ONLY a single number (1, 2, 3, 4, or 5). No explanation needed.

### Question:
{question}

### Answer:
{answer}

### Your Score:"""


# =============================================================================
# Data loading helpers
# =============================================================================

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
    """Safely convert any value to string."""
    if isinstance(val, str):
        return val
    if val is None:
        return ""
    return str(val)


def extract_qa(item: Dict) -> Tuple[str, str, str]:
    """Extract (question, answer, source) from various data formats."""
    q = get_text(item.get("question", item.get("instruction", "")))
    a = get_text(item.get("answer", item.get("output", item.get("response", item.get("solution", "")))))
    source = get_text(item.get("source", item.get("type", "unknown")))
    return q, a, source


def determine_source_from_file(filename: str) -> str:
    """Determine source category from filename."""
    name = filename.lower()
    if "code" in name:
        return "code_qa"
    elif "math" in name:
        return "math_qa"
    elif "complex" in name:
        return "complex_qa"
    elif "reasoning" in name:
        return "reasoning_qa"
    elif "creative" in name:
        return "creative_writing"
    elif "seed" in name or "fanno" in name:
        return "fanno_seed_qa"
    elif "multi_turn" in name or "dialog" in name:
        return "multi_turn"
    elif "invert" in name:
        return "self_inversion"
    elif "document" in name or "doc" in name:
        return "document_qa"
    return "unknown"


# =============================================================================
# Sampling strategy
# =============================================================================

def stratified_sample(
    data_by_source: Dict[str, List[Dict]],
    total_samples: int = 1000,
    min_per_source: int = 50,
) -> List[Tuple[Dict, str]]:
    """Stratified sampling: proportional to source size with minimum per source."""
    sources = list(data_by_source.keys())
    total_items = sum(len(v) for v in data_by_source.values())

    if total_items == 0:
        return []

    # Calculate proportional allocation with minimum
    allocations = {}
    remaining = total_samples
    for src in sources:
        allocations[src] = min(min_per_source, len(data_by_source[src]))
        remaining -= allocations[src]

    # Distribute remaining proportionally
    if remaining > 0:
        for src in sources:
            extra = int(remaining * len(data_by_source[src]) / total_items)
            allocations[src] = min(allocations[src] + extra, len(data_by_source[src]))

    # Sample
    sampled = []
    for src, count in allocations.items():
        items = data_by_source[src]
        selected = random.sample(items, min(count, len(items)))
        for item in selected:
            sampled.append((item, src))

    random.shuffle(sampled)
    return sampled[:total_samples]


# =============================================================================
# Main evaluation
# =============================================================================

def evaluate_answer_quality(
    total_samples: int = 1000,
    model: str = "gpt-4o",
    workers: int = 30,
    batch_size: int = 50,
    input_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate answer quality across all synthesized data sources.

    Returns a comprehensive report with per-source quality scores.
    """
    if input_dir is None:
        input_dir = OUTPUT_DIR

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ANSWER QUALITY EVALUATION (LLM-as-Judge)")
    logger.info("=" * 60)

    # 1. Load all data by source
    data_by_source: Dict[str, List[Dict]] = defaultdict(list)

    source_files = [
        "code_qa.jsonl",
        "math_qa.jsonl",
        "complex_qa.jsonl",
        "complex_qa_extra.jsonl",
        "reasoning_qa.jsonl",
        "creative_writing.jsonl",
        "fanno_seed_qa.jsonl",
        "self_inverted_qa.jsonl",
    ]

    for fname in source_files:
        fpath = input_dir / fname
        if fpath.exists():
            items = load_jsonl(fpath)
            source = determine_source_from_file(fname)
            # Only keep items with both Q and A
            for item in items:
                q, a, _ = extract_qa(item)
                if len(q.strip()) >= 10 and len(a.strip()) >= 20:
                    data_by_source[source].append(item)

    for src, items in data_by_source.items():
        logger.info(f"  {src}: {len(items)} items loaded")

    total_loaded = sum(len(v) for v in data_by_source.values())
    logger.info(f"Total loaded: {total_loaded}")

    if total_loaded == 0:
        logger.error("No data found!")
        return {}

    # 2. Stratified sampling
    sampled = stratified_sample(data_by_source, total_samples=total_samples)
    logger.info(f"Sampled {len(sampled)} items for evaluation")

    sample_counts = Counter(src for _, src in sampled)
    for src, cnt in sorted(sample_counts.items()):
        logger.info(f"  {src}: {cnt} samples")

    # 3. Build evaluation prompts
    prompts = []
    sample_meta = []  # Track (item, source) for each prompt

    for item, source in sampled:
        q, a, _ = extract_qa(item)
        # Truncate very long answers to save tokens
        a_truncated = a[:3000] if len(a) > 3000 else a
        prompt = ANSWER_QUALITY_PROMPT.format(question=q, answer=a_truncated)
        prompts.append(prompt)
        sample_meta.append((item, source))

    # 4. Run LLM-as-judge in batches
    logger.info(f"Running LLM-as-judge evaluation ({len(prompts)} prompts, {workers} workers)...")

    all_responses = []
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        logger.info(f"  Batch {batch_start//batch_size + 1}: prompts {batch_start+1}-{batch_end}")

        responses = parallel_call_gpt(
            prompts=batch_prompts,
            model_name=model,
            max_tokens=10,  # Only need a single digit
            temperature=0.0,  # Deterministic scoring
            system_prompt=ANSWER_QUALITY_SYSTEM_PROMPT,
            workers=workers,
            retries=3,
        )
        all_responses.extend(responses)

    # 5. Parse scores
    scores_by_source: Dict[str, List[int]] = defaultdict(list)
    failed_parses = 0
    detailed_results = []

    for (item, source), resp in zip(sample_meta, all_responses):
        q, a, _ = extract_qa(item)
        score = None

        if resp:
            resp_clean = resp.strip()
            # Extract the first digit 1-5
            for ch in resp_clean:
                if ch in "12345":
                    score = int(ch)
                    break

        if score is not None:
            scores_by_source[source].append(score)
            detailed_results.append({
                "question": q[:200],
                "answer_preview": a[:200],
                "source": source,
                "score": score,
                "raw_response": resp,
            })
        else:
            failed_parses += 1
            detailed_results.append({
                "question": q[:200],
                "answer_preview": a[:200],
                "source": source,
                "score": None,
                "raw_response": resp,
            })

    # 6. Analyze results
    logger.info("\n" + "=" * 60)
    logger.info("ANSWER QUALITY RESULTS")
    logger.info("=" * 60)

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "total_samples": len(sampled),
        "total_scored": sum(len(v) for v in scores_by_source.values()),
        "failed_parses": failed_parses,
        "per_source": {},
        "overall": {},
        "score_distribution": {},
        "low_quality_examples": [],
    }

    all_scores = []
    for source in sorted(scores_by_source.keys()):
        scores = scores_by_source[source]
        all_scores.extend(scores)

        avg = sum(scores) / len(scores) if scores else 0
        dist = Counter(scores)
        pct_good = sum(1 for s in scores if s >= 4) / len(scores) * 100 if scores else 0
        pct_bad = sum(1 for s in scores if s <= 2) / len(scores) * 100 if scores else 0

        source_report = {
            "count": len(scores),
            "mean_score": round(avg, 3),
            "median_score": sorted(scores)[len(scores) // 2] if scores else 0,
            "score_distribution": {str(k): v for k, v in sorted(dist.items())},
            "pct_good_4_5": round(pct_good, 1),
            "pct_bad_1_2": round(pct_bad, 1),
        }
        report["per_source"][source] = source_report

        logger.info(f"\n{source}:")
        logger.info(f"  Samples: {len(scores)}")
        logger.info(f"  Mean score: {avg:.2f}")
        logger.info(f"  Good (4-5): {pct_good:.1f}%")
        logger.info(f"  Bad (1-2): {pct_bad:.1f}%")
        logger.info(f"  Distribution: {dict(sorted(dist.items()))}")

    # Overall
    if all_scores:
        overall_avg = sum(all_scores) / len(all_scores)
        overall_dist = Counter(all_scores)
        overall_pct_good = sum(1 for s in all_scores if s >= 4) / len(all_scores) * 100
        overall_pct_bad = sum(1 for s in all_scores if s <= 2) / len(all_scores) * 100

        report["overall"] = {
            "mean_score": round(overall_avg, 3),
            "total_scored": len(all_scores),
            "pct_good_4_5": round(overall_pct_good, 1),
            "pct_bad_1_2": round(overall_pct_bad, 1),
        }
        report["score_distribution"] = {str(k): v for k, v in sorted(overall_dist.items())}

        logger.info(f"\n{'='*60}")
        logger.info(f"OVERALL:")
        logger.info(f"  Mean: {overall_avg:.2f}")
        logger.info(f"  Good (4-5): {overall_pct_good:.1f}%")
        logger.info(f"  Bad (1-2): {overall_pct_bad:.1f}%")
        logger.info(f"  Distribution: {dict(sorted(overall_dist.items()))}")

    # Collect low quality examples
    low_quality = [r for r in detailed_results if r.get("score") is not None and r["score"] <= 2]
    report["low_quality_examples"] = low_quality[:50]  # Save top 50 worst examples

    logger.info(f"\nLow quality examples (score 1-2): {len(low_quality)}")
    for ex in low_quality[:5]:
        logger.info(f"  [{ex['source']}] Score={ex['score']}: {ex['question'][:80]}...")

    # 7. Save report
    report_path = REPORT_DIR / "answer_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"\nReport saved to: {report_path}")

    # Save detailed results
    detailed_path = REPORT_DIR / "answer_quality_detailed.jsonl"
    with open(detailed_path, "w", encoding="utf-8") as f:
        for r in detailed_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Detailed results saved to: {detailed_path}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate answer quality with LLM-as-Judge")
    parser.add_argument("--samples", type=int, default=1000, help="Total samples to evaluate")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model for evaluation")
    parser.add_argument("--workers", type=int, default=30, help="Parallel workers")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for API calls")
    args = parser.parse_args()

    evaluate_answer_quality(
        total_samples=args.samples,
        model=args.model,
        workers=args.workers,
        batch_size=args.batch_size,
    )
