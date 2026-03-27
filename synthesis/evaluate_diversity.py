"""
Diversity Evaluation for FANNO-Dev synthesized data.
Evaluates data quality and diversity using multiple metrics.
"""
from __future__ import annotations

import json
import hashlib
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

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


# =============================================================================
# Basic Statistics
# =============================================================================

def compute_basic_stats(data: List[Dict]) -> Dict:
    """Compute basic statistics for a dataset."""
    stats = {
        "total_samples": len(data),
        "sources": Counter(),
        "domains": Counter(),
        "difficulties": Counter(),
        "types": Counter(),
    }

    text_lengths = []
    question_lengths = []
    answer_lengths = []

    for item in data:
        # Source distribution
        stats["sources"][item.get("source", "unknown")] += 1

        # Domain distribution
        if "domain" in item:
            stats["domains"][item["domain"]] += 1

        # Difficulty distribution
        if "difficulty" in item:
            stats["difficulties"][item["difficulty"]] += 1

        # Type distribution
        if "type" in item:
            stats["types"][item["type"]] += 1

        # Text lengths
        q = item.get("question", item.get("instruction", ""))
        a = item.get("answer", item.get("output", item.get("response", "")))
        if q and isinstance(q, str):
            question_lengths.append(len(q.split()))
        if a and isinstance(a, str):
            answer_lengths.append(len(a.split()))

    if question_lengths:
        stats["question_length"] = {
            "mean": np.mean(question_lengths),
            "std": np.std(question_lengths),
            "min": int(np.min(question_lengths)),
            "max": int(np.max(question_lengths)),
            "median": float(np.median(question_lengths)),
        }

    if answer_lengths:
        stats["answer_length"] = {
            "mean": np.mean(answer_lengths),
            "std": np.std(answer_lengths),
            "min": int(np.min(answer_lengths)),
            "max": int(np.max(answer_lengths)),
            "median": float(np.median(answer_lengths)),
        }

    # Convert Counters to dicts for JSON serialization
    stats["sources"] = dict(stats["sources"].most_common())
    stats["domains"] = dict(stats["domains"].most_common(30))
    stats["difficulties"] = dict(stats["difficulties"].most_common())
    stats["types"] = dict(stats["types"].most_common(30))

    return stats


# =============================================================================
# N-gram Diversity
# =============================================================================

def compute_ngram_diversity(texts: List[str], n: int = 3) -> Dict:
    """Compute n-gram diversity metrics."""
    all_ngrams = []
    unique_ngrams = set()

    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)
        unique_ngrams.update(ngrams)

    total = len(all_ngrams)
    unique = len(unique_ngrams)

    return {
        f"{n}gram_total": total,
        f"{n}gram_unique": unique,
        f"{n}gram_diversity_ratio": unique / total if total > 0 else 0,
    }


# =============================================================================
# Lexical Diversity (Type-Token Ratio variants)
# =============================================================================

def compute_lexical_diversity(texts: List[str]) -> Dict:
    """Compute various lexical diversity metrics."""
    all_words = []
    for text in texts:
        all_words.extend(text.lower().split())

    total_tokens = len(all_words)
    unique_tokens = len(set(all_words))

    # TTR (Type-Token Ratio)
    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0

    # Root TTR (Guiraud)
    root_ttr = unique_tokens / np.sqrt(total_tokens) if total_tokens > 0 else 0

    # Log TTR (Herdan)
    log_ttr = np.log(unique_tokens) / np.log(total_tokens) if total_tokens > 1 else 0

    # MTLD-style approximation (simplified)
    # Moving-average TTR with window
    window_size = 100
    mtld_scores = []
    for i in range(0, len(all_words) - window_size, window_size // 2):
        window = all_words[i:i+window_size]
        mtld_scores.append(len(set(window)) / len(window))

    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "ttr": ttr,
        "root_ttr": root_ttr,
        "log_ttr": log_ttr,
        "mtld_approximation": float(np.mean(mtld_scores)) if mtld_scores else 0,
    }


# =============================================================================
# Semantic Diversity (Hash-based approximation without embeddings)
# =============================================================================

def compute_hash_diversity(texts: List[str], num_hashes: int = 128) -> Dict:
    """Approximate semantic diversity using MinHash-style fingerprinting."""
    def minhash(text: str, num_hashes: int = 128) -> List[int]:
        words = set(text.lower().split())
        hashes = []
        for i in range(num_hashes):
            min_hash = float('inf')
            for word in words:
                h = int(hashlib.md5(f"{word}_{i}".encode()).hexdigest(), 16)
                min_hash = min(min_hash, h)
            hashes.append(min_hash)
        return hashes

    if len(texts) < 2:
        return {"estimated_jaccard_diversity": 0}

    # Sample pairs for efficiency
    sample_size = min(1000, len(texts))
    sampled = np.random.choice(len(texts), size=sample_size, replace=False)

    fingerprints = [minhash(texts[i], num_hashes) for i in sampled]

    # Compute pairwise Jaccard distances
    distances = []
    for i in range(min(500, len(fingerprints))):
        for j in range(i + 1, min(500, len(fingerprints))):
            agreement = sum(a == b for a, b in zip(fingerprints[i], fingerprints[j]))
            jaccard = agreement / num_hashes
            distances.append(1 - jaccard)  # Jaccard distance

    return {
        "mean_jaccard_distance": float(np.mean(distances)) if distances else 0,
        "std_jaccard_distance": float(np.std(distances)) if distances else 0,
        "min_jaccard_distance": float(np.min(distances)) if distances else 0,
        "estimated_diversity_score": float(np.mean(distances)) if distances else 0,
    }


# =============================================================================
# Topic/Domain Coverage
# =============================================================================

def compute_coverage_metrics(data: List[Dict]) -> Dict:
    """Compute topic and domain coverage metrics."""
    domains = [item.get("domain", "unknown") for item in data]
    types = [item.get("type", "unknown") for item in data]
    difficulties = [item.get("difficulty", "unknown") for item in data]

    domain_counts = Counter(domains)
    type_counts = Counter(types)
    diff_counts = Counter(difficulties)

    def entropy(counter: Counter) -> float:
        total = sum(counter.values())
        if total == 0:
            return 0
        probs = [c / total for c in counter.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def uniformity(counter: Counter) -> float:
        """How close to uniform distribution (1.0 = perfectly uniform)."""
        if not counter:
            return 0
        max_entropy = np.log2(len(counter))
        if max_entropy == 0:
            return 1.0
        return entropy(counter) / max_entropy

    return {
        "num_unique_domains": len(domain_counts),
        "domain_entropy": entropy(domain_counts),
        "domain_uniformity": uniformity(domain_counts),
        "num_unique_types": len(type_counts),
        "type_entropy": entropy(type_counts),
        "type_uniformity": uniformity(type_counts),
        "num_unique_difficulties": len(diff_counts),
        "difficulty_entropy": entropy(diff_counts),
        "difficulty_uniformity": uniformity(diff_counts),
    }


# =============================================================================
# Deduplication Analysis
# =============================================================================

def compute_dedup_stats(texts: List[str]) -> Dict:
    """Analyze potential duplicates."""
    # Exact duplicates
    unique_texts = set(texts)
    exact_dup_rate = 1 - len(unique_texts) / len(texts) if texts else 0

    # Near-duplicate detection (first 50 chars)
    prefixes = [t[:50].lower() for t in texts]
    prefix_counts = Counter(prefixes)
    near_dup_pairs = sum(c * (c - 1) // 2 for c in prefix_counts.values() if c > 1)

    # Hash-based near-duplicate
    hash_texts = [hashlib.md5(t.lower().strip().encode()).hexdigest() for t in texts]
    hash_unique = len(set(hash_texts))

    return {
        "total": len(texts),
        "exact_unique": len(unique_texts),
        "exact_duplicate_rate": exact_dup_rate,
        "prefix_near_duplicate_pairs": near_dup_pairs,
        "hash_unique": hash_unique,
        "hash_duplicate_rate": 1 - hash_unique / len(texts) if texts else 0,
    }


# =============================================================================
# Master Evaluation
# =============================================================================

def evaluate_all(output_dir: Path = None) -> Dict:
    """Run comprehensive diversity evaluation on all synthesized data."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    logger.info(f"Evaluating all data in {output_dir}")

    all_data = []
    per_file_stats = {}

    for jsonl_file in sorted(output_dir.glob("*.jsonl")):
        data = load_jsonl(jsonl_file)
        if not data:
            continue

        logger.info(f"Evaluating {jsonl_file.name}: {len(data)} samples")

        # Extract texts
        questions = [
            item.get("question", item.get("instruction", ""))
            for item in data if isinstance(item.get("question", item.get("instruction", "")), str) and item.get("question", item.get("instruction", ""))
        ]
        answers = [
            item.get("answer", item.get("output", item.get("response", "")))
            for item in data if isinstance(item.get("answer", item.get("output", item.get("response", ""))), str) and item.get("answer", item.get("output", item.get("response", "")))
        ]

        file_stats = {
            "count": len(data),
            "basic": compute_basic_stats(data),
        }

        if questions:
            file_stats["question_ngram_2"] = compute_ngram_diversity(questions, n=2)
            file_stats["question_ngram_3"] = compute_ngram_diversity(questions, n=3)
            file_stats["question_lexical"] = compute_lexical_diversity(questions)
            file_stats["question_dedup"] = compute_dedup_stats(questions)
            if len(questions) >= 10:
                file_stats["question_hash_diversity"] = compute_hash_diversity(questions)

        if answers:
            file_stats["answer_ngram_2"] = compute_ngram_diversity(answers, n=2)
            file_stats["answer_ngram_3"] = compute_ngram_diversity(answers, n=3)
            file_stats["answer_lexical"] = compute_lexical_diversity(answers)

        file_stats["coverage"] = compute_coverage_metrics(data)

        per_file_stats[jsonl_file.name] = file_stats
        all_data.extend(data)

    # Global stats
    global_stats = {}
    if all_data:
        all_questions = [
            item.get("question", item.get("instruction", ""))
            for item in all_data if item.get("question", item.get("instruction", ""))
        ]
        all_answers = [
            item.get("answer", item.get("output", item.get("response", "")))
            for item in all_data if item.get("answer", item.get("output", item.get("response", "")))
        ]

        global_stats = {
            "total_samples": len(all_data),
            "basic": compute_basic_stats(all_data),
            "coverage": compute_coverage_metrics(all_data),
        }

        if all_questions:
            global_stats["question_ngram_3"] = compute_ngram_diversity(all_questions, n=3)
            global_stats["question_lexical"] = compute_lexical_diversity(all_questions)
            global_stats["question_dedup"] = compute_dedup_stats(all_questions)
            if len(all_questions) >= 10:
                global_stats["question_hash_diversity"] = compute_hash_diversity(all_questions)

    result = {
        "timestamp": datetime.now().isoformat(),
        "per_file": per_file_stats,
        "global": global_stats,
    }

    # Save report
    report_path = output_dir / "diversity_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Diversity report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("FANNO-Dev Diversity Evaluation Report")
    print("=" * 70)
    print(f"Total samples: {len(all_data)}")
    print(f"\nPer-file breakdown:")
    for fname, stats in per_file_stats.items():
        print(f"  {fname}: {stats['count']} samples")

    if "question_ngram_3" in global_stats:
        d = global_stats["question_ngram_3"]
        print(f"\nGlobal Question 3-gram Diversity: {d.get('3gram_diversity_ratio', 0):.4f}")

    if "question_lexical" in global_stats:
        d = global_stats["question_lexical"]
        print(f"Global Question Lexical Diversity (TTR): {d.get('ttr', 0):.4f}")
        print(f"Global Question Root TTR: {d.get('root_ttr', 0):.4f}")

    if "question_hash_diversity" in global_stats:
        d = global_stats["question_hash_diversity"]
        print(f"Global Question Hash Diversity: {d.get('estimated_diversity_score', 0):.4f}")

    if "question_dedup" in global_stats:
        d = global_stats["question_dedup"]
        print(f"Global Question Exact Duplicate Rate: {d.get('exact_duplicate_rate', 0):.4f}")

    if "coverage" in global_stats:
        d = global_stats["coverage"]
        print(f"\nDomain Coverage: {d.get('num_unique_domains', 0)} domains")
        print(f"Domain Uniformity: {d.get('domain_uniformity', 0):.4f}")
        print(f"Type Coverage: {d.get('num_unique_types', 0)} types")
        print(f"Type Uniformity: {d.get('type_uniformity', 0):.4f}")

    print("=" * 70)
    return result


if __name__ == "__main__":
    evaluate_all()
