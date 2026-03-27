"""
Comprehensive quality report for FANNO-Dev synthesized data.
Analyzes data quality, diversity, and distribution at the final stage.
"""
from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_all_cleaned(output_dir: Path = None) -> tuple:
    """Load all cleaned data."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    single, multi = [], []
    for fname, target in [("cleaned_single_turn.jsonl", single), ("cleaned_multi_turn.jsonl", multi)]:
        fpath = output_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        target.append(json.loads(line))
    return single, multi


def analyze_length_distribution(data: List[Dict]) -> Dict:
    """Analyze question and answer length distributions."""
    q_lens, a_lens = [], []
    for item in data:
        q = item.get("question", item.get("instruction", ""))
        a = item.get("answer", item.get("output", item.get("response", item.get("solution", ""))))
        if isinstance(q, str) and q:
            q_lens.append(len(q.split()))
        if isinstance(a, str) and a:
            a_lens.append(len(a.split()))

    return {
        "question": {
            "count": len(q_lens),
            "mean": float(np.mean(q_lens)) if q_lens else 0,
            "std": float(np.std(q_lens)) if q_lens else 0,
            "min": int(np.min(q_lens)) if q_lens else 0,
            "p25": float(np.percentile(q_lens, 25)) if q_lens else 0,
            "median": float(np.median(q_lens)) if q_lens else 0,
            "p75": float(np.percentile(q_lens, 75)) if q_lens else 0,
            "max": int(np.max(q_lens)) if q_lens else 0,
        },
        "answer": {
            "count": len(a_lens),
            "mean": float(np.mean(a_lens)) if a_lens else 0,
            "std": float(np.std(a_lens)) if a_lens else 0,
            "min": int(np.min(a_lens)) if a_lens else 0,
            "p25": float(np.percentile(a_lens, 25)) if a_lens else 0,
            "median": float(np.median(a_lens)) if a_lens else 0,
            "p75": float(np.percentile(a_lens, 75)) if a_lens else 0,
            "max": int(np.max(a_lens)) if a_lens else 0,
        },
    }


def analyze_source_quality(data: List[Dict]) -> Dict:
    """Analyze quality indicators per source."""
    source_stats = defaultdict(lambda: {
        "count": 0,
        "q_lens": [],
        "a_lens": [],
        "domains": Counter(),
        "types": Counter(),
        "difficulties": Counter(),
    })

    for item in data:
        src = item.get("source", "unknown")
        stats = source_stats[src]
        stats["count"] += 1

        q = item.get("question", item.get("instruction", ""))
        a = item.get("answer", item.get("output", item.get("response", item.get("solution", ""))))
        if isinstance(q, str) and q:
            stats["q_lens"].append(len(q.split()))
        if isinstance(a, str) and a:
            stats["a_lens"].append(len(a.split()))

        if "domain" in item:
            stats["domains"][item["domain"]] += 1
        if "type" in item:
            stats["types"][item["type"]] += 1
        if "difficulty" in item:
            stats["difficulties"][item["difficulty"]] += 1

    result = {}
    for src, stats in sorted(source_stats.items(), key=lambda x: -x[1]["count"]):
        result[src] = {
            "count": stats["count"],
            "avg_q_len": float(np.mean(stats["q_lens"])) if stats["q_lens"] else 0,
            "avg_a_len": float(np.mean(stats["a_lens"])) if stats["a_lens"] else 0,
            "n_domains": len(stats["domains"]),
            "n_types": len(stats["types"]),
            "top_types": dict(stats["types"].most_common(5)),
            "top_difficulties": dict(stats["difficulties"].most_common()),
        }
    return result


def generate_report(output_dir: Path = None):
    """Generate comprehensive quality report."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    single, multi = load_all_cleaned(output_dir)
    all_data = single + multi

    print("=" * 80)
    print("FANNO-Dev COMPREHENSIVE QUALITY REPORT")
    print(f"Generated: {datetime.now().isoformat()}")
    print("=" * 80)

    print(f"\n📊 DATASET OVERVIEW")
    print(f"  Total cleaned samples: {len(all_data):,}")
    print(f"  Single-turn: {len(single):,}")
    print(f"  Multi-turn: {len(multi):,}")

    # Source distribution
    print(f"\n📁 SOURCE DISTRIBUTION")
    source_counts = Counter(item.get("source", "unknown") for item in all_data)
    for src, cnt in source_counts.most_common():
        pct = cnt / len(all_data) * 100
        bar = "█" * int(pct / 2)
        print(f"  {src:<30} {cnt:>8,} ({pct:5.1f}%) {bar}")

    # Length analysis
    print(f"\n📏 LENGTH DISTRIBUTION (words)")
    lengths = analyze_length_distribution(single)
    print(f"  Questions: mean={lengths['question']['mean']:.1f}, "
          f"median={lengths['question']['median']:.0f}, "
          f"p25-p75=[{lengths['question']['p25']:.0f}-{lengths['question']['p75']:.0f}], "
          f"max={lengths['question']['max']}")
    print(f"  Answers:   mean={lengths['answer']['mean']:.1f}, "
          f"median={lengths['answer']['median']:.0f}, "
          f"p25-p75=[{lengths['answer']['p25']:.0f}-{lengths['answer']['p75']:.0f}], "
          f"max={lengths['answer']['max']}")

    # Per-source quality
    print(f"\n📊 PER-SOURCE QUALITY")
    source_quality = analyze_source_quality(single)
    print(f"  {'Source':<30} {'Count':>8} {'Avg Q':>8} {'Avg A':>8} {'Domains':>8} {'Types':>6}")
    print(f"  {'-'*68}")
    for src, sq in source_quality.items():
        print(f"  {src:<30} {sq['count']:>8,} {sq['avg_q_len']:>8.1f} "
              f"{sq['avg_a_len']:>8.1f} {sq['n_domains']:>8} {sq['n_types']:>6}")

    # Type distribution
    print(f"\n🏷️ TYPE DISTRIBUTION (top 20)")
    type_counts = Counter(item.get("type", "unknown") for item in all_data)
    for tp, cnt in type_counts.most_common(20):
        pct = cnt / len(all_data) * 100
        print(f"  {tp:<35} {cnt:>8,} ({pct:5.1f}%)")

    # Difficulty distribution
    print(f"\n⚡ DIFFICULTY DISTRIBUTION")
    diff_counts = Counter(item.get("difficulty", "unknown") for item in all_data)
    for diff, cnt in diff_counts.most_common():
        pct = cnt / len(all_data) * 100
        print(f"  {diff:<25} {cnt:>8,} ({pct:5.1f}%)")

    # Domain coverage
    domain_counts = Counter(item.get("domain", "unknown") for item in all_data)
    print(f"\n🌐 DOMAIN COVERAGE")
    print(f"  Unique domains: {len(domain_counts):,}")
    print(f"  Top 15 domains:")
    for dom, cnt in domain_counts.most_common(15):
        pct = cnt / len(all_data) * 100
        print(f"    {dom:<40} {cnt:>6,} ({pct:4.1f}%)")

    # Multi-turn analysis
    if multi:
        print(f"\n💬 MULTI-TURN ANALYSIS")
        turn_counts = []
        for item in multi:
            conv = item.get("conversation", [])
            if isinstance(conv, list):
                turn_counts.append(len(conv))
        if turn_counts:
            print(f"  Conversations: {len(multi):,}")
            print(f"  Avg turns: {np.mean(turn_counts):.1f}")
            print(f"  Min turns: {np.min(turn_counts)}")
            print(f"  Max turns: {np.max(turn_counts)}")
            print(f"  Median turns: {np.median(turn_counts):.0f}")

        # Multi-turn patterns and scenarios
        patterns = Counter(item.get("pattern", "unknown") for item in multi)
        scenarios = Counter(item.get("scenario", "unknown") for item in multi)
        print(f"  Patterns: {dict(patterns.most_common(8))}")
        print(f"  Scenarios: {dict(scenarios.most_common(8))}")

    # Load diversity metrics if available
    vendi_path = output_dir / "vendi_diversity_report.json"
    if vendi_path.exists():
        with open(vendi_path) as f:
            vendi = json.load(f)
        print(f"\n🎯 DIVERSITY METRICS (from Vendi report)")
        print(f"  Vendi Score: {vendi.get('vendi_score', 'N/A')}")
        print(f"  Avg Pairwise Cosine Distance: {vendi.get('avg_pairwise_cosine_distance', 'N/A')}")
        if "scale_analysis" in vendi:
            print(f"  Scale Analysis:")
            for s in vendi["scale_analysis"]:
                print(f"    {s['fraction']:.0%} ({s['n_samples']:,}): Vendi={s['vendi_score']:.2f}")

    print("\n" + "=" * 80)

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total": len(all_data),
        "single_turn": len(single),
        "multi_turn": len(multi),
        "source_distribution": dict(source_counts.most_common()),
        "lengths": lengths,
        "source_quality": source_quality,
        "type_distribution": dict(type_counts.most_common()),
        "difficulty_distribution": dict(diff_counts.most_common()),
        "domain_coverage": len(domain_counts),
    }
    report_path = output_dir / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    generate_report()
