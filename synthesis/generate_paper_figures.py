"""
Generate data for paper-quality figures and tables.
Produces JSON data files that can be rendered with matplotlib/plotly.
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
FIGURE_DIR = Path(__file__).parent / "figures"


def generate_source_distribution_data(output_dir: Path = None):
    """Generate source distribution pie/bar chart data."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    data = []
    for f in ["cleaned_single_turn.jsonl", "cleaned_multi_turn.jsonl"]:
        fpath = output_dir / f
        if fpath.exists():
            with open(fpath) as fh:
                for line in fh:
                    if line.strip():
                        data.append(json.loads(line))

    sources = Counter(d.get("source", "unknown") for d in data)
    total = len(data)

    # Rename for paper readability
    name_map = {
        "fanno_complex_qa": "Complex QA",
        "fanno_reasoning_qa": "Reasoning QA",
        "fanno_code_qa": "Code QA",
        "fanno_multi_turn": "Multi-Turn Dialog",
        "fanno_math_qa": "Math QA",
        "fanno_seed_qa": "Document-Grounded QA",
        "fanno_creative_writing": "Creative Writing",
        "self_inversion": "Self-Inversion",
    }

    chart_data = []
    for src, cnt in sources.most_common():
        chart_data.append({
            "source": name_map.get(src, src),
            "count": cnt,
            "percentage": round(cnt / total * 100, 1),
        })

    return {"total": total, "sources": chart_data}


def generate_scaling_curve_data(output_dir: Path = None):
    """Generate scaling curve data from Vendi report."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    vendi_path = output_dir / "vendi_diversity_report.json"
    if not vendi_path.exists():
        return None

    with open(vendi_path) as f:
        report = json.load(f)

    scale = report.get("scale_analysis", [])
    if not scale:
        return None

    points = []
    for s in scale:
        points.append({
            "n_samples": s["n_samples"],
            "fraction": s["fraction"],
            "vendi_score": s["vendi_score"],
            "avg_pairwise_distance": s["avg_pairwise_distance"],
        })

    return {
        "total_vendi": report.get("vendi_score"),
        "total_avg_dist": report.get("avg_pairwise_cosine_distance"),
        "points": points,
    }


def generate_selection_strategy_data(output_dir: Path = None):
    """Generate selection strategy comparison data."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    report_path = output_dir / "selection_comparison.json"
    if not report_path.exists():
        return None

    with open(report_path) as f:
        report = json.load(f)

    return report.get("experiments", {})


def generate_length_distribution_data(output_dir: Path = None):
    """Generate length distribution histogram data."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    data = []
    fpath = output_dir / "cleaned_single_turn.jsonl"
    if fpath.exists():
        with open(fpath) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    q_lens, a_lens = [], []
    for item in data:
        q = item.get("question", item.get("instruction", ""))
        a = item.get("answer", item.get("output", item.get("response", item.get("solution", ""))))
        if isinstance(q, str) and q:
            q_lens.append(len(q.split()))
        if isinstance(a, str) and a:
            a_lens.append(len(a.split()))

    # Create histogram bins
    q_hist, q_bins = np.histogram(q_lens, bins=50, range=(0, 500))
    a_hist, a_bins = np.histogram(a_lens, bins=50, range=(0, 2000))

    return {
        "question_lengths": {
            "histogram": q_hist.tolist(),
            "bin_edges": q_bins.tolist(),
            "mean": float(np.mean(q_lens)),
            "median": float(np.median(q_lens)),
            "std": float(np.std(q_lens)),
            "count": len(q_lens),
        },
        "answer_lengths": {
            "histogram": a_hist.tolist(),
            "bin_edges": a_bins.tolist(),
            "mean": float(np.mean(a_lens)),
            "median": float(np.median(a_lens)),
            "std": float(np.std(a_lens)),
            "count": len(a_lens),
        },
    }


def generate_domain_coverage_data(output_dir: Path = None):
    """Generate domain coverage treemap/sunburst data."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    data = []
    for f in ["cleaned_single_turn.jsonl", "cleaned_multi_turn.jsonl"]:
        fpath = output_dir / f
        if fpath.exists():
            with open(fpath) as fh:
                for line in fh:
                    if line.strip():
                        data.append(json.loads(line))

    # Domain x Type matrix
    domain_type = defaultdict(Counter)
    for item in data:
        domain = item.get("domain", "unknown")
        qtype = item.get("type", "unknown")
        domain_type[domain][qtype] += 1

    # Top domains
    domain_counts = Counter(d.get("domain", "unknown") for d in data)
    top_domains = [d for d, _ in domain_counts.most_common(30) if d != "unknown"]

    # Top types
    type_counts = Counter(d.get("type", "unknown") for d in data)
    top_types = [t for t, _ in type_counts.most_common(15) if t != "unknown"]

    # Build matrix
    matrix = []
    for domain in top_domains:
        row = {"domain": domain, "total": domain_counts[domain]}
        for qtype in top_types:
            row[qtype] = domain_type[domain].get(qtype, 0)
        matrix.append(row)

    return {
        "n_domains": len(domain_counts),
        "n_types": len(type_counts),
        "top_domains": dict(domain_counts.most_common(30)),
        "top_types": dict(type_counts.most_common(20)),
        "domain_type_matrix": matrix,
        "tag_space": {
            "total_possible": len(domain_counts) * len(type_counts),
            "observed": sum(1 for d in domain_type for t in domain_type[d]),
            "utilization_pct": round(
                sum(1 for d in domain_type for t in domain_type[d])
                / max(1, len(domain_counts) * len(type_counts)) * 100, 1
            ),
        },
    }


def generate_per_source_vendi_data(output_dir: Path = None):
    """Generate per-source Vendi Score comparison data."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    vendi_path = output_dir / "vendi_diversity_report.json"
    if not vendi_path.exists():
        return None

    with open(vendi_path) as f:
        report = json.load(f)

    per_source = report.get("per_source", {})
    name_map = {
        "fanno_complex_qa": "Complex QA",
        "fanno_reasoning_qa": "Reasoning QA",
        "fanno_code_qa": "Code QA",
        "fanno_multi_turn": "Multi-Turn",
        "fanno_math_qa": "Math QA",
        "fanno_seed_qa": "Document QA",
        "fanno_creative_writing": "Creative Writing",
        "self_inversion": "Self-Inversion",
    }

    bars = []
    for src, metrics in sorted(per_source.items(), key=lambda x: -x[1].get("vendi_score", 0)):
        bars.append({
            "source": name_map.get(src, src),
            "raw_name": src,
            "count": metrics.get("count", 0),
            "vendi_score": metrics.get("vendi_score", 0),
            "avg_pairwise_distance": metrics.get("avg_pairwise_distance", 0),
            "diversity_efficiency": metrics.get("vendi_score", 0) / max(1, metrics.get("embedded", 1)) * 1000,
        })

    return {"per_source": bars, "overall_vendi": report.get("vendi_score")}


def generate_all_figure_data():
    """Generate all figure data and save to figures directory."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING PAPER FIGURE DATA")
    print("=" * 70)

    # 1. Source distribution
    print("\n📊 Generating source distribution data...")
    src_data = generate_source_distribution_data()
    with open(FIGURE_DIR / "fig_source_distribution.json", "w") as f:
        json.dump(src_data, f, indent=2)
    print(f"  Total: {src_data['total']:,} samples across {len(src_data['sources'])} sources")

    # 2. Scaling curve
    print("\n📈 Generating scaling curve data...")
    scale_data = generate_scaling_curve_data()
    if scale_data:
        with open(FIGURE_DIR / "fig_scaling_curve.json", "w") as f:
            json.dump(scale_data, f, indent=2)
        print(f"  {len(scale_data['points'])} data points, overall Vendi={scale_data['total_vendi']:.2f}")
    else:
        print("  No Vendi report found, skipping")

    # 3. Selection strategy comparison
    print("\n🏆 Generating selection strategy data...")
    strat_data = generate_selection_strategy_data()
    if strat_data:
        with open(FIGURE_DIR / "fig_selection_strategies.json", "w") as f:
            json.dump(strat_data, f, indent=2)
        print(f"  {len(strat_data)} selection sizes compared")
    else:
        print("  No selection comparison found, skipping")

    # 4. Length distribution
    print("\n📏 Generating length distribution data...")
    len_data = generate_length_distribution_data()
    with open(FIGURE_DIR / "fig_length_distribution.json", "w") as f:
        json.dump(len_data, f, indent=2)
    print(f"  Q mean={len_data['question_lengths']['mean']:.1f}, A mean={len_data['answer_lengths']['mean']:.1f}")

    # 5. Domain coverage
    print("\n🌐 Generating domain coverage data...")
    dom_data = generate_domain_coverage_data()
    with open(FIGURE_DIR / "fig_domain_coverage.json", "w") as f:
        json.dump(dom_data, f, indent=2)
    print(f"  {dom_data['n_domains']} domains × {dom_data['n_types']} types")
    print(f"  Tag space utilization: {dom_data['tag_space']['utilization_pct']}%")

    # 6. Per-source Vendi
    print("\n🎯 Generating per-source Vendi data...")
    vendi_data = generate_per_source_vendi_data()
    if vendi_data:
        with open(FIGURE_DIR / "fig_per_source_vendi.json", "w") as f:
            json.dump(vendi_data, f, indent=2)
        print(f"  {len(vendi_data['per_source'])} sources, overall Vendi={vendi_data['overall_vendi']:.2f}")
    else:
        print("  No Vendi report found, skipping")

    print("\n" + "=" * 70)
    print(f"All figure data saved to {FIGURE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_figure_data()
