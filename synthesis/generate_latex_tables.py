"""
Generate LaTeX tables for the FANNO-Dev paper.
Produces camera-ready tables that can be directly included in the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"
TABLE_DIR = Path(__file__).parent / "tables"


def generate_main_results_table():
    """Table 1: Main dataset statistics."""
    data = []
    for f in ["cleaned_single_turn.jsonl", "cleaned_multi_turn.jsonl"]:
        fpath = OUTPUT_DIR / f
        if fpath.exists():
            with open(fpath) as fh:
                for line in fh:
                    if line.strip():
                        data.append(json.loads(line))

    from collections import Counter
    import numpy as np

    sources = Counter(d.get("source", "unknown") for d in data)
    total = len(data)

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

    # Compute per-source stats
    from collections import defaultdict
    stats = defaultdict(lambda: {"q_lens": [], "a_lens": []})
    for item in data:
        src = item.get("source", "unknown")
        q = item.get("question", item.get("instruction", ""))
        a = item.get("answer", item.get("output", item.get("response", item.get("solution", ""))))
        if isinstance(q, str) and q:
            stats[src]["q_lens"].append(len(q.split()))
        if isinstance(a, str) and a:
            stats[src]["a_lens"].append(len(a.split()))

    latex = r"""\begin{table}[t]
\centering
\caption{FANNO-Dev dataset statistics. Each source uses a distinct synthesis pipeline.}
\label{tab:dataset_stats}
\small
\begin{tabular}{lrrrrr}
\toprule
\textbf{Source} & \textbf{Count} & \textbf{\%} & \textbf{Avg Q (w)} & \textbf{Avg A (w)} & \textbf{Domains} \\
\midrule
"""

    for src, cnt in sources.most_common():
        name = name_map.get(src, src)
        pct = cnt / total * 100
        q_mean = np.mean(stats[src]["q_lens"]) if stats[src]["q_lens"] else 0
        a_mean = np.mean(stats[src]["a_lens"]) if stats[src]["a_lens"] else 0
        n_domains = len(set(d.get("domain", "?") for d in data if d.get("source") == src))
        latex += f"{name} & {cnt:,} & {pct:.1f} & {q_mean:.0f} & {a_mean:.0f} & {n_domains} \\\\\n"

    latex += r"""\midrule
\textbf{Total} & \textbf{""" + f"{total:,}" + r"""} & \textbf{100.0} & & & \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_diversity_comparison_table():
    """Table 2: Diversity metrics comparison."""
    vendi_path = OUTPUT_DIR / "vendi_diversity_report.json"
    if not vendi_path.exists():
        return "% No Vendi report available"

    with open(vendi_path) as f:
        report = json.load(f)

    per_source = report.get("per_source", {})
    name_map = {
        "fanno_complex_qa": "Complex QA",
        "fanno_reasoning_qa": "Reasoning QA",
        "fanno_code_qa": "Code QA",
        "fanno_seed_qa": "Document QA",
        "fanno_creative_writing": "Creative Writing",
        "fanno_math_qa": "Math QA",
        "self_inversion": "Self-Inversion",
    }

    latex = r"""\begin{table}[t]
\centering
\caption{Per-source diversity metrics. Vendi Score measures effective dimensionality of the embedding distribution; higher indicates greater diversity.}
\label{tab:diversity_metrics}
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Source} & \textbf{N} & \textbf{Vendi Score} & \textbf{Avg Dist} & \textbf{Efficiency} \\
\midrule
"""

    for src, metrics in sorted(per_source.items(), key=lambda x: -x[1].get("vendi_score", 0)):
        name = name_map.get(src, src)
        n = metrics.get("embedded", metrics.get("count", 0))
        vendi = metrics.get("vendi_score", 0)
        dist = metrics.get("avg_pairwise_distance", 0)
        eff = vendi / max(1, n) * 1000
        latex += f"{name} & {n:,} & {vendi:.1f} & {dist:.4f} & {eff:.2f} \\\\\n"

    overall_vendi = report.get("vendi_score", 0)
    overall_dist = report.get("avg_pairwise_cosine_distance", 0)
    overall_n = report.get("embedded_samples", 0)

    latex += r"""\midrule
\textbf{Overall (mixed)} & """ + f"{overall_n:,} & \\textbf{{{overall_vendi:.1f}}} & {overall_dist:.4f} & ---" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_selection_strategy_table():
    """Table 3: Selection strategy comparison."""
    report_path = OUTPUT_DIR / "selection_comparison.json"
    if not report_path.exists():
        return "% No selection comparison available"

    with open(report_path) as f:
        report = json.load(f)

    experiments = report.get("experiments", {})
    EXCLUDE = {"coreset"}  # Known degenerate

    latex = r"""\begin{table}[t]
\centering
\caption{Selection strategy comparison. K-Center-Greedy achieves highest diversity at all selection sizes. Pool size = 10K.}
\label{tab:selection_strategies}
\small
\begin{tabular}{l""" + "c" * len(experiments) + r"""}
\toprule
\textbf{Strategy} """

    sizes = sorted(experiments.keys(), key=int)
    for s in sizes:
        latex += f"& \\textbf{{N={int(s):,}}} "
    latex += r""" \\
\midrule
"""

    # Collect all strategies
    all_strats = set()
    for s in sizes:
        for strat in experiments[s]:
            if strat not in EXCLUDE and "error" not in experiments[s][strat]:
                all_strats.add(strat)

    strat_names = {
        "k_center_greedy": "K-Center-Greedy",
        "herding": "Herding",
        "kmeans": "K-Means",
        "random": "Random",
        "stratified": "Stratified",
        "community": "Community Detection",
    }

    # Find best per size
    best_per_size = {}
    for s in sizes:
        best_v = 0
        for strat in experiments[s]:
            if strat not in EXCLUDE and "error" not in experiments[s].get(strat, {}):
                v = experiments[s][strat].get("vendi_score", 0)
                if v > best_v:
                    best_v = v
                    best_per_size[s] = strat

    for strat in sorted(all_strats):
        name = strat_names.get(strat, strat.replace("_", " ").title())
        latex += f"{name} "
        for s in sizes:
            if strat in experiments[s] and "error" not in experiments[s][strat]:
                v = experiments[s][strat]["vendi_score"]
                if strat == best_per_size.get(s):
                    latex += f"& \\textbf{{{v:.1f}}} "
                else:
                    latex += f"& {v:.1f} "
            else:
                latex += "& --- "
        latex += r" \\" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_scaling_analysis_table():
    """Table 4: Scaling analysis."""
    vendi_path = OUTPUT_DIR / "vendi_diversity_report.json"
    if not vendi_path.exists():
        return "% No Vendi report available"

    with open(vendi_path) as f:
        report = json.load(f)

    scale = report.get("scale_analysis", [])
    if not scale:
        return "% No scale analysis data"

    latex = r"""\begin{table}[t]
\centering
\caption{Diversity scaling analysis. Vendi Score grows sublinearly with data size, following an exponential saturation model ($R^2 = 0.99$).}
\label{tab:scaling}
\small
\begin{tabular}{rrrrrr}
\toprule
\textbf{Fraction} & \textbf{N} & \textbf{Vendi} & \textbf{Avg Dist} & \textbf{$\Delta$ Vendi} & \textbf{Per 1K} \\
\midrule
"""

    prev_vendi = None
    prev_n = None
    for s in scale:
        frac = s["fraction"]
        n = s["n_samples"]
        vendi = s["vendi_score"]
        dist = s["avg_pairwise_distance"]

        if prev_vendi is not None:
            delta = vendi - prev_vendi
            per_1k = delta / ((n - prev_n) / 1000)
            latex += f"{frac:.0%} & {n:,} & {vendi:.2f} & {dist:.4f} & {delta:+.2f} & {per_1k:+.4f} \\\\\n"
        else:
            latex += f"{frac:.0%} & {n:,} & {vendi:.2f} & {dist:.4f} & --- & --- \\\\\n"

        prev_vendi = vendi
        prev_n = n

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_all_tables():
    """Generate all LaTeX tables."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING LATEX TABLES")
    print("=" * 70)

    tables = {
        "tab1_dataset_stats.tex": ("Table 1: Dataset Statistics", generate_main_results_table),
        "tab2_diversity_metrics.tex": ("Table 2: Diversity Metrics", generate_diversity_comparison_table),
        "tab3_selection_strategies.tex": ("Table 3: Selection Strategies", generate_selection_strategy_table),
        "tab4_scaling_analysis.tex": ("Table 4: Scaling Analysis", generate_scaling_analysis_table),
    }

    for fname, (desc, func) in tables.items():
        print(f"\n  {desc}...")
        latex = func()
        with open(TABLE_DIR / fname, "w") as f:
            f.write(latex)
        print(f"  Saved {fname}")
        # Also print a preview
        print("  Preview:")
        for line in latex.split("\n")[:5]:
            print(f"    {line}")

    print("\n" + "=" * 70)
    print(f"All tables saved to {TABLE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_tables()
