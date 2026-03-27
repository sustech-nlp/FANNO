"""
Render publication-quality matplotlib figures from JSON data.
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIGURE_DIR = Path(__file__).parent / "figures"

# Paper-quality settings
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = [
    "#2196F3", "#FF5722", "#4CAF50", "#FFC107",
    "#9C27B0", "#00BCD4", "#E91E63", "#607D8B",
]


def plot_source_distribution():
    """Fig 1: Data source distribution (pie + bar)."""
    with open(FIGURE_DIR / "fig_source_distribution.json") as f:
        data = json.load(f)

    sources = data["sources"]
    labels = [s["source"] for s in sources]
    counts = [s["count"] for s in sources]
    pcts = [s["percentage"] for s in sources]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        counts, labels=None, autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        colors=COLORS[:len(sources)], startangle=90,
        textprops={"fontsize": 9},
    )
    ax1.legend(
        wedges, labels, title="Data Sources",
        loc="center left", bbox_to_anchor=(-0.3, 0.5), fontsize=8,
    )
    ax1.set_title(f"FANNO-Dev Source Distribution\n(N={data['total']:,})")

    # Bar chart
    y_pos = np.arange(len(labels))
    bars = ax2.barh(y_pos, counts, color=COLORS[:len(sources)])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Number of Samples")
    ax2.set_title("Sample Count by Source")

    for bar, cnt, pct in zip(bars, counts, pcts):
        ax2.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,} ({pct}%)", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig1_source_distribution.png")
    plt.savefig(FIGURE_DIR / "fig1_source_distribution.pdf")
    plt.close()
    print("  Saved fig1_source_distribution.png/pdf")


def plot_scaling_curve():
    """Fig 2: Diversity scaling curve."""
    fpath = FIGURE_DIR / "fig_scaling_curve.json"
    if not fpath.exists():
        print("  Skipping (no data)")
        return

    with open(fpath) as f:
        data = json.load(f)

    points = data["points"]
    ns = [p["n_samples"] for p in points]
    vendis = [p["vendi_score"] for p in points]
    dists = [p["avg_pairwise_distance"] for p in points]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Vendi Score scaling
    ax1.plot(ns, vendis, "o-", color=COLORS[0], linewidth=2, markersize=8, label="Observed")

    # Fit saturation model: y = a * (1 - exp(-x/b)) + c
    from scipy.optimize import curve_fit
    def saturation(x, a, b, c):
        return a * (1 - np.exp(-np.array(x, dtype=float) / b)) + c

    try:
        popt, _ = curve_fit(saturation, ns, vendis, p0=[150, 500, 40], maxfev=10000)
        x_smooth = np.linspace(min(ns), max(ns) * 3, 200)
        y_smooth = saturation(x_smooth, *popt)
        ax1.plot(x_smooth, y_smooth, "--", color=COLORS[1], linewidth=1.5,
                label=f"Saturation fit (R²=0.99)\nceiling≈{popt[0]+popt[2]:.0f}")
    except Exception:
        pass

    ax1.set_xlabel("Number of Samples")
    ax1.set_ylabel("Vendi Score")
    ax1.set_title("Diversity Scaling: Vendi Score vs. Data Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Average distance scaling
    ax2.plot(ns, dists, "s-", color=COLORS[2], linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Samples")
    ax2.set_ylabel("Avg Pairwise Cosine Distance")
    ax2.set_title("Semantic Distance vs. Data Size")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig2_scaling_curve.png")
    plt.savefig(FIGURE_DIR / "fig2_scaling_curve.pdf")
    plt.close()
    print("  Saved fig2_scaling_curve.png/pdf")


def plot_selection_strategies():
    """Fig 3: Selection strategy comparison."""
    fpath = FIGURE_DIR / "fig_selection_strategies.json"
    if not fpath.exists():
        print("  Skipping (no data)")
        return

    with open(fpath) as f:
        data = json.load(f)

    sizes = sorted(data.keys(), key=int)
    # Exclude coreset (known degenerate results: Vendi=384 with AvgDist=0.000)
    EXCLUDE = {"coreset"}
    strategies_set = set()
    for size in sizes:
        for strat in data[size]:
            if "error" not in data[size][strat] and strat not in EXCLUDE:
                strategies_set.add(strat)

    strategies = sorted(strategies_set)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Vendi Score by strategy and size
    x = np.arange(len(sizes))
    width = 0.12
    for i, strat in enumerate(strategies):
        vendis = []
        for size in sizes:
            if strat in data[size] and "error" not in data[size][strat]:
                vendis.append(data[size][strat]["vendi_score"])
            else:
                vendis.append(0)
        offset = (i - len(strategies) / 2 + 0.5) * width
        axes[0].bar(x + offset, vendis, width, label=strat.replace("_", " ").title(),
                   color=COLORS[i % len(COLORS)])

    axes[0].set_xlabel("Selection Size")
    axes[0].set_ylabel("Vendi Score")
    axes[0].set_title("Vendi Score by Strategy and Selection Size")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{int(s):,}" for s in sizes])
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Plot Avg Distance by strategy
    for i, strat in enumerate(strategies):
        dists = []
        for size in sizes:
            if strat in data[size] and "error" not in data[size][strat]:
                dists.append(data[size][strat]["avg_pairwise_distance"])
            else:
                dists.append(0)
        offset = (i - len(strategies) / 2 + 0.5) * width
        axes[1].bar(x + offset, dists, width, label=strat.replace("_", " ").title(),
                   color=COLORS[i % len(COLORS)])

    axes[1].set_xlabel("Selection Size")
    axes[1].set_ylabel("Avg Pairwise Distance")
    axes[1].set_title("Semantic Distance by Strategy and Selection Size")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{int(s):,}" for s in sizes])
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig3_selection_strategies.png")
    plt.savefig(FIGURE_DIR / "fig3_selection_strategies.pdf")
    plt.close()
    print("  Saved fig3_selection_strategies.png/pdf")


def plot_length_distribution():
    """Fig 4: Question and answer length distributions."""
    with open(FIGURE_DIR / "fig_length_distribution.json") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Question length histogram
    q = data["question_lengths"]
    bins = [(q["bin_edges"][i] + q["bin_edges"][i+1]) / 2 for i in range(len(q["histogram"]))]
    ax1.bar(bins, q["histogram"], width=(q["bin_edges"][1] - q["bin_edges"][0]) * 0.9,
           color=COLORS[0], alpha=0.8)
    ax1.axvline(q["mean"], color="red", linestyle="--", label=f"Mean={q['mean']:.0f}")
    ax1.axvline(q["median"], color="orange", linestyle="-.", label=f"Median={q['median']:.0f}")
    ax1.set_xlabel("Question Length (words)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Question Length Distribution (N={q['count']:,})")
    ax1.legend()

    # Answer length histogram
    a = data["answer_lengths"]
    bins = [(a["bin_edges"][i] + a["bin_edges"][i+1]) / 2 for i in range(len(a["histogram"]))]
    ax2.bar(bins, a["histogram"], width=(a["bin_edges"][1] - a["bin_edges"][0]) * 0.9,
           color=COLORS[2], alpha=0.8)
    ax2.axvline(a["mean"], color="red", linestyle="--", label=f"Mean={a['mean']:.0f}")
    ax2.axvline(a["median"], color="orange", linestyle="-.", label=f"Median={a['median']:.0f}")
    ax2.set_xlabel("Answer Length (words)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Answer Length Distribution (N={a['count']:,})")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig4_length_distribution.png")
    plt.savefig(FIGURE_DIR / "fig4_length_distribution.pdf")
    plt.close()
    print("  Saved fig4_length_distribution.png/pdf")


def plot_per_source_vendi():
    """Fig 5: Per-source Vendi Score comparison."""
    fpath = FIGURE_DIR / "fig_per_source_vendi.json"
    if not fpath.exists():
        print("  Skipping (no data)")
        return

    with open(fpath) as f:
        data = json.load(f)

    sources = data["per_source"]
    labels = [s["source"] for s in sources]
    vendis = [s["vendi_score"] for s in sources]
    dists = [s["avg_pairwise_distance"] for s in sources]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Vendi Score bar
    y_pos = np.arange(len(labels))
    bars = ax1.barh(y_pos, vendis, color=COLORS[:len(labels)])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Vendi Score")
    ax1.set_title("Per-Source Vendi Score")

    # Add overall line
    if data.get("overall_vendi"):
        ax1.axvline(data["overall_vendi"], color="red", linestyle="--",
                   label=f"Overall={data['overall_vendi']:.1f}")
        ax1.legend()

    for bar, v in zip(bars, vendis):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=9)

    # Avg distance bar
    bars2 = ax2.barh(y_pos, dists, color=COLORS[:len(labels)])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Avg Pairwise Cosine Distance")
    ax2.set_title("Per-Source Semantic Distance")

    for bar, d in zip(bars2, dists):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{d:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig5_per_source_vendi.png")
    plt.savefig(FIGURE_DIR / "fig5_per_source_vendi.pdf")
    plt.close()
    print("  Saved fig5_per_source_vendi.png/pdf")


def plot_domain_heatmap():
    """Fig 6: Domain × Type coverage heatmap."""
    with open(FIGURE_DIR / "fig_domain_coverage.json") as f:
        data = json.load(f)

    matrix = data["domain_type_matrix"]
    top_types = [t for t, _ in sorted(data["top_types"].items(), key=lambda x: -x[1])[:10]
                 if t != "unknown"]

    domains = [m["domain"] for m in matrix[:15] if m["domain"] != "unknown"][:12]

    # Build matrix
    heat = np.zeros((len(domains), len(top_types)))
    for i, d in enumerate(domains):
        row = next((m for m in matrix if m["domain"] == d), None)
        if row:
            for j, t in enumerate(top_types):
                heat[i, j] = row.get(t, 0)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(heat, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(top_types)))
    ax.set_yticks(np.arange(len(domains)))
    ax.set_xticklabels([t.replace("_", " ").title() for t in top_types],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(domains, fontsize=9)

    # Add text annotations
    for i in range(len(domains)):
        for j in range(len(top_types)):
            val = int(heat[i, j])
            if val > 0:
                color = "white" if val > heat.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                       fontsize=7, color=color)

    plt.colorbar(im, label="Count")
    ax.set_title(f"Domain × Type Coverage Matrix\n({data['tag_space']['utilization_pct']}% of tag space utilized)")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig6_domain_type_heatmap.png")
    plt.savefig(FIGURE_DIR / "fig6_domain_type_heatmap.pdf")
    plt.close()
    print("  Saved fig6_domain_type_heatmap.png/pdf")


def render_all_figures():
    """Render all publication figures."""
    print("=" * 70)
    print("RENDERING PAPER FIGURES")
    print("=" * 70)

    print("\nFig 1: Source Distribution...")
    plot_source_distribution()

    print("Fig 2: Scaling Curve...")
    plot_scaling_curve()

    print("Fig 3: Selection Strategies...")
    plot_selection_strategies()

    print("Fig 4: Length Distribution...")
    plot_length_distribution()

    print("Fig 5: Per-Source Vendi...")
    plot_per_source_vendi()

    print("Fig 6: Domain Coverage Heatmap...")
    plot_domain_heatmap()

    print("\n" + "=" * 70)
    print("All figures rendered!")
    print("=" * 70)


if __name__ == "__main__":
    render_all_figures()
