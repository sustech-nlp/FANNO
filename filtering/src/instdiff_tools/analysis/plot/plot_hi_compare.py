import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# =====================
# 1. 全局绘图风格
# =====================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'text.usetex': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
})

COLORS = {
    'base': '#E8975E',      # general: 橙色
    'highlight': '#D77272', # iter2: 红色
    'kde': '#6FAB7D',
    'zero': '#1976D2',
}

# =====================
# 2. 基础 histogram（general，不动）
# =====================
def save_hist(
    data, title, xlabel, filename, save_dir,
    bins=200, color=None, kde=True,
    cut_min_percentile=0.1, cut_max_percentile=0.1
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor('#F5F5F5')

    # percentile cut
    mask = np.ones_like(data, dtype=bool)
    mask &= data >= np.percentile(data, cut_min_percentile)
    mask &= data <= np.percentile(data, 100 - cut_max_percentile)
    data = data[mask]

    bin_edges = np.histogram_bin_edges(data, bins=bins)

    ax.hist(
        data,
        bins=bin_edges,
        density=True,
        alpha=0.7,
        color=color,
        edgecolor='none',
        label='General'
    )

    if kde and len(data) > 1:
        kde_fn = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 1000)
        ax.plot(x, kde_fn(x), color=COLORS['kde'], linewidth=3, label='KDE')

    ax.axvline(0, color=COLORS['zero'], linewidth=3, label='$x=0$')

    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    ax.set_ylabel("Density", fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    png = os.path.join(save_dir, filename)
    pdf = png.replace(".png", ".pdf")
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close()

def save_hist_with_highlight(
    base_data,
    highlight_data,
    title,
    xlabel,
    filename,
    save_dir,
    bins=200,
    kde=True,
    cut_min_percentile=0.1,
    cut_max_percentile=0.1,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor('#F5F5F5')

    # percentile cut（只基于 general）
    mask = np.ones_like(base_data, dtype=bool)
    mask &= base_data >= np.percentile(base_data, cut_min_percentile)
    mask &= base_data <= np.percentile(base_data, 100 - cut_max_percentile)
    base_data = base_data[mask]

    # highlight 只保留落在 base 区间内的
    highlight_data = highlight_data[
        (highlight_data >= base_data.min()) &
        (highlight_data <= base_data.max())
    ]

    bin_edges = np.histogram_bin_edges(base_data, bins=bins)

    # general
    ax.hist(
        base_data,
        bins=bin_edges,
        density=True,
        alpha=0.6,
        color=COLORS['base'],
        edgecolor='none',
        label='General',
        zorder=2
    )

    # iter2 highlight
    ax.hist(
        highlight_data,
        bins=bin_edges,
        density=True,
        alpha=0.85,
        color=COLORS['highlight'],
        edgecolor='none',
        label='Iter2 (selected)',
        zorder=4
    )

    if kde and len(base_data) > 1:
        kde_fn = gaussian_kde(base_data)
        x = np.linspace(base_data.min(), base_data.max(), 1000)
        ax.plot(x, kde_fn(x), color=COLORS['kde'], linewidth=3, label='KDE', zorder=5)

    ax.axvline(0, color=COLORS['zero'], linewidth=3, label='$x=0$', zorder=6)

    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    ax.set_ylabel("Density", fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    png = os.path.join(save_dir, filename)
    pdf = png.replace(".png", ".pdf")
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close()


# =====================
# 4. 主流程（instruction 匹配）
# =====================
if __name__ == "__main__":

    data_dir = "/volume/pt-train/users/wzhang/ghchen/zh/code/Data-Selection/plot/result_high"

    # general_path = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__general_general-1k-epoch1-2026/compared.jsonl"
    # iter2_path = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__general_general-1k-epoch1-iterEpoch2/selection_abalation/picked_diff_entropy_bytoken_reject_diff_nll_ratio_0.1_bireject_0.1.jsonl"
    
    
    math_path = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/qwen2_5_7b__math_math-1k/compared.jsonl"
    math_iter2_path = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/qwen2_5_7b__math_math-iter-2/selection/picked_diff_nll_bytoken_ratio_0.1_bireject_0.1.jsonl"

    # ---------- load ----------
    with open(math_path, "r", encoding="utf-8") as f:
        general_rows = [json.loads(l) for l in f]

    with open(math_iter2_path, "r", encoding="utf-8") as f:
        iter2_rows = [json.loads(l) for l in f]


    # ===== 用 instruction 做匹配 =====
    iter2_instructions = set(
        x["instruction"].strip() for x in iter2_rows if "instruction" in x
    )

    general_entropy = []
    iter2_entropy = []

    for x in general_rows:
        if "diff_entropy" not in x or not np.isfinite(x["diff_entropy"]):
            continue
        ent = x["diff_entropy"]
        general_entropy.append(ent)

        instr = x.get("instruction", "").strip()
        if instr in iter2_instructions:
            iter2_entropy.append(ent)

    general_entropy = np.array(general_entropy)
    iter2_entropy = np.array(iter2_entropy)

    print(f"Math samples: {len(general_entropy)}")
    print(f"Iter2 matched samples: {len(iter2_entropy)}")

    # ---------- general 原图 ----------
    save_hist(
        data=general_entropy,
        title="Distribution of ΔEntropy (math)",
        xlabel="ΔEntropy",
        filename="diff_entropy_math2.png",
        save_dir=data_dir,
        color=COLORS['base'],
        bins=200,
        kde=True,
    )

    # ---------- overlay 高亮 iter2 ----------
    save_hist_with_highlight(
        base_data=general_entropy,
        highlight_data=iter2_entropy,
        title="Distribution of ΔEntropy (math, iter2 highlighted)",
        xlabel="ΔEntropy",
        filename="diff_entropy_math_iter2_highlight2.png",
        save_dir=data_dir,
        bins=200,
        kde=True,
    )

    print("✅ Done. Check result_high directory.")
