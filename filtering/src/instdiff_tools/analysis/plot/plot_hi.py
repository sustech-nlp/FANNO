

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# =====================
# 1. 全局绘图风格设置 (更新为与柱状图一致的风格)
# =====================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 16,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'text.usetex': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
})

COLORS = {
    'base': '#5B7FA6',      # 蓝灰色
    'instruct': '#A23B72',  # 紫色
    'diff': '#E8975E',      # 橙色
    'highlight': '#D77272', # 红色
    'neutral': '#6C757D',   # 灰色
    'kde': '#6FAB7D'        # 绿色
}

# =====================
# 2. 核心绘图函数（x和y加粗，图例调整）
# =====================
def save_hist(data, title, xlabel, filename, save_dir, bins=50, color=None, kde=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 设置背景色
    ax.set_facecolor('#F5F5F5')
    fig.patch.set_facecolor('white')
    
    # 绘制直方图
    n, bins_, patches = ax.hist(
        data,
        bins=bins,
        alpha=0.7,
        color=color or COLORS['diff'],
        edgecolor='none',
        linewidth=0,
        density=True,
    )
    
    # 添加KDE曲线
    kde_line = None
    if kde:
        kde_obj = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 1000)
        kde_line = ax.plot(
            x_range, 
            kde_obj(x_range), 
            color=COLORS['kde'], 
            linewidth=2.5, 
            label='KDE'
        )[0]
    
    # 添加统计信息线
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)
    
    mean_line = ax.axvline(
        mean_val, 
        color=COLORS['highlight'], 
        linestyle="--", 
        linewidth=2, 
        label=f'Mean: {mean_val:.3f}'
    )
    median_line = ax.axvline(
        median_val, 
        color=COLORS['base'], 
        linestyle="--", 
        linewidth=2, 
        alpha=0.7, 
        label=f'Median: {median_val:.3f}'
    )
    
    # 图例移到右上角，包含KDE、Mean、Median
    legend_handles = []
    legend_labels = []
    if kde and kde_line:
        legend_handles.append(kde_line)
        legend_labels.append('KDE')
    legend_handles.extend([mean_line, median_line])
    legend_labels.extend([f'Mean: {mean_val:.3f}', f'Median: {median_val:.3f}'])
    
    legend = ax.legend(
        legend_handles, legend_labels,
        loc='upper right', 
        frameon=True, 
        fancybox=False,
        edgecolor='lightgray',
        framealpha=1.0
    )
    legend.get_frame().set_linewidth(1.0)
    
    # 添加统计文本框（仅包含基本统计信息）
    stats_text = (
        f'N = {len(data):,}\n'
        f'Std = {std_val:.3f}\n'
        f'Min = {data.min():.3f}\n'
        f'Max = {data.max():.3f}'
    )
    ax.text(
        0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray', linewidth=1)
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
    ax.set_ylabel("Density", fontsize=13, fontweight='bold')
    
    # 加粗刻度标签
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 添加网格
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.6, zorder=0, color='gray')
    ax.set_axisbelow(True)
    
    # 设置边框样式
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#CCCCCC')
    
    plt.tight_layout()
    
    # 保存PNG和PDF
    save_path_png = os.path.join(save_dir, filename)
    save_path_pdf = os.path.splitext(save_path_png)[0] + ".pdf"
    plt.savefig(save_path_png, bbox_inches='tight', dpi=300)
    plt.savefig(save_path_pdf, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Successfully saved histogram to: {save_path_png} and {save_path_pdf}")

# =====================
# 3. 支持dict的输入，分别画entropy, nll, ppl
# =====================
def run_histogram_analysis(input_dict):
    # 输出结果目录
    data_dir = "/volume/pt-train/users/wzhang/ghchen/zh/code/Data-Selection/plot/result_high"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_map = {
        # "ΔNLL": "diff_nll_{}.png",
        "ΔEntropy": "diff_entropy_{}.png",
        # "ΔPPL": "diff_ppl_{}.png"
    }
    xlabel_map = {
        # "ΔNLL": "ΔNLL",
        "ΔEntropy": "ΔEntropy",
        # "ΔPPL": "ΔPPL"
    }

    for name, jsonl_path in input_dict.items():
        if not os.path.exists(jsonl_path):
            print(f"File not found: {jsonl_path}")
            continue

        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        metrics = {
            # "ΔNLL": np.array([x["diff_nll"] for x in data]),
            "ΔEntropy": np.array([x["diff_entropy"] for x in data]),
            # "ΔPPL": np.array([x["diff_ppl"] for x in data]),
        }
        for metric_name, values in metrics.items():
            clean_values = values[np.isfinite(values)]
            title = f"Distribution of {metric_name} ({name.capitalize()})"
            save_hist(
                data=clean_values,
                title=title,
                xlabel=xlabel_map[metric_name],
                filename=file_map[metric_name].format(name),
                save_dir=data_dir,
                color=COLORS['diff'],
            )

# =====================
# 4. 执行 (实际数据路径)
# =====================
if __name__ == "__main__":
    # Math
    math_iter0_dataset = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/qwen2_5_7b__math_math-1k/compared_complexity.jsonl"
    math_iter1_dataset = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/qwen2_5_7b__math_math-iter-1/compared_complexity.jsonl"
    math_iter2_dataset = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/qwen2_5_7b__math_math-iter-2/compared_complexity.jsonl"

    # Code
    code_iter0_dataset = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__bigcode_bigcode-1k/compared.jsonl"
    code_iter1_dataset = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__bigcode_bigcode-iter1/compared.jsonl"
    code_iter2_dataset = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__bigcode_bigcode-iter2/compared.jsonl"

    # Med
    med_iter0_dataset = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__med_med-1k/compared.jsonl"
    med_iter1_dataset = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__med_med-1k-iter1-nll-0.1/compared.jsonl"
    med_iter2_dataset = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__med_med-1k-iter2-nll-0.1/compared.jsonl"

    # General
    general_iter0_dataset = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__general_general-1k-epoch1-2026/compared.jsonl"
    general_iter1_dataset = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__general_general-1k-epoch1-iterEpoch2/compared.jsonl"

    input_dict = {
        "math": math_iter0_dataset,
        # "math-iter1": math_iter1_dataset,
        # "math-iter2": math_iter2_dataset,
        "code": code_iter0_dataset,
        # "code-iter1": code_iter1_dataset,
        # "code-iter2": code_iter2_dataset,
        "med": med_iter0_dataset,
        # "med-iter1": med_iter1_dataset,
        # "med-iter2": med_iter2_dataset,
        "general": general_iter0_dataset,
        # "general-iter1": general_iter1_dataset,
    }

    print("Running histogram analysis with actual dataset paths...")
    run_histogram_analysis(input_dict)
    print("\nDone! Check /volume/pt-train/users/wzhang/ghchen/zh/code/Data-Selection/plot/result_high for generated plots.")

