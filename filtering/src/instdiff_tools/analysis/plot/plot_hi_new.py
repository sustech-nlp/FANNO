
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# =====================
# 1. 全局绘图风格设置 (图例字号缩小为12)
# =====================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 16,
    'legend.fontsize': 13,   # 图例字号缩小为12
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
    'kde': '#6FAB7D',       # 绿色
    'negative': '#8ecae6',  # 浅蓝色 for x<0
    'zero': '#1976D2',      # 蓝色 for x=0
}

# =====================
# 2. 核心绘图函数
#    - bin更加细致（bins=200）
#    - 过滤掉最大和最小的0.1%
# =====================
def save_hist(data, title, xlabel, filename, save_dir, bins=200, color=None, kde=True, cut_min_percentile=None, cut_max_percentile=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 设置背景色
    ax.set_facecolor('#F5F5F5')
    fig.patch.set_facecolor('white')

    # 过滤最大最小的percentile
    cut_mask = np.ones_like(data, dtype=bool)
    percent_info = []
    if cut_min_percentile is not None and cut_min_percentile > 0.0:
        lower = np.percentile(data, cut_min_percentile)
        cut_mask &= (data >= lower)
        percent_info.append(f">{cut_min_percentile:.3f}th")
    if cut_max_percentile is not None and cut_max_percentile > 0.0:
        upper = np.percentile(data, 100 - cut_max_percentile)
        cut_mask &= (data <= upper)
        percent_info.append(f"<{100 - cut_max_percentile:.3f}th")

    if percent_info:
        data = data[cut_mask]
        # 不加"percentile shown"，更简洁
        title = f"{title}"

    # --------先找到x=0的bin----------
    bin_edges = np.histogram_bin_edges(data, bins=bins)
    zero_bin_index = np.searchsorted(bin_edges, 0, side="right") - 1
    # ----------

    # 染色x<0区域（先填背景）
    min_xlim, max_xlim = data.min(), data.max()
    neg_xmin = min_xlim
    if min_xlim < 0:
        ax.axvspan(neg_xmin, 0, color=COLORS['negative'], alpha=0.28, zorder=1, label='x < 0')

    # 绘制直方图（主 bars)
    n, bins_, patches = ax.hist(
        data,
        bins=bin_edges, # 用细致的bins
        alpha=0.7,
        color=color or COLORS['diff'],
        edgecolor='none',
        linewidth=0,
        density=True,
        zorder=2,
    )

    # 单独给x<0的bins染浅蓝
    for i in range(len(bins_)-1):
        if bins_[i+1] <= 0:  # entire bin在x<0
            patches[i].set_facecolor(COLORS['negative'])
            patches[i].set_alpha(0.62)
            patches[i].set_zorder(3)

    # 添加KDE曲线
    kde_line = None
    if kde and len(data) > 1:
        kde_obj = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 1000)
        kde_dens = kde_obj(x_range)
        kde_line = ax.plot(
            x_range, 
            kde_dens, 
            color=COLORS['kde'], 
            linewidth=3.2,
            label='KDE',
            zorder=5
        )[0]

    # 添加统计信息线
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)

    mean_line = ax.axvline(
        mean_val, 
        color=COLORS['highlight'], 
        linestyle="--", 
        linewidth=3.2, 
        label=f'Mean: {mean_val:.3f}', 
        zorder=8
    )
    median_line = ax.axvline(
        median_val, 
        color=COLORS['base'], 
        linestyle="--", 
        linewidth=3.2, 
        alpha=0.7, 
        label=f'Median: {median_val:.3f}',
        zorder=8
    )

    # 标出x=0的线
    zero_line = ax.axvline(
        0,
        color=COLORS['zero'],
        linestyle='-', 
        linewidth=3.2, 
        label='$x = 0$', 
        zorder=15
    )

    # ===== 计算x<0占比 =====
    neg_ratio = (data < 0).sum() / len(data) if len(data) > 0 else 0.0

    # 图例移到右上角，包含KDE、Mean、Median、x=0和x<0区域
    legend_handles = []
    legend_labels = []

    # always show x=0 line
    legend_handles.append(zero_line)
    legend_labels.append('$x = 0$')

    # always show <0 区域的色块以及占比，无论数据是否有<0
    neg_proxy = plt.Rectangle((0, 0), 1, 1, fc=COLORS['negative'], alpha=0.45, edgecolor='none')
    legend_handles.append(neg_proxy)
    legend_labels.append(f'$x < 0$ ({neg_ratio*100:.2f}%)')

    # KDE
    if kde and kde_line:
        legend_handles.append(kde_line)
        legend_labels.append('KDE')

    # Mean, Median
    legend_handles.extend([mean_line, median_line])
    legend_labels.extend([f'Mean: {mean_val:.3f}', f'Median: {median_val:.3f}'])

    legend = ax.legend(
        legend_handles, legend_labels,
        loc='upper right', 
        frameon=True, 
        fancybox=False,
        edgecolor='lightgray',
        framealpha=1.0,
        fontsize=12    # 图例字号缩小为12
    )
    legend.get_frame().set_linewidth(1.0)

    # === 删除统计文本框 ===
    # stats_text = (
    #     f'N = {len(data):,}\n'
    #     f'Std = {std_val:.3f}\n'
    #     f'Min = {data.min():.3f}\n'
    #     f'Max = {data.max():.3f}'
    # )
    # ax.text(
    #     0.02, 0.98, stats_text, transform=ax.transAxes,
    #     fontsize=11, verticalalignment='top', horizontalalignment='left',
    #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray', linewidth=1)
    # )
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')  # 字体加1
    ax.set_ylabel("Density", fontsize=16, fontweight='bold')  # 字体加1
    
    # 加粗刻度标签
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 添加网格
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, zorder=0, color='gray')
    ax.set_axisbelow(True)
    
    # 设置边框样式
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
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
# 注意首先应该根据diff_nll只留下10%-90%这个区间的，再拿这个数据进行画图
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

        raw_data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                raw_data.append(json.loads(line))

        # 仅保留 diff_nll 合理（10%-90%）区间的数据
        diff_nll_values = np.array([x["diff_nll"] for x in raw_data])
        valid_mask = np.isfinite(diff_nll_values)
        diff_nll_values = diff_nll_values[valid_mask]
        lower = np.percentile(diff_nll_values, 5)
        upper = np.percentile(diff_nll_values, 95)
        # 按mask筛选原始数据行
        valid_rows = [x for x, v in zip(raw_data, valid_mask) if v]
        filtered_rows = [x for x in valid_rows if lower <= x["diff_nll"] <= upper]

        # 画图数据
        metrics = {
            # "ΔNLL": np.array([x["diff_nll"] for x in filtered_rows]),
            "ΔEntropy": np.array([x["diff_entropy"] for x in filtered_rows]),
            # "ΔPPL": np.array([x["diff_ppl"] for x in filtered_rows]),
        }
        # 对所有metric都过滤最大最小的0.1%
        for metric_name, values in metrics.items():
            clean_values = values[np.isfinite(values)]
            # 标题直接 Distribution of xxx (name)
            title = f"Distribution of {metric_name} ({name})"
            save_hist(
                data=clean_values,
                title=title,
                xlabel=xlabel_map[metric_name],
                filename=file_map[metric_name].format(name),
                save_dir=data_dir,
                color=COLORS['diff'],
                bins=200,  # 更细致
                kde=True,
                cut_min_percentile=0.1,
                cut_max_percentile=0.1,   # 两端各0.1%
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
        "math-iter1": math_iter1_dataset,
        "math-iter2": math_iter2_dataset,
        "code": code_iter0_dataset,
        "code-iter1": code_iter1_dataset,
        "code-iter2": code_iter2_dataset,
        "med": med_iter0_dataset,
        "med-iter1": med_iter1_dataset,
        "med-iter2": med_iter2_dataset,
        "general": general_iter0_dataset,
        "general-iter1": general_iter1_dataset,
    }

    print("Running histogram analysis with actual dataset paths...")
    run_histogram_analysis(input_dict)
    print("\nDone! Check /volume/pt-train/users/wzhang/ghchen/zh/code/Data-Selection/plot/result_high for generated plots.")

