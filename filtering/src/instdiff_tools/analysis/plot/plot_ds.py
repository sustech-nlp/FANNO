import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec

# =====================
# 全局绘图风格设置
# =====================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "patch.linewidth": 0.5,
    "axes.linewidth": 0.5,
})

# 创建调色板
COLORS = {
    'base': '#2E86AB',      # 蓝色
    'instruct': '#A23B72',  # 紫色
    'diff': '#F18F01',      # 橙色
    'highlight': '#C73E1D', # 红色
    'neutral': '#6C757D'    # 灰色
}

# =====================
# 读取 jsonl
# =====================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

jsonl_path = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__bigcode_bigcode-iter1/compared.jsonl"
data_dir = os.path.dirname(os.path.abspath(jsonl_path)) + "/plot_ds"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

data = load_jsonl(jsonl_path)

# # =====================
# # 提取数据
# # =====================
# base_nll = np.array([x["base_nll"] for x in data])
# instruct_nll = np.array([x["instruct_nll"] for x in data])
# diff_nll = np.array([x["diff_nll"] for x in data])

# base_entropy = np.array([x["base_avg_entropy"] for x in data])
# instruct_entropy = np.array([x["instruct_avg_entropy"] for x in data])
# diff_entropy = np.array([x["diff_entropy"] for x in data])

# base_ppl = np.array([x["base_ppl"] for x in data])
# instruct_ppl = np.array([x["instruct_ppl"] for x in data])
# diff_ppl = np.array([x["diff_ppl"] for x in data])
# =====================
# 提取原始数据
# =====================
base_nll = np.array([x["base_nll"] for x in data])
instruct_nll = np.array([x["instruct_nll"] for x in data])
diff_nll = np.array([x["diff_nll"] for x in data])

base_entropy = np.array([x["base_avg_entropy"] for x in data])
instruct_entropy = np.array([x["instruct_avg_entropy"] for x in data])
diff_entropy = np.array([x["diff_entropy"] for x in data])

base_ppl = np.array([x["base_ppl"] for x in data])
instruct_ppl = np.array([x["instruct_ppl"] for x in data])
diff_ppl = np.array([x["diff_ppl"] for x in data])

# =====================
# 保证长度一致，去掉 NaN/inf
# =====================
# 所有数组先堆成一个矩阵，每行是一个样本
matrix = np.stack([
    base_nll, instruct_nll, diff_nll,
    base_entropy, instruct_entropy, diff_entropy,
    base_ppl, instruct_ppl, diff_ppl
], axis=1)

# 找到所有行都是有限值的索引
finite_idx = np.all(np.isfinite(matrix), axis=1)

# 只保留有效行
matrix = matrix[finite_idx]

# 拆回原来的变量
(base_nll, instruct_nll, diff_nll,
 base_entropy, instruct_entropy, diff_entropy,
 base_ppl, instruct_ppl, diff_ppl) = matrix.T

# 计算相关系数
corr_nll_entropy, _ = stats.pearsonr(diff_nll, diff_entropy)
corr_nll_ppl, _ = stats.pearsonr(diff_nll, diff_ppl)
corr_entropy_ppl, _ = stats.pearsonr(diff_entropy, diff_ppl)

# =====================
# 工具函数：美化直方图
# =====================
def save_hist(data, title, xlabel, filename, bins=50, color=None, kde=True):
    plt.figure(figsize=(8, 5))
    
    # 绘制直方图
    n, bins, patches = plt.hist(data, bins=bins, alpha=0.7, 
                                color=color or COLORS['diff'],
                                edgecolor='white', linewidth=0.5,
                                density=True)
    
    # 添加KDE曲线
    if kde:
        from scipy.stats import gaussian_kde
        kde_obj = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 1000)
        plt.plot(x_range, kde_obj(x_range), 
                color=COLORS['highlight'], linewidth=2, label='KDE')
    
    # 添加统计信息
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)
    
    plt.axvline(mean_val, color=COLORS['highlight'], 
                linestyle="--", linewidth=2, label=f'Mean: {mean_val:.3f}')
    plt.axvline(median_val, color=COLORS['base'], 
                linestyle="--", linewidth=2, alpha=0.7, label=f'Median: {median_val:.3f}')
    
    # 添加文本框
    stats_text = f'N = {len(data):,}\nMean = {mean_val:.3f}\nStd = {std_val:.3f}\nMin = {data.min():.3f}\nMax = {data.max():.3f}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc='upper left', frameon=True, fancybox=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

# =====================
# 工具函数：美化散点图
# =====================
def save_scatter(x, y, title, xlabel, ylabel, filename, diag=False, hexbin=False, 
                 add_corr=True, color=None):
    plt.figure(figsize=(7, 6))
    
    if hexbin and len(x) > 1000:  # 大数据量使用hexbin
        hb = plt.hexbin(x, y, gridsize=50, cmap='viridis', alpha=0.8)
        plt.colorbar(hb, label='Count')
        alpha = 0  # 隐藏散点
    else:
        alpha = 0.5
    
    # 绘制散点
    plt.scatter(x, y, alpha=alpha, s=15, color=color or COLORS['neutral'])
    
    # 添加对角线
    if diag:
        min_v = min(x.min(), y.min())
        max_v = max(x.max(), y.max())
        margin = (max_v - min_v) * 0.05
        plt.plot([min_v - margin, max_v + margin], 
                [min_v - margin, max_v + margin], 
                linestyle="--", linewidth=1.5, color=COLORS['highlight'], 
                alpha=0.7, label='y = x')
    
    # 添加回归线
    if add_corr and len(x) > 2:
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        x_fit = np.array([x.min(), x.max()])
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, color=COLORS['highlight'], 
                linewidth=2, label=f'Fit (r = {r_value:.3f})')
    
    # 添加相关系数
    if add_corr:
        corr, _ = stats.pearsonr(x, y)
        plt.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    
    if diag or add_corr:
        plt.legend(loc='lower right', frameon=True, fancybox=True)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

# =====================
# 工具函数：并排对比图
# =====================
def save_comparison_hist(base_data, instruct_data, title, xlabel, filename):
    plt.figure(figsize=(10, 5))
    
    # 计算合适的bins
    all_data = np.concatenate([base_data, instruct_data])
    bins = np.linspace(all_data.min(), all_data.max(), 50)
    
    # 绘制两个分布
    plt.hist(base_data, bins=bins, alpha=0.6, color=COLORS['base'], 
             edgecolor='white', linewidth=0.5, density=True, label='Base')
    plt.hist(instruct_data, bins=bins, alpha=0.6, color=COLORS['instruct'], 
             edgecolor='white', linewidth=0.5, density=True, label='Instruct')
    
    # 添加统计信息
    base_mean = np.mean(base_data)
    instruct_mean = np.mean(instruct_data)
    
    plt.axvline(base_mean, color=COLORS['base'], linestyle='--', linewidth=2)
    plt.axvline(instruct_mean, color=COLORS['instruct'], linestyle='--', linewidth=2)
    
    # 添加文本框
    stats_text = f'Base:\n  Mean = {base_mean:.3f}\n  Std = {np.std(base_data):.3f}\n\nInstruct:\n  Mean = {instruct_mean:.3f}\n  Std = {np.std(instruct_data):.3f}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc='upper right', frameon=True, fancybox=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

# =====================
# 工具函数：箱线图
# =====================
def save_boxplot(data_dict, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    
    labels = list(data_dict.keys())
    data_values = list(data_dict.values())
    
    # 创建箱线图
    boxprops = dict(linewidth=1.5)
    medianprops = dict(linewidth=2, color=COLORS['highlight'])
    whiskerprops = dict(linewidth=1.5)
    capprops = dict(linewidth=1.5)
    
    bp = plt.boxplot(data_values, labels=labels, patch_artist=True,
                     boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops)
    
    # 设置颜色
    colors = [COLORS['base'], COLORS['instruct'], COLORS['diff']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 添加均值和离群点统计
    for i, (label, data) in enumerate(data_dict.items()):
        mean_val = np.mean(data)
        n_outliers = np.sum(np.abs(data - np.median(data)) > 3 * np.std(data))
        
        plt.text(i + 1, mean_val, f'μ={mean_val:.2f}', 
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=9, fontweight='bold')
        
        if n_outliers > 0:
            plt.text(i + 1, plt.ylim()[0], f'{n_outliers} outliers', 
                    horizontalalignment='center', verticalalignment='top',
                    fontsize=8, color='red', alpha=0.7)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

# =====================
# 工具函数：相关矩阵热图
# =====================
def save_correlation_heatmap(data_dict, title, filename):
    # 计算相关系数矩阵
    labels = list(data_dict.keys())
    n_vars = len(labels)
    corr_matrix = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            corr, _ = stats.pearsonr(data_dict[labels[i]], data_dict[labels[j]])
            corr_matrix[i, j] = corr
    
    # 创建热图
    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # 添加文本
    for i in range(n_vars):
        for j in range(n_vars):
            text = plt.text(j, i, f'{corr_matrix[i, j]:.3f}',
                           ha="center", va="center", 
                           color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                           fontsize=10, fontweight='bold')
    
    # 设置坐标轴
    plt.xticks(range(n_vars), labels, rotation=45, ha='right')
    plt.yticks(range(n_vars), labels)
    plt.colorbar(im, label='Pearson Correlation')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

# =====================
# 工具函数：综合摘要图
# =====================
def create_summary_figure(data, filename):
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. diff_nll 分布
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(diff_nll, bins=50, alpha=0.7, color=COLORS['diff'], 
             edgecolor='white', linewidth=0.5, density=True)
    ax1.axvline(np.mean(diff_nll), color=COLORS['highlight'], 
                linestyle='--', linewidth=2)
    ax1.set_title('ΔNLL Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('ΔNLL')
    ax1.set_ylabel('Density')
    ax1.grid(True, alpha=0.3)
    
    # 2. Base vs Instruct NLL 散点
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(base_nll, instruct_nll, alpha=0.4, s=10, color=COLORS['neutral'])
    min_v = min(base_nll.min(), instruct_nll.min())
    max_v = max(base_nll.max(), instruct_nll.max())
    ax2.plot([min_v, max_v], [min_v, max_v], '--', color=COLORS['highlight'], alpha=0.7)
    ax2.set_title('Base vs Instruct NLL', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Base NLL')
    ax2.set_ylabel('Instruct NLL')
    ax2.grid(True, alpha=0.3)
    
    # 3. ΔNLL vs ΔPPL 相关性
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(diff_nll, diff_ppl, alpha=0.5, s=10, color=COLORS['neutral'])
    corr = np.corrcoef(diff_nll, diff_ppl)[0, 1]
    ax3.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax3.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.set_title('ΔNLL vs ΔPPL', fontsize=12, fontweight='bold')
    ax3.set_xlabel('ΔNLL')
    ax3.set_ylabel('ΔPPL')
    ax3.grid(True, alpha=0.3)
    
    # 4. Base 和 Instruct 的箱线图对比
    ax4 = fig.add_subplot(gs[1, 0])
    bp = ax4.boxplot([base_nll, instruct_nll], labels=['Base', 'Instruct'], 
                     patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['base'])
    bp['boxes'][1].set_facecolor(COLORS['instruct'])
    ax4.set_title('NLL Distribution Comparison', fontsize=12, fontweight='bold')
    ax4.set_ylabel('NLL')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 累积分布函数 (CDF)
    ax5 = fig.add_subplot(gs[1, 1])
    sorted_diff = np.sort(diff_nll)
    cdf = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)
    ax5.plot(sorted_diff, cdf, linewidth=2, color=COLORS['diff'])
    ax5.axhline(0.5, linestyle='--', color='gray', alpha=0.5)
    ax5.axvline(np.median(diff_nll), linestyle='--', color=COLORS['highlight'], alpha=0.7)
    ax5.set_title('CDF of ΔNLL', fontsize=12, fontweight='bold')
    ax5.set_xlabel('ΔNLL')
    ax5.set_ylabel('Cumulative Probability')
    ax5.grid(True, alpha=0.3)
    
    # 6. 统计摘要文本
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_text = f"""Dataset Summary
----------------
Total Samples: {len(data):,}

ΔNLL Statistics:
  Mean: {np.mean(diff_nll):.4f}
  Std:  {np.std(diff_nll):.4f}
  Min:  {np.min(diff_nll):.4f}
  Max:  {np.max(diff_nll):.4f}
  Median: {np.median(diff_nll):.4f}

Correlations:
  ΔNLL vs ΔEntropy: {corr_nll_entropy:.3f}
  ΔNLL vs ΔPPL:     {corr_nll_ppl:.3f}
  ΔEntropy vs ΔPPL: {corr_entropy_ppl:.3f}

Percentage Analysis:
  ΔNLL > 0: {np.sum(diff_nll > 0)/len(diff_nll)*100:.1f}%
  ΔNLL < 0: {np.sum(diff_nll < 0)/len(diff_nll)*100:.1f}%
"""
    ax6.text(0, 1, summary_text, transform=ax6.transAxes,
             fontsize=10, fontfamily='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9))
    
    plt.suptitle(f'Data Analysis Summary - {os.path.basename(jsonl_path)}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

# =====================
# 1️⃣ diff 指标分布 (美化版)
# =====================
save_hist(
    diff_nll,
    title="Distribution of ΔNLL (Base − Instruct)",
    xlabel="ΔNLL",
    filename="diff_nll_distribution.png",
    color=COLORS['diff']
)

save_hist(
    diff_entropy,
    title="Distribution of ΔEntropy (Base − Instruct)",
    xlabel="ΔEntropy",
    filename="diff_entropy_distribution.png",
    color=COLORS['diff']
)

save_hist(
    diff_ppl,
    title="Distribution of ΔPPL (Base − Instruct)",
    xlabel="ΔPPL",
    filename="diff_ppl_distribution.png",
    color=COLORS['diff']
)

# =====================
# 2️⃣ Base vs Instruct 对比 (美化版)
# =====================
save_scatter(
    base_nll,
    instruct_nll,
    title="Base vs Instruct NLL",
    xlabel="Base NLL",
    ylabel="Instruct NLL",
    filename="base_vs_instruct_nll.png",
    diag=True,
    hexbin=True
)

save_scatter(
    base_entropy,
    instruct_entropy,
    title="Base vs Instruct Entropy",
    xlabel="Base Entropy",
    ylabel="Instruct Entropy",
    filename="base_vs_instruct_entropy.png",
    diag=True,
    hexbin=True
)

save_scatter(
    base_ppl,
    instruct_ppl,
    title="Base vs Instruct PPL",
    xlabel="Base PPL",
    ylabel="Instruct PPL",
    filename="base_vs_instruct_ppl.png",
    diag=True,
    hexbin=True
)

# =====================
# 3️⃣ diff 指标相关性 (美化版)
# =====================
save_scatter(
    diff_nll,
    diff_entropy,
    title="ΔNLL vs ΔEntropy",
    xlabel="ΔNLL",
    ylabel="ΔEntropy",
    filename="diff_nll_vs_entropy.png",
    add_corr=True
)

save_scatter(
    diff_nll,
    diff_ppl,
    title="ΔNLL vs ΔPPL",
    xlabel="ΔNLL",
    ylabel="ΔPPL",
    filename="diff_nll_vs_ppl.png",
    add_corr=True
)

# =====================
# 4️⃣ 新增：并排对比直方图
# =====================
save_comparison_hist(
    base_nll,
    instruct_nll,
    title="NLL Distribution: Base vs Instruct",
    xlabel="NLL",
    filename="nll_comparison_hist.png"
)

save_comparison_hist(
    base_entropy,
    instruct_entropy,
    title="Entropy Distribution: Base vs Instruct",
    xlabel="Entropy",
    filename="entropy_comparison_hist.png"
)

save_comparison_hist(
    base_ppl,
    instruct_ppl,
    title="PPL Distribution: Base vs Instruct",
    xlabel="PPL",
    filename="ppl_comparison_hist.png"
)

# =====================
# 5️⃣ 新增：箱线图
# =====================
save_boxplot(
    {"Base": base_nll, "Instruct": instruct_nll, "ΔNLL": diff_nll},
    title="NLL Distributions Comparison",
    ylabel="Value",
    filename="nll_boxplot.png"
)

save_boxplot(
    {"Base": base_entropy, "Instruct": instruct_entropy, "ΔEntropy": diff_entropy},
    title="Entropy Distributions Comparison",
    ylabel="Value",
    filename="entropy_boxplot.png"
)

# =====================
# 6️⃣ 新增：相关矩阵热图
# =====================
corr_data = {
    "ΔNLL": diff_nll,
    "ΔEntropy": diff_entropy,
    "ΔPPL": diff_ppl,
    "Base NLL": base_nll,
    "Instruct NLL": instruct_nll
}

save_correlation_heatmap(
    corr_data,
    title="Correlation Matrix of Metrics",
    filename="correlation_heatmap.png"
)

# =====================
# 7️⃣ 新增：综合摘要图
# =====================
create_summary_figure(data, "summary_figure.png")

# =====================
# 8️⃣ 新增：阈值分析图
# =====================
def create_threshold_analysis(diff_data, metric_name, filename):
    thresholds = np.linspace(diff_data.min(), diff_data.max(), 50)
    percentages = []
    
    for thresh in thresholds:
        if thresh >= 0:
            percentage = np.sum(diff_data >= thresh) / len(diff_data) * 100
        else:
            percentage = np.sum(diff_data <= thresh) / len(diff_data) * 100
        percentages.append(percentage)
    
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, percentages, linewidth=2, color=COLORS['diff'])
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(50, color='gray', linestyle='--', alpha=0.5)
    
    # 标记几个关键阈值
    for pct in [10, 25, 50, 75, 90]:
        idx = np.argmin(np.abs(np.array(percentages) - pct))
        plt.scatter(thresholds[idx], percentages[idx], color=COLORS['highlight'], s=50, zorder=5)
        plt.text(thresholds[idx], percentages[idx] + 2, f'{thresholds[idx]:.3f}\n({pct}%)',
                ha='center', fontsize=8)
    
    plt.title(f'Threshold Analysis for {metric_name}', fontsize=14, fontweight='bold')
    plt.xlabel(f'{metric_name} Threshold')
    plt.ylabel('Percentage of Samples (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

create_threshold_analysis(diff_nll, "ΔNLL", "threshold_analysis_delta_nll.png")

# =====================
# 统计摘要（终端）和保存到文件
# =====================
summary_stats = f"""
===== Summary Statistics =====
Dataset: {os.path.basename(jsonl_path)}
Total Samples: {len(data):,}

NLL Statistics:
  Base NLL:      mean={np.mean(base_nll):.4f}, std={np.std(base_nll):.4f}
  Instruct NLL:  mean={np.mean(instruct_nll):.4f}, std={np.std(instruct_nll):.4f}
  ΔNLL:          mean={np.mean(diff_nll):.4f}, std={np.std(diff_nll):.4f}

Entropy Statistics:
  Base Entropy:  mean={np.mean(base_entropy):.4f}, std={np.std(base_entropy):.4f}
  Instruct Entropy: mean={np.mean(instruct_entropy):.4f}, std={np.std(instruct_entropy):.4f}
  ΔEntropy:      mean={np.mean(diff_entropy):.4f}, std={np.std(diff_entropy):.4f}

PPL Statistics:
  Base PPL:      mean={np.mean(base_ppl):.4f}, std={np.std(base_ppl):.4f}
  Instruct PPL:  mean={np.mean(instruct_ppl):.4f}, std={np.std(instruct_ppl):.4f}
  ΔPPL:          mean={np.mean(diff_ppl):.4f}, std={np.std(diff_ppl):.4f}

Correlation Analysis:
  ΔNLL vs ΔEntropy:  ρ = {corr_nll_entropy:.4f}
  ΔNLL vs ΔPPL:      ρ = {corr_nll_ppl:.4f}
  ΔEntropy vs ΔPPL:  ρ = {corr_entropy_ppl:.4f}

Percentage Analysis:
  ΔNLL > 0: {np.sum(diff_nll > 0)/len(diff_nll)*100:.2f}% (Instruct worse than Base)
  ΔNLL < 0: {np.sum(diff_nll < 0)/len(diff_nll)*100:.2f}% (Instruct better than Base)
  ΔNLL = 0: {np.sum(diff_nll == 0)/len(diff_nll)*100:.2f}%

Figures saved to: {data_dir}
"""

print(summary_stats)

# 保存统计摘要到文件
with open(os.path.join(data_dir, "statistics_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary_stats)

print(f"\n✅ All visualizations completed! {len(os.listdir(data_dir))} files generated.")