# import json
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import gaussian_kde

# # =====================
# # 1. 全局绘图风格设置 (完全保留原样)
# # =====================
# plt.style.use('seaborn-v0_8-darkgrid')
# sns.set_palette("husl")

# plt.rcParams.update({
#     "figure.dpi": 150,
#     "savefig.dpi": 300,
#     "font.size": 11,
#     "font.family": "DejaVu Sans",
#     "axes.titlesize": 14,
#     "axes.labelsize": 12,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
#     "legend.fontsize": 10,
#     "axes.grid": True,
#     "grid.alpha": 0.3,
#     "grid.linewidth": 0.5,
# })

# COLORS = {
#     'base': '#2E86AB',      # 蓝色
#     'instruct': '#A23B72',  # 紫色
#     'diff': '#F18F01',      # 橙色
#     'highlight': '#C73E1D', # 红色
#     'neutral': '#6C757D'    # 灰色
# }

# # =====================
# # 2. 核心绘图函数（x和y加粗）
# # =====================
# def save_hist(data, title, xlabel, filename, save_dir, bins=50, color=None, kde=True):
#     plt.figure(figsize=(8, 5))
    
#     # 绘制直方图
#     n, bins_, patches = plt.hist(
#         data,
#         bins=bins,
#         alpha=0.7,
#         color=color or COLORS['diff'],
#         edgecolor='white',
#         linewidth=0.5,
#         density=True,
#     )
    
#     # 添加KDE曲线
#     if kde:
#         kde_obj = gaussian_kde(data)
#         x_range = np.linspace(data.min(), data.max(), 1000)
#         plt.plot(
#             x_range, 
#             kde_obj(x_range), 
#             color=COLORS['highlight'], 
#             linewidth=2, 
#             label='KDE'
#         )
    
#     # 添加统计信息线
#     mean_val = np.mean(data)
#     median_val = np.median(data)
#     std_val = np.std(data)
    
#     plt.axvline(
#         mean_val, 
#         color=COLORS['highlight'], 
#         linestyle="--", 
#         linewidth=2, 
#         label=f'Mean: {mean_val:.3f}'
#     )
#     plt.axvline(
#         median_val, 
#         color=COLORS['base'], 
#         linestyle="--", 
#         linewidth=2, 
#         alpha=0.7, 
#         label=f'Median: {median_val:.3f}'
#     )
    
#     # 添加统计文本框
#     stats_text = (
#         f'N = {len(data):,}\n'
#         f'Mean = {mean_val:.3f}\n'
#         f'Std = {std_val:.3f}\n'
#         f'Min = {data.min():.3f}\n'
#         f'Max = {data.max():.3f}'
#     )
#     plt.text(
#         0.95, 0.95, stats_text, transform=plt.gca().transAxes,
#         fontsize=9, verticalalignment='top', horizontalalignment='right',
#         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
#     )
    
#     plt.title(title, fontsize=14, fontweight='bold', pad=15)
#     plt.xlabel(xlabel, fontsize=13, fontweight='bold')
#     plt.ylabel("Density", fontsize=13, fontweight='bold')
#     plt.legend(loc='upper left', frameon=True, fancybox=True)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
    
#     # 保存
#     save_path = os.path.join(save_dir, filename)
#     plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     plt.close()
#     print(f"Successfully saved histogram to: {save_path}")

# # =====================
# # 3. 支持dict的输入，分别画entropy, nll, ppl
# # =====================
# def run_histogram_analysis(input_dict):
#     # 输出结果目录
#     data_dir = "/volume/pt-train/users/wzhang/ghchen/zh/code/Data-Selection/plot/result"
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)

#     file_map = {
#         "ΔNLL": "diff_nll_{}.png",
#         "ΔEntropy": "diff_entropy_{}.png",
#         "ΔPPL": "diff_ppl_{}.png"
#     }
#     xlabel_map = {
#         "ΔNLL": "ΔNLL",
#         "ΔEntropy": "ΔEntropy",
#         "ΔPPL": "ΔPPL"
#     }

#     for name, jsonl_path in input_dict.items():
#         if not os.path.exists(jsonl_path):
#             print(f"File not found: {jsonl_path}")
#             continue

#         data = []
#         with open(jsonl_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 data.append(json.loads(line))

#         metrics = {
#             "ΔNLL": np.array([x["diff_nll"] for x in data]),
#             "ΔEntropy": np.array([x["diff_entropy"] for x in data]),
#             "ΔPPL": np.array([x["diff_ppl"] for x in data]),
#         }
#         for metric_name, values in metrics.items():
#             clean_values = values[np.isfinite(values)]
#             title = f"Distribution of {metric_name} ({name.capitalize()})"
#             save_hist(
#                 data=clean_values,
#                 title=title,
#                 xlabel=xlabel_map[metric_name],
#                 filename=file_map[metric_name].format(name),
#                 save_dir=data_dir,
#                 color=COLORS['diff'],
#             )

# # =====================
# # 4. 执行
# # =====================
# if __name__ == "__main__":
#     # 输入格式如：{'code': '/path/to/code.jsonl', 'math': '/path/to/math.jsonl'}
#     input_dict = {
#         'code': "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__bigcode_bigcode-iter1/compared.jsonl",
#         'math': "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__math_math-iter1/compared.jsonl"
#     }
#     run_histogram_analysis(input_dict)

