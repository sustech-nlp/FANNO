import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

domains = ["Math", "General", "Medical", "Code"]
data = {
    "ALL":            [27.05, 4.93, 44.48, 43.0],
    "Random":         [25.80, 3.81, 44.76, 40.8],
    "IFD":            [21.99, 6.52, 44.59, 37.1],
    "Superfiltering": [22.61, 3.84, 38.29, 39.4],
    "SelectIT":       [24.35, 5.09, 46.81, 44.4],
    "ZIP":            [22.61, 3.84, 44.61, 39.4],
    "PPL":            [25.65, 5.66, 49.81, 40.3],
    "Entropy":        [23.98, 6.98, 50.63, 41.0],
    "Ours":           [31.63, 7.47, 56.42, 45.1],
}

df = pd.DataFrame(data, index=domains).T
domain_max = df.max(axis=0)
df_norm = df / domain_max

# 论文友好的浅色配色方案
colors = [
    '#B8B8B8',  # ALL - 浅灰
    '#C5E1A5',  # Random - 浅绿
    '#90CAF9',  # IFD - 浅蓝
    '#FFCC80',  # Superfiltering - 浅橙
    '#CE93D8',  # SelectIT - 浅紫
    '#FFE082',  # ZIP - 浅黄
    '#A5D6A7',  # PPL - 浅青绿
    '#EF9A9A',  # Entropy - 浅红
    '#FF6B6B',  # Ours - 强调色（稍深的红）
]

# 设置论文风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

angles = np.linspace(0, 2 * np.pi, len(domains), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(8, 8), dpi=300)
ax = plt.subplot(111, polar=True)

# 绘制每个方法
for idx, method in enumerate(df_norm.index):
    values = df_norm.loc[method, domains].tolist()
    values += values[:1]
    
    if method == "Ours":
        # Ours 使用更粗的线和填充
        ax.plot(angles, values, linewidth=2.5, label=method, 
                color=colors[idx], marker='o', markersize=6, zorder=10)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    else:
        ax.plot(angles, values, linewidth=1.8, label=method, 
                color=colors[idx], alpha=0.85)

# 设置网格样式
ax.set_thetagrids(np.degrees(angles[:-1]), domains, fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10, color='gray')
ax.grid(color='lightgray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_facecolor('#FAFAFA')

# 标题和图例
ax.set_title("Cross-Domain Generalization (Normalized per Domain Max)", 
             pad=25, fontsize=14, fontweight='bold')
ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.1), 
          frameon=True, fancybox=True, shadow=False,
          fontsize=10, edgecolor='gray')

plt.tight_layout()
plt.savefig('radar_chart.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('radar_chart.png', format='png', bbox_inches='tight', dpi=300)
plt.show()

# 打印归一化参考值
print("Domain Maximum Values (for normalization):")
print(domain_max.to_dict())