import matplotlib.pyplot as plt
import numpy as np

# =====================
# Global style (same as reference)
# =====================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 11,
    'text.usetex': False,
    'figure.figsize': (11, 6),
    'axes.linewidth': 1.2,
})

# =====================
# Data
# =====================
domains = ['General', 'Med', 'Math', 'Code']

high_values = [4.87, 48.24, 22.13, 39.76]
mid_values  = [5.64, 53.48, 26.80, 41.6]
low_values  = [7.47, 56.42, 31.63, 45.1]

# =====================
# Figure
# =====================
fig, ax = plt.subplots(figsize=(11, 6))

# Colors (consistent & muted)
colors = {
    'High': '#5B7FA6',   # blue-gray
    'Mid':  '#E8975E',   # orange
    'Low':  '#6FAB7D',   # green
}

# Bar layout
x = np.arange(len(domains))
width = 0.22
group_gap = 1.0
x_wide = x * group_gap

# =====================
# Plot bars
# =====================
bars_high = ax.bar(
    x_wide - width, high_values, width,
    label='Max 10%', color=colors['High'],
    edgecolor='none', zorder=3
)

bars_mid = ax.bar(
    x_wide, mid_values, width,
    label='Mid 10%', color=colors['Mid'],
    edgecolor='none', zorder=3
)

bars_low = ax.bar(
    x_wide + width, low_values, width,
    label='Min 10%', color=colors['Low'],
    edgecolor='none', zorder=3
)

# =====================
# Value labels
# =====================
def add_value_labels(bars, values):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f'{val:.2f}',
            ha='center', va='bottom',
            fontsize=9.5
        )

add_value_labels(bars_high, high_values)
add_value_labels(bars_mid, mid_values)
add_value_labels(bars_low, low_values)

# =====================
# Axes & title
# =====================
ax.set_xlabel('Domain', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Score', fontsize=13, fontweight='bold')
ax.set_title(
    'Performance under Different Entropy Levels',
    fontsize=16, fontweight='bold', pad=15
)

ax.set_xticks(x_wide)
ax.set_xticklabels(domains, fontweight='bold')

# Y ticks bold
ax.tick_params(axis='y', labelsize=11)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# =====================
# Grid & background
# =====================
ax.grid(
    True, axis='y',
    alpha=0.25, linestyle='-',
    linewidth=0.6, color='gray',
    zorder=0
)
ax.set_axisbelow(True)

ax.set_facecolor('#F5F5F5')
fig.patch.set_facecolor('white')

# =====================
# Legend
# =====================
# legend = ax.legend(
#     loc='upper left',
#     frameon=True,
#     fancybox=False,
#     edgecolor='lightgray',
#     framealpha=1.0,
#     ncol=3,
#     columnspacing=1.8
# )
legend = ax.legend(
    loc='upper left',
    frameon=True,
    fancybox=False,
    edgecolor='lightgray',
    framealpha=1.0,
    ncol=1,              # 关键：一列 → 上下排列
    labelspacing=0.6     # 行距，论文友好
)

legend.get_frame().set_linewidth(1.0)

# =====================
# Limits & spines
# =====================
ax.set_ylim(0, 65)
ax.set_xlim(-0.5, x_wide[-1] + 0.5)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color('#CCCCCC')

ax.tick_params(which='major', length=5, width=1.0, direction='out', color='#CCCCCC')

# =====================
# Save
# =====================
plt.tight_layout()
plt.savefig(
    '/volume/pt-train/users/wzhang/ghchen/zh/code/Data-Selection/plot/result/entropy_domain_bar.pdf',
    dpi=300, bbox_inches='tight', format='pdf'
)
plt.savefig(
    '/volume/pt-train/users/wzhang/ghchen/zh/code/Data-Selection/plot/result/entropy_domain_bar.png',
    dpi=300, bbox_inches='tight', format='png'
)

print("Entropy-domain bar chart saved successfully!")
plt.show()
