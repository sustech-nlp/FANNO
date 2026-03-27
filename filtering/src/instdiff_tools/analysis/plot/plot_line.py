import matplotlib.pyplot as plt
import numpy as np

# Set style to match the reference image
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

# Data for plotting - reordered as requested
categories = ['General', 'Med', 'Math', 'Code']
base_values = [0.81, 47.87, 12.61, 39.6]
iter_0_values = [5.98, 53.33, 27.93, 39.7]
iter_1_values = [7.47, 54.69, 31.63, 44.5]
iter_2_values = [7.03, 56.42, 27.01, 45.1]

# Create figure
fig, ax = plt.subplots(figsize=(11, 6))

# Define colors matching the reference image
colors = {
    'base': '#5B7FA6',      # Blue-gray (first bar)
    'iter0': '#E8975E',     # Orange (second bar)
    'iter1': '#6FAB7D',     # Green (third bar)
    'iter2': '#D77272',     # Red/coral (fourth bar)
}

# Bar settings - wider spacing between groups
x = np.arange(len(categories))
width = 0.18
group_gap = 1.0  # Wider gap between groups

# Adjust x positions for wider spacing
x_wide = x * group_gap

# Plot bars
bars1 = ax.bar(x_wide - 1.5*width, base_values, width, 
               label='Base', color=colors['base'], 
               edgecolor='none', zorder=3)

bars2 = ax.bar(x_wide - 0.5*width, iter_0_values, width, 
               label='Iter-0', color=colors['iter0'], 
               edgecolor='none', zorder=3)

bars3 = ax.bar(x_wide + 0.5*width, iter_1_values, width, 
               label='Iter-1', color=colors['iter1'], 
               edgecolor='none', zorder=3)

# Handle iter-2 with None values
iter_2_plot = []
iter_2_positions = []
for i, val in enumerate(iter_2_values):
    if val is not None:
        iter_2_plot.append(val)
        iter_2_positions.append(x_wide[i] + 1.5*width)

bars4 = ax.bar(iter_2_positions, iter_2_plot, width, 
               label='Iter-2', color=colors['iter2'], 
               edgecolor='none', zorder=3)

# Add value labels on top of bars
def add_value_labels(bars, values):
    for bar, value in zip(bars, values):
        if value is not None and not np.isnan(value):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.2,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontsize=9.5, fontweight='normal')

add_value_labels(bars1, base_values)
add_value_labels(bars2, iter_0_values)
add_value_labels(bars3, iter_1_values)

# Add labels for iter-2
for pos, val in zip(iter_2_positions, iter_2_plot):
    ax.text(pos, val + 1.2, f'{val:.2f}',
            ha='center', va='bottom', fontsize=9.5, fontweight='normal')

# Styling improvements
ax.set_xlabel('Task Category', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Performance Across Training Iterations', 
             fontsize=16, fontweight='bold', pad=15)

# Set x-axis with wider spacing
ax.set_xticks(x_wide)
ax.set_xticklabels(categories, fontweight='bold', fontsize=12)

# Make y-axis tick labels bold
ax.tick_params(axis='y', labelsize=11)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# Add subtle grid for better readability (matching reference)
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.6, axis='y', zorder=0, color='gray')
ax.set_axisbelow(True)

# Set background color to light gray (matching reference)
ax.set_facecolor('#F5F5F5')
fig.patch.set_facecolor('white')

# Improve legend
legend = ax.legend(loc='upper left', frameon=True, 
                   fancybox=False, shadow=False,
                   edgecolor='lightgray', framealpha=1.0,
                   ncol=4, columnspacing=1.8)
legend.get_frame().set_linewidth(1.0)

# Set y-axis range with some padding
ax.set_ylim(0, 65)

# Improve spines - lighter like reference
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color('#CCCCCC')

# Adjust x-axis limits for better spacing
ax.set_xlim(-0.5, x_wide[-1] + 0.5)

# Add minor ticks on y-axis
ax.tick_params(which='major', length=5, width=1.0, direction='out', color='#CCCCCC')
ax.tick_params(which='minor', length=0, width=0)

# Tight layout
plt.tight_layout()

# Save with high DPI for publication
plt.savefig('/volume/pt-train/users/wzhang/ghchen/zh/code/Data-Selection/plot/result/styled_bar_chart_3.pdf', 
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('/volume/pt-train/users/wzhang/ghchen/zh/code/Data-Selection/plot/result/styled_bar_chart_3.png', 
            dpi=300, bbox_inches='tight', format='png')

print("Styled bar chart saved successfully!")
plt.show()

