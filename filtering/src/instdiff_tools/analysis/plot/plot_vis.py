# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
# from matplotlib.patches import Rectangle

# # 创建图形 - 两个子图上下排列
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7))

# # ========== 第一个分布：完整的ΔNLL分布 ==========
# x1 = np.linspace(-3, 5, 1000)
# skewness1 = 3
# y1 = stats.skewnorm.pdf(x1, skewness1, loc=0.5, scale=1.2)

# # 计算分位点
# cdf1 = stats.skewnorm.cdf(x1, skewness1, loc=0.5, scale=1.2)
# left_idx1 = np.argmin(np.abs(cdf1 - 0.10))
# right_idx1 = np.argmin(np.abs(cdf1 - 0.90))

# # 绘制曲线
# ax1.plot(x1, y1, 'k-', linewidth=2, alpha=0.8)

# # 填充区域 - 使用非常浅的颜色
# # 左边α% - 浅红
# ax1.fill_between(x1[:left_idx1], 0, y1[:left_idx1], alpha=0.15, color='#ff6b6b')
# # 右边α% - 浅红
# ax1.fill_between(x1[right_idx1:], 0, y1[right_idx1:], alpha=0.15, color='#ff6b6b')
# # 中间部分 - 浅蓝
# ax1.fill_between(x1[left_idx1:right_idx1], 0, y1[left_idx1:right_idx1], alpha=0.2, color='#4dabf7')

# # 添加垂直虚线
# ax1.axvline(x1[left_idx1], color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
# ax1.axvline(x1[right_idx1], color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

# # 标注文字 - 调整位置避免和线重叠
# # 左侧Reject - 往左移，远离黑线
# ax1.text(x1[left_idx1]/2 - 0.5, np.max(y1)*0.25, r'$\mathbf{Reject}$' + '\n' + r'$\alpha$', 
#          ha='center', fontsize=11, color='#c92a2a', weight='bold')
# # 右侧Reject - 往右移，远离黑线
# ax1.text(x1[right_idx1] + (x1[-1]-x1[right_idx1])/2 + 0.5, np.max(y1)*0.25, r'$\mathbf{Reject}$' + '\n' + r'$\alpha$', 
#          ha='center', fontsize=11, color='#c92a2a', weight='bold')
# # Keep - 往左移动
# ax1.text((x1[left_idx1]+x1[right_idx1])/2 - 0.3, np.max(y1)*0.7, r'$\mathbf{Keep}$', 
#          ha='center', fontsize=13, color='#1971c2', weight='bold')

# # 标题在框内顶部
# ax1.text(0.5, 0.95, r'(1) $\Delta$NLL Bi-Filtering', transform=ax1.transAxes,
#          fontsize=13, weight='bold', ha='center', va='top')

# # 进一步减少左侧留白
# ax1.set_xlim(x1[0]+1.2, x1[-1])
# ax1.set_ylim(0, np.max(y1)*1.15)
# ax1.axis('off')

# # 添加框框
# rect1 = Rectangle((0, 0), 1, 1, transform=ax1.transAxes, 
#                   linewidth=2, edgecolor='black', facecolor='none')
# ax1.add_patch(rect1)

# # ========== 第二个分布：ΔEntropy（只显示被保留的80%部分）==========
# # 第二个分布只在第一个分布的中间80%范围内
# x2_start = x1[left_idx1]
# x2_end = x1[right_idx1]
# x2 = np.linspace(x2_start, x2_end, 1000)

# # 使用不同的skew参数，让分布看起来不一样
# skewness2 = -2  # 负的skew，向左倾斜
# y2 = stats.skewnorm.pdf(x2, skewness2, loc=(x2_start+x2_end)/2, scale=(x2_end-x2_start)/5)

# # 在这个新分布上计算bottom β%
# # 重新归一化cdf
# cdf2 = stats.skewnorm.cdf(x2, skewness2, loc=(x2_start+x2_end)/2, scale=(x2_end-x2_start)/5)
# # bottom β%的分界点
# bottom_idx2 = np.argmin(np.abs(cdf2 - 0.10))

# # 绘制曲线
# ax2.plot(x2, y2, 'k-', linewidth=2, alpha=0.8)

# # 填充区域
# # Bottom β% - 浅绿（目标选择）
# ax2.fill_between(x2[:bottom_idx2], 0, y2[:bottom_idx2], alpha=0.25, color='#51cf66')
# # Remaining - 浅蓝
# ax2.fill_between(x2[bottom_idx2:], 0, y2[bottom_idx2:], alpha=0.2, color='#4dabf7')

# # 添加垂直虚线
# ax2.axvline(x2[bottom_idx2], color='#2f9e44', linestyle='--', linewidth=1.5, alpha=0.6)

# # 标注文字 - 都往左移动
# # Select - 往左移动
# select_x = (x2[0]+x2[bottom_idx2])/2 - 0.05
# ax2.text(select_x, np.max(y2)*0.6, r'$\mathbf{Select}$' + '\n' + r'$\mathbf{Bottom}$' + ' ' + r'$\beta$', 
#          ha='center', fontsize=12, color='#2b8a3e', weight='bold')

# # Remaining - 往左移动
# remaining_x = (x2[bottom_idx2]+x2[-1])/2 - 0.4
# ax2.text(remaining_x, np.max(y2)*0.75, r'$\mathbf{Remaining}$', 
#          ha='center', fontsize=12, color='#1971c2', weight='bold')

# # 标题在框内顶部
# ax2.text(0.5, 0.95, r'(2) $\Delta$Entropy Selection', transform=ax2.transAxes,
#          fontsize=13, weight='bold', ha='center', va='top')

# # 减少右侧留白
# ax2.set_xlim(x2[0]-0.05, x2[-1]-0.6)
# ax2.set_ylim(0, np.max(y2)*1.15)
# ax2.axis('off')

# # 添加框框
# rect2 = Rectangle((0, 0), 1, 1, transform=ax2.transAxes, 
#                   linewidth=2, edgecolor='black', facecolor='none')
# ax2.add_patch(rect2)

# plt.tight_layout(pad=1.2)
# outdir = '/volume/pt-train/users/wzhang/ghchen/zh/code/Data-Selection/plot/result_vis'
# for ext in ['png', 'pdf']:
#     fname = f'{outdir}/paper_figure.{ext}'
#     plt.savefig(fname, dpi=300 if ext == 'png' else None, bbox_inches='tight', facecolor='white')
# print("Paper figure saved successfully!")





import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Rectangle

# 创建图形 - 两个子图上下排列
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7))

# ========== 第一个分布：完整的ΔNLL分布 ==========
x1 = np.linspace(-3, 5, 1000)
skewness1 = 3
y1 = stats.skewnorm.pdf(x1, skewness1, loc=0.5, scale=1.2)

# 计算分位点
cdf1 = stats.skewnorm.cdf(x1, skewness1, loc=0.5, scale=1.2)
left_idx1 = np.argmin(np.abs(cdf1 - 0.10))
right_idx1 = np.argmin(np.abs(cdf1 - 0.90))

# 绘制曲线
ax1.plot(x1, y1, 'k-', linewidth=2, alpha=0.8)

# 填充区域 - 使用非常浅的颜色
# 左边α% - 浅红
ax1.fill_between(x1[:left_idx1], 0, y1[:left_idx1], alpha=0.15, color='#ff6b6b')
# 右边α% - 浅红
ax1.fill_between(x1[right_idx1:], 0, y1[right_idx1:], alpha=0.15, color='#ff6b6b')
# 中间部分 - 浅蓝
ax1.fill_between(x1[left_idx1:right_idx1], 0, y1[left_idx1:right_idx1], alpha=0.2, color='#4dabf7')

# 添加垂直虚线
ax1.axvline(x1[left_idx1], color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
ax1.axvline(x1[right_idx1], color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

# 标注文字 - 调整位置避免和线重叠
# 左侧Reject - 往左移，远离黑线
ax1.text(x1[left_idx1]/2 - 0.5, np.max(y1)*0.25, r'$\mathbf{Reject}$' + '\n' + r'$\alpha$', 
         ha='center', fontsize=16, color='#c92a2a', weight='bold')
# 右侧Reject - 往右移，远离黑线
ax1.text(x1[right_idx1] + (x1[-1]-x1[right_idx1])/2 + 0.5, np.max(y1)*0.25, r'$\mathbf{Reject}$' + '\n' + r'$\alpha$', 
         ha='center', fontsize=16, color='#c92a2a', weight='bold')
# Keep - 往左移动, 并下移
ax1.text((x1[left_idx1]+x1[right_idx1])/2 - 0.3, np.max(y1)*0.48, r'$\mathbf{Keep}$', 
         ha='center', fontsize=19, color='#1971c2', weight='bold')

# 标题在框内顶部
ax1.text(0.5, 0.95, r'(1) $\Delta$NLL Bi-Filtering', transform=ax1.transAxes,
         fontsize=15, weight='bold', ha='center', va='top')

# 进一步减少左侧留白
ax1.set_xlim(x1[0]+1.2, x1[-1])
ax1.set_ylim(0, np.max(y1)*1.15)
ax1.axis('off')

# 添加框框
rect1 = Rectangle((0, 0), 1, 1, transform=ax1.transAxes, 
                  linewidth=2, edgecolor='black', facecolor='none')
ax1.add_patch(rect1)

# ========== 第二个分布：ΔEntropy（只显示被保留的80%部分）==========
# 第二个分布只在第一个分布的中间80%范围内
x2_start = x1[left_idx1]
x2_end = x1[right_idx1]
x2 = np.linspace(x2_start, x2_end, 1000)

# 使用不同的skew参数，让分布看起来不一样
skewness2 = -2  # 负的skew，向左倾斜
y2 = stats.skewnorm.pdf(x2, skewness2, loc=(x2_start+x2_end)/2, scale=(x2_end-x2_start)/5)

# 在这个新分布上计算bottom β%
# 重新归一化cdf
cdf2 = stats.skewnorm.cdf(x2, skewness2, loc=(x2_start+x2_end)/2, scale=(x2_end-x2_start)/5)
# bottom β%的分界点
bottom_idx2 = np.argmin(np.abs(cdf2 - 0.10))

# 绘制曲线
ax2.plot(x2, y2, 'k-', linewidth=2, alpha=0.8)

# 填充区域
# Bottom β% - 浅绿（目标选择）
ax2.fill_between(x2[:bottom_idx2], 0, y2[:bottom_idx2], alpha=0.25, color='#51cf66')
# Remaining - 浅蓝
ax2.fill_between(x2[bottom_idx2:], 0, y2[bottom_idx2:], alpha=0.2, color='#4dabf7')

# 添加垂直虚线
ax2.axvline(x2[bottom_idx2], color='#2f9e44', linestyle='--', linewidth=1.5, alpha=0.6)

# 标注文字 - 都往左移动，但是再往下移动
# Select - 往左移动，往下移动
select_x = (x2[0]+x2[bottom_idx2])/2 - 0.05
ax2.text(select_x, np.max(y2)*0.40, r'$\mathbf{Select}$' + '\n' + r'$\mathbf{Bottom}$' + ' ' + r'$\beta$', 
         ha='center', fontsize=17, color='#2b8a3e', weight='bold')

# Remaining - 往左移动，往下移动
remaining_x = (x2[bottom_idx2]+x2[-1])/2 - 0.4
ax2.text(remaining_x, np.max(y2)*0.54, r'$\mathbf{Remaining}$', 
         ha='center', fontsize=17, color='#1971c2', weight='bold')

# 标题在框内顶部
ax2.text(0.5, 0.95, r'(2) $\Delta$Entropy Selection', transform=ax2.transAxes,
         fontsize=15, weight='bold', ha='center', va='top')

# 减少右侧留白
ax2.set_xlim(x2[0]-0.05, x2[-1]-0.6)
ax2.set_ylim(0, np.max(y2)*1.15)
ax2.axis('off')

# 添加框框
rect2 = Rectangle((0, 0), 1, 1, transform=ax2.transAxes, 
                  linewidth=2, edgecolor='black', facecolor='none')
ax2.add_patch(rect2)

plt.tight_layout(pad=1.2)
outdir = '/volume/pt-train/users/wzhang/ghchen/zh/code/Data-Selection/plot/result_vis'
for ext in ['png', 'pdf']:
    fname = f'{outdir}/paper_figure.{ext}'
    plt.savefig(fname, dpi=300 if ext == 'png' else None, bbox_inches='tight', facecolor='white')
print("Paper figure saved successfully!")
