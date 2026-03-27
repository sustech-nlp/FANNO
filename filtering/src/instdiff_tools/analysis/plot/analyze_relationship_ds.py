import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from matplotlib.colors import LinearSegmentedColormap

# =====================
# 参数区
# =====================
folder_path = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/qwen2_5_7b__math_math-1k"
jsonl_path = f"{folder_path}/compared_complexity.jsonl"
output_dir = f"{folder_path}/correlation_outputs_acl_ds2"

os.makedirs(output_dir, exist_ok=True)

# =====================
# 设置与plot_npmi_heatmap一致的全局样式
# =====================
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica Neue']

# =====================
# 读取 jsonl
# =====================
records = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading jsonl"):
        obj = json.loads(line)

        inst_len = obj.get("instruct_token_lens", None)
        resp_len = obj.get("response_token_lens", None)

        # 基本过滤
        if inst_len is None or resp_len is None:
            continue
        if inst_len == 0 or resp_len == 0:
            continue

        records.append({
            # diff signals
            "Base NLL": obj.get("base_nll"),
            "Calib NLL": obj.get("instruct_nll"),
            "Diff NLL": obj.get("diff_nll"),
            "Base Entropy": obj.get("base_avg_entropy"),
            "Calib Entropy": obj.get("instruct_avg_entropy"),
            "Diff Entropy": obj.get("diff_entropy"),
            
            # length
            "Inst Len": inst_len,
            "Resp Len": resp_len,
            
            # ratio
            "Resp/Inst": resp_len / inst_len,
            "Inst/Resp": inst_len / resp_len,
            
            # complexity
            "Complexity": obj.get("complexity"),
        })

df = pd.DataFrame(records)
print(f"Loaded samples: {len(df)}")

# =====================
# 清洗数据
# =====================
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print(f"Samples after cleaning: {len(df)}")

# =====================
# 相关性计算
# =====================
pearson_corr = df.corr(method="pearson")
spearman_corr = df.corr(method="spearman")

pearson_corr.to_csv(os.path.join(output_dir, "pearson_correlation.csv"))
spearman_corr.to_csv(os.path.join(output_dir, "spearman_correlation.csv"))

# =====================
# 创建与plot_npmi_heatmap风格一致的热力图
# =====================
def create_npmi_style_heatmap(corr_matrix, title, filename, cmap_style="diverging"):
    """
    创建与plot_npmi_heatmap风格一致的相关性热力图
    """
    # 创建图形 - 使用较大的尺寸
    plt.figure(figsize=(16, 14))
    
    # 选择颜色映射 - 使用与plot_npmi_heatmap相似的颜色方案
    if cmap_style == "diverging":
        # 使用seaborn的diverging调色板，类似于plot_npmi_heatmap
        cmap = sns.diverging_palette(240, 10, as_cmap=True)  # 蓝色到红色
    elif cmap_style == "coolwarm":
        cmap = plt.cm.coolwarm
    else:
        cmap = plt.cm.RdBu_r
    
    # 创建热力图
    ax = sns.heatmap(
        corr_matrix,
        annot=True,  # 显示数值
        fmt=".2f",   # 保留两位小数
        cmap=cmap,
        center=0,    # 以0为中心
        linewidths=0.5,  # 单元格之间的线条宽度
        linecolor='white',  # 线条颜色
        square=True,  # 保持单元格为正方形
        cbar_kws={
            "shrink": 0.8,
            "label": "Correlation Coefficient",
            "extend": "both"
        }
    )
    
    # 调整颜色条 - 与plot_npmi_heatmap一致
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel("Correlation Coefficient", fontsize=22, weight="bold")
    
    # 设置标题 - 加粗
    plt.title(title, fontsize=26, weight="bold", pad=20)
    
    # 设置坐标轴标签 - 旋转并调整大小，与plot_npmi_heatmap一致
    plt.xticks(
        fontsize=22, 
        rotation=45, 
        ha="right",
        weight="bold"
    )
    plt.yticks(
        fontsize=22,
        rotation=0,
        weight="bold"
    )
    
    # 设置x轴和y轴标签
    plt.xlabel("Features", fontsize=24, weight="bold", labelpad=15)
    plt.ylabel("Features", fontsize=24, weight="bold", labelpad=15)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()
    
    print(f"Saved: {filename}")

# =====================
# 创建简化的热力图（不显示数值，更清晰）
# =====================
def create_simple_npmi_heatmap(corr_matrix, title, filename):
    """
    创建简化版热力图，不显示数值，更专注于模式识别
    """
    plt.figure(figsize=(14, 12))
    
    # 使用与plot_npmi_heatmap相同的颜色映射
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # 创建热力图，不显示数值
    ax = sns.heatmap(
        corr_matrix,
        annot=False,  # 不显示数值
        cmap=cmap,
        center=0,
        linewidths=0.5,
        linecolor='white',
        square=True,
        cbar_kws={
            "shrink": 0.8,
            "label": "Correlation",
            "extend": "both"
        }
    )
    
    # 调整颜色条
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel("Correlation", fontsize=20, weight="bold")
    
    # 设置标题
    plt.title(title, fontsize=24, weight="bold", pad=20)
    
    # 设置坐标轴标签
    plt.xticks(fontsize=20, rotation=45, ha="right", weight="bold")
    plt.yticks(fontsize=20, rotation=0, weight="bold")
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    
    print(f"Saved: {filename}")

# =====================
# 创建下三角热力图（避免重复）
# =====================
def create_triangular_heatmap(corr_matrix, title, filename):
    """
    创建下三角热力图，避免重复显示
    """
    plt.figure(figsize=(14, 12))
    
    # 创建mask隐藏上三角
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # 使用与plot_npmi_heatmap相同的颜色映射
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # 创建热力图
    ax = sns.heatmap(
        corr_matrix,
        mask=mask,  # 应用mask
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        linecolor='white',
        square=True,
        cbar_kws={
            "shrink": 0.8,
            "label": "Correlation",
            "extend": "both"
        }
    )
    
    # 调整颜色条
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel("Correlation", fontsize=20, weight="bold")
    
    # 设置标题
    plt.title(title, fontsize=24, weight="bold", pad=20)
    
    # 设置坐标轴标签
    plt.xticks(fontsize=20, rotation=45, ha="right", weight="bold")
    plt.yticks(fontsize=20, rotation=0, weight="bold")
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    
    print(f"Saved: {filename}")

# =====================
# 创建高相关性热力图（只显示强相关）
# =====================
def create_high_correlation_heatmap(corr_matrix, title, filename, threshold=0.5):
    """
    创建只显示高相关性的热力图
    """
    plt.figure(figsize=(14, 12))
    
    # 创建只显示高相关性的mask
    high_corr_mask = np.abs(corr_matrix) < threshold
    high_corr_mask = high_corr_mask & ~np.eye(len(corr_matrix), dtype=bool)  # 保留对角线
    
    # 使用与plot_npmi_heatmap相同的颜色映射
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # 创建热力图
    ax = sns.heatmap(
        corr_matrix,
        mask=high_corr_mask,  # 应用mask
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        linecolor='white',
        square=True,
        cbar_kws={
            "shrink": 0.8,
            "label": "Correlation",
            "extend": "both"
        }
    )
    
    # 调整颜色条
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel("Correlation", fontsize=20, weight="bold")
    
    # 设置标题
    plt.title(f"{title}\n(|Correlation| ≥ {threshold})", fontsize=24, weight="bold", pad=20)
    
    # 设置坐标轴标签
    plt.xticks(fontsize=20, rotation=45, ha="right", weight="bold")
    plt.yticks(fontsize=20, rotation=0, weight="bold")
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    
    print(f"Saved: {filename}")

# =====================
# 生成所有热力图
# =====================
print("\n" + "="*60)
print("Creating NPMI-style Correlation Heatmaps")
print("="*60)

# 1. 完整Pearson相关性热力图（带数值）
create_npmi_style_heatmap(
    pearson_corr,
    "Pearson Correlation Matrix",
    "heatmap_pearson_full.png",
    cmap_style="diverging"
)

# 2. 简化版Pearson相关性热力图（不带数值）
create_simple_npmi_heatmap(
    pearson_corr,
    "Pearson Correlation Matrix",
    "heatmap_pearson_simple.png"
)

# 3. 下三角Pearson相关性热力图
create_triangular_heatmap(
    pearson_corr,
    "Pearson Correlation Matrix (Lower Triangle)",
    "heatmap_pearson_triangular.png"
)

# 4. 高相关性热力图（|r| ≥ 0.5）
create_high_correlation_heatmap(
    pearson_corr,
    "High Pearson Correlations",
    "heatmap_pearson_high_corr.png",
    threshold=0.5
)

# 5. Spearman相关性热力图
create_npmi_style_heatmap(
    spearman_corr,
    "Spearman Correlation Matrix",
    "heatmap_spearman_full.png",
    cmap_style="diverging"
)

# =====================
# 创建与Complexity相关的子热力图
# =====================
def create_complexity_focused_heatmap(corr_matrix, title, filename):
    """
    创建专注于Complexity相关性的热力图
    """
    # 选择与Complexity最相关的特征
    complexity_corrs = corr_matrix["Complexity"].sort_values(key=abs, ascending=False)
    top_features = complexity_corrs.index[:8].tolist()  # 选择前8个特征
    
    # 提取子矩阵
    sub_matrix = corr_matrix.loc[top_features, top_features]
    
    plt.figure(figsize=(12, 10))
    
    # 使用与plot_npmi_heatmap相同的颜色映射
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # 创建热力图
    ax = sns.heatmap(
        sub_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        linecolor='white',
        square=True,
        cbar_kws={
            "shrink": 0.8,
            "label": "Correlation",
            "extend": "both"
        }
    )
    
    # 调整颜色条
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel("Correlation", fontsize=18, weight="bold")
    
    # 设置标题
    plt.title(title, fontsize=22, weight="bold", pad=20)
    
    # 设置坐标轴标签
    plt.xticks(fontsize=18, rotation=45, ha="right", weight="bold")
    plt.yticks(fontsize=18, rotation=0, weight="bold")
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    
    print(f"Saved: {filename}")

# 6. Complexity相关特征热力图
create_complexity_focused_heatmap(
    pearson_corr,
    "Correlations with Complexity",
    "heatmap_complexity_focused.png"
)

# =====================
# 创建统计摘要
# =====================
def create_correlation_summary(corr_matrix, df, filename):
    """
    创建相关性统计摘要
    """
    summary_path = os.path.join(output_dir, filename)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("CORRELATION ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Dataset Information:\n")
        f.write(f"- Total samples: {len(df):,}\n")
        f.write(f"- Number of features: {len(df.columns)}\n")
        f.write(f"- Features analyzed: {', '.join(df.columns)}\n\n")
        
        # 强相关性分析
        f.write("Strong Correlations (|r| > 0.7):\n")
        f.write("-"*40 + "\n")
        
        strong_corrs = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_corrs.append((abs(corr_value), corr_value, 
                                         corr_matrix.index[i], corr_matrix.columns[j]))
        
        if strong_corrs:
            strong_corrs.sort(reverse=True)
            for abs_corr, corr, feat1, feat2 in strong_corrs:
                direction = "positive" if corr > 0 else "negative"
                f.write(f"{feat1} & {feat2}: r = {corr:.3f} ({direction})\n")
        else:
            f.write("No strong correlations found (|r| > 0.7)\n")
        
        f.write("\n")
        
        # Complexity相关性分析
        if "Complexity" in corr_matrix.index:
            f.write("Correlations with Complexity:\n")
            f.write("-"*40 + "\n")
            
            complexity_corrs = []
            for feat in corr_matrix.columns:
                if feat != "Complexity":
                    corr_value = corr_matrix.loc["Complexity", feat]
                    complexity_corrs.append((abs(corr_value), corr_value, feat))
            
            complexity_corrs.sort(reverse=True)
            
            for abs_corr, corr, feat in complexity_corrs:
                direction = "positive" if corr > 0 else "negative"
                f.write(f"{feat}: r = {corr:.3f} ({direction})\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"Summary saved: {filename}")

# 创建统计摘要
create_correlation_summary(pearson_corr, df, "correlation_summary.txt")

# =====================
# 完成提示
# =====================
print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
print("  • Correlation matrices (CSV): pearson_correlation.csv, spearman_correlation.csv")
print("  • Heatmaps (PNG):")
print("     1. heatmap_pearson_full.png - 完整Pearson相关性热力图")
print("     2. heatmap_pearson_simple.png - 简化版Pearson相关性热力图")
print("     3. heatmap_pearson_triangular.png - 下三角Pearson相关性热力图")
print("     4. heatmap_pearson_high_corr.png - 高相关性热力图")
print("     5. heatmap_spearman_full.png - Spearman相关性热力图")
print("     6. heatmap_complexity_focused.png - Complexity相关特征热力图")
print("  • Summary report: correlation_summary.txt")
print("\nAll visualizations use the same style as plot_npmi_heatmap.")
print("="*60)