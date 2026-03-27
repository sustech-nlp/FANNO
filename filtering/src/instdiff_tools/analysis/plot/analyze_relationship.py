import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================
# 参数区
# =====================
folder_path = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/qwen2_5_7b__math_math-1k"
jsonl_path = f"{folder_path}/compared_complexity.jsonl"
output_dir = f"{folder_path}/correlation_outputs_acl"

os.makedirs(output_dir, exist_ok=True)

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
            "Base Nll": obj.get("base_nll"),
            "Calibration Nll": obj.get("instruct_nll"),
            "Diff Nll": obj.get("diff_nll"),
            # "diff_ppl": obj.get("diff_ppl"),
            "Base Entropy": obj.get("base_avg_entropy"),
            "Calibration Entropy": obj.get("instruct_avg_entropy"),
            "Diff Entropy": obj.get("diff_entropy"),

            # length
            "Inst lens": inst_len,
            "Resp lens": resp_len,

            # ratio
            "Resp/Inst": resp_len / inst_len,
            "Inst/Resp": inst_len / resp_len,

            # complexity
            "Complexity": obj.get("complexity"),

            # base / instruct
            # "base_ppl": obj.get("base_ppl"),
            # "Instruct Ppl": obj.get("instruct_ppl"),
            
            
            
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
# Heatmap 1: 全变量
# =====================
plt.figure(figsize=(14, 12))
sns.heatmap(
    pearson_corr,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    annot=False
)
plt.title("Pearson Correlation Heatmap (All Features)", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "heatmap_all_features.png"), dpi=300)
plt.close()
# =====================
# Heatmap 1（带数字）：全变量
# =====================
plt.figure(figsize=(16, 14))

sns.heatmap(
    pearson_corr,
    cmap="coolwarm",
    center=0,
    linewidths=0.4,
    square=False,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 8},
    cbar_kws={"shrink": 0.75}
)

plt.title("Pearson Correlation Heatmap (All Features)", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "heatmap_all_features_with_numbers.png"),
    dpi=300
)
plt.close()

# # =====================
# # Heatmap 2: 核心变量
# # =====================
# # focus_cols = [
# #     "diff_nll",
# #     "diff_ppl",
# #     "diff_entropy",
# #     "complexity",
# #     "instruct_token_lens",
# #     "response_token_lens",
# #     "resp_div_inst",
# #     "inst_div_resp",
# # ]
# focus_cols = [
#     "Diff Nll",
#     "Diff Entropy",
#     "complexity",
#     "Inst lens",
#     "Resp lens",
#     "resp/inst",
#     "inst/resp",
# ]


# focus_corr = df[focus_cols].corr(method="pearson")

# plt.figure(figsize=(10, 8))
# sns.heatmap(
#     focus_corr,
#     cmap="coolwarm",
#     center=0,
#     square=True,
#     linewidths=0.5,
#     annot=True,
#     fmt=".2f"
# )
# plt.title("Focused Correlation Heatmap", fontsize=15)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "heatmap_focus_features.png"), dpi=300)
# plt.close()

# # =====================
# # Heatmap 3: 按 complexity 分组
# # =====================
# for c in sorted(df["complexity"].unique()):
#     sub_df = df[df["complexity"] == c]

#     if len(sub_df) < 20:
#         print(f"Skip complexity={c}, too few samples ({len(sub_df)})")
#         continue

#     corr_c = sub_df[focus_cols].corr(method="pearson")

#     plt.figure(figsize=(9, 7))
#     sns.heatmap(
#         corr_c,
#         cmap="coolwarm",
#         center=0,
#         square=True,
#         linewidths=0.5,
#         annot=False
#     )
#     plt.title(f"Correlation Heatmap (complexity={c})", fontsize=14)
#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(output_dir, f"heatmap_complexity_{c}.png"),
#         dpi=300
#     )
#     plt.close()

# print("All correlation analysis and heatmaps are saved to:", output_dir)
