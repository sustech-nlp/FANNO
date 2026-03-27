# Diversity Comparison Experiment

指令数据多样化实验的核心代码：从嵌入提取、子集选择，到多样性指标与训练/评估挂钩。主入口 `python -m src.experiment` 可一键跑完多样化流程。

## 环境与安装
- 需要 `python>=3.10`，GPU 强烈推荐（梯度嵌入/Prismatic 更依赖算力）。
- 基础依赖：
  ```bash
  pip install torch transformers datasets scikit-learn pandas pyyaml sentence-transformers
  ```
- 可选加速：`faiss-cpu` 或 `faiss-gpu`（大规模距离/聚类），`python-igraph`（社区检测）。
- 数据：默认从 Hugging Face 拉取 `Magpie-Align/Magpie-Pro-300K-Filtered`；离线环境在配置里设置 `dataset.local_path` 指向本地 JSONL。

## 快速开始
1) 按需修改 `configs/experiment_config.yaml`（嵌入模型、选择策略、样本规模等）。
2) 运行全流程（会缓存嵌入到 `data/embeddings`，输出指标 CSV）：
   ```bash
   python -m src.experiment --config configs/experiment_config.yaml
   ```
   - 已有缓存时可加 `--skip-embeddings` 复用。
3) 结果查看：
   - 选择索引：`results/selections/*_indices.json`，同时有 `selection_timings.json`。
   - 多样性指标：`results/diversity_scores.csv`，对应耗时 `results/diversity_scores.times.json`。

## 配置要点（`configs/experiment_config.yaml`）
- `dataset`: 训练/测试规模、随机种子、`hf_id` 或 `local_path`。`n_select` 控制每个策略选取的样本数。
- `embedding_configs`: 多个命名嵌入配置；`type` 支持 `semantic` / `gradient` / `hybrid`，`model_id` 为 HF 权重或本地路径；`proj_dim`、`batch_size`、`max_length` 可调。
- `selection_strategies`: 逐个策略及参数（`random`、`kmeans`、`community`、`k_center_greedy_fast`、`novelsum_fast`、`prismatic` 等）。
- `metrics`: 多样性评估参数（参考嵌入键、采样规模、NovelSum 模式/样本数、KMeans 簇数等）。

## 目录结构
```
diversity/
├── configs/                 # YAML 配置
├── data/embeddings/         # 缓存的嵌入 (运行后生成)
├── results/
│   ├── selections/          # 选择索引与计时
│   └── diversity_scores.csv # 多样性指标 (运行后生成)
└── src/
    ├── embedding_extraction.py  # 语义/梯度/混合嵌入
    ├── selection_strategies.py  # Random/K-Means/Community/K-Center/NovelSum/Prismatic
    ├── diversity_metrics.py     # NovelSum/G-Vendi/Vendi/Cluster Inertia/等
    ├── training.py              # HF Trainer 微调入口
    └── evaluation.py            # 困惑度与自定义基准注册
```

## 扩展与提示
- 跑大规模前先用小样本（如 5k/1k）验证配置。
- 梯度嵌入与 Prismatic 计算量大，可多卡并行；`--skip-embeddings` 可避免重复算。
- `evaluation.py` 的 `register_benchmark` 可挂接 MT-Bench/AlpacaEval/IFEval/MMLU-Pro 等，`train_on_subset` 可在主脚本中插入以跑端到端闭环。
- `community_detect_select` 在缺少 `sentence-transformers` 或 `igraph` 时会回退到 O(n^2) BFS，数据特别大时可调高阈值或改用其他策略。
