#!/bin/bash
set -e
# 环境：
# # 创建并激活环境
# conda create -n BFCL python=3.10 -y && conda activate BFCL

# # 克隆并进入项目
# git clone https://github.com/ShishirPatil/gorilla.git
# cd gorilla/berkeley-function-call-leaderboard
# pip install -e .[oss_eval_sglang]
# pip install -e .[oss_eval_vllm]
# pip install bfcl-eval
# 安装核心包
# pip install -e .

# 使用:
# simple, parallel, multiple, live_data, multi_turn。

# # 示例：使用 sglang 运行多轮对话评估
# bfcl generate --model <模型名> --test-category multi_turn --backend sglang
# bfcl evaluate --model <模型名> --test-category multi_turn
#文件名,核心指标
# data_overall.csv,总榜单：各模型综合准确率。
# data_multi_turn.csv,Agent能力：多轮对话中的连贯性。
# data_live.csv,真实智商：防刷榜（未见过的 API）得分。
# data_non_live.csv,基础能力：标准静态任务得分。




# 实验标识
# experiment_prefix="qwen3-8b_bfcl_eval"
experiment_prefix="all_model_bfcl_eval"
timestamp=$(date +"%Y%m%d_%H%M%S")
# BFCL 模型名（逻辑名，不影响本地权重）
model="Qwen/Qwen3-8B"
experiment_name="${experiment_prefix}_${timestamp}"

# ⚠️【关键】保存模型的父目录 例如：saves/qwen3_sft_xxx
MODEL_ROOT="/home/aiscuser/FANNO-Tool-Dev/LLaMA-Factory/saves/sft_tools"
category="single_turn,multi_turn"
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
GPU_UTIL=0.7

RESULT_BASE="results/$experiment_name"
GEN_BASE="$RESULT_BASE/generation"
EVAL_BASE="$RESULT_BASE/evaluation"

mkdir -p "$GEN_BASE"
mkdir -p "$EVAL_BASE"

# 备份脚本本身
cp "$0" "$RESULT_BASE/run_all_models.sh"

############################
# 遍历模型目录
############################

echo "🚀 Start BFCL evaluation for all models in: $MODEL_ROOT"
echo "Results will be saved to: $RESULT_BASE"
echo

for MODEL_DIR in "$MODEL_ROOT"/*; do
  # 只处理目录
  [ -d "$MODEL_DIR" ] || continue

  MODEL_NAME=$(basename "$MODEL_DIR")
  echo "=============================="
  echo "▶ Testing model: $MODEL_NAME"
  echo "▶ Model path: $MODEL_DIR"

  MODEL_GEN_DIR="$GEN_BASE/$MODEL_NAME"
  MODEL_EVAL_DIR="$EVAL_BASE/$MODEL_NAME"

  mkdir -p "$MODEL_GEN_DIR"
  mkdir -p "$MODEL_EVAL_DIR"

  ############################
  # 生成
  ############################
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bfcl generate \
    --model "$model" \
    --test-category "$category" \
    --backend vllm \
    --num-gpus $NUM_GPUS \
    --gpu-memory-utilization $GPU_UTIL \
    --local-model-path "$MODEL_DIR" \
    --result-dir "$MODEL_GEN_DIR"

  if [ $? -ne 0 ]; then
    echo "❌ Generation failed for $MODEL_NAME, skip evaluation"
    continue
  fi

  ############################
  # 评估
  ############################
  bfcl evaluate \
    --model "$model" \
    --test-category "$category" \
    --result-dir "$MODEL_GEN_DIR" \
    --score-dir "$MODEL_EVAL_DIR"

  if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed for $MODEL_NAME"
    continue
  fi

  echo "✅ Finished model: $MODEL_NAME"
  echo
done

echo "🎉 All models evaluated!"
echo "📁 Results saved at: $RESULT_BASE"
