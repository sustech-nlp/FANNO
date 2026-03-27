#!/bin/bash
set -e

# ==============================================================================
# 环境参考 (需提前手动执行):
# conda create -n tau_bench python=3.10 -y && conda activate tau_bench
# git clone https://github.com/sierra-research/tau-bench && cd tau-bench
# pip install -e .
# pip install vllm  # 脚本依赖 vllm 启动服务
# export OPENAI_API_KEY="sk-4FsGXuXXiyoBFaFg49E44c2c0eF34c50B6669dC4613d9177" # 运行评估通常需要一个强模型作为 User Simulator
# ==============================================================================

# 实验标识
timestamp=$(date +"%Y%m%d_%H%M%S")
experiment_name="all_model_tau_bench_eval"

# ⚠️【关键】本地模型路径列表
MODEL_PATHS=(
  "/mnt/msranlphot_intern/zhuhe/models/Qwen3-0.6B"
#   "/mnt/msranlphot_intern/zhuhe/models/Qwen3-1.7B"
#   "/mnt/msranlphot_intern/zhuhe/models/Qwen3-4B"
#   "/mnt/msranlphot_intern/zhuhe/models/Qwen3-8B"
#   "/mnt/msranlphot_intern/zhuhe/models/Qwen3-14B"
#   "/mnt/msranlphot_intern/zhuhe/models/Qwen3-32B"
)

# 评估配置
ENVIRONMENTS="retail airline"   # 测试领域：零售、航空
AGENT_STRATEGY="tool-calling"  # Agent 策略
USER_MODEL="gpt-4o"            # 用户模拟器模型（建议用强模型以保证测试公平性）
USER_STRATEGY="llm"            # 用户模拟策略：llm, react, verify, reflection
MAX_CONCURRENCY=10             # 并发数

export CUDA_VISIBLE_DEVICES=0,1,2,3

# 结果存储路径
RESULT_BASE="results/$experiment_name"
mkdir -p "$RESULT_BASE"

# 备份脚本
cp "$0" "$RESULT_BASE/run_tau_bench.sh"

echo "🚀 Start Tau-Bench evaluation for listed models"
echo "Results will be saved to: $RESULT_BASE"
echo "User Simulator: $USER_MODEL ($USER_STRATEGY)"
echo

############################
# 遍历模型目录
############################

for MODEL_DIR in "${MODEL_PATHS[@]}"; do
  # 只处理存在的目录
  if [ ! -d "$MODEL_DIR" ]; then
    echo "⚠️ Skip missing dir: $MODEL_DIR"
    continue
  fi

  MODEL_NAME=$(basename "$MODEL_DIR")
  echo "=============================="
  echo "▶ Testing model: $MODEL_NAME"
  
  MODEL_RESULT_DIR="$RESULT_BASE/$MODEL_NAME"
  mkdir -p "$MODEL_RESULT_DIR"

  for ENV in $ENVIRONMENTS; do
    echo "--- Running Environment: $ENV ---"
    
    # 1. 运行 Benchmark 生成轨迹 (Trajectories)
    # 注意：tau-bench 的 run.py 可能需要根据你的推理框架（如 vLLM）做简单适配
    # 这里假设你使用其提供的标准接口，通过 --model 传递本地路径
    python run.py \
      --env "$ENV" \
      --model "$MODEL_DIR" \
      --model-provider "openai" \
      --agent-strategy "$AGENT_STRATEGY" \
      --user-model "$USER_MODEL" \
      --user-model-provider "openai" \
      --user-strategy "$USER_STRATEGY" \
      --max-concurrency "$MAX_CONCURRENCY" \
      --output-path "$MODEL_RESULT_DIR/${ENV}_results.json"

    if [ $? -ne 0 ]; then
      echo "❌ Run failed for $MODEL_NAME in $ENV"
      continue
    fi

    # 2. 自动错误识别 (Auto Error Identification)
    # 这一步会分析失败的会话，判断是 Agent、User 还是环境的问题
    echo "--- Starting Auto Error Identification for $ENV ---"
    python auto_error_identification.py \
      --env "$ENV" \
      --platform "openai" \
      --results-path "$MODEL_RESULT_DIR/${ENV}_results.json" \
      --output-path "$MODEL_RESULT_DIR/${ENV}_error_analysis" \
      --max-concurrency "$MAX_CONCURRENCY"

    echo "✅ Finished $ENV for $MODEL_NAME"
  done

  echo "🏁 Completed model: $MODEL_NAME"
  echo
done

echo "🎉 All models in Tau-Bench evaluated!"
echo "📁 Final Results: $RESULT_BASE"


