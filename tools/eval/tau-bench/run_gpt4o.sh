#!/bin/bash
set -e

# 实验标识
timestamp=$(date +"%Y%m%d_%H%M%S")
experiment_name="gpt4o_tau_bench_eval_$timestamp"

# 模型列表
MODEL_NAMES=(
  "gpt-4o-mini"
  "gpt-4o"
)

# 评估配置
ENVIRONMENTS="retail airline"
AGENT_STRATEGY="tool-calling"
USER_MODEL="gpt-4o"
USER_STRATEGY="llm"
MAX_CONCURRENCY=5  # 如果是 Tier 1 账号，建议设为 2

# 结果存储根路径
RESULT_BASE="results/$experiment_name"
mkdir -p "$RESULT_BASE"

echo "🚀 Start Tau-Bench evaluation"
echo "Results will be saved to: $RESULT_BASE"

for M_NAME in "${MODEL_NAMES[@]}"; do
  echo "=============================="
  echo "▶ Testing model: $M_NAME"
  
  MODEL_LOG_DIR="$RESULT_BASE/$M_NAME"
  mkdir -p "$MODEL_LOG_DIR"

  for ENV in $ENVIRONMENTS; do
    echo "--- Running Environment: $ENV ---"
    
    # 【关键修正】使用小写的 openai
    python run.py \
      --env "$ENV" \
      --model "$M_NAME" \
      --model-provider "openai" \
      --agent-strategy "$AGENT_STRATEGY" \
      --user-model "$USER_MODEL" \
      --user-model-provider "openai" \
      --user-strategy "$USER_STRATEGY" \
      --max-concurrency "$MAX_CONCURRENCY" \
      --log-dir "$MODEL_LOG_DIR"

    if [ $? -ne 0 ]; then
      echo "❌ Run failed for $M_NAME in $ENV"
      continue
    fi

    # 2. 自动错误识别
    echo "--- Starting Auto Error Identification for $ENV ---"
    
    # 查找刚刚生成的 json 结果文件
    # tau-bench 通常生成类似 retail_gpt-4o-mini_...json 的文件
    TRAJ_FILE=$(ls "$MODEL_LOG_DIR" | grep "${ENV}" | grep ".json" | head -n 1)
    
    if [ -n "$TRAJ_FILE" ]; then
        echo "Analyzing trajectory: $TRAJ_FILE"
        python auto_error_identification.py \
          --env "$ENV" \
          --model "$USER_MODEL" \
          --model-provider "openai" \
          --results-path "$MODEL_LOG_DIR/$TRAJ_FILE" \
          --output-path "$MODEL_LOG_DIR/${ENV}_error_analysis.json"
    else
        echo "⚠️ No trajectory file found in $MODEL_LOG_DIR to analyze."
    fi

    echo "✅ Finished $ENV for $M_NAME"
  done
  echo "🏁 Completed model: $M_NAME"
done

echo "🎉 Evaluation Complete!"
echo "📁 Results in: $RESULT_BASE"