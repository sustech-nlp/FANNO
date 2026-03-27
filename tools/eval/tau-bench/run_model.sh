#!/bin/bash
set -e

# ==============================================================================
# 配置区域
# ==============================================================================
experiment_name="qwen3_tau_bench_vllm"
MODEL_PATHS=(
  "/mnt/msranlphot_intern/zhuhe/models/Qwen3-7B"
  "/mnt/msranlphot_intern/zhuhe/models/Qwen3-14B"
)

# vLLM 服务配置
PORT=8000
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# Tau-Bench 配置
ENVIRONMENTS="retail airline"
USER_MODEL="gpt-4o"  # 建议保留一个强模型做 User Simulator
MAX_CONCURRENCY=8

RESULT_BASE="results/$experiment_name"
mkdir -p "$RESULT_BASE"

############################
# 核心逻辑：遍历模型
############################

for MODEL_DIR in "${MODEL_PATHS[@]}"; do
  if [ ! -d "$MODEL_DIR" ]; then continue; fi
  MODEL_NAME=$(basename "$MODEL_DIR")
  
  echo "======================================================"
  echo "🚀 正在启动 vLLM 服务: $MODEL_NAME"
  
  # 1. 后台启动 vLLM 服务
  # 使用 OpenAI 兼容接口，这样 tau-bench 就能通过 openai provider 访问它
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size "$NUM_GPUS" \
    --port "$PORT" \
    --gpu-memory-utilization 0.9 &
  
  VLLM_PID=$!

  # 2. 等待服务就绪
  echo "⏳ 等待 vLLM 服务就绪..."
  while ! curl -s "http://localhost:$PORT/v1/models" > /dev/null; do
    sleep 5
    if ! kill -0 $VLLM_PID 2>/dev/null; then
      echo "❌ vLLM 启动失败"
      exit 1
    fi
  done
  echo "✅ 服务已就绪"

  # 3. 运行 Tau-Bench
  # 关键：通过修改 API_BASE 指向本地 vLLM
  for ENV in $ENVIRONMENTS; do
    echo "▶ 正在评估环境: $ENV"
    
    # 我们设置 OPENAI_API_BASE 指向本地 vLLM
    # 设置 OPENAI_API_KEY 为任意值（vLLM 不需要真实 Key）
    # 注意：User Model 仍需真实的 OpenAI Key，所以这里直接在命令行覆盖环境变量
    
    OPENAI_API_BASE="http://localhost:$PORT/v1" \
    OPENAI_API_KEY="dummy" \
    python run.py \
      --env "$ENV" \
      --model "$MODEL_NAME" \
      --model-provider "openai" \
      --user-model "$USER_MODEL" \
      --user-model-provider "openai" \
      --max-concurrency "$MAX_CONCURRENCY" \
      --output-path "$RESULT_BASE/${MODEL_NAME}_${ENV}.json"
  done

  # 4. 运行错误识别 (使用真实 OpenAI，因为需要强模型分析报告)
  echo "🔍 正在进行自动错误分析..."
  python auto_error_identification.py \
    --env "retail" \
    --results-path "$RESULT_BASE/${MODEL_NAME}_retail.json" \
    --output-path "$RESULT_BASE/${MODEL_NAME}_analysis"

  # 5. 关闭当前 vLLM 服务，释放显存给下一个模型
  echo "🛑 停止 vLLM 服务 (PID: $VLLM_PID)"
  kill $VLLM_PID
  sleep 10
done

echo "🎉 所有模型评估完成！结果见: $RESULT_BASE"

