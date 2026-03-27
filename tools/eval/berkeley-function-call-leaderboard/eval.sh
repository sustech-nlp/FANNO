# # CUDA_VISIBLE_DEVICES=0 bfcl generate \
# #   --model meta-llama/Llama-3.1-{8B,70B}-Instruct-FC \
# #   --test-category simple_python,parallel,live_multiple,multi_turn \
# #   --backend vllm \
# #   --num-gpus 1 \
# #   --gpu-memory-utilization 0.9 \
# #   --local-model-path /volume/pt-train/users/wzhang/ghchen/zh/code/Jy/LLaMA-Factory/saves/llama3__1-8b-Instruct-apigen-5k/full/sft


# # CUDA_VISIBLE_DEVICES=1,2,3,5 bfcl generate \
# #   --model meta-llama/Llama-3.1-8B-Instruct-FC \
# #   --test-category single_turn,multi_turn \
# #   --backend vllm \
# #   --num-gpus 4 \
# #   --gpu-memory-utilization 0.9 \
# #   --local-model-path /volume/pt-train/users/wzhang/ghchen/zh/models/LLM-Research/Meta-Llama-3___1-8B-Instruct

# # TODO: setup experiment_name and parameters

# CUDA_VISIBLE_DEVICES=3,6 bfcl generate \
#   --model Qwen/Qwen3-8B-FC \
#   --test-category single_turn,multi_turn \
#   --backend vllm \
#   --num-gpus 2 \
#   --gpu-memory-utilization 0.7 \
#   --local-model-path /volume/pt-train/users/wzhang/ghchen/zh/code/Jy/LLaMA-Factory/saves/qwen3-8b-tools_sft1k

# bfcl evaluate --model Qwen/Qwen3-8B-FC --test-category single_turn,multi_turn

# # TODO: move result and sorce dir to results/experiment_name


# CUDA_VISIBLE_DEVICES=1,7 bfcl generate \
#   --model meta-llama/Llama-3.1-8B-Instruct-FC \
#   --test-category single_turn,multi_turn \
#   --backend vllm \
#   --num-gpus 2 \
#   --gpu-memory-utilization 0.7 \
#   --local-model-path /volume/pt-train/users/wzhang/ghchen/zh/models/LLM-Research/Meta-Llama-3___1-8B-Instruct


# CUDA_VISIBLE_DEVICES=3 bfcl generate \
#   --model meta-llama/Llama-3.1-8B-Instruct-FC \
#   --test-category single_turn,multi_turn \
#   --backend vllm \
#   --num-gpus 1 \
#   --gpu-memory-utilization 0.8 \
#   --local-model-path /volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ms-swift/output/llama3__1_8b_instruct_1k_toucan3/v0-20251218-135148/checkpoint-4

#   CUDA_VISIBLE_DEVICES=0 bfcl generate \
#   --model meta-llama/Llama-3.1-8B-Instruct \
#   --test-category single_turn,multi_turn \
#   --backend vllm \
#   --num-gpus 1 \
#   --gpu-memory-utilization 0.9 \
#   --local-model-path /volume/pt-train/users/wzhang/ghchen/zh/models/LLM-Research/Meta-Llama-3___1-8B-Instruct


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bfcl generate \
#   --model Qwen/Qwen2.5-7B-Instruct-FC \
#   --test-category single_turn,multi_turn \
#   --backend vllm \
#   --num-gpus 8 \
#   --gpu-memory-utilization 0.9 \
#   --local-model-path /volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ms-swift/output/v2-20251217-192738/checkpoint-32


# bfcl evaluate --model meta-llama/Llama-3.1-8B-Instruct-FC --test-category single_turn,multi_turn

# bfcl evaluate --model meta-llama/Llama-3.1-8B-Instruct --test-category single_turn,multi_turn

# bfcl evaluate --model Qwen/Qwen3-8B-FC --test-category single_turn,multi_turn


#!/bin/bash
# 设置实验名称（包含模型标识、日期时间戳）
experiment_name="qwen3-8b_20251219_214722"
model="Qwen/Qwen3-8B-FC"
model_path="/volume/pt-train/users/wzhang/ghchen/zh/models/Qwen3-8B"
category="single_turn,multi_turn"

# 创建结果目录结构
result_base="results/$experiment_name"
mkdir -p "$result_base/generation"
mkdir -p "$result_base/evaluation"

# 保存实验配置（复制当前脚本）
cp "$0" "$result_base/experiment_script.sh"

# 生成预测结果 - 直接指定输出目录
CUDA_VISIBLE_DEVICES=3,6 bfcl generate \
  --model "$model" \
  --test-category "$category" \
  --backend vllm \
  --num-gpus 2 \
  --gpu-memory-utilization 0.7 \
  --local-model-path "$model_path" \
  --result-dir "$result_base/generation"  # 关键参数：指定生成结果目录

# 检查生成步骤是否成功
if [ $? -ne 0 ]; then
  echo "Error: Generation step failed"
  exit 1
fi

# 执行评估 - 指定输入结果和输出评分目录
bfcl evaluate \
  --model "$model" \
  --test-category "$category" \
  --result-dir "$result_base/generation" \
  --score-dir "$result_base/evaluation"     # 关键参数：指定评估结果目录

# 检查评估步骤是否成功
if [ $? -ne 0 ]; then
  echo "Error: Evaluation step failed"
  exit 1
fi

echo "实验完成! 结果保存在: $result_base"
echo "生成结果: $result_base/generation"
echo "评估结果: $result_base/evaluation"
echo "实验配置: $result_base/experiment_script.sh"