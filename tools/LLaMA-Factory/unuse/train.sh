# # CUDA_VISIBLE_DEVICES=5 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_apigen.yaml


# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_xlam.yaml


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_tool.yaml


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen3_full_sft_tool.yaml


#!/bin/bash

# GPU 设置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FORCE_TORCHRUN=1

# 模型和数据集基础配置
# MODEL_PATH="/volume/pt-train/users/wzhang/ghchen/zh/models/Qwen3-8B"

MODEL_PATH="/mnt/msranlphot_intern/zhuhe/models/Meta-Llama-3.1-8B-Instruct"
DATASET="tools_sft"
TEMPLATE="llama3"
YAML_BASE="train_config"

# 参数网格
TOTAL_BATCHS=(16 32 64 128)          # 总 batch size
PER_DEVICE_BATCH=4               # 每卡 batch size
LEARNING_RATES=(5e-6 1e-5 2e-5 5e-5)
EPOCHS=(2 4)

# 遍历网格
for TOTAL_BATCH in "${TOTAL_BATCHS[@]}"; do
  for LR in "${LEARNING_RATES[@]}"; do
    for EPOCH in "${EPOCHS[@]}"; do

      # 自动生成唯一输出目录和 YAML 文件名
      OUTPUT_DIR="saves/qwen3_sft_tb${TOTAL_BATCH}_lr${LR}_ep${EPOCH}"
      YAML_FILE="${YAML_BASE}_tb${TOTAL_BATCH}_lr${LR}_ep${EPOCH}.yaml"

      # 调用 Python 脚本生成 YAML 并训练
      python train.py \
        --model_path $MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --total_batch_size $TOTAL_BATCH \
        --per_device_batch_size $PER_DEVICE_BATCH \
        --dataset $DATASET \
        --template $TEMPLATE \
        --learning_rate $LR \
        --num_epochs $EPOCH \
        --yaml_path $YAML_FILE \
        --cuda_devices "0,1,2,3"

    done
  done
done
