## start model 
# MODEL_ID=/mnt/msranlphot_intern/zhuhe/models/Qwen3-8B

# vllm serve $MODEL_ID \
#     --tensor-parallel-size 1 \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \
#     --host 0.0.0.0 \
#     --port 8000 


## start model 
MODEL_ID=/mnt/msranlphot_intern/zhuhe/models/Qwen3-8B

vllm serve $MODEL_ID \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu-memory-utilization 0.9 \
    --served-model-name Qwen3-8B
    # --pipeline-parallel-size 1 \
    # --max-model-len 8192 \
    # --max-num-seqs 256 \
    # --dtype auto \
    # --enable-prefix-caching \
    # --disable-log-requests \
    
    # --chat-template-kwargs '{"enable_thinking": true}' \
    # --max-log-len 1000 \
    # --response-role assistant \
    # --trust-remote-code \
    # --download-dir /tmp/vllm_cache \
    # --load-format auto \
    # --tokenizer-mode auto \
    # --enable-chunked-prefill \
    # --guided-decoding-backend outlines \
    # --swap-space 4 \
    # --disable-sliding-window \
    # --max-parallel-loading-workers 2


# Note:
# --pipeline-parallel-size 1 - 流水线并行度
# --enable-auto-tool-choice - 启用自动工具调用
# --tool-call-parser hermes - Hermes风格工具解析器
# --gpu-memory-utilization 0.9 - GPU内存利用率90%
# --max-model-len 8192 - 最大上下文长度
# --max-num-seqs 256 - 最大批处理序列数
# --dtype auto - 自动选择数据类型
# --enable-prefix-caching - 启用前缀缓存
# --disable-log-requests - 禁用请求日志
# --api-key your-secret-key - API密钥（请替换）
# --served-model-name Qwen3-8B - API中的模型名称
# --chat-template-kwargs - 聊天模板参数（启用思考模式）
# --max-log-len 1000 - 最大日志长度
# --response-role assistant - 响应角色名称
# --trust-remote-code - 信任远程代码
# --download-dir - 模型下载缓存目录
# --load-format auto - 自动选择权重加载格式
# --tokenizer-mode auto - 自动选择tokenizer模式
# --enable-chunked-prefill - 启用分块预填充
# --guided-decoding-backend outlines - 指导解码后端
# --swap-space 4 - 交换空间大小（GB）
# --disable-sliding-window - 禁用滑动窗口
# --max-parallel-loading-workers 2 - 最大并行加载工作线程数