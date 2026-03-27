# uv pip install openai azure.identity
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Sequence, Tuple

from azure.identity import AzureCliCredential, get_bearer_token_provider
from loguru import logger
from openai import AzureOpenAI
from src.config import InferenceConfig

def get_endpoints() -> dict:
    gpt_4o = [
        {"endpoints": "https://conversationhubeastus.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://conversationhubeastus2.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://conversationhubnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://conversationhubsouthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://conversationhubwestus.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://readineastus.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://readineastus2.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://readinnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://readinwestus.openai.azure.com/", "speed": 150, "model": "gpt-4o"},
        {"endpoints": "https://conversationhubeastus.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
        {"endpoints": "https://conversationhubeastus2.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
        {"endpoints": "https://conversationhubnorthcentralus.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
        {"endpoints": "https://conversationhubsouthcentralus.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
        {"endpoints": "https://readineastus.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
        {"endpoints": "https://readineastus2.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
        {"endpoints": "https://readinnorthcentralus.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
        {"endpoints": "https://readinwestus.openai.azure.com/", "speed": 450, "model": "gpt-4o-global"},
    ]
    gpt_4o_mini = [
        {"endpoints": "https://conversationhubeastus.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://conversationhubeastus2.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://conversationhubnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://conversationhubsouthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://conversationhubswedencentral.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://conversationhubwestus.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://readineastus.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://readineastus2.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://readinnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://readinwestus.openai.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
        {"endpoints": "https://malicata-azure-ai-foundry.cognitiveservices.azure.com/", "speed": 150, "model": "gpt-4o-mini"},
    ]
    gpt_4_turbo = [
        {"endpoints": "https://conversationhubeastus2.openai.azure.com/", "speed": 150, "model": "gpt-4-turbo"},
        {"endpoints": "https://conversationhubswedencentral.openai.azure.com/", "speed": 150, "model": "gpt-4-turbo"},
        {"endpoints": "https://readineastus2.openai.azure.com/", "speed": 150, "model": "gpt-4-turbo"},
        {"endpoints": "https://readinswedencentral.openai.azure.com/", "speed": 150, "model": "gpt-4-turbo"},
    ]
    gpt_4_1 = [
        {"endpoints": "https://conversationhubnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4.1-DZS"},
        {"endpoints": "https://conversationhubsouthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4.1-DZS"},
        {"endpoints": "https://conversationhubswedencentral.openai.azure.com/", "speed": 150, "model": "gpt-4.1-DZS"},
        {"endpoints": "https://readinnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4.1-DZS"},
        {"endpoints": "https://conversationhubeastus2.openai.azure.com/", "speed": 150, "model": "gpt-4.1-global"},
        {"endpoints": "https://conversationhubnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4.1-global"},
        {"endpoints": "https://conversationhubswedencentral.openai.azure.com/", "speed": 150, "model": "gpt-4.1-global"},
        {"endpoints": "https://readinnorthcentralus.openai.azure.com/", "speed": 150, "model": "gpt-4.1-global"},
    ]
    gpt_5 = [
        {"endpoints": "https://conversationhubeastus2.openai.azure.com/", "speed": 150, "model": "gpt-5-global"},
    ]
    return {
        "gpt-4o": gpt_4o,
        "gpt-4o-mini": gpt_4o_mini,
        "gpt-4-turbo": gpt_4_turbo,
        "gpt-4.1": gpt_4_1,
        "gpt-5": gpt_5,
    }

def select_endpoint(model_name: str) -> dict:
    azure_endpoints = get_endpoints()
    entries = azure_endpoints[model_name]
    candidates = [e for e in entries if e.get("speed", 0) > 0 and e.get("endpoints")]
    weights = [e["speed"] for e in candidates]
    chosen = random.choices(candidates, weights=weights, k=1)[0]
    return chosen

def get_client(
    model_name: str | None = None,
    tenant_id: str | None = None,
    api_version: str | None = None,
    max_retries: int | None = None,
    config: InferenceConfig | None = None,
) -> Tuple[AzureOpenAI, str]:
    cfg = config or InferenceConfig()
    resolved_model_name = model_name or cfg.model
    azure_ad_token_provider = get_bearer_token_provider(
        AzureCliCredential(tenant_id=tenant_id or cfg.tenant_id),
        "https://cognitiveservices.azure.com/.default",
    )
    selected = select_endpoint(resolved_model_name)
    client = AzureOpenAI(
        azure_endpoint=selected["endpoints"],
        azure_ad_token_provider=azure_ad_token_provider,
        api_version=api_version or cfg.api_version,
        max_retries=max_retries if max_retries is not None else cfg.max_retries,
    )
    return client, selected["model"]

def client_parallel_inference(
    prompts: Sequence[str],
    model_name: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    tenant_id: str | None = None,
    api_version: str | None = None,
    max_retries: int | None = None,
    workers: int | None = None,
    config: InferenceConfig | None = None,
) -> List[str]:
    """Parallel-style interface using Azure OpenAI chat completions."""
    if not prompts:
        return []
    cfg = config or InferenceConfig()
    target_model = model_name or cfg.model
    client, resolved_model = get_client(
        model_name=target_model,
        tenant_id=tenant_id,
        api_version=api_version,
        max_retries=max_retries,
        config=cfg,
    )
    resolved_temperature = cfg.temperature if temperature is None else temperature
    send_temperature = None if str(target_model).startswith("gpt-5") else resolved_temperature
    resolved_max_tokens = max_tokens if max_tokens is not None else cfg.parallel_max_tokens
    resolved_workers = workers if workers is not None else cfg.workers

    def _infer(idx: int, prompt: str) -> Tuple[int, str]:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": resolved_model,
            "max_completion_tokens": resolved_max_tokens,
            "messages": messages,
        }
        if send_temperature is not None:
            kwargs["temperature"] = send_temperature
        resp = client.chat.completions.create(**kwargs)
        return idx, resp.choices[0].message.content

    outputs: List[str] = ["" for _ in prompts]
    with ThreadPoolExecutor(max_workers=resolved_workers) as ex:
        futures = [ex.submit(_infer, idx, prompt) for idx, prompt in enumerate(prompts)]
        for fut in as_completed(futures):
            idx, text = fut.result()
            outputs[idx] = text
    logger.info(f"Azure inference completed with model {resolved_model}")
    return outputs

if __name__ == "__main__":
    # 1. 模拟一个简单的配置类 (如果 src.config 导入失败)
    try:
        from config import InferenceConfig
    except ImportError:
        class InferenceConfig:
            model: str = "gpt-5"
            tenant_id: str | None = None
            api_version: str = "2024-05-01-preview"
            max_retries: int = 2
            parallel_max_tokens: int = 1000
            workers: int = 5
            temperature: float = 0.7

    # 2. 准备测试数据
    test_prompts = [
        "你好，请问你是 GPT-5 吗？",
        "请简述一下量子计算的原理。",
        "写一首关于人工智能的短诗。"
    ]

    logger.info("正在启动 GPT-5 并行推理测试...")

    try:
        # 3. 调用并行推理函数
        # 注意：这里 model_name 传入 "gpt-5"，函数内部会通过 select_endpoint 
        # 匹配到 https://conversationhubeastus2.openai.azure.com/ 这个端点
        results = client_parallel_inference(
            prompts=test_prompts,
            model_name="gpt-5",
            workers=3
        )

        # 4. 打印结果
        for i, res in enumerate(results):
            print(f"\n--- 响应 {i+1} ---")
            print(res)

    except Exception as e:
        logger.error(f"推理过程中发生错误: {e}")
        print("\n提示：请确保你已执行 'az login' 并且有权访问指定的 Azure 端点。")

__all__ = ["client_parallel_inference", "get_client", "select_endpoint", "get_endpoints"]
