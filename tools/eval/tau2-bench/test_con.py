#!/usr/bin/env python3
"""
测试 LiteLLM 连接 vLLM 服务
"""
import os
from litellm import completion

# 配置
VLLM_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "Qwen3-8B"

# 设置环境变量
os.environ["OPENAI_API_BASE"] = VLLM_API_BASE
os.environ["OPENAI_API_KEY"] = "dummy"

def test_litellm():
    """测试 LiteLLM 调用"""
    try:
        print(f"[测试] 连接到: {VLLM_API_BASE}")
        print(f"[测试] 模型名称: {MODEL_NAME}")
        
        # 方式1: 使用 hosted_vllm 前缀
        print("\n--- 方式1: hosted_vllm ---")
        response = completion(
            model=f"hosted_vllm/{MODEL_NAME}",
            messages=[{"role": "user", "content": "Hello! Say 'Hi' back."}],
            max_tokens=10
        )
        print(f"✓ 成功: {response.choices[0].message.content}")
        
        # 方式2: 使用 openai 前缀
        print("\n--- 方式2: openai ---")
        response = completion(
            model=f"openai/{MODEL_NAME}",
            messages=[{"role": "user", "content": "Say '1+1=' and answer."}],
            max_tokens=10
        )
        print(f"✓ 成功: {response.choices[0].message.content}")
        
        print("\n✅ 所有测试通过！LiteLLM 可以正常连接 vLLM 服务")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("\n请检查:")
        print("1. vLLM 服务是否在运行？")
        print(f"2. 地址是否正确？{VLLM_API_BASE}")
        print(f"3. 模型名称是否正确？{MODEL_NAME}")
        return False

if __name__ == "__main__":
    test_litellm()