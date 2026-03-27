# # from litellm import LLM, CompletionRequest

# # # 初始化 LLM，指定 OpenAI 兼容接口的 URL
# # llm = LLM(
# #     model="Meta-Llama-3___1-8B-Instruct",
# #     llm_type="openai",        # 使用 OpenAI 兼容接口
# #     api_base="http://127.0.0.1:8000/v1",  # vLLM API 地址
# # )

# # # 构造一个简单的请求
# # request = CompletionRequest(
# #     prompt="Write a short poem about AI in 4 lines.",
# #     max_tokens=100,
# #     temperature=0.7
# # )

# # # 调用模型
# # response = llm.generate(request)

# # # 输出生成结果
# # print("=== Model Output ===")
# # print(response.text)


# from litellm import completion
# import os

# # os.environ["OPENAI_API_KEY"] = "your-openai-key"
# # os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# # OpenAI
# response = completion(model="hosted_vllm/Meta-Llama-3___1-8B-Instruct", messages=[{"role": "user", "content": "Hello!"}])

# print(response)

# # Anthropic  
# # response = completion(model="anthropic/claude-sonnet-4-20250514", messages=[{"role": "user", "content": "Hello!"}])


import litellm 

response = litellm.completion(
            model="hosted_vllm/Meta-Llama-3___1-8B-Instruct", # pass the vllm model name
            messages=[{"role": "user", "content": "Hello!"}],
            api_base="http://127.0.0.1:8000/v1",
            temperature=0.2,
            max_tokens=80)

print(response)

