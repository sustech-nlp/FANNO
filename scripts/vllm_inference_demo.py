from fanno.config import InferenceConfig
from fanno.inference.vllm_inference import parallel_inference


def main():
    prompts = [
        "Explain what a binary search tree is.",
        "Write a short haiku about the ocean.",
    ]
    cfg = InferenceConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        max_tokens=128,
        temperature=0.7,
    )
    outputs = parallel_inference(prompts, config=cfg, template_type="direct")
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}\nResponse: {output}\n{'-'*40}")


if __name__ == "__main__":
    main()
