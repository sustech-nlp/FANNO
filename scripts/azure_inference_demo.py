from fanno.inference.client_inference import client_parallel_inference


def main():
    prompts = [
        "What is the capital of France?",
        "Give me a two-sentence summary of reinforcement learning.",
    ]
    outputs = client_parallel_inference(prompts, model_name="gpt-4o", max_tokens=128, temperature=0.7)
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}\nResponse: {output}\n{'-'*40}")


if __name__ == "__main__":
    main()
