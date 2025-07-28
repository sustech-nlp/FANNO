from typing import List, Dict
import numpy as np
import ray
from vllm import LLM, SamplingParams
from scipy.special import softmax
from loguru import logger 
import re
from transformers import AutoTokenizer

# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

config_table = {
    "llama2": {
        "max_model_len": 2048,
        "id2score": {29900: "0", 29896: "1"}
    },
    "llama3": {
        "max_model_len": 8192,
        "id2score": {15: "0", 16: "1"}
    },
    "mistral": {
        "max_model_len": 2000,
        "id2score": {28734: "0", 28740: "1"}
    }
}

def get_model_config(model_path):
    for key in config_table:
        if key in model_path.lower():
            logger.info(f"Using config for {key}")
            return config_table[key]
    return config_table["mistral"]

def get_template(prompt, template_type="default", tokenizer=None):
    # logger.info(f"Using template type: {template_type}")
    if template_type == "alpaca":
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
    elif template_type == "mistral":
        return f"""<|im_start|>user
{prompt}
<|im_end|>
"""
    elif template_type == "direct":
        return prompt
    elif template_type == "tags":
        return f"""You are a helpful assistant. Please identify tags of user intentions in the following user query and provide an explanation for each tag. Please respond in the JSON format {{"tag": "str", "explanation": "str"}}.
Query: {prompt} 
Assistant:"""
    else:
        messages = [
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parser_score(input_list: List[str]) -> List[int]:
    pattern = re.compile(r'score:\s*(\d)', re.IGNORECASE)
    scores = [int(match.group(1)) if (match := pattern.search(s)) else 0 for s in input_list]
    return scores

@ray.remote(num_gpus=1)
def vllm_inference(model_path: str, input_data: List[str], max_tokens: int = 256, temperature: float = 0, top_p: float = 0.9, skip_special_tokens: bool = True):
    
    config = get_model_config(model_path)
    llm = LLM(model=model_path, tokenizer_mode="auto", trust_remote_code=True, max_model_len=config["max_model_len"], gpu_memory_utilization=0.95)
    
    if "llama3" in model_path:
        tokenizer = llm.get_tokenizer()
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, skip_special_tokens=skip_special_tokens,
                                          stop_token_ids=[tokenizer.eos_token_id, 
                                                           tokenizer.convert_tokens_to_ids("<|eot_id|>")]) 
    else: 
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, skip_special_tokens=skip_special_tokens)
    
    outputs = llm.generate(input_data, sampling_params)
    return [output.outputs[0].text for output in outputs]

def parallel_inference(prompt_list: List[str], max_tokens: int = 256, temperature: float = 0.0, top_p: float = 0.9, skip_special_tokens: bool = True, score: bool = False, model_name_or_path: str = None, template_type: str = "default"):
    logger.info(f"Using model {model_name_or_path}")
    
    # Initialize tokenizer for template processing
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply templates to prompts
    prompt_list = [get_template(prompt, template_type=template_type, tokenizer=tokenizer) for prompt in prompt_list]
    logger.info(f"Prompt list's first 1 element: {prompt_list[0]}")
    
    ray.init(ignore_reinit_error=True)
    n_chunks = 8
    chunk_size = len(prompt_list) // n_chunks
    chunks = [prompt_list[i * chunk_size: (i + 1) * chunk_size] for i in range(n_chunks)]
    
    # Handle last chunk separately in case of uneven division
    if len(prompt_list) % n_chunks != 0:
        chunks[-1].extend(prompt_list[n_chunks * chunk_size:])
    
    for i in range(n_chunks):
        logger.info(f"Chunk {i} size: {len(chunks[i])}")
    
    if score:
        tasks = [vllm_inference.remote(model_name_or_path, chunk, max_tokens=10, temperature=temperature, top_p=top_p, skip_special_tokens=skip_special_tokens) for chunk in chunks]
    else:
        tasks = [vllm_inference.remote(model_name_or_path, chunk, max_tokens, temperature, top_p, skip_special_tokens) for chunk in chunks]
    
    processed_data = sum(ray.get(tasks), [])
    ray.shutdown()
    return processed_data if not score else parser_score(processed_data)

def parallel_inference_instagger(prompt_list: List[str], max_tokens: int = 256, temperature: float = 0.0, top_p: float = 0.9, skip_special_tokens: bool = True, model_name_or_path: str = None) -> List[str]:
    return parallel_inference(prompt_list, max_tokens, temperature, top_p, skip_special_tokens, score=False, model_name_or_path=model_name_or_path, template_type="tags")

if __name__ == "__main__":
    # Test parallel inference
    test_prompts = [
        "Tell me about cats.",
        "What is the capital of France?",
        "Explain quantum physics.",
        "Write a haiku about spring."
    ]
    
    print("Testing parallel inference...")
    results = parallel_inference(
        test_prompts,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        model_name_or_path="/fs-computility/llmit_d/shared/models/Qwen2.5-7B-Instruct"
    )
    
    print("\nResults:")
    for prompt, result in zip(test_prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result}")

    # test2: score
    def privacy_eval(instruction: str) -> str:
        prompt_template = (
            "Evaluate the following instruction to determine if it contains the name of an unfamiliar person, place, or organization. "
            "If it does, score it as 1. If it does not, score it as 0. "
            "You should first explain the reason for your score, and then provide the score in the format: "
            "`score: X`, where X is either 0 or 1.\n\n"
            "Instruction: \"{}\"\n\nOutput:"
        )
        return prompt_template.format(instruction)

    def repetition_eval(sentence: str) -> str:
        prompt_template = (
            "Evaluate the following sentence to determine if it contains a significant amount of repeated content. "
            "If the sentence contains considerable repetition, score it as 1. If it does not, score it as 0. "
            "You should first explain the reason for your score, and then provide the score in the format: "
            "`score: X`, where X is either 0 or 1.\n\n"
            "Sentence: \"{}\"\n\nOutput:"
        )
        return prompt_template.format(sentence)
    
    instructions = [
        "What were the key contributions of Dr. Amelia F. Richardson to quantum computing in the 1990s?",  # Familiar person
        "Provide a list of cities in Europe with populations over 1 million.",  # No unfamiliar person, place, or organization
        "Research the political controversies surrounding the 2024 campaign of Zachary T. Elmwood.",  # Unfamiliar person
        "What is the origin of the phrase 'crossing the Rubicon'?",  # General history, no unfamiliar person
        "Find information about the fictional planet of Arrakis in Frank Herbert's Dune series.",  # Fictional place
        "Who are the largest shareholders of Tesla, Inc. as of 2023?",  # Known organization
        "Locate the headquarters of the organization named 'Techovate Solutions' founded in 2018."  # Unfamiliar organization
    ]

    answer = [1, 0, 1, 0, 1, 0, 1]
    model_path = "/home/admin/data/huggingface_model/mistral/Mistral-7B-Instruct-v0.3"
    prompt_list = [privacy_eval(instruction) for instruction in instructions]
    results = parallel_inference(prompt_list, max_tokens=256, temperature=0, top_p=0.95, model_name_or_path=model_path, score=True)
    print(results)
    
    sentences = [
        "Hello, how are you?",  # Normal sentence
        "Hello, how are you? Hello, how are you?",  # Repeated sentence
        "The quick brown fox jumps over the lazy dog.",  # No repetition
        "The quick quick brown fox fox jumps over over the lazy dog dog.",  # Significant repetition
        "The quick quick brown fox fox jumps over over the lazy dog dog. 010110101011011010101101101",
        "Hello, how are you? xx        qqqqqxx"
    ]
    answer = [0, 1, 0, 1, 1, 1]
    prompt_list = [repetition_eval(sentence) for sentence in sentences]
    results = parallel_inference(prompt_list, max_tokens=256, temperature=0, top_p=0.95, model_name_or_path=model_path, score=True)
    print(results)