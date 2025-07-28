import asyncio
import httpx
from typing import List
from loguru import logger
import re
from transformers import AutoTokenizer


from regex import T

def parser_score(input_list: List[str]) -> List[int]:
    pattern = re.compile(r'score:\s*(\d)', re.IGNORECASE)
    scores = [int(match.group(1)) if (match := pattern.search(s)) else 0 for s in input_list]
    return scores


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
    # messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)





async def distribute_requests(prompt_list: List[str], 
                              max_tokens: int = 256, 
                              temperature: float = 0.0, 
                              top_p: float = 0.9, 
                              skip_special_tokens: bool = True, 
                              score = False, 
                              servers: List[str] = None,
                              template_type: str = "default",
                              tokenizer: str = None
                              ) -> List[str]:
    
    prompt_list = [get_template(prompt, template_type=template_type, tokenizer=tokenizer) for prompt in prompt_list]
    logger.info(f"Prompt list's first 1 element: {prompt_list[0]}")
    
    n_chunks = len(servers)
    chunk_size = len(prompt_list) // n_chunks
    chunks = [prompt_list[i * chunk_size: (i + 1) * chunk_size] for i in range(n_chunks)]
    
    if len(prompt_list) % n_chunks != 0:
        chunks[-1].extend(prompt_list[n_chunks * chunk_size:])
    
    for i in range(n_chunks):
        logger.info(f"Chunk {i} size: {len(chunks[i])}")
    
    
    tasks = [fetch_results(servers[i], chunks[i], max_tokens, temperature, top_p, skip_special_tokens) for i in range(n_chunks)]
   
    results = await asyncio.gather(*tasks)
    results = sum(results, [])
    return results if not score else parser_score(results)

async def fetch_results(server_url: str, 
                        chunk: List[str], 
                        max_tokens: int, 
                        temperature: float, 
                        top_p: float, 
                        skip_special_tokens: bool):
    async with httpx.AsyncClient(timeout=3600.0) as client:  # 将超时时间设置为30秒
        response = await client.post(f"{server_url}/inference", json={
            "input_data": chunk,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "skip_special_tokens": skip_special_tokens
        })
        response.raise_for_status()
        return response.json()["outputs"]
       


def parallel_inference(prompt_list: List[str], max_tokens: int = 256, temperature: float = 0.0, top_p: float = 0.9, skip_special_tokens: bool = True, model_name_or_path: str = None, score:bool = False) -> List[str]:
    gpu_ids = [4,5,6,7]
    servers = ["http://localhost:800{}".format(i) for i in gpu_ids]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return asyncio.run(distribute_requests(prompt_list, max_tokens, temperature, top_p, skip_special_tokens, servers=servers, tokenizer=tokenizer,score = False ))


# def parallel_inference_instagger(prompt_list: List[str], max_tokens: int = 256, temperature: float = 0.0, top_p: float = 0.9, skip_special_tokens: bool = True, model_name_or_path: str = None) -> List[str]:
#     gpu_ids = [4, 5, 6,7]
#     servers = ["http://localhost:800{}".format(i) for i in gpu_ids]
#     return asyncio.run(distribute_requests(prompt_list, max_tokens, temperature, top_p, skip_special_tokens, servers=servers, template_type="tags"))



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
        model_name_or_path="/fs-computility/llmit_d/shared/models/Qwen2.5-0.5B-Instruct"
    )
    
    print("\nResults:")
    for prompt, result in zip(test_prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result}")


    # test2 : score
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
    model_path = "/fs-computility/llmit_d/shared/models/Qwen2.5-0.5B-Instruct"
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
    
    
    
    
    # python start.py --model_path /home/zhe/models/lukeminglkm/instagger_llama2 --base_port 8000 --gpu_list 1,2,3,4,5,6,7
    


    

    