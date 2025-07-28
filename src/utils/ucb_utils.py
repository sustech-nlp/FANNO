from typing import List, Dict
import math
import random
from tqdm import trange
from config import InferenceConfig   
from transformers import AutoTokenizer

random.seed(42)

tokenizer = AutoTokenizer.from_pretrained(InferenceConfig().model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    


def ucb_judge(seeds: List[Dict[str, str]], 
              top_k: int = 3, 
              N: int = 5000) -> List[List[Dict[str, str]]]:
    '''
    UCB (Upper Confidence Bound) selection for seeds.

    This function selects top_k seeds in each round using a UCB-based strategy, 
    where the score is based on the normalized response length (tokenized output).
    In each round, it samples from the top 5% of seeds (by UCB value).

    Args:
        seeds (List[Dict[str, str]]): List of seed dictionaries, each with at least an 'output' key.
        top_k (int): Number of seeds to select per round.
        N (int): Number of rounds.

    Returns:
        List[List[Dict[str, str]]]: A list of lists, each containing the selected seeds for that round.
    '''
    for seed in seeds:
        seed.setdefault('cnt', 0)
        
    # version: use response length
    len_values = [len(tokenizer.tokenize(seed['output'])) for seed in seeds]
    min_len, max_len = min(len_values), max(len_values)
    
    for seed in seeds:
        normalized_len = (len(tokenizer.tokenize(seed['output'])) - min_len) / (max_len - min_len) if max_len > min_len else 0.5
        seed['score'] = normalized_len

    results = []  
    total_attempts = sum(seed['cnt'] for seed in seeds) + len(seeds)

    for _ in trange(N):
        for seed in seeds:
            seed['value'] = seed['score'] + 3 * math.sqrt(2 * math.log(total_attempts) / (seed['cnt'] + 1))
        
        sorted_seeds = sorted(seeds, key=lambda x: x['value'], reverse=True)
        # Select the top 5% of seeds, then sample top_k from them
        top_p = 0.05 * len(sorted_seeds)
        top_k_seed = random.sample(sorted_seeds[:int(top_p)], top_k)
        top_k_instructions = [seed for seed in top_k_seed]

        for seed in top_k_seed:
            seed['cnt'] += 1  
            total_attempts += 1 

        results.append(top_k_instructions) 
    
    return results



