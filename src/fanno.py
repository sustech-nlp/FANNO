from typing import List, Dict, Any 
from template.gen.seed_template import generate_seed_prompt
from template.gen.ucb_template import TD

from utils.inference_utils import parallel_inference
from utils.ucb_utils import * 
from utils.data_utils import load_jsonlines, save_jsonlines, instruction_cleaning,get_unlabeled_data,save_or_skip,save_or_skip_dynamic
from evaluator import Evaluator
from template.gen.response_template import q2a, qdoc2a
from loguru import logger
from config import Config, InferenceConfig



GLOBAL_IDX = 0 

def get_idx_range(length):
    global GLOBAL_IDX
    idx_range = range(GLOBAL_IDX, GLOBAL_IDX + length)
    GLOBAL_IDX += length
    return idx_range


@save_or_skip(Config().seed_pth)
def seed_generate(docs: List[str], evaluator: Evaluator) -> List[Dict[str, Any]]:
    logger.info("Start generating seeds")
    prompts =[]
    raw_doc = []
    for doc in docs:
        new_prompts = generate_seed_prompt(doc)
        prompts += new_prompts
        length = len(new_prompts) 
        raw_doc += [doc] * length
    gen_results =  instruction_cleaning(parallel_inference(prompts, max_tokens=2048, **vars(InferenceConfig())))
    gen_instruction = [part[0] for part in gen_results]
    gen_input = [part[1] for part in gen_results]

    seeds = [{"idx": idx, 
              "instruction": instruction, 
              "input": input,
              "value": 0,
              "doc": doc} for idx, instruction, input, doc in zip(get_idx_range(len(gen_instruction)), gen_instruction, gen_input, raw_doc)]
    
    logger.info(seeds[0])
    seeds = evaluator.evaluate_and_filter(seeds)
    seeds = response_generate(seeds)
    return seeds
    
    
def Think_Different(texts: List[str], seeds: List[Dict[str, str]]) -> List[str]:
    
    few_shots_list = ucb_judge(seeds, 3, len(texts)) 

    cut_num = 100
    for few_shots in few_shots_list:
        for i, few_shot in enumerate(few_shots):
            temp_instruction = few_shot["instruction"].split()
            if len(temp_instruction) > cut_num:
                few_shots[i]['instruction'] = " ".join(temp_instruction[:cut_num])
    few_shots_list = [tuple(few_shot['instruction'] for few_shot in few_shots) for few_shots in few_shots_list]
    prompts_list = [] 
    for text, (seed1, seed2, seed3) in zip(texts, few_shots_list):
        prompts_list.append(TD(text=text, seed1=seed1, seed2=seed2, seed3=seed3))
    return prompts_list



# @save_or_skip(Config().final_data_pth)
def response_generate(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    prompts = [q2a(i['instruction']+"\n"+i.get("input", "") if i.get("input", "")!="" else i['instruction']) for i in data]
    response =  parallel_inference(prompts, max_tokens=2048, **vars(InferenceConfig()))
    for instruction, response in zip(data, response):
        instruction["output"] = response
    return data 


def shots_generator_ucb(docs: List[str], seeds: List[str]) -> List[Dict[str,Any]]: 
    prompts =Think_Different(docs, seeds)
    logger.debug(f"prompts: {prompts[0]}")
    gen_results =  instruction_cleaning(parallel_inference(prompts, max_tokens=2048, **vars(InferenceConfig())))
    gen_instruction = [part[0] for part in gen_results]
    gen_input = [part[1] for part in gen_results]
        
    gen_data = [{"idx": idx,
                    "instruction": instruction,
                    "input": input,
                    "doc": doc,
                    "value": 0,
                    "cnt":0} 
                    for idx, instruction, input, doc in zip(get_idx_range(len(gen_instruction)), gen_instruction, gen_input, docs)]
    return gen_data


@save_or_skip_dynamic('file_pth')
def ucb_instruction_generate(docs: List[str], seeds: List[str], evaluator: Evaluator, **kwargs) -> List[str]:
    instruction_gen = shots_generator_ucb(docs, seeds)
    instruction_gen = evaluator.evaluate_and_filter(instruction_gen, old_data=seeds)
    instruction_gen = response_generate(instruction_gen)
    seeds += instruction_gen
    return seeds


if __name__ == '__main__':
    config = Config()
    docs = get_unlabeled_data(config, merge_bool=True)
    seed_docs_num , window_size, limit_size = config.seed_docs_num, config.window_size, config.limit_size
    evaluator = Evaluator(config)

    seeds = seed_generate(docs[:seed_docs_num],  evaluator)
    for idx, i in enumerate(range(seed_docs_num, len(docs), window_size)):
        file_path = config.custom_pth + f"/ucb_aug_{idx}.jsonl"
        seeds = ucb_instruction_generate( docs[i:i+window_size], seeds, evaluator, file_pth=file_path)
        if len(seeds) > limit_size: break
        
        
    

    

    
    
    

        
    