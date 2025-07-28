#NEW CODE
from typing import List, Dict, Any
from loguru import logger
import numpy as np
import re
import ray
from utils.inference_utils import parallel_inference
from utils.data_utils import load_jsonlines, save_jsonlines
from template.eval.eval_template import * 
from sentence_transformers import SentenceTransformer, util
from config import Config, InferenceConfig
import logging

@ray.remote(num_gpus=1)
def diversity_filter_remote(encode_model_path: str, new_data: List[Dict[str, Any]], old_data: List[Dict[str, Any]], config) -> List[Dict[str, Any]]:
    model = SentenceTransformer(encode_model_path, trust_remote_code=True).to(config.device)
    data = new_data + old_data
    instructions = [d["instruction"] for d in data]
    embeddings = model.encode(instructions, convert_to_tensor=True, show_progress_bar=True, device=config.device, batch_size=config.batch_size)
    diversity_result = Evaluator.community_detection(data, old_data, embeddings, min_community_size=config.min_community_size, threshold=config.threshold)
    logger.info("Diversity filter ratio: {} , remain ratio: {}".format( 1 - len(diversity_result) / len(new_data), len(diversity_result) / len(new_data)) ) if len(new_data) > 0 else None
    return diversity_result


class Evaluator:
    def __init__(self, config): 
        self.encode_model_path = config.encode_model_pth 
        self.model_name_or_pth = config.model_name_or_pth
        self.config = config
        
    def get_basic_infomation(self, value: List[int]) -> List[int]:
        p95 = np.percentile(value, 95)
        p5 = np.percentile(value, 5)
        mean = np.mean(value)
        med = np.median(value)
        logger.info("p95: {}, p5: {}, mean: {}, med: {}".format(p95, p5, mean, med))
        return p95, p5, mean, med
    
    
    
    def hard_filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        
        ref_key_word = ["based on", "according", "given", "mentioned", "refer", "provided", "passage", "text", "paragraph"]
        time_key_word = ["recent", "current", "now", "today", "yesterday", "tomorrow", "soon", "upcoming", "recently", "coming", "currently"]
        obj_key_word = ["name"]
        key_words = ref_key_word + time_key_word + obj_key_word
    
        
        length = [len(item['instruction'].split()) for item in data]
        p95, p5, mean, med = self.get_basic_infomation(length)
        
        remaining_data = []
        for item in data:
            instruction = item['instruction']
            # 标准1: 如果instruction为空, 则过滤掉
            if len(instruction) == 0: # or instruction[-1].isalpha(): # 这个判断条件是为了过滤掉以字母结尾的句子
                continue
            # 标准2. 禁止出现中文等其他语言
            if not all(ord(c) < 128 for c in instruction):
                continue
            # 标准3: 把指令长度<5且结尾不是.或?的过滤掉
            if len(instruction.split()) < 5 and instruction[-1] not in [".", "?"]:
                continue
            # 标准4: 如果instruction中关键词, 则过滤掉 
            if any(re.search(key, instruction, re.IGNORECASE) for key in key_words):
                continue
            
             # 标准5: 如果instruction中英文字母比标点符号<1:1, 则过滤掉
            if sum([1 for c in instruction if c.isalpha()]) < sum([1 for c in instruction if not c.isalpha()]):
                continue
            
            # 标准6: 95%之外的数据
            # if len(instruction.split()) > p95:
            #     continue
            
            # TODO: Add a response filter
            # response = item['response']
            
    
            remaining_data.append(item)
        logger.info("Hard filter ratio: {}, remain ratio: {}".format(1 - len(remaining_data) / len(data), len(remaining_data) / len(data))) if len(data) > 0 else None
        return remaining_data
    
    def llm_filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        instructions = [d['instruction'] for d in data]
        # OLD Version
        # responses = [d['response'] for d in data]
        # difficult_prompts = [difficult_eval(instruction) for instruction in instructions]
        # insjudge_prompts = [insjudge_eval(instruction) for instruction in instructions]
        # all_prompts = difficult_prompts + insjudge_prompts 
        # 
        # return np.array(scores).reshape(len(instructions), 4) Bug code
        # scores_array = np.array([scores[i::len(instructions)] for i in range(len(instructions))]) # Correct code
        
                
        privacy_prompts = [privacy_eval(instruction) for instruction in instructions]
        privacy_scores = parallel_inference(privacy_prompts, max_tokens=512, **vars(InferenceConfig()), score=True)
        
        # repetition_prompts = [repetition_eval(response) for response in responses]
        # repetition_scores = parallel_inference_complex(repetition_prompts, max_tokens=512, **vars(InferenceConfig()), score=True)
        
        remaining_data = []
        for item, ps in zip(data, privacy_scores):
            if ps == 0:
                remaining_data.append(item)
        logger.info("LLM filter ratio: {}, remain ratio: {}".format(1 - len(remaining_data) / len(data), len(remaining_data) / len(data))) if len(data) > 0 else None
        return remaining_data
    
    
    # @staticmethod
    # def community_detection(data: List[Dict[str, Any]], corpus_embeddings: np.ndarray, min_community_size: int = 1, threshold: float = 0.7) -> List[Dict[str, Any]]:
    #     clusters = util.community_detection(corpus_embeddings, min_community_size=min_community_size, threshold=threshold)
    #     new_data = []
    #     for cluster in clusters:
    #         sorted_cluster = sorted((data[idx].copy() for idx in cluster), key=lambda x: x["value"], reverse=True)
    #         new_data.extend(sorted_cluster[:1])
    #     return new_data
    
    # def diversify(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     model = SentenceTransformer(self.encode_model_path, trust_remote_code=True).to(self.config.device)
    #     instructions = [d["instruction"] for d in data]
    #     cutted_instructions = [" ".join(instruction.split()[:self.config.words_num]) if len(instruction.split()) > self.config.words_num else instruction for instruction in instructions]  
    #     embeddings = model.encode(cutted_instructions, convert_to_tensor=True, show_progress_bar=True, device=self.config.device, batch_size=self.config.batch_size)
    #     remaining_data = Evaluator.community_detection(data, embeddings, min_community_size=self.config.min_community_size, threshold=self.config.threshold)
    #     logger.info("Diversify ratio: {}, remain ratio: {}".format(1 - len(remaining_data) / len(data), len(remaining_data) / len(data))) if len(data) > 0 else None
    #     return remaining_data
    
    
    # def evaluate_and_filter(self, new_data: List[Dict[str, Any]], old_data: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    #     hard_filtered_data = self.hard_filter(new_data)
    #     if len(hard_filtered_data) == 0:
    #         return []
    #     llm_filtered_data=  self.llm_filter(hard_filtered_data)
    #     return self.diversify(llm_filtered_data)
    
    
    @staticmethod
    def community_detection(data: List[Dict[str, Any]],old_data:List[Dict[str, Any]], corpus_embeddings: np.ndarray, min_community_size: int = 1, threshold: float = 0.8) -> List[Dict[str, Any]]:
        # 只过滤新的数据; last_data是之前的数据
        clusters = util.community_detection(corpus_embeddings, min_community_size=min_community_size, threshold=threshold)
        new_data = []
        for cluster in clusters: 
            sorted_cluster = [data[idx].copy() for idx in cluster]
            if any([d in old_data for d in sorted_cluster]):
                continue
            else:
                new_data.append(sorted_cluster[0])
        return new_data 
    
    def diversity_filter(self, new_data: List[Dict[str, Any]], old_data: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        ray.init(log_to_driver=False, logging_level=logging.WARNING)
        diversity_result_future = diversity_filter_remote.remote(
            self.encode_model_path, new_data, old_data, self.config
        )
        diversity_result = ray.get(diversity_result_future)
        ray.shutdown()
        return diversity_result
    
    def evaluate_and_filter(self, new_data: List[Dict[str, Any]], old_data: List[Dict[str, Any]] = []) -> List[Dict[str, Any]]:
        new_data = self.hard_filter(new_data)
        if len(new_data) == 0:
            return []
        # new_data = self.llm_filter(new_data)
        return self.diversity_filter(new_data, old_data)
    
    
    

if __name__ == "__main__":
    config =Config()
    evaluator = Evaluator(config)
    
    data = load_jsonlines("/home/admin/research/Fanno/FANNO/src/template/eval/privacy_test.jsonl")
    for item in data:
        print(item["instruction"])
        
        
    instructions = [
        "What were the key contributions of Dr. Amelia F. Richardson to quantum computing in the 1990s?",  
        # Likely unfamiliar person, depending on context (score: 0).
        
        "Provide a list of cities in Europe with populations over 1 million.",  
        # General geographic query, no names (score: 1).
        
        "Research the political controversies surrounding the 2024 campaign of Zachary T. Elmwood.",  
        # Likely unfamiliar person involved in politics (score: 0).
        
        "Find information about the fictional planet of Arrakis in Frank Herbert's Dune series.",  
        # Contains a fictional place, might not qualify as unfamiliar (score: 1).
        
        "Who are the largest shareholders of Tesla, Inc. as of 2023?",  
        # Well-known organization but may involve unfamiliar names in the shareholders list (score: 0 or 1).
        
        "Locate the headquarters of the organization named 'Techovate Solutions' founded in 2018.",  
        # Likely an unfamiliar organization (score: 0).
        
        "What are the traditional recipes used by the Hmong community in Southeast Asia?",  
        # Refers to a specific ethnic group, no unfamiliar names (score: 1).
        
        "Can you provide the biography of novelist Alexander Pryce from New Zealand?",  
        # Potentially unfamiliar person (score: 0).
        
        "What is the origin of the phrase 'crossing the Rubicon'?",  
        # No unfamiliar names or entities, general history query (score: 1).
        
        "Analyze the case law involving Sinclair and Braeburn Corp. in intellectual property disputes.",  
        # Potentially unfamiliar organization or legal context (score: 0).
        
        "Explain the philosophy of Hegbert Jones regarding environmental ethics.",  
        # Unfamiliar person (score: 0).
        
        "Compare the policies of Nordic countries regarding renewable energy production.",  
        # General geographic query, no unfamiliar names (score: 1).
        
        "Who were the key donors to the campaign of Mikhail Ivanovich Petrov in 2022?",  
        # Likely unfamiliar person (score: 0).
        
        "Provide an analysis of the impact of DenoCorp's AI solutions in healthcare.",  
        # Likely an unfamiliar organization (score: 0).
        
        "Identify the cultural practices of the Tlingit people in the Pacific Northwest.",  
        # Refers to an ethnic group, no unfamiliar names (score: 1).
    ]
    answer = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]

    data = [{"instruction": instruction} for instruction in instructions]

    
    # print(data)
    # 进行llm_filter
    data, drop_data,scores = evaluator.llm_filter(data)
    
    # 计算正确率：
    correct = 0
    for i in range(len(data)):
        if scores[i] == answer[i]:
            correct += 1
    print(correct / len(data))
    