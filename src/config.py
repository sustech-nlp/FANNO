import random
random.seed(42)
from dataclasses import dataclass

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

current_dir = os.getcwd()
seed_docs_num = 10
window_size = 200
limit_size = 1000
custom_name = "response-100k"
model_name_or_path = "/fs-computility/llmit_d/shared/models/Qwen2.5-7B-Instruct"
encode_model_pth = "/fs-computility/llmit_d/shared/models/stella_en_400M_v5"


@dataclass
class PipelineConfig:
    seed_docs_num: int = seed_docs_num
    window_size: int = window_size
    limit_size: int = limit_size
    model_name_or_pth: str = model_name_or_path 
    encode_model_pth: str = encode_model_pth
    
    
@dataclass
class InferenceConfig:
    temperature: float = 0.0
    top_p: float = 0.9
    skip_special_tokens: bool = True
    model_name_or_path: str = model_name_or_path


@dataclass
class FileConfig:
    # unlabeled_data
    unlabeled_data_pth: str = "/fs-computility/llmit_d/shared/zhuhe/FANNO/data/unlabel_data.jsonl"
    com_unlabeled_data_pth: str = "/fs-computility/llmit_d/shared/zhuhe/FANNO/data/unlabel_data_com.jsonl"
    custom_name: str = custom_name
    data_dir: str = "/fs-computility/llmit_d/shared/zhuhe/FANNO/experiment"
    custom_pth: str = f'{data_dir}/{custom_name}'
    seed_pth: str = f"{custom_pth}/initial_seed.jsonl"
    final_data_pth: str = f"{custom_pth}/final_data.jsonl"
    
    
@dataclass
class EvaluatorConfig:
    min_community_size: int = 1
    threshold: float = 0.9
    words_num: int = 4
    device: str = 'cuda'
    batch_size: int = 128

class Config(FileConfig, PipelineConfig, EvaluatorConfig):
    pass
