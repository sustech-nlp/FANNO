from copy import deepcopy

from numpy import save
from inference.vllm_client import parallel_inference_instagger
from dataclasses import dataclass
import re
from typing import List
from loguru import logger 
import os
from instdiff_tools.analysis import load_json, write_jsonlines
import json
from transformers import AutoTokenizer
from matplotlib import pyplot as plt




@dataclass
class InferenceConfig:
    temperature: float = 0.0
    top_p: float = 0.9
    skip_special_tokens: bool = True
    model_name_or_path: str =  "/volume/pt-train/users/wzhang/ghchen/zh/models/InsTagger"


def extract_tags(input_string):
    pattern = r'"tag":\s*"([^"]*)",\s*"explanation":\s*"([^"]*)"'
    matches = re.findall(pattern, input_string)
    return [{"tag": tag if tag else None, "explanation": explanation if explanation else None} 
            for tag, explanation in matches]

inference_config = InferenceConfig()

def get_tags(data: List[dict]) -> List[dict]:
    prompts = [entry["instruction"] for entry in data]
    responses = parallel_inference_instagger(prompts, max_tokens=1024, use_vllm=True, **vars(inference_config))
    for i, response in enumerate(responses):
        data[i]["instags"] = response
    return data

def get_instagger_tags(data: List[dict]) -> List[List[dict]]:
    data = get_tags(data)
    tags_str = [entry["instags"] for entry in data]
    tags_list = []
    for tags in tags_str:
        tags_list.append(extract_tags(tags))
    return tags_list


def get_experiment_tags(data: List[dict]) -> List[List[dict]]:
    data = get_tags(data)
    tags_str = [entry["instags"] for entry in data]
    tags_list = []
    for tags in tags_str:
        tags_list.append(extract_tags(tags))
        
        
    # data['tags'] = tags_list
    for i, entry in enumerate(data):    
        tag_str = ",".join([tag['tag'] for tag in tags_list[i]])
        entry['tag'] = tag_str
    return data

        
        

def get_complexity_diversity(data: List[dict]) -> List[dict]:
    
    data = get_tags(data)
    tags_str = [entry["instags"] for entry in data]
    tags_list = []
    for tags in tags_str:
        tags_list.append(extract_tags(tags))
    # 计算复杂性
    complexity = [len(tag) for tag in tags_list]
    avg_complexity = sum(complexity) / len(complexity) if len(complexity) > 0 else 0
    
    # 计算多样性
    try:
        diversity = len(set(tag["tag"] for tags in tags_list for tag in tags))
    except KeyError as e:
        logger.error(f"Missing 'tag' key in one of the tags. Error: {e}")
        diversity = 0
    
    # 输出调试信息
    logger.debug(f"complexity: {complexity}")
    logger.debug(f"avg_complexity: {avg_complexity}")
    logger.debug(f"diversity: {diversity}")
    
    return avg_complexity, diversity, len(data)


def add_complexity_to_items(data: List[dict], tokenizer=None) -> List[dict]:
    """为每个数据项添加 complexity 字段，并可选记录 token 长度"""
    # 获取 tags
    data = get_tags(data)
    tags_list = [extract_tags(entry["instags"]) for entry in data]

    # 为每个 item 添加 complexity（tag 的数量）
    for i, entry in enumerate(data):
        entry["complexity"] = len(tags_list[i])
        entry["tags"] = tags_list[i]

        if tokenizer is not None:
            instruction = entry.get("instruction", "")
            response = entry.get("response", "")
            entry["instruction_len"] = len(tokenizer.tokenize(instruction))
            entry["response_len"] = len(tokenizer.tokenize(response))

    return data

def calculate_average_tokens(data: List[dict], tokenizer) -> tuple[float, float]:
    if 'instruction' not in data[0] or 'response' not in data[0]:
        raise ValueError("The data does not contain 'instruction' or 'response' key.")
    
    instruction_token_counts = []
    response_token_counts = []
    for entry in data:
        instruction = entry["instruction"]
        response = entry["response"]
        instruction_tokens = tokenizer.tokenize(instruction)
        response_tokens = tokenizer.tokenize(response)
        instruction_token_counts.append(len(instruction_tokens))
        response_token_counts.append(len(response_tokens))
    avg_instruction_tokens = sum(instruction_token_counts) / len(instruction_token_counts) if instruction_token_counts else 0
    avg_response_tokens = sum(response_token_counts) / len(response_token_counts) if response_token_counts else 0
    return avg_instruction_tokens, avg_response_tokens


def get_infomation_request(data: List[dict], tokenizer=None) -> List[dict]:
    complex_values = [entry.get("complexity", 0) for entry in data if entry.get("complexity") is not None]
    avg_complexity = sum(complex_values) / len(complex_values) if complex_values else 0

    tag_names = [tag["tag"] for entry in data for tag in entry.get("tags", []) if tag.get("tag")]
    diversity = len(set(tag_names))

    instruction_lengths = []
    response_lengths = []
    for entry in data:
        instr_len = entry.get("instruction_len")
        resp_len = entry.get("response_len")
        if instr_len is None and tokenizer is not None:
            instr_len = len(tokenizer.tokenize(entry.get("instruction", "")))
            entry["instruction_len"] = instr_len
        if resp_len is None and tokenizer is not None:
            resp_len = len(tokenizer.tokenize(entry.get("response", "")))
            entry["response_len"] = resp_len
        if instr_len is not None:
            instruction_lengths.append(instr_len)
        if resp_len is not None:
            response_lengths.append(resp_len)

    avg_instruction_tokens = sum(instruction_lengths) / len(instruction_lengths) if instruction_lengths else 0
    avg_response_tokens = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    return avg_complexity, diversity, avg_instruction_tokens, avg_response_tokens



if __name__ == "__main__":
    import glob
    from tqdm import tqdm

    # 设置文件夹路径
    # folder_path = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/dataSelect/idea4_diff_size/qwen2_5_7b__math_math-1k/split5_strict_diff_nll"
    folder_path = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__general_general-1k"
    # folder_path = "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/dataSelect/idea4_diff_size"  # 可以设置为更上层的文件夹

    # 长度统计使用的 tokenizer（复用模型 tokenizer，避免重复加载）
    length_tokenizer = AutoTokenizer.from_pretrained(inference_config.model_name_or_path, use_fast=False)
    if length_tokenizer.pad_token is None:
        length_tokenizer.pad_token = length_tokenizer.eos_token

    # 创建输出子文件夹
    output_subfolder = os.path.join(folder_path, "complexity_annotated")
    os.makedirs(output_subfolder, exist_ok=True)
    logger.info(f"Output subfolder: {output_subfolder}")

    # 输出文件路径（汇总统计）
    output_file = f"{folder_path}/tag_analysis_summary.jsonl"
    summary_file = f"{folder_path}/tag_analysis_summary.json"  # 汇总所有结果的 JSON 文件

    jsonl_files = [f for f in glob.glob(os.path.join(folder_path, "*.jsonl"))]
    
    if not jsonl_files:
        logger.warning(f"No .jsonl files found in {folder_path}")
        exit(1)
    
    logger.info(f"Found {len(jsonl_files)} .jsonl files in {folder_path}")
    
    results = []
    
    # 如果输出文件已存在，先清空（或者可以选择追加模式）
    if os.path.exists(output_file):
        os.remove(output_file)
        logger.info(f"Removed existing output file: {output_file}")
    
    # 处理每个文件
    for filepath in tqdm(jsonl_files, desc="Processing files"):
        try:
            data = load_json(filepath)
            if not data:
                logger.warning(f"Empty file: {filepath}")
                continue
            
            # 为每个 item 添加 complexity，顺便记录长度
            logger.info(f"Adding complexity to items in: {os.path.basename(filepath)}")
            data_with_complexity = add_complexity_to_items(data, tokenizer=length_tokenizer)
            
            # 保存带 complexity 的数据到子文件夹
            rel_path = os.path.relpath(filepath, folder_path)
            name = os.path.basename(filepath).split('.')[0]
            output_filepath = os.path.join(output_subfolder, os.path.basename(filepath))
            
            # 如果文件已存在，先删除（因为 write_jsonlines 会跳过已存在的文件）
            if os.path.exists(output_filepath):
                os.remove(output_filepath)
            
            write_jsonlines(data_with_complexity, output_filepath)
            logger.info(f"Saved complexity-annotated data to: {output_filepath} ({len(data_with_complexity)} items)")
            
            # 计算统计信息（用于汇总）
            avg_complexity, diversity, avg_instruction_tokens, avg_response_tokens = get_infomation_request(
                data_with_complexity, tokenizer=length_tokenizer)
            
            result = {
                "filepath": filepath,
                "relative_path": rel_path,
                "name": name,
                "complexity": avg_complexity,
                "diversity": diversity, 
                "instruction_token": avg_instruction_tokens,
                "response_token": avg_response_tokens,
                "num_files": len(data),
                "output_filepath": output_filepath
            }
            results.append(result)
            
            # Save result immediately after processing each file (JSONL format)
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            logger.info(f"Processed: {name}")
            logger.info(f"  Average Complexity: {avg_complexity:.2f}")
            logger.info(f"  Diversity: {diversity}")
            logger.info(f"  Average Instruction Tokens: {avg_instruction_tokens:.2f}")
            logger.info(f"  Average Response Tokens: {avg_response_tokens:.2f}")
            logger.info(f"  Number of Samples: {len(data)}")
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 保存汇总结果到 JSON 文件
    summary = {
        "folder_path": folder_path,
        "output_subfolder": output_subfolder,
        "total_files_processed": len(results),
        "results": results,
        "statistics": {
            "avg_complexity": sum(r["complexity"] for r in results) / len(results) if results else 0,
            "avg_diversity": sum(r["diversity"] for r in results) / len(results) if results else 0,
            "avg_instruction_tokens": sum(r["instruction_token"] for r in results) / len(results) if results else 0,
            "avg_response_tokens": sum(r["response_token"] for r in results) / len(results) if results else 0,
            "total_samples": sum(r["num_files"] for r in results)
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Summary:")
    logger.info(f"  Total files processed: {len(results)}")
    logger.info(f"  Average complexity: {summary['statistics']['avg_complexity']:.2f}")
    logger.info(f"  Average diversity: {summary['statistics']['avg_diversity']:.2f}")
    logger.info(f"  Average instruction tokens: {summary['statistics']['avg_instruction_tokens']:.2f}")
    logger.info(f"  Average response tokens: {summary['statistics']['avg_response_tokens']:.2f}")
    logger.info(f"  Total samples: {summary['statistics']['total_samples']}")
    logger.info(f"\nResults saved to:")
    logger.info(f"  - Complexity-annotated data: {output_subfolder}")
    logger.info(f"  - JSONL format: {output_file}")
    logger.info(f"  - Summary JSON: {summary_file}")
    logger.info(f"{'='*60}")
    

