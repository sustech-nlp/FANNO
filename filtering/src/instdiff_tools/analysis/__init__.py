import json
from tqdm import tqdm
import os
from loguru import logger
import re

def load_jsonlines(file_path):
    '''
    Load data from a JSONLines file
    '''
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in tqdm(lines, desc="Loading JSONLines")]
    return data

def write_jsonlines(data, file_path):
    '''
    Write data to a JSONLines file
    '''
    if os.path.exists(file_path):
        logger.info(f"Skipping operation because {file_path} already exists.")
        return
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in tqdm(data, desc="Writing JSONLines"):
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

def load_json(file_path):
    '''
    Load data from a JSON file
    '''
    if file_path.endswith(".jsonl"):
        return load_jsonlines(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(data, file_path):
    '''
    Write data to a JSON file
    '''
    with open(file_path, 'w', encoding='utf-8') as f:
        if isinstance(data, list):  # 如果 data 是列表，逐项写入以显示进度
            f.write('[')
            for i, entry in enumerate(tqdm(data, desc="Writing JSON")):
                json.dump(entry, f, ensure_ascii=False)
                if i < len(data) - 1:
                    f.write(', ')
            f.write(']')
        else:  # 非列表情况下直接写入
            json.dump(data, f, ensure_ascii=False)
            
def parser_json_string(json_string):
    '''
    Parse a JSON string and return the corresponding object
    '''
    try:
        data = json.loads(json_string)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON string: {e}")
        return None
