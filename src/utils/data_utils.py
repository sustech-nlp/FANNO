from tqdm import tqdm
import os
import json
import random
from typing import List, Dict, Tuple
import re
from functools import wraps
from loguru import logger

def instruction_cleaning(texts: List[str]) -> List[Tuple[str, str]]:
    '''
    Cleans a list of instruction texts by removing leading/trailing special characters,
    and splits each text into two parts at the first newline character.
    Returns a list of (part1, part2) tuples.
    '''
    # Only process leading and trailing symbols
    cleaned_texts = [re.sub(r'^[*"\n]+|[*"\n]+$', '', text).strip() for text in texts]
    
    def process_text(text: str) -> tuple:
        # If there is a newline, split into two parts
        if '\n' in text:
            part1, part2 = text.split('\n', 1)
            return part1.strip(), part2.strip()
        else:
            return text.strip(), ""
    
    return [process_text(text) for text in cleaned_texts]


def load_jsonlines(filepath):
    '''
    Loads a JSONL (JSON Lines) file and returns a list of parsed JSON objects.
    Skips lines that cannot be decoded as JSON.
    '''
    data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {i + 1}: {e}")
                print(f"Problematic line: {line}")
                continue  # Skip this line and continue with the next
    return data

def save_jsonlines(data, filepath):
    '''
    Saves a list of data (dicts) to a JSONL (JSON Lines) file.
    If the file already exists, does nothing and prints a warning.
    '''
    if os.path.exists(filepath):
        print("File exists, please check the path.")
        return
    with open(filepath, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def load_json(filepath):
    '''
    Loads a JSON or JSONL file and returns the parsed data.
    Uses load_jsonlines for .jsonl files, and json.load for .json files.
    '''
    if filepath.endswith(".jsonl"):
        return load_jsonlines(filepath)
    with open(filepath, "r") as f:
        return json.load(f)
    
def save_json(data, filepath):
    '''
    Saves data to a JSON or JSONL file.
    Uses save_jsonlines for .jsonl files, and json.dump for .json files.
    If the file already exists, does nothing and prints a warning.
    '''
    if os.path.exists(filepath):
        print("File exists, please check the path.")
        return
    if filepath.endswith(".jsonl"):
        save_jsonlines(data, filepath)
        return
    with open(filepath, "w") as f:
        f.write(json.dumps(data))


def load_unlabeled_data(unlabeled_pth: str, sample_k: int, random_bool: bool = False) -> List[Dict[str, str]]:    
    '''
    Loads unlabeled data from a JSONL file and returns a list of dicts.
    If random_bool is True, samples sample_k items randomly (with a fixed seed).
    Otherwise, returns the first sample_k items.
    '''
    data = load_jsonlines(unlabeled_pth)
    if random_bool:
        if len(data) < sample_k:
            sample_k = len(data)
        random.seed(42)
        data = random.sample(data, sample_k)
    else:
        if len(data) < sample_k:
            sample_k = len(data)
        data = data[:sample_k]
    return data


def load_documents(file_path: str) -> List[str]:
    '''
    Loads documents from a JSONL file, extracting the 'doc' field from each line.
    Cleans up newlines and removes unwanted characters.
    Returns a list of cleaned document strings.
    '''
    documents = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Loading documents"):
            data = json.loads(line)
            doc = data.get('doc', '').strip()
            doc = re.sub(r'\n+', '\n', doc)
            doc = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\n,.!?<>:，、：《》。！？\s]+', '', doc)
            if doc:
                documents.append(doc)
    return documents

def save_document_embeddings(embeddings, output_path):
    '''
    Saves document embeddings (numpy array) to a .npy file.
    '''
    np.save(output_path, embeddings)
    print(f"Document embeddings saved to {output_path}")
    
def load_document_embeddings(file_path):
    '''
    Loads document embeddings from a .npy file and returns a torch tensor on CUDA.
    '''
    print(f"Document embeddings loaded from {file_path}")
    return torch.tensor(np.load(file_path)).to('cuda')



def get_unlabeled_data(config, merge_bool):
    '''
    Loads unlabeled data from config paths.
    If merge_bool is True, merges data from multiple sources.
    Returns a list of document strings.
    '''
    unlabeled_pth = config.unlabeled_data_pth
    com_unlabeled_pth = config.com_unlabeled_data_pth
    data = load_json(unlabeled_pth)
    com_data = load_json(com_unlabeled_pth)
    docs = [d["doc"] for d in data]
    com_docs = [d["doc"] for d in com_data] 
    if merge_bool:
        docs += com_docs
    return docs
    


# Decorator to check if a file exists before executing a function
def save_or_skip(file_pth):
    '''
    Decorator that checks if a file exists before executing the decorated function.
    If the file exists, loads and returns its contents.
    Otherwise, executes the function, saves the result, and returns it.
    '''
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(file_pth):
                logger.info(f"Skipping operation because {file_pth} already exists.")
                return load_jsonlines(file_pth)
            result = func(*args, **kwargs)
        
            os.makedirs(os.path.dirname(file_pth), exist_ok=True)
            save_jsonlines(result, file_pth)
            return result
        return wrapper
    return decorator

# Decorator to check if a file exists before executing a function, with dynamic file path
def save_or_skip_dynamic(parameter_name):
    '''
    Decorator that checks if a file exists before executing the decorated function.
    The file path is determined dynamically from the function's kwargs using parameter_name.
    If the file exists, loads and returns its contents.
    Otherwise, executes the function, saves the result, and returns it.
    '''
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the file_path parameter
            file_pth = kwargs.get(parameter_name)
            if not file_pth:
                raise ValueError(f"Parameter '{parameter_name}' not provided in kwargs.")
            
            if os.path.exists(file_pth):
                logger.info(f"Skipping operation because {file_pth} already exists.")
                return load_jsonlines(file_pth)
            
            result = func(*args, **kwargs)
        
            os.makedirs(os.path.dirname(file_pth), exist_ok=True)
            save_jsonlines(result, file_pth)
            return result
        return wrapper
    return decorator



if __name__ == "__main__":
    # Test: instruction_cleaning
    texts = ["\"Hello you are a good. \n I am a \n tSET\""]
    print(instruction_cleaning(texts))
