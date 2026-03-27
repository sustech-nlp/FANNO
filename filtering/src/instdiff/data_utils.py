from typing import List, Dict, Any, Optional
import random

from .io_utils import load_jsonl, write_jsonl


def sample_data(
    data_path: str,
    output_path: str,
    max_samples: int,
    seed: int = 42,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    data = load_jsonl(data_path)
    if limit is not None:
        data = data[:limit]
    random.seed(seed)
    if max_samples >= len(data):
        sampled = data
    else:
        sampled = random.sample(data, max_samples)
    write_jsonl(sampled, output_path)
    return sampled
