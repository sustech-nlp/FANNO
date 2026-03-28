"""JSONL / JSON file loading and saving utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

from loguru import logger
from tqdm import tqdm


def load_jsonlines(filepath: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL file, skipping malformed lines."""
    data: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {Path(filepath).name}")):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {i + 1} in {filepath}: {e}")
    return data


def save_jsonlines(
    data: List[Dict[str, Any]],
    filepath: str | Path,
    overwrite: bool = False,
) -> None:
    """Save data to a JSONL file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists() and not overwrite:
        logger.warning(f"File {filepath} already exists; set overwrite=True to replace.")
        return
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} records to {filepath}")


def load_json(filepath: str | Path) -> Union[List, Dict]:
    """Load a JSON or JSONL file."""
    filepath = str(filepath)
    if filepath.endswith(".jsonl"):
        return load_jsonlines(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    data: Union[List, Dict],
    filepath: str | Path,
    overwrite: bool = False,
) -> None:
    """Save data to a JSON or JSONL file."""
    filepath_str = str(filepath)
    if filepath_str.endswith(".jsonl"):
        save_jsonlines(data, filepath, overwrite=overwrite)
        return
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        logger.warning(f"File {p} already exists; set overwrite=True to replace.")
        return
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


__all__ = ["load_json", "load_jsonlines", "save_json", "save_jsonlines"]
