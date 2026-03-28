"""Training data preparation: load, filter, mix, and format datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


def load_fanno_synthesized(
    data_dir: str | Path,
    max_samples: int = 50000,
) -> List[Dict[str, Any]]:
    """Load FANNO synthesized data from outputs directory.

    Scans for JSONL files, quality-filters, and converts to Alpaca format.
    """
    data_dir = Path(data_dir)
    all_data: List[Dict[str, Any]] = []

    # Load all JSONL files in the directory
    jsonl_files = sorted(data_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No JSONL files found in {data_dir}")
        return []

    from fanno.data.loader import load_jsonlines

    for f in jsonl_files:
        items = load_jsonlines(f)
        all_data.extend(items)
        if len(all_data) >= max_samples * 2:
            break

    logger.info(f"Loaded {len(all_data)} raw samples from {len(jsonl_files)} files")

    # Quality filter: keep items with instruction + output
    filtered = [
        item for item in all_data
        if item.get("instruction", "").strip()
        and item.get("output", "").strip()
        and len(item["instruction"].split()) >= 5
        and len(item["output"].split()) >= 10
    ]

    # Sort by value score if available, take top samples
    filtered.sort(key=lambda x: x.get("value", 0), reverse=True)
    selected = filtered[:max_samples]

    logger.info(f"Quality-filtered: {len(selected)}/{len(all_data)} samples")
    return selected


def load_alpaca_cleaned(
    max_samples: int = 20000,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load yahma/alpaca-cleaned from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return []

    logger.info(f"Loading alpaca-cleaned (max {max_samples})")
    ds = load_dataset("yahma/alpaca-cleaned", split="train", cache_dir=cache_dir)

    data: List[Dict[str, Any]] = []
    for item in ds:
        data.append({
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["output"],
            "source": "alpaca-cleaned",
        })
        if len(data) >= max_samples:
            break

    logger.info(f"Loaded {len(data)} Alpaca samples")
    return data


def load_arena_hard(
    max_samples: int = 10000,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load ArenaHard v2 conversations and convert to Alpaca format."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return []

    logger.info(f"Loading ArenaHard v2 (max {max_samples})")
    try:
        ds = load_dataset("lmsys/arena-hard-auto-v0.1", split="train", cache_dir=cache_dir)
    except Exception:
        try:
            ds = load_dataset("lmarena-ai/arena-hard-auto", split="train", cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Could not load ArenaHard: {e}. Trying alternative...")
            return []

    data: List[Dict[str, Any]] = []
    for item in ds:
        # Extract instruction from the first user turn
        turns = item.get("turns", [])
        if turns:
            instruction = turns[0] if isinstance(turns[0], str) else str(turns[0])
        elif item.get("question"):
            instruction = item["question"]
        else:
            continue

        data.append({
            "instruction": instruction,
            "input": "",
            "output": "",  # ArenaHard provides questions only; output generated separately
            "source": "arena-hard",
        })
        if len(data) >= max_samples:
            break

    logger.info(f"Loaded {len(data)} ArenaHard samples")
    return data


def load_bfcl_v4(
    max_samples: int = 15000,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load BFCLv4 function-calling data from gorilla-llm."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return []

    logger.info(f"Loading BFCLv4 (max {max_samples})")
    try:
        ds = load_dataset(
            "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
            split="train",
            cache_dir=cache_dir,
        )
    except Exception as e:
        logger.warning(f"Could not load BFCLv4: {e}")
        return []

    data: List[Dict[str, Any]] = []
    for item in ds:
        # BFCLv4 has various formats; extract instruction and function definitions
        entry: Dict[str, Any] = {
            "instruction": item.get("question", item.get("instruction", "")),
            "input": "",
            "output": item.get("answer", item.get("output", "")),
            "source": "bfcl-v4",
        }
        # Preserve function definitions if available
        if item.get("function"):
            entry["functions"] = item["function"]
        if item.get("tools"):
            entry["tools"] = item["tools"]
        data.append(entry)
        if len(data) >= max_samples:
            break

    logger.info(f"Loaded {len(data)} BFCLv4 samples")
    return data


def mix_datasets(
    fanno_qa: List[Dict[str, Any]],
    fanno_agent: Optional[List[Dict[str, Any]]] = None,
    alpaca: Optional[List[Dict[str, Any]]] = None,
    arena: Optional[List[Dict[str, Any]]] = None,
    bfcl: Optional[List[Dict[str, Any]]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Combine and shuffle multiple datasets.

    Args:
        fanno_qa: FANNO synthesized QA data.
        fanno_agent: FANNO agent trajectory data.
        alpaca: Alpaca-cleaned data.
        arena: ArenaHard data.
        bfcl: BFCLv4 data.
        seed: Random seed for reproducibility.

    Returns:
        Shuffled combined dataset.
    """
    combined: List[Dict[str, Any]] = list(fanno_qa)

    for dataset, name in [
        (fanno_agent, "fanno-agent"),
        (alpaca, "alpaca"),
        (arena, "arena-hard"),
        (bfcl, "bfcl-v4"),
    ]:
        if dataset:
            combined.extend(dataset)
            logger.info(f"Added {len(dataset)} samples from {name}")

    random.seed(seed)
    random.shuffle(combined)

    logger.info(f"Mixed dataset: {len(combined)} total samples")

    # Log source distribution
    sources: Dict[str, int] = {}
    for item in combined:
        src = item.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items()):
        logger.info(f"  {src}: {count} ({count / len(combined):.1%})")

    return combined


def save_training_data(
    data: List[Dict[str, Any]],
    output_path: str | Path,
    fmt: str = "jsonl",
) -> None:
    """Save training data in specified format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jsonl":
        from fanno.data.loader import save_jsonlines
        save_jsonlines(data, output_path, overwrite=True)
    elif fmt == "json":
        with open(output_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    logger.info(f"Saved {len(data)} training samples to {output_path}")


def main():
    """CLI entry point for data preparation."""
    parser = argparse.ArgumentParser(description="Prepare FANNO training data")
    parser.add_argument("--output-dir", type=str, default="./train_data")
    parser.add_argument("--fanno-dir", type=str, default="./outputs")
    parser.add_argument("--max-fanno-qa", type=int, default=50000)
    parser.add_argument("--max-fanno-agent", type=int, default=5000)
    parser.add_argument("--max-alpaca", type=int, default=20000)
    parser.add_argument("--max-arena", type=int, default=10000)
    parser.add_argument("--max-bfcl", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-external", action="store_true", help="Skip HuggingFace downloads")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load FANNO data
    fanno_qa = load_fanno_synthesized(args.fanno_dir, max_samples=args.max_fanno_qa)

    # Load external datasets
    alpaca_data = [] if args.skip_external else load_alpaca_cleaned(max_samples=args.max_alpaca)
    arena_data = [] if args.skip_external else load_arena_hard(max_samples=args.max_arena)
    bfcl_data = [] if args.skip_external else load_bfcl_v4(max_samples=args.max_bfcl)

    # Mix
    mixed = mix_datasets(
        fanno_qa=fanno_qa,
        alpaca=alpaca_data,
        arena=arena_data,
        bfcl=bfcl_data,
        seed=args.seed,
    )

    # Save
    save_training_data(mixed, output_dir / "train.jsonl")

    # Save a small validation set
    val_size = min(1000, len(mixed) // 20)
    save_training_data(mixed[:val_size], output_dir / "val.jsonl")
    save_training_data(mixed[val_size:], output_dir / "train.jsonl")

    logger.info(f"Training data prepared: {len(mixed) - val_size} train, {val_size} val")


if __name__ == "__main__":
    main()
