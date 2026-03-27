"""
End-to-end orchestration for the diversity comparison experiment.

The script mirrors the design in the project README:
1) Load dataset (Magpie or user-provided).
2) Extract embeddings (semantic + gradient).
3) Run diversification strategies to pick subsets.
4) Compute diversity metrics for each (embedding, strategy) pair.
5) Optionally fine-tune and evaluate models (hook points provided).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm

from .diversity_metrics import compute_all_diversity_metrics
from .embedding_extraction import (
    concat_embeddings,
    extract_gradient_embeddings,
    extract_semantic_embeddings,
)
from .selection_strategies import select_diverse_samples

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_data(
    dataset_cfg: Dict,
) -> Tuple[Iterable[dict], Iterable[dict]]:
    """
    Returns (train_pool, test_set) iterables.
    """
    seed = dataset_cfg.get("seed", 42)
    train_size = dataset_cfg.get("train_size", 90_000)
    test_size = dataset_cfg.get("test_size", 10_000)
    if test_size < 0:
        test_size = 0

    if "local_path" in dataset_cfg:
        data = _load_jsonl(Path(dataset_cfg["local_path"]))
        logger.info("Loaded local dataset from %s (%d samples)", dataset_cfg["local_path"], len(data))
        if len(data) < train_size + test_size:
            raise ValueError("Local dataset is smaller than requested split.")
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(data))
        train_idx = indices[:train_size]
        test_idx = indices[train_size : train_size + test_size]
        train = [data[i] for i in train_idx]
        test = [data[i] for i in test_idx]
        return train, test

    # Hugging Face dataset path
    dataset_id = dataset_cfg.get("hf_id", "Magpie-Align/Magpie-Pro-300K-Filtered")
    subset_size = train_size + test_size
    logger.info("Fetching HF dataset %s (subset=%d)", dataset_id, subset_size)
    ds = load_dataset(dataset_id)["train"].shuffle(seed=seed).select(range(subset_size))
    train = ds.select(range(train_size))
    if test_size > 0:
        test = ds.select(range(train_size, subset_size))
    else:
        test = []
    return train, test


def maybe_load_embeddings(path: Path) -> np.ndarray | None:
    if path.exists():
        return np.load(path)
    return None


def run_embeddings(
    train_pool: Iterable[dict],
    embedding_cfgs: Dict[str, Dict],
    cache_dir: Path,
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings for all configured methods, caching to disk.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings: Dict[str, np.ndarray] = {}
    for name, cfg in tqdm(list(embedding_cfgs.items()), desc="Embeddings"):
        out_path = cache_dir / f"{name}.npy"
        cached = maybe_load_embeddings(out_path)
        if cached is not None:
            expected_len = None
            try:
                expected_len = len(train_pool)  # type: ignore[arg-type]
            except Exception:
                expected_len = None
            if expected_len is not None and cached.shape[0] != expected_len:
                logger.warning(
                    "Cached embeddings for %s at %s have length %d != expected %d; recomputing.",
                    name,
                    out_path,
                    cached.shape[0],
                    expected_len,
                )
            else:
                logger.info("Using cached embeddings for %s at %s", name, out_path)
                embeddings[name] = cached
                continue

        emb_type = cfg.get("type", "semantic")
        model_id = cfg["model_id"]
        logger.info("Extracting %s embeddings [%s] with model=%s", emb_type, name, model_id)
        if emb_type == "semantic":
            embs = extract_semantic_embeddings(
                train_pool,
                model_id=model_id,
                batch_size=cfg.get("batch_size", 8),
                max_length=cfg.get("max_length", 256),
                device=cfg.get("device"),
            )
        elif emb_type == "gradient":
            embs = extract_gradient_embeddings(
                train_pool,
                proxy_model_id=model_id,
                proj_dim=cfg.get("proj_dim", 1024),
                max_length=cfg.get("max_length", 256),
                device=cfg.get("device"),
            )
        elif emb_type == "hybrid":
            # Build semantic + gradient concatenation
            sem = extract_semantic_embeddings(
                train_pool,
                model_id=cfg["semantic_model_id"],
                batch_size=cfg.get("batch_size", 8),
                max_length=cfg.get("max_length", 256),
                device=cfg.get("device"),
            )
            grad = extract_gradient_embeddings(
                train_pool,
                proxy_model_id=cfg["proxy_model_id"],
                proj_dim=cfg.get("proj_dim", 1024),
                max_length=cfg.get("max_length", 256),
                device=cfg.get("device"),
            )
            embs = concat_embeddings(grad, sem)
        else:
            raise ValueError(f"Unknown embedding type: {emb_type}")

        np.save(out_path, embs)
        logger.info("Saved embeddings for %s to %s", name, out_path)
        embeddings[name] = embs
    return embeddings


def run_selection(
    embeddings: Dict[str, np.ndarray],
    train_pool: Iterable[dict],
    selection_cfgs: List[Dict],
    n_select: int,
    timings: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict]:
    """
    Run each diversification strategy for each embedding space.
    """
    results: Dict[str, Dict] = {}
    total_steps = len(embeddings) * len(selection_cfgs)
    with tqdm(total=total_steps, desc="Selection") as pbar:
        for emb_name, emb_array in embeddings.items():
            for cfg in selection_cfgs:
                strategy = cfg["name"]
                params = cfg.get("params", {})
                key = f"{emb_name}_{strategy}"
                logger.info("Selecting %d samples using %s on %s", n_select, strategy, emb_name)
                start = perf_counter()
                indices = select_diverse_samples(
                    emb_array,
                    strategy=strategy,
                    n_select=n_select,
                    **params,
                )
                elapsed = perf_counter() - start
                if timings is not None:
                    timings.append(
                        {"embedding": emb_name, "strategy": strategy, "seconds": elapsed}
                    )
                subset = train_pool.select(indices) if hasattr(train_pool, "select") else [train_pool[i] for i in indices]  # type: ignore
                results[key] = {"indices": indices, "dataset": subset}
                pbar.update(1)
    return results


def run_diversity_metrics(
    results: Dict[str, Dict],
    embeddings: Dict[str, np.ndarray],
    reference_key: str,
    metrics_cfg: Dict | None = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    metrics_cfg = metrics_cfg or {}
    novelsum_mode = metrics_cfg.get("novelsum_mode", "fast")
    novelsum_sample_size = metrics_cfg.get("novelsum_sample_size", 5000)
    pairwise_sample_size = metrics_cfg.get("pairwise_sample_size", 2000)
    kmeans_clusters = metrics_cfg.get("kmeans_clusters", 100)

    records = []
    timings: List[Dict[str, Any]] = []
    for key, data in tqdm(list(results.items()), desc="Diversity metrics"):
        indices = data["indices"]
        emb_subset = {name: emb[indices] for name, emb in embeddings.items()}
        logger.info("Computing diversity metrics for %s", key)
        metric_times: Dict[str, float] = {}
        scores = compute_all_diversity_metrics(
            emb_subset,
            reference_key=reference_key,
            kmeans_clusters=kmeans_clusters,
            novelsum_sample_size=novelsum_sample_size,
            pairwise_sample_size=pairwise_sample_size,
            novelsum_mode=novelsum_mode,
            timings=metric_times,
        )
        scores["config"] = key
        records.append(scores)
        metric_times["config"] = key
        timings.append(metric_times)
    return pd.DataFrame(records), timings


def save_indices(results: Dict[str, Dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, data in results.items():
        path = out_dir / f"{key}_indices.json"
        with path.open("w") as f:
            json.dump(data["indices"], f)


def main():
    parser = argparse.ArgumentParser(description="Run diversity comparison experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Reuse cached embeddings if present.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger.info("Loading data using config: %s", args.config)
    train_pool, test_set = load_data(cfg["dataset"])
    if hasattr(train_pool, "__len__"):
        logger.info("Train pool size: %d", len(train_pool))  # type: ignore[arg-type]
    if hasattr(test_set, "__len__"):
        logger.info("Test set size: %d", len(test_set))  # type: ignore[arg-type]

    embedding_cfgs = cfg["embedding_configs"]
    cache_dir = Path(cfg.get("embedding_cache_dir", "data/embeddings"))
    if args.skip_embeddings:
        embeddings = {
            name: np.load(cache_dir / f"{name}.npy")
            for name in embedding_cfgs.keys()
        }
        logger.info("Skipping embedding extraction; loaded from cache at %s", cache_dir)
    else:
        embeddings = run_embeddings(train_pool, embedding_cfgs, cache_dir)

    selection_cfgs = cfg["selection_strategies"]
    n_select = cfg["dataset"].get("n_select", 10_000)
    logger.info("Running selection strategies for %d samples per config", n_select)
    selection_timings: List[Dict[str, Any]] = []
    selection_results = run_selection(
        embeddings,
        train_pool,
        selection_cfgs,
        n_select=n_select,
        timings=selection_timings,
    )
    selection_dir = Path(cfg.get("selection_dir", "results/selections"))
    save_indices(selection_results, selection_dir)
    # Record selection timings
    (selection_dir).mkdir(parents=True, exist_ok=True)
    with (selection_dir / "selection_timings.json").open("w") as f:
        json.dump(selection_timings, f, indent=2)

    logger.info("Computing diversity metrics...")
    diversity_df, metrics_timings = run_diversity_metrics(
        selection_results,
        embeddings,
        reference_key=cfg.get("reference_embedding", cfg.get("metrics", {}).get("reference_embedding", "llama")),
        metrics_cfg=cfg.get("metrics"),
    )
    diversity_scores_path = Path(cfg.get("diversity_scores_path", "results/diversity_scores.csv"))
    diversity_df.to_csv(diversity_scores_path, index=False)
    metrics_timings_path = diversity_scores_path.with_suffix(".times.json")
    with metrics_timings_path.open("w") as f:
        json.dump(metrics_timings, f, indent=2)

    # Training/evaluation hooks can be added here as needed.
    logger.info("Diversity metrics written to %s", cfg.get("diversity_scores_path", "results/diversity_scores.csv"))


if __name__ == "__main__":
    main()
