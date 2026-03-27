"""
Quick debug utility to inspect embedding quality on a small sample.

For a given embedding config name (from configs/experiment_config.yaml by default),
the script:
  1) loads the dataset with the experiment loader,
  2) takes the first N examples,
  3) computes embeddings for those examples,
  4) reports cosine-similarity stats to surface degenerate cases.

Usage:
    python scripts/debug_embeddings.py \
        --config configs/experiment_config.yaml \
        --embedding-name bert \
        --n 100
"""
from __future__ import annotations

import argparse
import numpy as np
import yaml
from pathlib import Path
import sys
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.experiment import load_data
from src.embedding_extraction import (
    concat_embeddings,
    extract_gradient_embeddings,
    extract_semantic_embeddings,
)


def _build_embeddings(name: str, cfg: dict, dataset) -> np.ndarray:
    emb_type = cfg.get("type", "semantic")
    if emb_type == "semantic":
        return extract_semantic_embeddings(
            dataset,
            model_id=cfg["model_id"],
            batch_size=cfg.get("batch_size", 8),
            max_length=cfg.get("max_length", 256),
            device=cfg.get("device"),
        )
    if emb_type == "gradient":
        return extract_gradient_embeddings(
            dataset,
            proxy_model_id=cfg["model_id"],
            proj_dim=cfg.get("proj_dim", 1024),
            max_length=cfg.get("max_length", 256),
            device=cfg.get("device"),
        )
    if emb_type == "hybrid":
        sem = extract_semantic_embeddings(
            dataset,
            model_id=cfg["semantic_model_id"],
            batch_size=cfg.get("batch_size", 8),
            max_length=cfg.get("max_length", 256),
            device=cfg.get("device"),
        )
        grad = extract_gradient_embeddings(
            dataset,
            proxy_model_id=cfg["proxy_model_id"],
            proj_dim=cfg.get("proj_dim", 1024),
            max_length=cfg.get("max_length", 256),
            device=cfg.get("device"),
        )
        return concat_embeddings(grad, sem)
    raise ValueError(f"Unsupported embedding type '{emb_type}' for {name}")


def _cosine_stats(embs: np.ndarray) -> dict:
    # Normalize rows to unit norm for cosine.
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    normed = embs / norms

    if len(normed) < 2:
        return {"note": "need at least 2 rows for cosine stats"}

    dists = pairwise_distances(normed, metric="cosine")
    mask = np.triu(np.ones_like(dists, dtype=bool), k=1)
    cos_vals = 1.0 - dists[mask]

    unique_exact = np.unique(normed, axis=0).shape[0]
    # Near-duplicate detection: round to 6 decimals.
    rounded = np.unique(np.round(normed, 6), axis=0).shape[0]

    return {
        "pairs": cos_vals.size,
        "cos_mean": float(cos_vals.mean()),
        "cos_min": float(cos_vals.min()),
        "cos_max": float(cos_vals.max()),
        "row_norm_mean": float(norms.mean()),
        "unique_rows_exact": int(unique_exact),
        "unique_rows_round6": int(rounded),
    }


def main():
    parser = argparse.ArgumentParser(description="Debug embedding cosine similarity.")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--embedding-name", default="bert", help="Name in embedding_configs")
    parser.add_argument("--n", type=int, default=100, help="Number of samples to embed")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.embedding_name not in cfg.get("embedding_configs", {}):
        available = ", ".join(cfg.get("embedding_configs", {}).keys())
        raise SystemExit(f"embedding-name '{args.embedding_name}' not found. Available: {available}")

    dataset_cfg = dict(cfg["dataset"])
    # Keep dataset load light but ensure we have at least n samples.
    dataset_cfg["train_size"] = max(args.n, dataset_cfg.get("train_size", args.n))
    dataset_cfg["test_size"] = 0

    train_pool, _ = load_data(dataset_cfg)
    subset = (
        train_pool.select(range(min(args.n, len(train_pool))))  # type: ignore[arg-type]
        if hasattr(train_pool, "select")
        else list(train_pool)[: args.n]
    )
    print(f"Loaded {len(subset)} samples for debugging.")

    emb_cfg = cfg["embedding_configs"][args.embedding_name]
    print(f"Computing embeddings for '{args.embedding_name}' ({emb_cfg.get('type', 'semantic')})...")
    embs = _build_embeddings(args.embedding_name, emb_cfg, subset)
    print(f"Embeddings shape: {embs.shape}")

    stats = _cosine_stats(embs)
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
