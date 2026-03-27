"""
Selection Strategy Comparison Experiment.
Compares diversity of subsets selected by different strategies.

Uses FANNO-Dev synthesized data + diversity_metric toolkit's
selection strategies and Vendi Score metrics.
"""
from __future__ import annotations

import json
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from diversity_metric.src.selection_strategies import select_diverse_samples
from diversity_metric.src.diversity_metrics import (
    compute_vendi_score,
    compute_avg_pairwise_distance_fast,
    compute_cluster_inertia,
    compute_dominance_score,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_and_embed(
    output_dir: Path = None,
    model_name: str = "all-MiniLM-L6-v2",
    max_samples: int = 20000,
) -> tuple:
    """Load data, extract texts, compute embeddings."""
    from sentence_transformers import SentenceTransformer

    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Load cleaned data
    data = []
    texts = []
    for fname in ["cleaned_single_turn.jsonl", "cleaned_multi_turn.jsonl"]:
        fpath = output_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        data.append(item)
                        q = item.get("question", item.get("instruction", ""))
                        if isinstance(q, str) and q.strip():
                            texts.append(q.strip())
                        elif "conversation" in item:
                            conv = item["conversation"]
                            if isinstance(conv, list):
                                for t in conv:
                                    if isinstance(t, dict) and t.get("role") == "user":
                                        texts.append(t.get("content", "").strip())
                                        break

    logger.info(f"Loaded {len(data)} samples, {len(texts)} texts")

    # Subsample for tractability
    if len(texts) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(texts), size=max_samples, replace=False)
        texts = [texts[i] for i in indices]
        logger.info(f"Subsampled to {max_samples} texts")

    # Compute embeddings
    logger.info(f"Computing embeddings with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")

    return texts, embeddings


def evaluate_subset(embeddings: np.ndarray, indices: List[int], n_clusters: int = 30) -> Dict:
    """Evaluate diversity metrics for a selected subset."""
    subset = embeddings[indices]
    return {
        "n_selected": len(indices),
        "vendi_score": compute_vendi_score(subset),
        "avg_pairwise_distance": compute_avg_pairwise_distance_fast(subset, sample_size=min(2000, len(subset))),
        "cluster_inertia": compute_cluster_inertia(subset, n_clusters=min(n_clusters, len(subset) - 1)),
    }


def run_selection_comparison(
    output_dir: Path = None,
    n_pool: int = 10000,
    select_sizes: List[int] = None,
    strategies: List[str] = None,
) -> Dict:
    """Compare selection strategies on FANNO-Dev data."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if select_sizes is None:
        select_sizes = [500, 1000, 2000, 5000]
    if strategies is None:
        strategies = [
            "random",
            "kmeans",
            "community",
            "k_center_greedy",
            "herding",
            "coreset",
            "stratified",
        ]

    texts, embeddings = load_and_embed(output_dir, max_samples=n_pool)

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_pool": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "strategies": strategies,
        "select_sizes": select_sizes,
        "experiments": {},
    }

    for n_select in select_sizes:
        if n_select > len(embeddings):
            logger.warning(f"Skipping n_select={n_select} (larger than pool)")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Selection size: {n_select}")
        logger.info(f"{'='*60}")

        size_results = {}

        for strategy in strategies:
            logger.info(f"  Running {strategy}...")
            try:
                start = time.time()
                indices = select_diverse_samples(
                    embeddings,
                    strategy=strategy,
                    n_select=n_select,
                    n_clusters=min(100, n_select),
                )
                elapsed = time.time() - start

                metrics = evaluate_subset(embeddings, indices)
                metrics["time_seconds"] = elapsed
                size_results[strategy] = metrics

                logger.info(
                    f"    {strategy}: Vendi={metrics['vendi_score']:.2f}, "
                    f"AvgDist={metrics['avg_pairwise_distance']:.4f}, "
                    f"Time={elapsed:.2f}s"
                )
            except Exception as e:
                logger.error(f"    {strategy} failed: {e}")
                size_results[strategy] = {"error": str(e)}

        results["experiments"][str(n_select)] = size_results

    # Print comparison table
    print("\n" + "=" * 100)
    print("SELECTION STRATEGY COMPARISON")
    print("=" * 100)
    print(f"Pool size: {len(embeddings):,}")
    print()

    for n_select in select_sizes:
        key = str(n_select)
        if key not in results["experiments"]:
            continue
        exp = results["experiments"][key]
        print(f"\n--- n_select = {n_select:,} ---")
        print(f"{'Strategy':<25} {'Vendi':>10} {'AvgDist':>10} {'Inertia':>12} {'Time(s)':>10}")
        print("-" * 67)

        sorted_strategies = sorted(
            exp.items(),
            key=lambda x: x[1].get("vendi_score", 0) if "error" not in x[1] else 0,
            reverse=True,
        )
        for strategy, metrics in sorted_strategies:
            if "error" in metrics:
                print(f"  {strategy:<23} {'ERROR':>10}")
            else:
                print(
                    f"  {strategy:<23} {metrics['vendi_score']:>10.2f} "
                    f"{metrics['avg_pairwise_distance']:>10.4f} "
                    f"{metrics['cluster_inertia']:>12.2f} "
                    f"{metrics['time_seconds']:>10.2f}"
                )

    print("\n" + "=" * 100)

    # Save results
    report_path = output_dir / "selection_comparison.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {report_path}")

    return results


if __name__ == "__main__":
    run_selection_comparison(
        n_pool=10000,
        select_sizes=[500, 1000, 2000, 5000],
    )
