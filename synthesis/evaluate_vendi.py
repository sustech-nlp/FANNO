"""
Embedding-based diversity evaluation using Vendi Score and other metrics.
Uses sentence-transformers for efficient embedding extraction,
then computes Vendi Score, pairwise distance, and dominance metrics
from diversity_metric toolkit.
"""
from __future__ import annotations

import json
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from loguru import logger

# Add parent dir to path for diversity_metric imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from diversity_metric.src.diversity_metrics import (
    compute_vendi_score,
    compute_avg_pairwise_distance_fast,
    compute_dominance_score,
    compute_cluster_inertia,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_cleaned_data(output_dir: Path = None) -> List[Dict]:
    """Load cleaned data from output directory."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    data = []
    for fname in ["cleaned_single_turn.jsonl", "cleaned_multi_turn.jsonl"]:
        fpath = output_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
    return data


def extract_texts(data: List[Dict]) -> List[str]:
    """Extract question/instruction texts from data."""
    texts = []
    for item in data:
        q = item.get("question", item.get("instruction", ""))
        if isinstance(q, str) and q.strip():
            texts.append(q.strip())
        elif "conversation" in item:
            conv = item["conversation"]
            if isinstance(conv, list):
                for t in conv:
                    if isinstance(t, dict) and t.get("role") == "user":
                        content = t.get("content", "")
                        if content:
                            texts.append(content.strip())
                        break
    return texts


def compute_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """Compute embeddings using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    if max_samples and len(texts) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(texts), size=max_samples, replace=False)
        texts = [texts[i] for i in indices]
        logger.info(f"Sampled {max_samples} texts for embedding")

    logger.info(f"Computing embeddings for {len(texts)} texts with {model_name}")
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def evaluate_vendi_diversity(
    output_dir: Path = None,
    model_name: str = "all-MiniLM-L6-v2",
    max_samples: int = 10000,
    kmeans_clusters: int = 50,
) -> Dict:
    """Run full Vendi Score-based diversity evaluation."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    logger.info("Loading cleaned data...")
    data = load_cleaned_data(output_dir)
    logger.info(f"Loaded {len(data)} cleaned samples")

    texts = extract_texts(data)
    logger.info(f"Extracted {len(texts)} texts")

    # Compute embeddings
    embeddings = compute_embeddings(
        texts, model_name=model_name, max_samples=max_samples
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "total_samples": len(data),
        "embedded_samples": len(embeddings),
        "embedding_dim": embeddings.shape[1],
    }

    # Vendi Score
    logger.info("Computing Vendi Score...")
    vendi = compute_vendi_score(embeddings)
    results["vendi_score"] = vendi
    logger.info(f"Vendi Score: {vendi:.4f}")

    # Average Pairwise Cosine Distance
    logger.info("Computing avg pairwise distance...")
    avg_dist = compute_avg_pairwise_distance_fast(embeddings, sample_size=2000)
    results["avg_pairwise_cosine_distance"] = avg_dist
    logger.info(f"Avg pairwise cosine distance: {avg_dist:.4f}")

    # Cluster Inertia
    logger.info(f"Computing cluster inertia (k={kmeans_clusters})...")
    inertia = compute_cluster_inertia(embeddings, n_clusters=kmeans_clusters)
    results["cluster_inertia"] = inertia
    results["kmeans_clusters"] = kmeans_clusters
    logger.info(f"Cluster inertia: {inertia:.4f}")

    # Dominance Score
    logger.info("Computing dominance score...")
    dominance = compute_dominance_score(embeddings)
    results["dominance_scores"] = dominance
    logger.info(f"Dominance scores: {dominance}")

    # Subsample analysis: how does diversity scale?
    logger.info("\n=== Scale Analysis ===")
    scale_results = []
    for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
        n = int(len(embeddings) * frac)
        if n < 10:
            continue
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embeddings), size=n, replace=False)
        sub_emb = embeddings[idx]
        sub_vendi = compute_vendi_score(sub_emb)
        sub_dist = compute_avg_pairwise_distance_fast(sub_emb, sample_size=min(2000, n))
        scale_results.append({
            "fraction": frac,
            "n_samples": n,
            "vendi_score": sub_vendi,
            "avg_pairwise_distance": sub_dist,
        })
        logger.info(f"  {frac:.0%} ({n:,} samples): Vendi={sub_vendi:.2f}, AvgDist={sub_dist:.4f}")
    results["scale_analysis"] = scale_results

    # Per-source analysis
    logger.info("\n=== Per-Source Analysis ===")
    source_texts = {}
    for item in data:
        src = item.get("source", "unknown")
        q = item.get("question", item.get("instruction", ""))
        if isinstance(q, str) and q.strip():
            if src not in source_texts:
                source_texts[src] = []
            source_texts[src].append(q.strip())

    source_results = {}
    for src, src_texts in sorted(source_texts.items(), key=lambda x: -len(x[1])):
        if len(src_texts) < 50:
            continue
        sample_n = min(2000, len(src_texts))
        src_emb = compute_embeddings(src_texts, model_name=model_name, max_samples=sample_n, batch_size=256)
        src_vendi = compute_vendi_score(src_emb)
        src_dist = compute_avg_pairwise_distance_fast(src_emb, sample_size=min(1000, len(src_emb)))
        source_results[src] = {
            "count": len(src_texts),
            "embedded": len(src_emb),
            "vendi_score": src_vendi,
            "avg_pairwise_distance": src_dist,
        }
        logger.info(f"  {src} ({len(src_texts):,}): Vendi={src_vendi:.2f}, AvgDist={src_dist:.4f}")
    results["per_source"] = source_results

    # Save report
    report_path = output_dir / "vendi_diversity_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nReport saved to {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("FANNO-Dev Vendi Score Diversity Report")
    print("=" * 70)
    print(f"Total cleaned samples: {len(data):,}")
    print(f"Embedded samples: {len(embeddings):,}")
    print(f"Embedding model: {model_name}")
    print(f"\nVendi Score: {vendi:.4f}")
    print(f"Avg Pairwise Cosine Distance: {avg_dist:.4f}")
    print(f"Cluster Inertia (k={kmeans_clusters}): {inertia:.4f}")
    print(f"Dominance Scores: {dominance}")
    print(f"\nScale Analysis:")
    for s in scale_results:
        print(f"  {s['fraction']:.0%} ({s['n_samples']:,}): Vendi={s['vendi_score']:.2f}, AvgDist={s['avg_pairwise_distance']:.4f}")
    print(f"\nPer-Source Vendi Scores:")
    for src, sr in sorted(source_results.items(), key=lambda x: -x[1]["vendi_score"]):
        print(f"  {src} ({sr['count']:,}): Vendi={sr['vendi_score']:.2f}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    evaluate_vendi_diversity()
