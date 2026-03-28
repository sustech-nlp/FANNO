"""Streamlined diversity metrics: Vendi Score, pairwise distance, k-center greedy."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from loguru import logger


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix from row-normalized embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms
    return normed @ normed.T


def vendi_score(embeddings: np.ndarray) -> float:
    """Vendi Score: exp(Shannon entropy of eigenvalues of the normalized kernel matrix).

    Reference: Friedman & Dieng (2022) "The Vendi Score".
    Higher values indicate more diversity.

    Args:
        embeddings: (N, D) array of embedding vectors.

    Returns:
        Vendi Score (float). For N identical items → 1.0, for N orthogonal items → N.
    """
    if len(embeddings) <= 1:
        return 1.0

    K = _cosine_similarity_matrix(embeddings)
    # Normalize: K_norm = K / trace(K) so eigenvalues sum to 1
    trace_K = np.trace(K)
    if trace_K < 1e-10:
        return 1.0
    K_norm = K / trace_K

    eigvals = np.linalg.eigvalsh(K_norm)
    # Filter to positive eigenvalues only
    eigvals = eigvals[eigvals > 1e-10]

    if len(eigvals) == 0:
        return 1.0

    # Shannon entropy
    entropy = -np.sum(eigvals * np.log(eigvals))
    return float(np.exp(entropy))


def avg_pairwise_distance(
    embeddings: np.ndarray,
    metric: str = "cosine",
) -> float:
    """Average pairwise distance between all embedding pairs.

    Higher values indicate more diversity.

    Args:
        embeddings: (N, D) array of embedding vectors.
        metric: "cosine" or "euclidean".

    Returns:
        Average distance (float). Range [0, 1] for cosine, [0, inf) for euclidean.
    """
    n = len(embeddings)
    if n <= 1:
        return 0.0

    if metric == "cosine":
        sim_matrix = _cosine_similarity_matrix(embeddings)
        # Distance = 1 - similarity
        dist_matrix = 1.0 - sim_matrix
    elif metric == "euclidean":
        # Compute pairwise euclidean distances
        diff = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Average over upper triangle (exclude diagonal)
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    num_pairs = n * (n - 1) / 2
    return float(np.sum(dist_matrix[upper_mask]) / num_pairs)


def k_center_greedy(
    embeddings: np.ndarray,
    k: int,
    seed_index: Optional[int] = None,
) -> List[int]:
    """K-center greedy selection for maximum coverage.

    Greedily selects k points that maximize the minimum distance from any
    point to its nearest selected center.

    Args:
        embeddings: (N, D) array of embedding vectors.
        k: Number of points to select.
        seed_index: Starting index. If None, uses the centroid-farthest point.

    Returns:
        List of k selected indices.
    """
    n = len(embeddings)
    if k >= n:
        return list(range(n))
    if k <= 0:
        return []

    # Precompute cosine distance matrix
    sim_matrix = _cosine_similarity_matrix(embeddings)
    dist_matrix = 1.0 - sim_matrix

    # Initialize: pick the point farthest from the centroid (or use seed)
    if seed_index is not None:
        selected = [seed_index]
    else:
        centroid = embeddings.mean(axis=0, keepdims=True)
        norms_c = np.linalg.norm(centroid, keepdims=True)
        norms_e = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if norms_c.item() < 1e-10:
            selected = [0]
        else:
            cos_to_centroid = (embeddings @ centroid.T).flatten() / (norms_e.flatten() * norms_c.item())
            selected = [int(np.argmin(cos_to_centroid))]

    # Greedy loop
    min_dists = dist_matrix[selected[0]].copy()  # distance to nearest selected

    for _ in range(1, k):
        # Pick the point with the largest min-distance to any selected point
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        # Update min distances
        min_dists = np.minimum(min_dists, dist_matrix[next_idx])
        # Set selected points' distance to -inf so they're not re-selected
        min_dists[next_idx] = -np.inf

    return selected


def embed_texts(
    texts: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Embed texts using sentence-transformers. Returns (N, D) numpy array.

    Lazy import to avoid requiring sentence-transformers at module load time.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, trust_remote_code=True)
    model = model.to(device)
    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 100,
        device=device,
        batch_size=batch_size,
    )
    return embeddings


__all__ = [
    "vendi_score",
    "avg_pairwise_distance",
    "k_center_greedy",
    "embed_texts",
]
