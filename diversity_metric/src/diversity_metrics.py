"""
Diversity metrics for evaluating selected instruction subsets.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from time import perf_counter
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from .selection_strategies import compute_density

TensorLike = np.ndarray | torch.Tensor

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional accel dependency
    faiss = None


def _to_numpy(arr: TensorLike) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(arr.detach().cpu().numpy())


def _l2_normalize_inplace(arr: np.ndarray) -> None:
    """In-place L2 normalization; prefers FAISS accel when available."""
    if faiss is not None:
        faiss.normalize_L2(arr)
        return
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    np.divide(arr, np.clip(norms, 1e-12, None), out=arr)


def _maybe_sample(embs: np.ndarray, sample_size: Optional[int], seed: int = 42) -> np.ndarray:
    """Subsample embeddings if sample_size is provided and smaller than n."""
    if sample_size is None:
        return embs
    n = embs.shape[0]
    if n <= sample_size:
        return embs
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_size, replace=False)
    return embs[idx]


def compute_vendi_score(embeddings: TensorLike, eps: float = 1e-8) -> float:
    """
    Vendi score: exp(-sum_i p_i log p_i) where p_i are normalized eigenvalues.
    Higher is more diverse.
    """
    embs = _to_numpy(embeddings)
    embs = embs - embs.mean(axis=0, keepdims=True)
    cov = np.cov(embs, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, a_min=eps, a_max=None)
    p = eigvals / eigvals.sum()
    entropy = -np.sum(p * np.log(p + eps))
    return float(np.exp(entropy))


def compute_g_vendi(gradient_embeddings: TensorLike) -> float:
    """Alias for gradient-space Vendi."""
    return compute_vendi_score(gradient_embeddings)


def compute_cluster_inertia(
    embeddings: TensorLike, n_clusters: int = 100, seed: int = 42
) -> float:
    """K-means inertia (within-cluster variance). Lower is more compact."""
    embs = _to_numpy(embeddings)
    km = KMeans(
        n_clusters=n_clusters, random_state=seed, n_init="auto", verbose=0
    ).fit(embs)
    return float(km.inertia_)


def compute_avg_pairwise_distance(embeddings: TensorLike, sample_size: int = 2000) -> float:
    """
    Average cosine distance over a subsample (to keep computation tractable).
    """
    embs = _to_numpy(embeddings)
    n = embs.shape[0]
    if n > sample_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=sample_size, replace=False)
        embs = embs[indices]
    dists = pairwise_distances(embs, metric="cosine")
    return float(dists.mean())


def compute_avg_pairwise_distance_fast(
    embeddings: TensorLike, sample_size: int = 2000
) -> float:
    """
    FAISS-accelerated average pairwise cosine distance with optional sampling.
    Falls back to the sklearn implementation if FAISS is unavailable.
    """
    embs = _to_numpy(embeddings).astype("float32")
    embs = _maybe_sample(embs, sample_size)
    n = embs.shape[0]
    if n < 2:
        return 0.0

    if faiss is None:
        dists = pairwise_distances(embs, metric="cosine")
        mask = np.triu(np.ones_like(dists, dtype=bool), k=1)
        return float(dists[mask].mean())

    _l2_normalize_inplace(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    D, _ = index.search(embs, n)
    distances = 1.0 - D
    mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
    return float(distances[mask].mean())


def compute_novelsum_score(
    embeddings: TensorLike,
    reference_embeddings: Optional[TensorLike] = None,
    k_density: int = 10,
) -> float:
    """
    Aggregate NovelSum-style novelty across a set using density-aware distances.
    """
    embs = _to_numpy(embeddings)
    if reference_embeddings is None:
        reference_embeddings = embs
    densities = compute_density(embs, reference_embeddings, k=k_density)
    n = embs.shape[0]
    total = 0.0
    for i in range(n):
        # novelty of sample i relative to others
        dists = pairwise_distances(embs[[i]], embs, metric="cosine")[0]
        total += np.mean((densities ** 0.5) * dists)
    return float(total)


def compute_novelsum_score_fast(
    embeddings: TensorLike,
    reference_embeddings: Optional[TensorLike] = None,
    k_density: int = 10,
    sample_size: Optional[int] = 5000,
    batch_size: int = 512,
) -> float:
    """
    NovelSum with sampling + FAISS acceleration (dot-product cosine). Falls back to
    slower sklearn distances if FAISS is unavailable.
    """
    embs = _to_numpy(embeddings).astype("float32")
    ref = _to_numpy(reference_embeddings) if reference_embeddings is not None else embs
    n = embs.shape[0]

    # Sample rows whose novelty we evaluate
    sampled_embs = _maybe_sample(embs, sample_size)

    if faiss is not None:
        ref = ref.astype("float32")
        _l2_normalize_inplace(ref)
        _l2_normalize_inplace(embs)
        _l2_normalize_inplace(sampled_embs)

        index = faiss.IndexFlatIP(ref.shape[1])
        index.add(ref)
        k = min(k_density + 1, len(ref))
        D, _ = index.search(embs, k)
        if k > 1:
            densities = 1 - D[:, 1:].mean(axis=1)
        else:
            densities = np.zeros(n, dtype=np.float32)

        root_dens = densities ** 0.5
        scores: List[np.ndarray] = []
        for start in range(0, sampled_embs.shape[0], batch_size):
            end = min(sampled_embs.shape[0], start + batch_size)
            sims = sampled_embs[start:end] @ embs.T
            dists = 1.0 - sims
            scores.append((dists * root_dens).mean(axis=1))
        return float(np.concatenate(scores).mean())

    # Fallback: use sklearn to avoid dependency on FAISS; also sample reference to cap cost.
    ref_sampled = _maybe_sample(ref, sample_size)
    densities = compute_density(embs, ref_sampled, k=min(k_density, max(1, len(ref_sampled) - 1)))
    root_dens = densities ** 0.5
    scores: List[np.ndarray] = []
    for start in range(0, sampled_embs.shape[0], batch_size):
        end = min(sampled_embs.shape[0], start + batch_size)
        dists = pairwise_distances(sampled_embs[start:end], embs, metric="cosine")
        scores.append((dists * root_dens).mean(axis=1))
    return float(np.concatenate(scores).mean())


def compute_novelsum_score_approx(
    embeddings: TensorLike,
    reference_embeddings: Optional[TensorLike] = None,
    k_density: int = 10,
    k_neighbors: int = 100,
) -> float:
    """
    Approximate NovelSum using only k-nearest neighbors for each point.
    Requires FAISS; falls back to compute_novelsum_score_fast otherwise.
    """
    if faiss is None:
        return compute_novelsum_score_fast(
            embeddings, reference_embeddings, k_density=k_density, sample_size=k_neighbors
        )

    embs = _to_numpy(embeddings).astype("float32")
    ref = _to_numpy(reference_embeddings) if reference_embeddings is not None else embs
    _l2_normalize_inplace(embs)
    ref = ref.astype("float32")
    _l2_normalize_inplace(ref)

    # Density via top-k in reference
    index_ref = faiss.IndexFlatIP(ref.shape[1])
    index_ref.add(ref)
    k_den = min(k_density + 1, len(ref))
    D_den, _ = index_ref.search(embs, k_den)
    if k_den > 1:
        densities = 1 - D_den[:, 1:].mean(axis=1)
    else:
        densities = np.zeros(len(embs), dtype=np.float32)
    root_dens = densities ** 0.5

    # Neighbor graph on embs
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    k_n = min(k_neighbors + 1, len(embs))
    D_nei, I_nei = index.search(embs, k_n)

    scores: List[float] = []
    for i in range(len(embs)):
        # skip self at position 0
        neighbor_dists = 1.0 - D_nei[i, 1:]
        neighbor_weights = root_dens[I_nei[i, 1:]]
        scores.append(float(np.mean(neighbor_weights * neighbor_dists)))
    return float(np.mean(scores))


def compute_dominance_score(
    embeddings: TensorLike, topk_ratios: Sequence[float] = (0.1, 0.2, 0.3, 0.5)
) -> Dict[str, float]:
    """
    Feature dominance based on eigenvalue mass captured by the top-k principal components.
    Lower dominance => more evenly spread variance.
    """
    if isinstance(embeddings, torch.Tensor):
        feat_mat = embeddings.clone()
    else:
        feat_mat = torch.from_numpy(_to_numpy(embeddings))
    feat_mat = feat_mat - feat_mat.mean(dim=0, keepdim=True)
    feat_mat = feat_mat / (feat_mat.std(dim=0, keepdim=True) + 1e-8)

    n = feat_mat.shape[0]
    corr = (feat_mat.T @ feat_mat) / max(1, n - 1)
    eigenvalues, _ = torch.linalg.eigh(corr)
    eigenvalues = torch.sort(eigenvalues, descending=True)[0]
    total_var = eigenvalues.sum() + 1e-8

    scores: Dict[str, float] = {}
    for ratio in topk_ratios:
        k = max(1, int(len(eigenvalues) * ratio))
        topk_var = eigenvalues[:k].sum()
        scores[f"dom_{int(ratio * 100)}"] = float((topk_var / total_var).item())
    return scores


def compute_all_diversity_metrics(
    embeddings: Dict[str, TensorLike],
    reference_key: str = "llama",
    kmeans_clusters: int = 100,
    novelsum_sample_size: Optional[int] = 5000,
    pairwise_sample_size: int = 2000,
    novelsum_mode: str = "fast",
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Convenience wrapper to compute the full suite of diversity metrics for a subset.

    Args:
        embeddings: mapping from embedding name -> array/tensor for the same subset.
        reference_key: which embedding to use for distance-based metrics.
        novelsum_sample_size: subsample size for NovelSum (None to disable sampling).
        pairwise_sample_size: subsample size for pairwise distance.
        novelsum_mode: {"fast", "approx", "exact"} to control NovelSum implementation.
    """
    if reference_key not in embeddings:
        raise ValueError(f"reference_key '{reference_key}' not found in embeddings.")
    ref = embeddings[reference_key]

    scores: Dict[str, float] = {}
    if timings is not None:
        timings.clear()

    start = perf_counter()
    if novelsum_mode == "approx":
        scores["novelsum"] = compute_novelsum_score_approx(ref, ref)
    elif novelsum_mode == "exact":
        scores["novelsum"] = compute_novelsum_score(ref, ref)
    else:
        scores["novelsum"] = compute_novelsum_score_fast(
            ref, ref, sample_size=novelsum_sample_size
        )
    if timings is not None:
        timings["novelsum"] = perf_counter() - start

    start = perf_counter()
    scores["g_vendi"] = compute_g_vendi(embeddings.get("gradient", ref))
    if timings is not None:
        timings["g_vendi"] = perf_counter() - start

    start = perf_counter()
    scores["emb_vendi"] = compute_vendi_score(ref)
    if timings is not None:
        timings["emb_vendi"] = perf_counter() - start

    start = perf_counter()
    scores["inertia"] = compute_cluster_inertia(ref, n_clusters=kmeans_clusters)
    if timings is not None:
        timings["inertia"] = perf_counter() - start

    start = perf_counter()
    scores["avg_pairwise_dist"] = compute_avg_pairwise_distance_fast(
        ref, sample_size=pairwise_sample_size
    )
    if timings is not None:
        timings["avg_pairwise_dist"] = perf_counter() - start

    start = perf_counter()
    dom = compute_dominance_score(ref)
    if timings is not None:
        timings["dominance"] = perf_counter() - start
    scores.update(dom)
    return scores
