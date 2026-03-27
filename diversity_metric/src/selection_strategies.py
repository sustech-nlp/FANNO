"""
Selection strategies for building diverse instruction datasets.

Implements the methods described in the experiment design:
- Random sampling
- K-means centroids sampling
- Community detection with cosine similarity
- K-Center-Greedy
- NovelSum-style density-aware greedy selection
- Prismatic Synthesis inspired gradient-space clustering

All functions accept embeddings as a NumPy array or a torch tensor of shape
[num_samples, dim] and return a list of selected indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional accel dependency
    faiss = None

TensorLike = np.ndarray | torch.Tensor


def _to_numpy(embeddings: TensorLike) -> np.ndarray:
    """Convert embeddings to a contiguous NumPy array."""
    if isinstance(embeddings, np.ndarray):
        return np.ascontiguousarray(embeddings)
    return np.ascontiguousarray(embeddings.detach().cpu().numpy())


def _l2_normalize_inplace(arr: np.ndarray) -> None:
    """In-place L2 normalization with optional FAISS acceleration."""
    if faiss is not None:
        faiss.normalize_L2(arr)
        return
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    np.divide(arr, np.clip(norms, 1e-12, None), out=arr)


def random_select(embeddings: TensorLike, n_select: int, seed: int = 42) -> List[int]:
    """Uniformly sample indices without replacement."""
    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]
    if n_select > n:
        raise ValueError(f"Cannot select {n_select} from only {n} samples.")
    return rng.choice(n, size=n_select, replace=False).tolist()


def kmeans_select(
    embeddings: TensorLike,
    n_select: int,
    n_clusters: int = 100,
    seed: int = 42,
) -> List[int]:
    """
    Cluster embeddings with K-Means and sample uniformly from each cluster.
    """
    embs = _to_numpy(embeddings)
    kmeans = KMeans(
        n_clusters=n_clusters, random_state=seed, n_init="auto", verbose=0
    )
    labels = kmeans.fit_predict(embs)

    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    rng = np.random.default_rng(seed)
    selected: List[int] = []
    per_cluster = max(1, n_select // n_clusters)
    for cluster_indices in clusters.values():
        rng.shuffle(cluster_indices)
        selected.extend(cluster_indices[:per_cluster])
        if len(selected) >= n_select:
            break

    # If we are short due to small clusters, top up randomly.
    if len(selected) < n_select:
        remaining = [i for i in range(len(embs)) if i not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: n_select - len(selected)])

    return selected[:n_select]


def _fallback_community_detection(embs: np.ndarray, threshold: float) -> List[List[int]]:
    """
    Lightweight community detection based on cosine similarity.
    This is O(n^2); intended for moderate n (e.g., 10k).
    """
    sim = 1 - pairwise_distances(embs, metric="cosine")
    n = sim.shape[0]
    visited = np.zeros(n, dtype=bool)
    clusters: List[List[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        queue = [i]
        visited[i] = True
        cluster = [i]
        while queue:
            cur = queue.pop()
            neighbors = np.where(sim[cur] >= threshold)[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
                    cluster.append(nb)
        clusters.append(cluster)
    return clusters


def _community_from_edges(edges: List[Tuple[int, int]], n: int) -> Optional[List[List[int]]]:
    """Try igraph community detection; return None if unavailable."""
    try:
        import igraph as ig  # type: ignore
    except Exception:
        return None

    if not edges:
        return [[i] for i in range(n)]
    graph = ig.Graph(n=n, edges=edges)
    comms = graph.community_leiden()
    return [list(comm) for comm in comms]


def community_detect_fast(
    embeddings: TensorLike,
    threshold: float = 0.8,
    n_select: int = 10_000,
    k_neighbors: int = 50,
) -> List[int]:
    """
    FAISS + igraph accelerated community detection.

    - Build a sparse similarity graph via top-k FAISS search.
    - Run Leiden community detection (igraph).
    - Pick one representative per community; pad randomly if needed.
    """
    if faiss is None:
        raise ImportError("FAISS not available for community_detect_fast.")

    embs = _to_numpy(embeddings).astype("float32")
    n, dim = embs.shape
    k_neighbors = min(k_neighbors, n)

    _l2_normalize_inplace(embs)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    D, I = index.search(embs, k_neighbors)
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j, sim in zip(I[i], D[i]):
            if j == i or j < 0:
                continue
            if i < j and sim >= threshold:
                edges.append((i, j))

    clusters = _community_from_edges(edges, n)
    if clusters is None:
        # fallback to original path if igraph missing
        raise ImportError("igraph not available for community_detect_fast.")

    selected: List[int] = []
    for cluster in clusters:
        if len(selected) >= n_select:
            break
        selected.append(cluster[0])

    if len(selected) < n_select:
        remaining = [i for i in range(n) if i not in selected]
        rng = np.random.default_rng(42)
        rng.shuffle(remaining)
        selected.extend(remaining[: n_select - len(selected)])

    return selected[:n_select]


def community_detect_select(
    embeddings: TensorLike, threshold: float = 0.8, n_select: int = 10_000
) -> List[int]:
    """
    Community detection over cosine similarity graph, then pick one per community.
    Preferred order:
    1) sentence-transformers community_detection,
    2) BFS cosine clustering fallback.
    """
    embs = _to_numpy(embeddings)
    try:
        from sentence_transformers import util

        # sentence-transformers expects torch tensors
        torch_embs = torch.from_numpy(embs)
        clusters = util.community_detection(
            torch_embs,
            min_community_size=1,
            threshold=threshold,
        )
        clusters = [list(c) for c in clusters]
    except Exception:
        clusters = _fallback_community_detection(embs, threshold)

    selected: List[int] = []
    for cluster in clusters:
        if len(selected) >= n_select:
            break
        # pick the first element as representative; could be replaced by quality scoring
        selected.append(cluster[0])

    # If not enough communities, pad with random indices
    if len(selected) < n_select:
        remaining = [i for i in range(len(embs)) if i not in selected]
        rng = np.random.default_rng(42)
        rng.shuffle(remaining)
        selected.extend(remaining[: n_select - len(selected)])

    return selected[:n_select]


def k_center_greedy_fast(
    embeddings: TensorLike, n_select: int, seed: int = 42
) -> List[int]:
    """
    FAISS-accelerated K-Center-Greedy using inner-product search on L2-normalized embeddings.
    Falls back to the standard implementation if FAISS is unavailable.
    """
    if faiss is None:
        return k_center_greedy(embeddings, n_select=n_select, seed=seed)

    embs = _to_numpy(embeddings).astype("float32")
    n, dim = embs.shape
    if n_select > n:
        raise ValueError(f"Cannot select {n_select} from only {n} samples.")

    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(0, n))]
    # Track the maximum distance (1 - similarity) to the selected set
    min_distances = np.full(n, -np.inf, dtype=np.float32)

    for _ in range(n_select - 1):
        # Distance to the latest center; FAISS returns similarity because vectors are normalized
        D, _ = index.search(embs[selected[-1 :]], n)
        min_distances = np.maximum(min_distances, -D[0])

        farthest = int(np.argmax(min_distances))
        selected.append(farthest)
        min_distances[farthest] = -np.inf

    return selected


def k_center_greedy(
    embeddings: TensorLike, n_select: int, seed: int = 42
) -> List[int]:
    """
    Iteratively pick the point farthest from the current selected set
    under cosine distance (K-Center-Greedy).
    """
    embs = _to_numpy(embeddings)
    n = embs.shape[0]
    if n_select > n:
        raise ValueError(f"Cannot select {n_select} from only {n} samples.")

    rng = np.random.default_rng(seed)
    first = int(rng.integers(low=0, high=n))
    selected = [first]

    # Precompute pairwise cosine distances incrementally
    dists = pairwise_distances(embs, embs[selected], metric="cosine").reshape(-1)
    while len(selected) < n_select:
        farthest = int(np.argmax(dists))
        selected.append(farthest)
        # Update min distances to the selected set
        new_dists = pairwise_distances(
            embs, embs[[farthest]], metric="cosine"
        ).reshape(-1)
        dists = np.minimum(dists, new_dists)

    return selected


def compute_density(
    embeddings: TensorLike, reference_embeddings: TensorLike, k: int = 10
) -> np.ndarray:
    """Average cosine distance to k-nearest neighbors in the reference set."""
    embs = _to_numpy(embeddings)
    ref_embs = _to_numpy(reference_embeddings)
    k = min(k, len(ref_embs) - 1)
    if k < 1:
        return np.zeros(len(embs))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(ref_embs)
    distances, _ = nbrs.kneighbors(embs, n_neighbors=k + 1, return_distance=True)
    # drop self-distance at index 0
    return distances[:, 1:].mean(axis=1)


def novelsum_select(
    embeddings: TensorLike,
    reference_embeddings: Optional[TensorLike],
    n_select: int,
    alpha: float = 1.0,
    beta: float = 0.5,
    k_density: int = 10,
) -> List[int]:
    """
    NovelSum-inspired greedy selection with proximity weighting and
    density-aware distance.
    """
    embs = _to_numpy(embeddings)
    n = embs.shape[0]
    if reference_embeddings is None:
        reference_embeddings = embs
    densities = compute_density(embs, reference_embeddings, k=k_density)

    candidates = set(range(n))
    selected: List[int] = []

    def _rank_weight(index: int) -> float:
        # Earlier selections contribute more; avoid rank=0
        rank = selected.index(index) + 1 if index in selected else len(selected) + 1
        return 1.0 / rank

    while len(selected) < n_select and candidates:
        best_idx = None
        best_score = -np.inf
        for idx in candidates:
            if not selected:
                score = densities[idx]
            else:
                score = 0.0
                for j in selected:
                    w = _rank_weight(j)
                    d_sem = pairwise_distances(
                        embs[[idx]], embs[[j]], metric="cosine"
                    )[0, 0]
                    score += (w ** alpha) * (densities[j] ** beta) * d_sem
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(int(best_idx))
        candidates.remove(int(best_idx))

    return selected[:n_select]


def novelsum_select_fast(
    embeddings: TensorLike,
    reference_embeddings: Optional[TensorLike],
    n_select: int,
    alpha: float = 1.0,
    beta: float = 0.5,
    k_density: int = 10,
) -> List[int]:
    """
    FAISS-accelerated NovelSum variant:
    - density via top-k cosine neighbors
    - vectorized scoring against selected set
    """
    if faiss is None:
        return novelsum_select(
            embeddings, reference_embeddings, n_select, alpha, beta, k_density
        )

    embs = _to_numpy(embeddings).astype("float32")
    n, dim = embs.shape
    if reference_embeddings is None:
        ref_embs = embs.copy()
    else:
        ref_embs = _to_numpy(reference_embeddings).astype("float32")

    _l2_normalize_inplace(ref_embs)
    _l2_normalize_inplace(embs)

    index = faiss.IndexFlatIP(dim)
    index.add(ref_embs)
    k = min(k_density + 1, len(ref_embs))
    D, _ = index.search(embs, k)
    densities = 1 - D[:, 1:].mean(axis=1) if k > 1 else np.zeros(n, dtype=np.float32)

    candidates = list(range(n))
    selected: List[int] = []

    while len(selected) < n_select and candidates:
        if not selected:
            best_idx = candidates[int(np.argmax(densities[candidates]))]
        else:
            sel_embs = embs[selected]
            cand_embs = embs[candidates]
            sims = cand_embs @ sel_embs.T  # cosine similarities
            dists = 1.0 - sims

            rank_weights = 1.0 / (np.arange(1, len(selected) + 1, dtype=np.float32) ** alpha)
            density_weights = (densities[selected] ** beta)
            weights = rank_weights * density_weights

            scores = (dists * weights).sum(axis=1)
            best_idx = candidates[int(np.argmax(scores))]

        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected[:n_select]


def coreset_select(embeddings: TensorLike, n_select: int, seed: int = 42) -> List[int]:
    """
    Frank-Wolfe 风格的核心集选择，优先覆盖几何中心。
    """
    embs = _to_numpy(embeddings).astype("float32")
    n, dim = embs.shape
    _l2_normalize_inplace(embs)

    weights = np.ones(n, dtype=np.float32) / n
    selected: List[int] = []
    rng = np.random.default_rng(seed)

    for _ in range(min(n_select, n)):
        center = (weights[:, None] * embs).sum(axis=0, keepdims=True)
        _l2_normalize_inplace(center)

        sims = embs @ center.T
        farthest = int(np.argmin(sims))
        selected.append(farthest)

        sim_selected = embs @ embs[farthest]
        weights *= (1 - sim_selected * 0.1)
        weights = np.clip(weights, 0, None)
        weights_sum = weights.sum()
        if weights_sum > 0:
            weights /= weights_sum
        else:  # degenerate
            weights = np.ones(n, dtype=np.float32) / n

    return selected[:n_select]


def herding_select(embeddings: TensorLike, n_select: int) -> List[int]:
    """
    Herding: 逐步逼近全局均值的确定性采样。
    """
    embs = _to_numpy(embeddings).astype("float32")
    n, dim = embs.shape
    mu = embs.mean(axis=0, keepdims=True)

    selected: List[int] = []
    running_sum = np.zeros_like(mu)
    embs_norm = embs.copy()
    _l2_normalize_inplace(embs_norm)
    available = np.ones(n, dtype=bool)

    for t in range(min(n_select, n)):
        target = (t + 1) * mu - running_sum
        _l2_normalize_inplace(target)
        sims = (embs_norm @ target.T).flatten()
        sims[~available] = -np.inf
        best_idx = int(np.argmax(sims))
        selected.append(best_idx)
        running_sum += embs[best_idx]
        available[best_idx] = False

    return selected[:n_select]


def semantic_dedup_select(
    embeddings: TensorLike, n_select: int, threshold: float = 0.95
) -> List[int]:
    """
    基于语义相似度去重后再抽样。
    """
    if faiss is None:
        raise ImportError("FAISS required for semantic_dedup_select.")

    embs = _to_numpy(embeddings).astype("float32")
    n, dim = embs.shape
    _l2_normalize_inplace(embs)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    k = min(100, n)
    D, I = index.search(embs, k)

    to_remove = set()
    for i in range(n):
        if i in to_remove:
            continue
        for j, sim in zip(I[i, 1:], D[i, 1:]):  # skip self
            if j < 0:
                continue
            if sim > threshold and j not in to_remove:
                to_remove.add(int(j))

    kept = [i for i in range(n) if i not in to_remove]
    rng = np.random.default_rng(42)
    if len(kept) > n_select:
        return rng.choice(kept, size=n_select, replace=False).tolist()
    return kept


def stratified_select(
    embeddings: TensorLike, n_select: int, n_strata: int = 100, seed: int = 42
) -> List[int]:
    """
    分层采样：先聚类，再按簇大小比例采样。
    """
    embs = _to_numpy(embeddings).astype("float32")
    n, dim = embs.shape

    if faiss is not None:
        kmeans = faiss.Kmeans(dim, n_strata, niter=20, verbose=False, seed=seed)
        kmeans.train(embs)
        _, labels = kmeans.index.search(embs, 1)
        labels = labels.flatten()
    else:
        kmeans = KMeans(n_clusters=n_strata, random_state=seed, n_init="auto", verbose=0)
        labels = kmeans.fit_predict(embs)

    unique, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()

    rng = np.random.default_rng(seed)
    selected: List[int] = []

    for cluster_id, prop in zip(unique, proportions):
        cluster_indices = np.where(labels == cluster_id)[0]
        n_from_cluster = max(1, int(n_select * prop))

        if len(cluster_indices) <= n_from_cluster:
            selected.extend(cluster_indices.tolist())
        else:
            sampled = rng.choice(cluster_indices, size=n_from_cluster, replace=False)
            selected.extend(sampled.tolist())

    if len(selected) > n_select:
        rng.shuffle(selected)
        selected = selected[:n_select]

    return selected


@dataclass
class PrismaticConfig:
    k_clusters_ratio: float = 0.01
    sparse_ratio: float = 0.2
    samples_per_cluster: int = 5
    seed: int = 42


def prismatic_select(
    embeddings: TensorLike,
    n_select: int,
    config: Optional[PrismaticConfig] = None,
) -> List[int]:
    """
    Prismatic Synthesis-inspired iterative selection:
    1) cluster current pool in gradient space,
    2) identify sparse clusters,
    3) sample from sparse clusters to promote rare reasoning modes.
    """
    if config is None:
        config = PrismaticConfig()

    embs = _to_numpy(embeddings)
    pool_indices = list(range(len(embs)))
    rng = np.random.default_rng(config.seed)
    selected: List[int] = []

    while len(selected) < n_select and pool_indices:
        k = max(10, int(len(pool_indices) * config.k_clusters_ratio))
        subset = embs[pool_indices]
        km = KMeans(
            n_clusters=k, random_state=config.seed, n_init="auto", verbose=0
        )
        labels = km.fit_predict(subset)

        # cluster -> indices in the original space
        clusters: Dict[int, List[int]] = {}
        for local_idx, label in enumerate(labels):
            global_idx = pool_indices[local_idx]
            clusters.setdefault(label, []).append(global_idx)

        cluster_sizes = np.array([len(c) for c in clusters.values()])
        n_sparse = max(1, int(k * config.sparse_ratio))
        sparse_cluster_ids = cluster_sizes.argsort()[:n_sparse]
        sparse_clusters = [
            list(clusters[list(clusters.keys())[i]]) for i in sparse_cluster_ids
        ]

        for cluster_indices in sparse_clusters:
            rng.shuffle(cluster_indices)
            take = cluster_indices[: config.samples_per_cluster]
            selected.extend(take)
            if len(selected) >= n_select:
                break
        # remove selected from pool
        selected_set = set(selected)
        pool_indices = [i for i in pool_indices if i not in selected_set]

    return selected[:n_select]


def select_diverse_samples(
    embeddings: TensorLike,
    strategy: str,
    n_select: int,
    reference_embeddings: Optional[TensorLike] = None,
    **kwargs,
) -> List[int]:
    """
    Dispatch function that routes to the desired diversification strategy.
    """
    name = strategy.lower()
    if name == "random":
        return random_select(embeddings, n_select, seed=kwargs.get("seed", 42))
    if name == "kmeans":
        return kmeans_select(
            embeddings,
            n_select=n_select,
            n_clusters=kwargs.get("n_clusters", 100),
            seed=kwargs.get("seed", 42),
        )
    if name in {"community_fast", "community_detection_fast"}:
        return community_detect_fast(
            embeddings,
            threshold=kwargs.get("threshold", 0.8),
            n_select=n_select,
            k_neighbors=kwargs.get("k_neighbors", 50),
        )
    if name in {"community", "community_detection"}:
        return community_detect_select(
            embeddings,
            threshold=kwargs.get("threshold", 0.8),
            n_select=n_select,
        )
    if name in {"k_center_greedy_fast", "k-center-fast", "kcenter-fast"}:
        return k_center_greedy_fast(
            embeddings,
            n_select=n_select,
            seed=kwargs.get("seed", 42),
        )
    if name in {"k_center_greedy", "k-center", "kcenter"}:
        if kwargs.get("fast", True):
            try:
                return k_center_greedy_fast(
                    embeddings,
                    n_select=n_select,
                    seed=kwargs.get("seed", 42),
                )
            except Exception:
                pass
        return k_center_greedy(
            embeddings,
            n_select=n_select,
            seed=kwargs.get("seed", 42),
        )
    if name in {"novelsum_fast", "novel_sum_fast"}:
        return novelsum_select_fast(
            embeddings,
            reference_embeddings=reference_embeddings,
            n_select=n_select,
            alpha=kwargs.get("alpha", 1.0),
            beta=kwargs.get("beta", 0.5),
            k_density=kwargs.get("K", kwargs.get("k_density", 10)),
        )
    if name in {"novelsum", "novel_sum"}:
        return novelsum_select(
            embeddings,
            reference_embeddings=reference_embeddings,
            n_select=n_select,
            alpha=kwargs.get("alpha", 1.0),
            beta=kwargs.get("beta", 0.5),
            k_density=kwargs.get("K", kwargs.get("k_density", 10)),
        )
    if name in {"coreset"}:
        return coreset_select(
            embeddings,
            n_select=n_select,
            seed=kwargs.get("seed", 42),
        )
    if name in {"herding"}:
        return herding_select(
            embeddings,
            n_select=n_select,
        )
    if name in {"semantic_dedup", "dedup"}:
        return semantic_dedup_select(
            embeddings,
            n_select=n_select,
            threshold=kwargs.get("threshold", 0.95),
        )
    if name in {"stratified"}:
        return stratified_select(
            embeddings,
            n_select=n_select,
            n_strata=kwargs.get("n_strata", 100),
            seed=kwargs.get("seed", 42),
        )
    if name in {"prismatic", "prismatic_synthesis"}:
        config = kwargs.get("config")
        if config is None:
            config = PrismaticConfig(
                k_clusters_ratio=kwargs.get("k_ratio", 0.01),
                sparse_ratio=kwargs.get("sparse_ratio", 0.2),
                samples_per_cluster=kwargs.get("samples_per_cluster", 5),
                seed=kwargs.get("seed", 42),
            )
        return prismatic_select(embeddings, n_select=n_select, config=config)

    raise ValueError(f"Unknown strategy: {strategy}")
