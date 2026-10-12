"""
Case 04 — Phase 4: Asymmetric Distance Computation (ADC) Primitive

Implements the fundamental search primitive for PQ-assisted search:
    For query q = [q_0, q_1, ..., q_{M-1}], each subvector q_m has dimension dsub = D / M.
    Look-Up Table (LUT):
        LUT[m, k] = ||q_m - centroid[m, k]||^2   for m in 0..M-1, k in 0..Ks-1
    Distance to database vector with PQ code [c_0, c_1, ..., c_{M-1}]:
        ADC_distance(q, code_i) = sum_{m=0}^{M-1} LUT[m, code_i[m]]

Properties:
1. Query vectors remain in full float32 precision (Asymmetric).
2. Database vectors are represented purely by compact uint8 codes (no vector reconstruction needed).
3. Distance calculation reduces to M = 16 table lookups and additions per vector candidate.
4. Mathematically equivalent to ||q - x_hat_i||^2 (distance from query to reconstructed vector).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

try:
    import numpy as np
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import NumPy: {err}", file=sys.stderr)
    sys.exit(1)


def compute_adc_lut_single_query(
    query_vector: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Computes the Look-Up Table (LUT) for a single query vector against all subquantizer centroids.
    Args:
        query_vector: float32 array of shape (dimension,)
        centroids: float32 array of shape (M, Ks, dsub)
    Returns:
        lut: float32 array of shape (M, Ks)
    """
    m, ks, dsub = centroids.shape
    q_sub = query_vector.reshape(m, dsub)  # Shape (M, dsub)
    # Broadcast: (M, 1, dsub) - (M, Ks, dsub) -> (M, Ks, dsub)
    diff = q_sub[:, None, :] - centroids
    # Sum of squared differences over dimension axis (axis=2)
    lut = np.sum(diff ** 2, axis=2).astype(np.float32)  # Shape (M, Ks)
    return lut


def compute_adc_lut_batch(
    query_vectors: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Computes LUTs for a batch of query vectors.
    Args:
        query_vectors: float32 array of shape (Q, dimension)
        centroids: float32 array of shape (M, Ks, dsub)
    Returns:
        luts: float32 array of shape (Q, M, Ks)
    """
    q, d = query_vectors.shape
    m, ks, dsub = centroids.shape
    if d != m * dsub:
        raise ValueError(f"Query dim {d} does not match M*dsub = {m * dsub}")

    q_sub = query_vectors.reshape(q, m, 1, dsub)
    diff = q_sub - centroids[None, :, :, :]  # Shape (Q, M, Ks, dsub)
    luts = np.sum(diff ** 2, axis=3).astype(np.float32)  # Shape (Q, M, Ks)
    return luts


def compute_adc_distances(
    lut: np.ndarray,
    codes: np.ndarray,
) -> np.ndarray:
    """
    Computes ADC distances from a query's LUT to all database PQ codes.
    Fast vectorized accumulation over M subquantizers:
        dist[i] = sum_{m=0}^{M-1} lut[m, codes[i, m]]
    Args:
        lut: float32 array of shape (M, Ks)
        codes: uint8 array of shape (N, M)
    Returns:
        distances: float32 array of shape (N,)
    """
    n, m = codes.shape
    distances = np.zeros(n, dtype=np.float32)
    for sub_idx in range(m):
        distances += lut[sub_idx, codes[:, sub_idx]]
    return distances


class ADCSearcher:
    """
    Asymmetric Distance Computation (ADC) search engine over DRAM-resident PQ codes.
    """

    def __init__(self, centroids: np.ndarray, codes: np.ndarray):
        """
        Args:
            centroids: float32 array of shape (M, Ks, dsub)
            codes: uint8 array of shape (N, M)
        """
        if centroids.ndim != 3:
            raise ValueError(f"Centroids must be 3D array (M, Ks, dsub), got {centroids.shape}")
        if codes.ndim != 2:
            raise ValueError(f"Codes must be 2D array (N, M), got {codes.shape}")
        if centroids.shape[0] != codes.shape[1]:
            raise ValueError(
                f"Subquantizers mismatch: centroids have M={centroids.shape[0]}, codes have M={codes.shape[1]}"
            )

        self.centroids = centroids.astype(np.float32)
        self.codes = codes
        self.num_vectors, self.num_subquantizers = codes.shape
        self.centroids_per_sub = centroids.shape[1]
        self.subvector_dim = centroids.shape[2]
        self.dimension = self.num_subquantizers * self.subvector_dim

    def compute_lut(self, query: np.ndarray) -> np.ndarray:
        """Computes (M, Ks) LUT for a single query."""
        return compute_adc_lut_single_query(query, self.centroids)

    def search_single_query(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs ADC search for a single query vector.
        Returns:
            (top_k_indices, top_k_distances)
        """
        lut = self.compute_lut(query)
        dists = compute_adc_distances(lut, self.codes)

        if k >= self.num_vectors:
            top_k_idx = np.argsort(dists)
        else:
            partitioned = np.argpartition(dists, k)[:k]
            top_k_idx = partitioned[np.argsort(dists[partitioned])]

        top_k_dists = dists[top_k_idx]
        return top_k_idx.astype(np.int32), top_k_dists

    def search_batch(
        self,
        queries: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs ADC search for a batch of query vectors.
        Args:
            queries: float32 array of shape (num_queries, dimension)
            k: number of nearest neighbors
        Returns:
            (indices, distances) of shape (num_queries, k)
        """
        num_queries, d = queries.shape
        if d != self.dimension:
            raise ValueError(f"Query dim {d} does not match index dim {self.dimension}")

        all_indices = np.empty((num_queries, k), dtype=np.int32)
        all_dists = np.empty((num_queries, k), dtype=np.float32)

        for q_idx in range(num_queries):
            lut = compute_adc_lut_single_query(queries[q_idx], self.centroids)
            dists = compute_adc_distances(lut, self.codes)

            if k >= self.num_vectors:
                top_k = np.argsort(dists)[:k]
            else:
                part = np.argpartition(dists, k)[:k]
                top_k = part[np.argsort(dists[part])]

            all_indices[q_idx] = top_k
            all_dists[q_idx] = dists[top_k]

        return all_indices, all_dists

    def verify_mathematical_identity(
        self,
        sample_query: np.ndarray,
        reconstructed_vectors: np.ndarray,
        num_check: int = 100,
        atol: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Verifies that ADC distance D_ADC(q, c_i) is mathematically identical
        to ||q - x_hat_i||^2 for a sample of vectors.
        """
        lut = self.compute_lut(sample_query)
        adc_dists = compute_adc_distances(lut, self.codes[:num_check])

        rec_sample = reconstructed_vectors[:num_check]
        direct_l2_sq = np.sum((sample_query - rec_sample) ** 2, axis=1).astype(np.float32)

        abs_diff = np.abs(adc_dists - direct_l2_sq)
        max_diff = float(np.max(abs_diff))
        mean_diff = float(np.mean(abs_diff))

        matches = max_diff < atol
        return {
            "identity_verified": matches,
            "num_vectors_checked": num_check,
            "max_absolute_difference": max_diff,
            "mean_absolute_difference": mean_diff,
            "tolerance": atol,
            "sample_adc_distances": adc_dists[:5].tolist(),
            "sample_direct_l2_sq": direct_l2_sq[:5].tolist(),
        }
