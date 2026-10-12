"""
Case 04 — Phase 4: Product Quantization (PQ) Vector Reconstruction & Error Evaluation

Reconstructs full-precision vectors from PQ codes and codebooks:
    code[i] (16 uint8 bytes) + codebooks (16 x 256 x 8 float32) -> reconstructed_x[i] (128 float32)

Evaluates reconstruction fidelity on the full 100,000 vector dataset:
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - Relative L2 reconstruction error: ||X - X_hat||_F / ||X||_F
    - Per-vector error distribution (min, p50, p95, p99, max)
"""

from __future__ import annotations

import json
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


def reconstruct_vectors_numpy(
    codes: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Vectorized reconstruction of full vectors from PQ codes and centroids.
    Args:
        codes: uint8 array of shape (N, M)
        centroids: float32 array of shape (M, Ks, dsub)
    Returns:
        reconstructed: float32 array of shape (N, M * dsub)
    """
    if codes.ndim != 2:
        raise ValueError(f"Codes must be 2D array (N, M), got shape {codes.shape}")
    if centroids.ndim != 3:
        raise ValueError(f"Centroids must be 3D array (M, Ks, dsub), got shape {centroids.shape}")

    n, m = codes.shape
    cm, ks, dsub = centroids.shape
    if m != cm:
        raise ValueError(f"Number of subquantizers mismatch: codes have {m}, centroids have {cm}")

    dimension = m * dsub
    reconstructed = np.empty((n, dimension), dtype=np.float32)

    for sub_idx in range(m):
        start_col = sub_idx * dsub
        end_col = start_col + dsub
        sub_codes = codes[:, sub_idx]
        reconstructed[:, start_col:end_col] = centroids[sub_idx, sub_codes, :]

    return reconstructed


class PQReconstructor:
    """Manages PQ vector reconstruction and error measurement."""

    def __init__(self, centroids: np.ndarray):
        if centroids.ndim != 3:
            raise ValueError(f"Centroids must be (M, Ks, dsub), got {centroids.shape}")
        self.centroids = centroids.astype(np.float32)
        self.num_subquantizers = centroids.shape[0]
        self.subvector_dim = centroids.shape[2]
        self.dimension = self.num_subquantizers * self.subvector_dim

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        """Reconstructs float32 vectors from PQ codes."""
        return reconstruct_vectors_numpy(codes, self.centroids)

    def evaluate_reconstruction_error(
        self,
        original_vectors: np.ndarray,
        codes: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluates reconstruction error on the provided dataset.
        Computes MSE, RMSE, relative L2 error, and per-vector distributions.
        """
        if original_vectors.shape[0] != codes.shape[0]:
            raise ValueError(
                f"Count mismatch: original={original_vectors.shape[0]}, codes={codes.shape[0]}"
            )

        t0 = time.perf_counter()
        reconstructed = self.reconstruct(codes)
        reconstruction_time = time.perf_counter() - t0

        # Validate no NaN/Inf
        if np.isnan(reconstructed).any():
            raise ValueError("Reconstructed vectors contain NaN!")
        if np.isinf(reconstructed).any():
            raise ValueError("Reconstructed vectors contain Infinite values!")

        # Element-wise diff
        diff = original_vectors - reconstructed

        # Mean Squared Error (per-dimension element-wise)
        mse = float(np.mean(diff ** 2))
        rmse = float(np.sqrt(mse))

        # Relative Frobenius norm error
        norm_orig = float(np.linalg.norm(original_vectors))
        norm_diff = float(np.linalg.norm(diff))
        relative_l2_error = norm_diff / norm_orig if norm_orig > 0 else 0.0

        # Per-vector L2 squared errors
        per_vector_l2_sq = np.sum(diff ** 2, axis=1)
        per_vector_l2 = np.sqrt(per_vector_l2_sq)
        per_vector_norm = np.linalg.norm(original_vectors, axis=1)
        per_vector_rel = per_vector_l2 / np.maximum(per_vector_norm, 1e-12)

        return {
            "num_vectors_evaluated": original_vectors.shape[0],
            "dimension": self.dimension,
            "reconstruction_time_seconds": round(reconstruction_time, 4),
            "reconstruction_throughput_vps": round(original_vectors.shape[0] / max(reconstruction_time, 1e-9), 1),
            "aggregate_metrics": {
                "mse": round(mse, 6),
                "rmse": round(rmse, 6),
                "relative_l2_error": round(relative_l2_error, 6),
            },
            "per_vector_l2_error": {
                "mean": round(float(np.mean(per_vector_l2)), 6),
                "std": round(float(np.std(per_vector_l2)), 6),
                "min": round(float(np.min(per_vector_l2)), 6),
                "p50": round(float(np.percentile(per_vector_l2, 50)), 6),
                "p95": round(float(np.percentile(per_vector_l2, 95)), 6),
                "p99": round(float(np.percentile(per_vector_l2, 99)), 6),
                "max": round(float(np.max(per_vector_l2)), 6),
            },
            "per_vector_relative_error": {
                "mean": round(float(np.mean(per_vector_rel)), 6),
                "std": round(float(np.std(per_vector_rel)), 6),
                "min": round(float(np.min(per_vector_rel)), 6),
                "p50": round(float(np.percentile(per_vector_rel, 50)), 6),
                "p95": round(float(np.percentile(per_vector_rel, 95)), 6),
                "p99": round(float(np.percentile(per_vector_rel, 99)), 6),
                "max": round(float(np.max(per_vector_rel)), 6),
            },
        }


def save_reconstruction_metrics(metrics: Dict[str, Any], output_path: Path | str) -> Path:
    """Saves reconstruction metrics to JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return out_p
