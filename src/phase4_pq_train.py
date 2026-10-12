"""
Case 04 — Phase 4: Deterministic Product Quantization (PQ) Training

Trains PQ codebooks using FAISS ProductQuantizer with strictly deterministic parameters:
    Dimension D = 128
    Number of subquantizers M = 16
    Centroids per subquantizer Ks = 256 (nbits = 8)
    Subvector dimension dsub = 128 / 16 = 8
    Random Seed = 42

Output Artifacts:
    results/phase4_pq/codebooks/pq_codebooks.npy
    results/phase4_pq/codebooks/faiss_pq_quantizer.bin
    results/phase4_pq/codebooks/codebook_metadata.json
"""

from __future__ import annotations

import json
import sys
import time
import hashlib
from dataclasses import dataclass, asdict
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

try:
    import faiss
except ImportError as err:
    print(f"CRITICAL ERROR: Failed to import FAISS: {err}", file=sys.stderr)
    sys.exit(1)

from src.phase1_2_evaluation import load_fvecs, compute_sha256, EXPECTED_CHECKSUMS


@dataclass
class PQConfig:
    """Explicit PQ configuration parameters."""
    dimension: int = 128
    num_subquantizers: int = 16  # M
    centroids_per_subquantizer: int = 256  # Ks = 2^8
    nbits: int = 8
    random_seed: int = 42
    niter: int = 25
    training_sample_count: int = 100_000

    @property
    def subvector_dimension(self) -> int:
        if self.dimension % self.num_subquantizers != 0:
            raise ValueError(
                f"Dimension {self.dimension} is not divisible by num_subquantizers {self.num_subquantizers}"
            )
        return self.dimension // self.num_subquantizers

    @property
    def code_bytes_per_vector(self) -> int:
        return self.num_subquantizers

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["subvector_dimension"] = self.subvector_dimension
        d["code_bytes_per_vector"] = self.code_bytes_per_vector
        return d


class PQTrainer:
    """
    Deterministic trainer for Product Quantizer codebooks.
    Uses FAISS ProductQuantizer with pinned random seed.
    """

    def __init__(self, config: Optional[PQConfig] = None):
        self.config = config or PQConfig()
        self._validate_config()

        self.pq = faiss.ProductQuantizer(
            self.config.dimension,
            self.config.num_subquantizers,
            self.config.nbits,
        )
        self.pq.cp.seed = self.config.random_seed
        self.pq.cp.niter = self.config.niter
        self.pq.cp.verbose = False

        self.is_trained = False
        self.centroids: Optional[np.ndarray] = None
        self.training_time_seconds: float = 0.0

    def _validate_config(self) -> None:
        if self.config.dimension <= 0:
            raise ValueError(f"Dimension must be positive: {self.config.dimension}")
        if self.config.num_subquantizers <= 0:
            raise ValueError(f"M must be positive: {self.config.num_subquantizers}")
        if self.config.dimension % self.config.num_subquantizers != 0:
            raise ValueError(
                f"Dimension D={self.config.dimension} must be divisible by M={self.config.num_subquantizers}"
            )
        if self.config.centroids_per_subquantizer != (1 << self.config.nbits):
            raise ValueError(
                f"Ks={self.config.centroids_per_subquantizer} does not match 2^nbits (2^{self.config.nbits})"
            )

    def train(self, base_vectors: np.ndarray) -> np.ndarray:
        """
        Trains the PQ codebook on base vectors.
        Args:
            base_vectors: float32 array of shape (N, dimension)
        Returns:
            centroids array of shape (M, Ks, dsub) float32
        """
        if not isinstance(base_vectors, np.ndarray):
            raise TypeError(f"Expected numpy ndarray, got {type(base_vectors)}")
        if base_vectors.dtype != np.float32:
            base_vectors = base_vectors.astype(np.float32)

        n_samples, d = base_vectors.shape
        if d != self.config.dimension:
            raise ValueError(f"Vector dimension {d} does not match config {self.config.dimension}")
        if n_samples < self.config.centroids_per_subquantizer * 39:
            # FAISS recommends at least 39 * Ks points for k-means
            pass

        t0 = time.perf_counter()
        self.pq.train(base_vectors)
        self.training_time_seconds = time.perf_counter() - t0
        self.is_trained = True

        # Extract centroids into numpy array: shape (M, Ks, dsub)
        raw_centroids = faiss.vector_to_array(self.pq.centroids)
        expected_elements = self.config.num_subquantizers * self.config.centroids_per_subquantizer * self.config.subvector_dimension
        if len(raw_centroids) != expected_elements:
            raise ValueError(
                f"Centroids array size {len(raw_centroids)} does not match expected {expected_elements}"
            )

        self.centroids = raw_centroids.reshape(
            self.config.num_subquantizers,
            self.config.centroids_per_subquantizer,
            self.config.subvector_dimension,
        ).astype(np.float32)

        # Validate numerical integrity
        if np.isnan(self.centroids).any():
            raise ValueError("PQ Codebooks contain NaN values!")
        if np.isinf(self.centroids).any():
            raise ValueError("PQ Codebooks contain Infinite values!")

        return self.centroids

    def save_artifacts(self, output_dir: Path | str) -> Dict[str, Path]:
        """
        Saves trained codebooks and metadata.
        """
        if not self.is_trained or self.centroids is None:
            raise RuntimeError("Cannot save artifacts: PQ quantizer has not been trained yet.")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save portable numpy centroids array
        codebooks_npy_path = out_dir / "pq_codebooks.npy"
        np.save(codebooks_npy_path, self.centroids)

        # 2. Save serialized FAISS quantizer
        faiss_bin_path = out_dir / "faiss_pq_quantizer.bin"
        faiss.write_ProductQuantizer(self.pq, str(faiss_bin_path))

        # Compute cryptographic checksums
        npy_hash = compute_sha256(codebooks_npy_path)
        bin_hash = compute_sha256(faiss_bin_path)

        # 3. Save metadata JSON
        metadata = {
            "component": "Phase 4 PQ Codebooks",
            "configuration": self.config.to_dict(),
            "environment": {
                "python_version": sys.version,
                "numpy_version": np.__version__,
                "faiss_version": faiss.__version__,
            },
            "training_stats": {
                "training_time_seconds": round(self.training_time_seconds, 4),
                "total_centroids": self.config.num_subquantizers * self.config.centroids_per_subquantizer,
                "total_centroid_floats": int(self.centroids.size),
                "codebook_memory_bytes": int(self.centroids.nbytes),
                "centroid_min": float(self.centroids.min()),
                "centroid_max": float(self.centroids.max()),
                "centroid_mean": float(self.centroids.mean()),
                "centroid_std": float(self.centroids.std()),
            },
            "artifacts": {
                "pq_codebooks_npy": {
                    "file": codebooks_npy_path.name,
                    "sha256": npy_hash,
                    "bytes": codebooks_npy_path.stat().st_size,
                    "shape": list(self.centroids.shape),
                    "dtype": str(self.centroids.dtype),
                },
                "faiss_pq_quantizer_bin": {
                    "file": faiss_bin_path.name,
                    "sha256": bin_hash,
                    "bytes": faiss_bin_path.stat().st_size,
                },
            },
        }

        meta_path = out_dir / "codebook_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "codebooks_npy": codebooks_npy_path,
            "faiss_bin": faiss_bin_path,
            "metadata_json": meta_path,
        }


def load_trained_codebooks(codebooks_dir: Path | str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads saved codebooks and metadata from disk."""
    cb_dir = Path(codebooks_dir)
    npy_path = cb_dir / "pq_codebooks.npy"
    meta_path = cb_dir / "codebook_metadata.json"

    if not npy_path.exists():
        raise FileNotFoundError(f"Codebooks file not found: {npy_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Codebook metadata not found: {meta_path}")

    centroids = np.load(npy_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Verify checksum
    expected_hash = meta["artifacts"]["pq_codebooks_npy"]["sha256"]
    actual_hash = compute_sha256(npy_path)
    if actual_hash != expected_hash:
        raise ValueError(f"Codebook hash mismatch! Expected {expected_hash}, got {actual_hash}")

    return centroids, meta
